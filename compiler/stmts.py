from __future__ import annotations

from typing import Any
import emmy.compiler.ir.expr as expr
from .context import CodegenContext
from .expr import lower_expr, float_to_ptx_hex


class StmtLowerer:
    def __init__(
        self,
        ctx: CodegenContext,
        buffer_shapes: dict[str, list[int]],
        smem_shapes: dict[str, list[int]],
        smem_offsets: dict[str, int],
        dynamic_smem: bool,
    ) -> None:
        self.ctx = ctx
        self.buffer_shapes = buffer_shapes
        self.smem_shapes = smem_shapes
        self.smem_offsets = smem_offsets
        self.dynamic_smem = dynamic_smem
        self.init_vars: set[str] = set()

    def _init_accumulators_for_axis(self, axis_name: str, body: Any) -> None:
        ctx = self.ctx
        for s in body:
            s_type = type(s).__name__
            if s_type == "Accum":
                if s.name in self.init_vars:
                    continue
                if s.axes and axis_name in s.axes:
                    if s.name not in ctx.var_map:
                        reg = ctx.alloc_f32()
                        ctx.bind_var(s.name, reg, "f32")
                    else:
                        reg, _ = ctx.get_var(s.name)

                    op_name = getattr(s.op, "name", str(s.op))
                    if op_name == "maximum":
                        ctx.emit(f"mov.f32 {reg}, {float_to_ptx_hex(-1e30)};")
                    else:
                        ctx.emit(f"mov.f32 {reg}, 0f00000000;")
            if hasattr(s, "body") and s.body:
                self._init_accumulators_for_axis(axis_name, s.body)

    def lower_statements(self, statements: Any) -> None:
        for s in statements:
            if type(s).__name__ == "Init":
                self.init_vars.add(s.name)
        for s in statements:
            self.lower_statement(s)

    def _calc_flat_index(self, indices: tuple[Any, ...], shape: list[int]) -> str:
        ctx = self.ctx
        if not indices:
            r = ctx.alloc_u32()
            ctx.emit(f"mov.u32 {r}, 0;")
            return r

        strides = [1] * len(shape)
        for i in range(len(shape) - 2, -1, -1):
            strides[i] = strides[i + 1] * shape[i + 1]

        flat_terms: list[str] = []
        for idx_expr, stride in zip(indices, strides):
            idx_reg, _ = lower_expr(ctx, idx_expr)
            if stride == 1:
                flat_terms.append(idx_reg)
            else:
                s_reg = ctx.alloc_u32()
                ctx.emit(f"mul.lo.s32 {s_reg}, {idx_reg}, {stride};")
                flat_terms.append(s_reg)

        if not flat_terms:
            r = ctx.alloc_u32()
            ctx.emit(f"mov.u32 {r}, 0;")
            return r

        cur = flat_terms[0]
        for next_term in flat_terms[1:]:
            dst = ctx.alloc_u32()
            ctx.emit(f"add.s32 {dst}, {cur}, {next_term};")
            cur = dst
        return cur

    def _get_buffer_addr(self, buf_name: str, flat_idx_reg: str) -> str:
        ctx = self.ctx
        base_reg, _ = ctx.get_var(buf_name)
        addr_reg = ctx.alloc_u64()
        ctx.emit(f"mad.wide.s32 {addr_reg}, {flat_idx_reg}, 4, {base_reg};")
        return addr_reg

    def _get_smem_addr(self, smem_name: str, flat_idx_reg: str) -> str:
        ctx = self.ctx
        byte_off = ctx.alloc_u32()
        ctx.emit(f"shl.b32 {byte_off}, {flat_idx_reg}, 2;")

        if self.dynamic_smem:
            pool_base = ctx.alloc_u32()
            ctx.emit(f"mov.u32 {pool_base}, _smem_pool;")
            offset = self.smem_offsets.get(smem_name, 0)
            if offset > 0:
                base_reg = ctx.alloc_u32()
                ctx.emit(f"add.s32 {base_reg}, {pool_base}, {offset};")
            else:
                base_reg = pool_base
        else:
            base_reg = ctx.alloc_u32()
            ctx.emit(f"mov.u32 {base_reg}, {smem_name};")

        addr_reg = ctx.alloc_u32()
        ctx.emit(f"add.s32 {addr_reg}, {base_reg}, {byte_off};")
        return addr_reg

    def _emit_barrier(self, barrier_id: int, count: int | None, warp: bool) -> None:
        if warp:
            self.ctx.emit("bar.warp.sync 0xffffffff;")
        elif count is None:
            self.ctx.emit(f"bar.sync {barrier_id};")
        else:
            self.ctx.emit(f"bar.sync {barrier_id}, {count};")

    @staticmethod
    def _require_f32(dtype: Any, what: str) -> None:
        name = getattr(dtype, "name", dtype)
        if name not in (None, "f32", "float"):
            raise NotImplementedError(f"{what}: dtype {name!r} (only f32)")

    @staticmethod
    def _require_no_swizzle(swizzle: Any, what: str) -> None:
        if str(swizzle) not in ("NONE", "None", "SwizzleKind.NONE"):
            raise NotImplementedError(f"{what}: swizzle {swizzle!r}")

    def _lower_init(self, s: Any) -> None:
        ctx = self.ctx
        reg = ctx.alloc_f32()
        if isinstance(s.identity, (int, float)):
            ctx.emit(f"mov.f32 {reg}, {float_to_ptx_hex(s.identity)};")
        else:
            ctx.emit(f"mov.f32 {reg}, 0f00000000;")
        ctx.bind_var(s.name, reg, "f32")

    def _lower_let(self, s: Any) -> None:
        reg, dtype = lower_expr(self.ctx, s.value)
        self.ctx.bind_var(s.name, reg, dtype)

    def _assign_dst(self, s: Any) -> str:
        if s.name in self.init_vars and s.name in self.ctx.var_map:
            return self.ctx.get_var(s.name)[0]
        return self.ctx.alloc_f32()

    def _lower_assign(self, s: Any) -> None:
        ctx = self.ctx
        op_name = getattr(s.op, "name", str(s.op))
        arg_regs = [ctx.get_var(a)[0] for a in s.args]

        if op_name == "copy":
            dst = self._assign_dst(s)
            ctx.emit(f"mov.f32 {dst}, {arg_regs[0]};")
            ctx.bind_var(s.name, dst, "f32")
        elif op_name == "add":
            dst = self._assign_dst(s)
            ctx.emit(f"add.f32 {dst}, {arg_regs[0]}, {arg_regs[1]};")
            ctx.bind_var(s.name, dst, "f32")
        elif op_name == "subtract":
            dst = self._assign_dst(s)
            ctx.emit(f"sub.f32 {dst}, {arg_regs[0]}, {arg_regs[1]};")
            ctx.bind_var(s.name, dst, "f32")
        elif op_name == "multiply":
            dst = self._assign_dst(s)
            ctx.emit(f"mul.f32 {dst}, {arg_regs[0]}, {arg_regs[1]};")
            ctx.bind_var(s.name, dst, "f32")
        elif op_name == "divide":
            dst = self._assign_dst(s)
            ctx.emit(f"div.rn.f32 {dst}, {arg_regs[0]}, {arg_regs[1]};")
            ctx.bind_var(s.name, dst, "f32")
        elif op_name == "reciprocal":
            dst = self._assign_dst(s)
            ctx.emit(f"rcp.rn.f32 {dst}, {arg_regs[0]};")
            ctx.bind_var(s.name, dst, "f32")
        elif op_name == "sqrt":
            dst = self._assign_dst(s)
            ctx.emit(f"sqrt.rn.f32 {dst}, {arg_regs[0]};")
            ctx.bind_var(s.name, dst, "f32")
        elif op_name == "rsqrt":
            dst = self._assign_dst(s)
            ctx.emit(f"rsqrt.approx.f32 {dst}, {arg_regs[0]};")
            ctx.bind_var(s.name, dst, "f32")
        elif op_name in ("exp", "exp_fast"):
            dst = self._assign_dst(s)
            t = ctx.alloc_f32()
            # 0f3fb8aa3b is log2(e)
            ctx.emit(f"mul.f32 {t}, {arg_regs[0]}, 0f3fb8aa3b;")
            ctx.emit(f"ex2.approx.f32 {dst}, {t};")
            ctx.bind_var(s.name, dst, "f32")
        elif op_name == "maximum":
            dst = self._assign_dst(s)
            ctx.emit(f"max.f32 {dst}, {arg_regs[0]}, {arg_regs[1]};")
            ctx.bind_var(s.name, dst, "f32")
        elif op_name == "minimum":
            dst = self._assign_dst(s)
            ctx.emit(f"min.f32 {dst}, {arg_regs[0]}, {arg_regs[1]};")
            ctx.bind_var(s.name, dst, "f32")
        elif op_name == "abs":
            dst = self._assign_dst(s)
            ctx.emit(f"abs.f32 {dst}, {arg_regs[0]};")
            ctx.bind_var(s.name, dst, "f32")
        elif op_name in ("negative", "neg"):
            dst = self._assign_dst(s)
            ctx.emit(f"neg.f32 {dst}, {arg_regs[0]};")
            ctx.bind_var(s.name, dst, "f32")
        elif op_name == "relu":
            dst = self._assign_dst(s)
            ctx.emit(f"max.f32 {dst}, {arg_regs[0]}, 0f00000000;")
            ctx.bind_var(s.name, dst, "f32")
        else:
            raise NotImplementedError(f"unsupported assign op: {op_name}")

    def _lower_accum(self, s: Any) -> None:
        ctx = self.ctx
        op_name = getattr(s.op, "name", str(s.op))
        val_reg, _ = ctx.get_var(s.value) if isinstance(s.value, str) else lower_expr(ctx, s.value)

        lhs_name = s.base if s.base is not None else s.name
        lhs_reg, _ = ctx.get_var(lhs_name) if isinstance(lhs_name, str) else lower_expr(ctx, lhs_name)

        if s.name in ctx.var_map:
            acc_reg, _ = ctx.get_var(s.name)
        else:
            acc_reg = ctx.alloc_f32()
            ctx.bind_var(s.name, acc_reg, "f32")

        if op_name == "add":
            ctx.emit(f"add.f32 {acc_reg}, {lhs_reg}, {val_reg};")
        elif op_name == "maximum":
            ctx.emit(f"max.f32 {acc_reg}, {lhs_reg}, {val_reg};")
        else:
            raise NotImplementedError(f"unsupported accum op: {op_name}")

    def _lower_load(self, s: Any) -> None:
        ctx = self.ctx
        self._require_f32(s.dtype, f"Load {s.input}")
        is_smem = s.input in self.smem_shapes
        shape = self.smem_shapes[s.input] if is_smem else self.buffer_shapes[s.input]
        flat_idx = self._calc_flat_index(s.index, shape)

        space = "shared" if is_smem else "global"
        addr = self._get_smem_addr(s.input, flat_idx) if is_smem else self._get_buffer_addr(s.input, flat_idx)

        width = len(s.names)
        if width == 1:
            dst = ctx.alloc_f32()
            ctx.emit(f"ld.{space}.f32 {dst}, [{addr}];")
            ctx.bind_var(s.names[0], dst, "f32")
        elif width in (2, 4):
            dsts = [ctx.alloc_f32() for _ in range(width)]
            vec = f"{{{', '.join(dsts)}}}"
            ctx.emit(f"ld.{space}.v{width}.f32 {vec}, [{addr}];")
            for name, d in zip(s.names, dsts):
                ctx.bind_var(name, d, "f32")
        else:
            raise NotImplementedError(f"unsupported load width: {width}")

    def _lower_write(self, s: Any) -> None:
        ctx = self.ctx
        self._require_f32(s.value_dtype, f"Write {s.output}")
        self._require_no_swizzle(s.swizzle, f"Write {s.output}")
        is_smem = s.output in self.smem_shapes
        shape = self.smem_shapes[s.output] if is_smem else self.buffer_shapes[s.output]
        flat_idx = self._calc_flat_index(s.index, shape)

        val_regs: list[str] = []
        for v in s.values:
            if isinstance(v, str):
                val_regs.append(ctx.get_var(v)[0])
            else:
                vr, _ = lower_expr(ctx, v)
                val_regs.append(vr)

        space = "shared" if is_smem else "global"
        addr = self._get_smem_addr(s.output, flat_idx) if is_smem else self._get_buffer_addr(s.output, flat_idx)

        if not is_smem and getattr(s, "atomic", False):
            ctx.emit(f"red.global.add.f32 [{addr}], {val_regs[0]};")
            return

        width = len(val_regs)
        if width == 1:
            ctx.emit(f"st.{space}.f32 [{addr}], {val_regs[0]};")
        elif width in (2, 4):
            vec = f"{{{', '.join(val_regs)}}}"
            ctx.emit(f"st.{space}.v{width}.f32 [{addr}], {vec};")
        else:
            raise NotImplementedError(f"unsupported write width: {width}")

    def _lower_strided_loop(self, s: Any) -> None:
        ctx = self.ctx
        if isinstance(s.start, expr.Expr):
            start_reg, _ = lower_expr(ctx, s.start)
        else:
            start_reg = ctx.alloc_u32()
            val = int(s.start.value) if hasattr(s.start, "value") else int(s.start)
            ctx.emit(f"mov.u32 {start_reg}, {val};")

        var_reg = ctx.alloc_u32()
        ctx.emit(f"mov.u32 {var_reg}, {start_reg};")
        ctx.bind_var(s.axis.name, var_reg, "s32")

        self._init_accumulators_for_axis(s.axis.name, s.body)

        lbl_start = ctx.new_label("LOOP")
        lbl_end = ctx.new_label("LOOP_END")
        ctx.emit_label(lbl_start)

        end_val = s.end if s.end is not None else s.axis.extent
        if isinstance(end_val, expr.Expr):
            end_reg, _ = lower_expr(ctx, end_val)
        else:
            end_reg = ctx.alloc_u32()
            val = int(end_val.value) if hasattr(end_val, "value") else int(end_val)
            ctx.emit(f"mov.u32 {end_reg}, {val};")

        p_exit = ctx.alloc_pred()
        ctx.emit(f"setp.ge.s32 {p_exit}, {var_reg}, {end_reg};")
        ctx.emit(f"@{p_exit} bra {lbl_end};")

        self.lower_statements(s.body)

        if isinstance(s.step, expr.Expr):
            step_reg, _ = lower_expr(ctx, s.step)
            ctx.emit(f"add.s32 {var_reg}, {var_reg}, {step_reg};")
        else:
            val = int(s.step.value) if hasattr(s.step, "value") else int(s.step)
            ctx.emit(f"add.s32 {var_reg}, {var_reg}, {val};")

        ctx.emit(f"bra {lbl_start};")
        ctx.emit_label(lbl_end)

    def _lower_loop(self, s: Any) -> None:
        ctx = self.ctx
        extent_val = s.axis.extent
        extent = int(extent_val.value) if hasattr(extent_val, "value") else int(extent_val)

        var_reg = ctx.alloc_u32()
        ctx.emit(f"mov.u32 {var_reg}, 0;")
        ctx.bind_var(s.axis.name, var_reg, "s32")

        self._init_accumulators_for_axis(s.axis.name, s.body)

        lbl_start = ctx.new_label("LOOP")
        lbl_end = ctx.new_label("LOOP_END")
        ctx.emit_label(lbl_start)

        p_exit = ctx.alloc_pred()
        ctx.emit(f"setp.ge.s32 {p_exit}, {var_reg}, {extent};")
        ctx.emit(f"@{p_exit} bra {lbl_end};")

        self.lower_statements(s.body)

        ctx.emit(f"add.s32 {var_reg}, {var_reg}, 1;")
        ctx.emit(f"bra {lbl_start};")
        ctx.emit_label(lbl_end)

    def _lower_cond(self, s: Any) -> None:
        ctx = self.ctx
        cond_reg, cond_type = lower_expr(ctx, s.cond)
        if cond_type != "pred":
            p = ctx.alloc_pred()
            ctx.emit(f"setp.ne.s32 {p}, {cond_reg}, 0;")
            cond_reg = p

        if s.else_body:
            lbl_else = ctx.new_label("ELSE")
            lbl_end = ctx.new_label("END")
            ctx.emit(f"@!{cond_reg} bra {lbl_else};")
            self.lower_statements(s.body)
            ctx.emit(f"bra {lbl_end};")
            ctx.emit_label(lbl_else)
            self.lower_statements(s.else_body)
            ctx.emit_label(lbl_end)
        else:
            lbl_end = ctx.new_label("END")
            ctx.emit(f"@!{cond_reg} bra {lbl_end};")
            self.lower_statements(s.body)
            ctx.emit_label(lbl_end)

    def _lower_warp_shuffle(self, s: Any) -> None:
        ctx = self.ctx
        length = getattr(s, "length", 32)
        step = length // 2
        deltas = []
        while step >= 1:
            deltas.append(step)
            step //= 2

        for delta in deltas:
            for s_name, sb_name in zip(s.state, s.state_b):
                src_reg, src_dtype = ctx.get_var(s_name)
                sb_reg = ctx.alloc_typed(src_dtype)
                ctx.emit(f"shfl.sync.bfly.b32 {sb_reg}, {src_reg}, {delta}, 0x1f, 0xffffffff;")
                ctx.bind_var(sb_name, sb_reg, src_dtype)
            self.lower_statements(s.combine_states)

    def _lower_tree_halve(self, s: Any) -> None:
        ctx = self.ctx
        if s.inner is not None:
            raise NotImplementedError(f"TreeHalve.inner={s.inner!r} (segmented tree)")
        tid_reg, _ = ctx.get_var(s.tid_var)
        cur = s.length

        while cur > 1:
            stride = (cur + 1) // 2
            p_active = ctx.alloc_pred()
            ctx.emit(f"setp.lt.s32 {p_active}, {tid_reg}, {cur - stride};")

            lbl_skip = ctx.new_label("TREE_SKIP")
            ctx.emit(f"@!{p_active} bra {lbl_skip};")

            partner_tid = ctx.alloc_u32()
            ctx.emit(f"add.s32 {partner_tid}, {tid_reg}, {stride};")

            for buf_name, s_name, sb_name in zip(s.bufs, s.state, s.state_b):
                addr_self = self._get_smem_addr(buf_name, tid_reg)
                addr_partner = self._get_smem_addr(buf_name, partner_tid)

                reg_self = ctx.alloc_f32()
                reg_partner = ctx.alloc_f32()
                ctx.emit(f"ld.shared.f32 {reg_self}, [{addr_self}];")
                ctx.emit(f"ld.shared.f32 {reg_partner}, [{addr_partner}];")

                ctx.bind_var(s_name, reg_self, "f32")
                ctx.bind_var(sb_name, reg_partner, "f32")

            self.lower_statements(s.combine_states)

            for buf_name, s_name in zip(s.bufs, s.state):
                addr_self = self._get_smem_addr(buf_name, tid_reg)
                res_reg, _ = ctx.get_var(s_name)
                ctx.emit(f"st.shared.f32 [{addr_self}], {res_reg};")

            ctx.emit_label(lbl_skip)
            self._emit_barrier(s.barrier_id, s.barrier_count, False)
            cur = stride

        zero_reg = ctx.alloc_u32()
        ctx.emit(f"mov.u32 {zero_reg}, 0;")
        for buf_name, s_name in zip(s.bufs, s.state):
            addr_zero = self._get_smem_addr(buf_name, zero_reg)
            bcast_reg = ctx.alloc_f32()
            ctx.emit(f"ld.shared.f32 {bcast_reg}, [{addr_zero}];")
            ctx.bind_var(s_name, bcast_reg, "f32")
        self._emit_barrier(s.barrier_id, s.barrier_count, False)

    def _lower_cp_async_copy(self, s: Any) -> None:
        ctx = self.ctx
        self._require_no_swizzle(s.swizzle, f"CpAsyncCopy {s.smem}")
        src_shape = self.buffer_shapes[s.src]
        src_flat = self._calc_flat_index(s.src_index, src_shape)
        src_addr = self._get_buffer_addr(s.src, src_flat)

        smem_shape = self.smem_shapes[s.smem]
        smem_flat = self._calc_flat_index(s.smem_index, smem_shape)
        if s.lane_index is not None:
            lane_flat = self._calc_flat_index(s.lane_index, smem_shape[-len(s.lane_index):])
            comb_flat = ctx.alloc_u32()
            ctx.emit(f"add.s32 {comb_flat}, {smem_flat}, {lane_flat};")
            smem_flat = comb_flat

        smem_addr = self._get_smem_addr(s.smem, smem_flat)
        if s.nbytes == 16:
            ctx.emit(f"cp.async.cg.shared.global [{smem_addr}], [{src_addr}], 16;")
        else:
            ctx.emit(f"cp.async.ca.shared.global [{smem_addr}], [{src_addr}], {s.nbytes};")

    def lower_statement(self, s: Any) -> None:
        s_type = type(s).__name__
        handlers = {
            "Init": self._lower_init,
            "Let": self._lower_let,
            "Assign": self._lower_assign,
            "Accum": self._lower_accum,
            "Load": self._lower_load,
            "Write": self._lower_write,
            "StridedLoop": self._lower_strided_loop,
            "Loop": self._lower_loop,
            "Cond": self._lower_cond,
            "Sync": lambda st: self._emit_barrier(st.barrier_id, st.count, st.warp),
            "Smem": lambda _: None,
            "WarpShuffle": self._lower_warp_shuffle,
            "TreeHalve": self._lower_tree_halve,
            "CpAsyncCopy": self._lower_cp_async_copy,
            "CpAsyncCommit": lambda _: self.ctx.emit("cp.async.commit_group;"),
            "CpAsyncWait": lambda stmt: self.ctx.emit(f"cp.async.wait_group {stmt.group};"),
        }

        handler = handlers.get(s_type)
        if handler is None:
            raise NotImplementedError(f"unsupported statement type: {s_type}")
        handler(s)

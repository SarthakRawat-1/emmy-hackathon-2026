from __future__ import annotations

from typing import Any
import emmy.compiler.ir.expr as expr
from .context import CodegenContext
from .expr import lower_expr, float_to_ptx_hex, calc_flat_index


class StmtLowerer:
    IDENTITY_MAP = {
        "add": 0.0,
        "sum": 0.0,
        "maximum": -1e30,
        "amax": -1e30,
        "minimum": 1e30,
        "multiply": 1.0,
        "prod": 1.0,
    }

    def __init__(
        self,
        ctx: CodegenContext,
        buffer_shapes: dict[str, list[int]],
        smem_shapes: dict[str, list[int]],
        smem_offsets: dict[str, int],
        dynamic_smem: bool,
        buffer_elem_bytes: dict[str, int] | None = None,
    ) -> None:
        self.ctx = ctx
        self.buffer_shapes = buffer_shapes
        self.smem_shapes = smem_shapes
        self.smem_offsets = smem_offsets
        self.dynamic_smem = dynamic_smem
        self.buffer_elem_bytes = buffer_elem_bytes or {}
        self.init_vars: set[str] = set()
        self.handlers = {
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
            "Select": self._lower_select,
            "ZeroPrologue": self._lower_zero_prologue,
            "CpAsyncCopy": self._lower_cp_async_copy,
            "CpAsyncCommit": lambda _: self.ctx.emit("cp.async.commit_group;"),
            "CpAsyncWait": lambda stmt: self.ctx.emit(f"cp.async.wait_group {stmt.group};"),
        }

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
                    identity_val = self.IDENTITY_MAP.get(op_name, 0.0)
                    ctx.emit(f"mov.f32 {reg}, {float_to_ptx_hex(identity_val)};")
            if hasattr(s, "body") and s.body:
                self._init_accumulators_for_axis(axis_name, s.body)

    def lower_statements(self, statements: Any) -> None:
        for s in statements:
            if type(s).__name__ == "Init":
                self.init_vars.add(s.name)
        for s in statements:
            self.lower_statement(s)

    def _calc_flat_index(self, indices: tuple[Any, ...], shape: list[int]) -> str:
        return calc_flat_index(self.ctx, indices, shape)

    def _get_buffer_addr(self, buf_name: str, flat_idx_reg: str) -> str:
        ctx = self.ctx
        base_reg, _ = ctx.get_var(buf_name)
        elem_bytes = self.buffer_elem_bytes.get(buf_name, 4)
        addr_reg = ctx.alloc_u64()
        ctx.emit(f"mad.wide.s32 {addr_reg}, {flat_idx_reg}, {elem_bytes}, {base_reg};")
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
    def _resolve_dtype(dtype: Any) -> tuple[str, int]:
        dt_str = getattr(dtype, "name", str(dtype) if dtype is not None else "f32")
        dtype_map = {
            "f32": ("f32", 4),
            "float": ("f32", 4),
            "float32": ("f32", 4),
            "s32": ("s32", 4),
            "int32": ("s32", 4),
            "int": ("s32", 4),
            "u32": ("u32", 4),
            "uint32": ("u32", 4),
            "f16": ("b16", 2),
            "float16": ("b16", 2),
            "half": ("b16", 2),
            "bf16": ("b16", 2),
            "bfloat16": ("b16", 2),
            "s64": ("s64", 8),
            "int64": ("s64", 8),
            "u64": ("u64", 8),
        }
        return dtype_map.get(dt_str.lower(), ("f32", 4))

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

    UNARY_OPS = {
        "copy": "mov.f32",
        "reciprocal": "rcp.rn.f32",
        "sqrt": "sqrt.rn.f32",
        "rsqrt": "rsqrt.approx.f32",
        "abs": "abs.f32",
        "negative": "neg.f32",
        "neg": "neg.f32",
        "sin": "sin.approx.f32",
        "cos": "cos.approx.f32",
        "tanh": "tanh.approx.f32",
    }

    BINARY_OPS = {
        "add": "add.f32",
        "subtract": "sub.f32",
        "multiply": "mul.f32",
        "divide": "div.rn.f32",
        "maximum": "max.f32",
        "minimum": "min.f32",
    }

    ACCUM_OPS = {
        "add": "add.f32",
        "sum": "add.f32",
        "maximum": "max.f32",
        "amax": "max.f32",
        "minimum": "min.f32",
        "multiply": "mul.f32",
        "prod": "mul.f32",
    }

    def _assign_dst(self, s: Any) -> str:
        if s.name in self.init_vars and s.name in self.ctx.var_map:
            return self.ctx.get_var(s.name)[0]
        return self.ctx.alloc_f32()

    def _lower_assign(self, s: Any) -> None:
        ctx = self.ctx
        op_name = getattr(s.op, "name", str(s.op))
        arg_regs = [ctx.get_var(a)[0] for a in s.args]
        dst = self._assign_dst(s)

        if op_name in self.UNARY_OPS:
            ctx.emit(f"{self.UNARY_OPS[op_name]} {dst}, {arg_regs[0]};")
        elif op_name in self.BINARY_OPS:
            ctx.emit(f"{self.BINARY_OPS[op_name]} {dst}, {arg_regs[0]}, {arg_regs[1]};")
        elif op_name in ("exp", "exp_fast"):
            t = ctx.alloc_f32()
            # 0f3fb8aa3b is log2(e)
            ctx.emit(f"mul.f32 {t}, {arg_regs[0]}, 0f3fb8aa3b;")
            ctx.emit(f"ex2.approx.f32 {dst}, {t};")
        elif op_name == "log":
            t = ctx.alloc_f32()
            # lg2.approx computes log2(x); multiply by ln(2) (0f3f317218) for natural log
            ctx.emit(f"lg2.approx.f32 {t}, {arg_regs[0]};")
            ctx.emit(f"mul.f32 {dst}, {t}, 0f3f317218;")
        elif op_name == "relu":
            ctx.emit(f"max.f32 {dst}, {arg_regs[0]}, 0f00000000;")
        elif op_name == "square":
            ctx.emit(f"mul.f32 {dst}, {arg_regs[0]}, {arg_regs[0]};")
        else:
            raise NotImplementedError(f"unsupported assign op: {op_name}")

        ctx.bind_var(s.name, dst, "f32")

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

        if op_name not in self.ACCUM_OPS:
            raise NotImplementedError(f"unsupported accum op: {op_name}")
        ctx.emit(f"{self.ACCUM_OPS[op_name]} {acc_reg}, {lhs_reg}, {val_reg};")

    def _lower_load(self, s: Any) -> None:
        ctx = self.ctx
        ptx_dt, elem_bytes = self._resolve_dtype(s.dtype)
        is_smem = s.input in self.smem_shapes
        shape = self.smem_shapes[s.input] if is_smem else self.buffer_shapes[s.input]
        flat_idx = self._calc_flat_index(s.index, shape)

        space = "shared" if is_smem else "global"
        addr = self._get_smem_addr(s.input, flat_idx) if is_smem else self._get_buffer_addr(s.input, flat_idx)

        width = len(s.names)
        if ptx_dt == "b16":
            if width == 1:
                r16 = ctx.alloc_u32()
                ctx.emit(f"ld.{space}.b16 {r16}, [{addr}];")
                f32_dst = ctx.alloc_f32()
                ctx.emit(f"cvt.f32.f16 {f32_dst}, {r16};")
                ctx.bind_var(s.names[0], f32_dst, "f32")
            elif width in (2, 4):
                r16s = [ctx.alloc_u32() for _ in range(width)]
                vec = f"{{{', '.join(r16s)}}}"
                ctx.emit(f"ld.{space}.v{width}.b16 {vec}, [{addr}];")
                for name, r in zip(s.names, r16s):
                    f32_dst = ctx.alloc_f32()
                    ctx.emit(f"cvt.f32.f16 {f32_dst}, {r};")
                    ctx.bind_var(name, f32_dst, "f32")
            else:
                raise NotImplementedError(f"unsupported load width: {width}")
            return

        if width == 1:
            dst = ctx.alloc_typed(ptx_dt)
            ctx.emit(f"ld.{space}.{ptx_dt} {dst}, [{addr}];")
            ctx.bind_var(s.names[0], dst, ptx_dt)
        elif width in (2, 4):
            dsts = [ctx.alloc_typed(ptx_dt) for _ in range(width)]
            vec = f"{{{', '.join(dsts)}}}"
            ctx.emit(f"ld.{space}.v{width}.{ptx_dt} {vec}, [{addr}];")
            for name, d in zip(s.names, dsts):
                ctx.bind_var(name, d, ptx_dt)
        else:
            raise NotImplementedError(f"unsupported load width: {width}")

    def _lower_write(self, s: Any) -> None:
        ctx = self.ctx
        ptx_dt, elem_bytes = self._resolve_dtype(s.value_dtype)
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
        if ptx_dt == "b16":
            r16s = []
            for vr in val_regs:
                r16 = ctx.alloc_u32()
                ctx.emit(f"cvt.rn.f16.f32 {r16}, {vr};")
                r16s.append(r16)
            if width == 1:
                ctx.emit(f"st.{space}.b16 [{addr}], {r16s[0]};")
            elif width in (2, 4):
                vec = f"{{{', '.join(r16s)}}}"
                ctx.emit(f"st.{space}.v{width}.b16 [{addr}], {vec};")
            else:
                raise NotImplementedError(f"unsupported write width: {width}")
            return

        width = len(val_regs)
        if width == 1:
            ctx.emit(f"st.{space}.{ptx_dt} [{addr}], {val_regs[0]};")
        elif width in (2, 4):
            vec = f"{{{', '.join(val_regs)}}}"
            ctx.emit(f"st.{space}.v{width}.{ptx_dt} [{addr}], {vec};")
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
        tid_reg, _ = ctx.get_var(s.tid_var)
        cur = s.length

        inner_scale = None
        inner_reg = None
        if s.inner is not None:
            iv_name, inner_scale = s.inner
            inner_reg, _ = ctx.get_var(iv_name)

        while cur > 1:
            stride = (cur + 1) // 2
            p_active = ctx.alloc_pred()
            ctx.emit(f"setp.lt.s32 {p_active}, {tid_reg}, {cur - stride};")

            lbl_skip = ctx.new_label("TREE_SKIP")
            ctx.emit(f"@!{p_active} bra {lbl_skip};")

            partner_tid = ctx.alloc_u32()
            ctx.emit(f"add.s32 {partner_tid}, {tid_reg}, {stride};")

            if inner_reg is not None and inner_scale is not None:
                slot_self = ctx.alloc_u32()
                slot_partner = ctx.alloc_u32()
                ctx.emit(f"mad.lo.s32 {slot_self}, {tid_reg}, {inner_scale}, {inner_reg};")
                ctx.emit(f"mad.lo.s32 {slot_partner}, {partner_tid}, {inner_scale}, {inner_reg};")
            else:
                slot_self = tid_reg
                slot_partner = partner_tid

            for buf_name, s_name, sb_name in zip(s.bufs, s.state, s.state_b):
                addr_self = self._get_smem_addr(buf_name, slot_self)
                addr_partner = self._get_smem_addr(buf_name, slot_partner)

                reg_self = ctx.alloc_f32()
                reg_partner = ctx.alloc_f32()
                ctx.emit(f"ld.shared.f32 {reg_self}, [{addr_self}];")
                ctx.emit(f"ld.shared.f32 {reg_partner}, [{addr_partner}];")

                ctx.bind_var(s_name, reg_self, "f32")
                ctx.bind_var(sb_name, reg_partner, "f32")

            self.lower_statements(s.combine_states)

            for buf_name, s_name in zip(s.bufs, s.state):
                addr_self = self._get_smem_addr(buf_name, slot_self)
                res_reg, _ = ctx.get_var(s_name)
                ctx.emit(f"st.shared.f32 [{addr_self}], {res_reg};")

            ctx.emit_label(lbl_skip)
            self._emit_barrier(s.barrier_id, s.barrier_count, False)
            cur = stride

        if inner_reg is not None:
            root_slot = inner_reg
        else:
            zero_reg = ctx.alloc_u32()
            ctx.emit(f"mov.u32 {zero_reg}, 0;")
            root_slot = zero_reg

        for buf_name, s_name in zip(s.bufs, s.state):
            addr_root = self._get_smem_addr(buf_name, root_slot)
            bcast_reg = ctx.alloc_f32()
            ctx.emit(f"ld.shared.f32 {bcast_reg}, [{addr_root}];")
            ctx.bind_var(s_name, bcast_reg, "f32")
        self._emit_barrier(s.barrier_id, s.barrier_count, False)

    def _lower_select(self, s: Any) -> None:
        branches = list(s.branches)
        res_expr: expr.Expr = expr.CastExpr("float", expr.Var(branches[-1].value))
        for b in reversed(branches[:-1]):
            res_expr = expr.TernaryExpr(cond=b.select, if_true=expr.CastExpr("float", expr.Var(b.value)), if_false=res_expr)
        reg, dtype = lower_expr(self.ctx, res_expr)
        self.ctx.bind_var(s.name, reg, dtype)

    def _lower_zero_prologue(self, s: Any) -> None:
        ctx = self.ctx
        base_addr, _ = ctx.get_var(s.dst)
        lbl_skip = ctx.new_label("ZP_SKIP")
        p_cta = ctx.alloc_pred()
        r_ctaid = ctx.alloc_u32()
        ctx.emit(f"mov.u32 {r_ctaid}, %ctaid.x;")
        ctx.emit(f"setp.ne.s32 {p_cta}, {r_ctaid}, 0;")
        ctx.emit(f"@{p_cta} bra {lbl_skip};")

        r_zi = ctx.alloc_u32()
        ctx.emit(f"mov.u32 {r_zi}, %tid.x;")
        lbl_loop = ctx.new_label("ZP_LOOP")
        lbl_end = ctx.new_label("ZP_END")
        ctx.emit_label(lbl_loop)

        p_exit = ctx.alloc_pred()
        ctx.emit(f"setp.ge.s32 {p_exit}, {r_zi}, {s.words};")
        ctx.emit(f"@{p_exit} bra {lbl_end};")

        addr = ctx.alloc_u64()
        ctx.emit(f"mad.wide.s32 {addr}, {r_zi}, 4, {base_addr};")
        ctx.emit(f"st.global.u32 [{addr}], 0;")

        r_ntid = ctx.alloc_u32()
        ctx.emit(f"mov.u32 {r_ntid}, %ntid.x;")
        ctx.emit(f"add.s32 {r_zi}, {r_zi}, {r_ntid};")
        ctx.emit(f"bra {lbl_loop};")
        ctx.emit_label(lbl_end)
        ctx.emit_label(lbl_skip)

    def _lower_cp_async_copy(self, s: Any) -> None:
        ctx = self.ctx
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
        handler = self.handlers.get(s_type)
        if handler is None:
            raise NotImplementedError(f"unsupported statement type: {s_type}")
        handler(s)

from __future__ import annotations

import struct
from typing import Any
import emmy.compiler.ir.expr as expr
from .context import CodegenContext


def float_to_ptx_hex(val: float) -> str:
    b = struct.pack(">f", float(val))
    return f"0f{b.hex()}"


def as_f32(ctx: CodegenContext, reg: str, dtype: str) -> str:
    if dtype in ("f32", "float"):
        return reg
    pf = ctx.alloc_f32()
    ctx.emit(f"cvt.rn.f32.s32 {pf}, {reg};")
    return pf


def calc_flat_index(ctx: CodegenContext, indices: tuple[Any, ...], shape: list[int]) -> str:
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


def lower_expr(ctx: CodegenContext, e: expr.Expr) -> tuple[str, str]:
    if isinstance(e, expr.Var):
        return ctx.get_var(e.name)

    if isinstance(e, expr.Literal):
        if e.dtype in ("float", "f32") or isinstance(e.value, float):
            reg = ctx.alloc_f32()
            ctx.emit(f"mov.f32 {reg}, {float_to_ptx_hex(e.value)};")
            return reg, "f32"
        reg = ctx.alloc_u32()
        ctx.emit(f"mov.u32 {reg}, {int(e.value)};")
        return reg, "s32"

    if isinstance(e, expr.BinaryExpr):
        l_reg, l_type = lower_expr(ctx, e.left)
        r_reg, r_type = lower_expr(ctx, e.right)

        is_float = (l_type == "f32" or r_type == "f32")
        ptx_type = "f32" if is_float else "s32"

        if is_float:
            if e.op in ("//", "%"):
                raise TypeError(f"Operator {e.op!r} is not supported on float operands")
            l_reg = as_f32(ctx, l_reg, l_type)
            r_reg = as_f32(ctx, r_reg, r_type)

        if e.op == "+":
            dst = ctx.alloc_f32() if is_float else ctx.alloc_u32()
            ctx.emit(f"add.{ptx_type} {dst}, {l_reg}, {r_reg};")
            return dst, ptx_type
        if e.op == "-":
            dst = ctx.alloc_f32() if is_float else ctx.alloc_u32()
            ctx.emit(f"sub.{ptx_type} {dst}, {l_reg}, {r_reg};")
            return dst, ptx_type
        if e.op == "*":
            dst = ctx.alloc_f32() if is_float else ctx.alloc_u32()
            op = "mul.f32" if is_float else "mul.lo.s32"
            ctx.emit(f"{op} {dst}, {l_reg}, {r_reg};")
            return dst, ptx_type
        if e.op in ("/", "//"):
            dst = ctx.alloc_f32() if (is_float and e.op == "/") else ctx.alloc_u32()
            op = "div.rn.f32" if (is_float and e.op == "/") else "div.s32"
            ctx.emit(f"{op} {dst}, {l_reg}, {r_reg};")
            return dst, ("f32" if (is_float and e.op == "/") else "s32")
        if e.op == "%":
            dst = ctx.alloc_u32()
            ctx.emit(f"rem.s32 {dst}, {l_reg}, {r_reg};")
            return dst, "s32"
        if e.op in ("<", "<=", ">", ">=", "==", "!="):
            dst = ctx.alloc_pred()
            cmp_map = {
                "<": "lt",
                "<=": "le",
                ">": "gt",
                ">=": "ge",
                "==": "eq",
                "!=": "neu" if is_float else "ne",
            }
            ctx.emit(f"setp.{cmp_map[e.op]}.{ptx_type} {dst}, {l_reg}, {r_reg};")
            return dst, "pred"
        if e.op in ("&", "&&"):
            if l_type == "pred" or r_type == "pred":
                lp = l_reg if l_type == "pred" else ctx.alloc_pred()
                if l_type != "pred":
                    ctx.emit(f"setp.ne.s32 {lp}, {l_reg}, 0;")
                rp = r_reg if r_type == "pred" else ctx.alloc_pred()
                if r_type != "pred":
                    ctx.emit(f"setp.ne.s32 {rp}, {r_reg}, 0;")
                dst = ctx.alloc_pred()
                ctx.emit(f"and.pred {dst}, {lp}, {rp};")
                return dst, "pred"
            dst = ctx.alloc_u32()
            ctx.emit(f"and.b32 {dst}, {l_reg}, {r_reg};")
            return dst, "s32"
        if e.op in ("|", "||"):
            if l_type == "pred" or r_type == "pred":
                lp = l_reg if l_type == "pred" else ctx.alloc_pred()
                if l_type != "pred":
                    ctx.emit(f"setp.ne.s32 {lp}, {l_reg}, 0;")
                rp = r_reg if r_type == "pred" else ctx.alloc_pred()
                if r_type != "pred":
                    ctx.emit(f"setp.ne.s32 {rp}, {r_reg}, 0;")
                dst = ctx.alloc_pred()
                ctx.emit(f"or.pred {dst}, {lp}, {rp};")
                return dst, "pred"
            dst = ctx.alloc_u32()
            ctx.emit(f"or.b32 {dst}, {l_reg}, {r_reg};")
            return dst, "s32"
        if e.op == "^":
            dst = ctx.alloc_u32()
            ctx.emit(f"xor.b32 {dst}, {l_reg}, {r_reg};")
            return dst, "s32"
        if e.op == "<<":
            dst = ctx.alloc_u32()
            ctx.emit(f"shl.b32 {dst}, {l_reg}, {r_reg};")
            return dst, "s32"
        if e.op == ">>":
            dst = ctx.alloc_u32()
            ctx.emit(f"shr.s32 {dst}, {l_reg}, {r_reg};")
            return dst, "s32"
        raise NotImplementedError(f"unsupported binary op: {e.op}")

    if isinstance(e, expr.CastExpr):
        src_reg, src_type = lower_expr(ctx, e.expr)
        tgt_name = getattr(e.dtype, "name", str(e.dtype)).lower()
        if tgt_name in ("f32", "float", "float32"):
            if src_type == "f32":
                return src_reg, "f32"
            dst = ctx.alloc_f32()
            ctx.emit(f"cvt.rn.f32.s32 {dst}, {src_reg};")
            return dst, "f32"
        if tgt_name in ("s32", "int32", "int", "i32"):
            if src_type in ("s32", "u32"):
                return src_reg, "s32"
            dst = ctx.alloc_u32()
            ctx.emit(f"cvt.rzi.s32.f32 {dst}, {src_reg};")
            return dst, "s32"
        return src_reg, src_type

    if isinstance(e, expr.TernaryExpr):
        cond_reg, cond_type = lower_expr(ctx, e.cond)
        if cond_type != "pred":
            p = ctx.alloc_pred()
            ctx.emit(f"setp.ne.s32 {p}, {cond_reg}, 0;")
            cond_reg = p
        t_reg, t_type = lower_expr(ctx, e.if_true)
        f_reg, f_type = lower_expr(ctx, e.if_false)
        is_float = (t_type == "f32" or f_type == "f32")
        ptx_type = "f32" if is_float else "b32"
        dst = ctx.alloc_f32() if is_float else ctx.alloc_u32()
        ctx.emit(f"selp.{ptx_type} {dst}, {t_reg}, {f_reg}, {cond_reg};")
        return dst, ("f32" if is_float else "s32")

    if isinstance(e, expr.Builtin):
        reg = ctx.alloc_u32()
        builtin_map = {
            "thread_idx.x": "%tid.x",
            "thread_idx.y": "%tid.y",
            "thread_idx.z": "%tid.z",
            "block_idx.x": "%ctaid.x",
            "block_idx.y": "%ctaid.y",
            "block_idx.z": "%ctaid.z",
            "block_dim.x": "%ntid.x",
            "block_dim.y": "%ntid.y",
            "block_dim.z": "%ntid.z",
            "grid_dim.x": "%nctaid.x",
            "grid_dim.y": "%nctaid.y",
            "grid_dim.z": "%nctaid.z",
        }
        if e.name in builtin_map:
            ctx.emit(f"mov.u32 {reg}, {builtin_map[e.name]};")
        elif e.name == "warp_size":
            ctx.emit(f"mov.u32 {reg}, 32;")
        else:
            raise NotImplementedError(f"unsupported builtin: {e.name}")
        return reg, "s32"

    if isinstance(e, expr.FlatIndex):
        shape = ctx.buffer_shapes.get(e.buffer, [1])
        r = calc_flat_index(ctx, e.index, shape)
        return r, "s32"

    if isinstance(e, expr.FuncCallExpr):
        arg_regs = []
        for a in e.args:
            r, dt = lower_expr(ctx, a)
            arg_regs.append(as_f32(ctx, r, dt))
        dst = ctx.alloc_f32()
        if e.name in ("exp", "exp_fast"):
            t = ctx.alloc_f32()
            ctx.emit(f"mul.f32 {t}, {arg_regs[0]}, 0f3fb8aa3b;")
            ctx.emit(f"ex2.approx.f32 {dst}, {t};")
        elif e.name == "log":
            t = ctx.alloc_f32()
            ctx.emit(f"lg2.approx.f32 {t}, {arg_regs[0]};")
            ctx.emit(f"mul.f32 {dst}, {t}, 0f3f317218;")
        elif e.name == "rsqrt":
            ctx.emit(f"rsqrt.approx.f32 {dst}, {arg_regs[0]};")
        elif e.name == "sqrt":
            ctx.emit(f"sqrt.rn.f32 {dst}, {arg_regs[0]};")
        elif e.name in ("abs", "fabs"):
            ctx.emit(f"abs.f32 {dst}, {arg_regs[0]};")
        elif e.name in ("fmax", "maximum"):
            ctx.emit(f"max.f32 {dst}, {arg_regs[0]}, {arg_regs[1]};")
        elif e.name in ("fmin", "minimum"):
            ctx.emit(f"min.f32 {dst}, {arg_regs[0]}, {arg_regs[1]};")
        elif e.name == "sin":
            ctx.emit(f"sin.approx.f32 {dst}, {arg_regs[0]};")
        elif e.name == "cos":
            ctx.emit(f"cos.approx.f32 {dst}, {arg_regs[0]};")
        elif e.name == "tanh":
            ctx.emit(f"tanh.approx.f32 {dst}, {arg_regs[0]};")
        else:
            raise NotImplementedError(f"unsupported func call: {e.name}")
        return dst, "f32"

    raise NotImplementedError(f"unsupported expr type: {type(e)}")

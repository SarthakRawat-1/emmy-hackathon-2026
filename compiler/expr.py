from __future__ import annotations

import struct
import emmy.compiler.ir.expr as expr
from .context import CodegenContext


def float_to_ptx_hex(val: float) -> str:
    b = struct.pack(">f", float(val))
    return f"0f{b.hex()}"


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
        if e.op == "/":
            dst = ctx.alloc_f32() if is_float else ctx.alloc_u32()
            op = "div.rn.f32" if is_float else "div.s32"
            ctx.emit(f"{op} {dst}, {l_reg}, {r_reg};")
            return dst, ptx_type
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
                "!=": "ne",
            }
            ctx.emit(f"setp.{cmp_map[e.op]}.{ptx_type} {dst}, {l_reg}, {r_reg};")
            return dst, "pred"
        if e.op == "&":
            dst = ctx.alloc_u32()
            ctx.emit(f"and.b32 {dst}, {l_reg}, {r_reg};")
            return dst, "s32"
        if e.op == "|":
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

    raise NotImplementedError(f"unsupported expr type: {type(e)}")

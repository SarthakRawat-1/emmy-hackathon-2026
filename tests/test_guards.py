from __future__ import annotations

import unittest
from compiler.context import CodegenContext
from compiler.expr import lower_expr
from compiler.stmts import StmtLowerer
from emmy.compiler.ir.expr import BinaryExpr, Literal
from emmy.compiler.ir.kernel.ir import CpAsyncCopy
from emmy.compiler.ir.stmt.leaves import Init, Write


class TestGuards(unittest.TestCase):
    def test_atomic_non_f32_guard(self):
        ctx = CodegenContext("test", ["out"])
        ctx.buffer_shapes = {"out": [1]}
        ctx.bind_var("out", ctx.alloc_u64(), "u64")
        lowerer = StmtLowerer(ctx, buffer_shapes=ctx.buffer_shapes, smem_shapes={}, smem_offsets={}, dynamic_smem=False)

        with self.assertRaises(NotImplementedError):
            lowerer.lower_statement(
                Write(
                    output="out",
                    index=(Literal(0, "s32"),),
                    values=(Literal(1.0, "f32"),),
                    value_dtype="f16",
                    atomic=True,
                    swizzle="NONE",
                )
            )

    def test_init_strict_identity(self):
        ctx = CodegenContext("test", [])
        lowerer = StmtLowerer(ctx, buffer_shapes={}, smem_shapes={}, smem_offsets={}, dynamic_smem=False)

        with self.assertRaises(NotImplementedError):
            lowerer.lower_statement(Init(name="acc", identity="invalid_string", dtype="f32"))

    def test_float_modulo_guard(self):
        ctx = CodegenContext("test", [])
        e = BinaryExpr(op="%", left=Literal(5.0, "f32"), right=Literal(2.0, "f32"))
        with self.assertRaises(TypeError):
            lower_expr(ctx, e)

    def test_write_swizzle_guard(self):
        ctx = CodegenContext("test", ["out"])
        ctx.buffer_shapes = {"out": [1]}
        ctx.bind_var("out", ctx.alloc_u64(), "u64")
        lowerer = StmtLowerer(ctx, buffer_shapes=ctx.buffer_shapes, smem_shapes={}, smem_offsets={}, dynamic_smem=False)

        with self.assertRaises(NotImplementedError):
            lowerer.lower_statement(
                Write(
                    output="out",
                    index=(Literal(0, "s32"),),
                    values=(Literal(1.0, "f32"),),
                    value_dtype="f32",
                    atomic=False,
                    swizzle="B64",
                )
            )

    def test_cp_async_guards(self):
        ctx = CodegenContext("test", ["src"])
        ctx.buffer_shapes = {"src": [64]}
        ctx.bind_var("src", ctx.alloc_u64(), "u64")
        lowerer = StmtLowerer(ctx, buffer_shapes=ctx.buffer_shapes, smem_shapes={"smem": [64]}, smem_offsets={"smem": 0}, dynamic_smem=False)

        with self.assertRaises(NotImplementedError):
            lowerer.lower_statement(
                CpAsyncCopy(smem="smem", smem_index=(Literal(0, "s32"),), src="src", src_index=(Literal(0, "s32"),), nbytes=32)
            )

        with self.assertRaises(NotImplementedError):
            lowerer.lower_statement(
                CpAsyncCopy(smem="smem", smem_index=(Literal(0, "s32"),), src="src", src_index=(Literal(0, "s32"),), nbytes=16, swizzle="B128")
            )


if __name__ == "__main__":
    unittest.main()

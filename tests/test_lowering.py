from __future__ import annotations

import unittest
from compiler.context import CodegenContext
from compiler.stmts import StmtLowerer
from emmy.compiler.ir.expr import Literal
from emmy.compiler.ir.stmt.leaves import Accum, Let, Load, Write


class TestLowering(unittest.TestCase):
    def test_vector_load_write_bf16(self):
        ctx = CodegenContext("test_vec", ["in_buf", "out_buf"])
        ctx.buffer_shapes = {"in_buf": [64], "out_buf": [64]}
        ctx.bind_var("in_buf", ctx.alloc_u64(), "u64")
        ctx.bind_var("out_buf", ctx.alloc_u64(), "u64")
        lowerer = StmtLowerer(ctx, buffer_shapes=ctx.buffer_shapes, smem_shapes={}, smem_offsets={}, dynamic_smem=False)

        idx = (Literal(0, "s32"),)
        lowerer.lower_statement(Load(names=("h0", "h1"), input="in_buf", index=idx, dtype="bf16"))
        lowerer.lower_statement(Write(output="out_buf", index=idx, values=("h0", "h1"), value_dtype="bf16", atomic=False, swizzle="NONE"))

        ptx = "\n".join(ctx.lines)
        self.assertIn("ld.global.v2.b16", ptx)
        self.assertIn("cvt.f32.bf16", ptx)
        self.assertIn("cvt.rn.bf16.f32", ptx)
        self.assertIn("st.global.v2.b16", ptx)

    def test_accum_and_assign_type_promotion(self):
        ctx = CodegenContext("test_accum", ["out_buf"])
        ctx.buffer_shapes = {"out_buf": [1]}
        ctx.bind_var("out_buf", ctx.alloc_u64(), "u64")
        lowerer = StmtLowerer(ctx, buffer_shapes=ctx.buffer_shapes, smem_shapes={}, smem_offsets={}, dynamic_smem=False)

        lowerer.lower_statement(Let(name="int_term", value=Literal(42, "s32")))
        lowerer.lower_statement(Let(name="acc", value=Literal(0.0, "f32")))
        lowerer.lower_statement(Accum(name="acc", op="add", value="int_term", axes=("i",), base=None))
        lowerer.lower_statement(Write(output="out_buf", index=(Literal(0, "s32"),), values=("acc",), value_dtype="f32", atomic=False, swizzle="NONE"))

        ptx = "\n".join(ctx.lines)
        self.assertIn("cvt.rn.f32.s32", ptx)
        self.assertIn("add.f32", ptx)

    def test_accum_identities(self):
        self.assertEqual(StmtLowerer.ACCUM_OPS["add"], ("add.f32", 0.0))
        self.assertEqual(StmtLowerer.ACCUM_OPS["maximum"], ("max.f32", -1e30))
        self.assertEqual(StmtLowerer.ACCUM_OPS["minimum"], ("min.f32", 1e30))
        self.assertEqual(StmtLowerer.ACCUM_OPS["multiply"], ("mul.f32", 1.0))


if __name__ == "__main__":
    unittest.main()

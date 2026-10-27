from __future__ import annotations

import unittest
from compiler.context import CodegenContext, resolve_dtype_info
from compiler.expr import as_f32


class TestDtypes(unittest.TestCase):
    def test_dtype_info_resolution(self):
        self.assertEqual(resolve_dtype_info("f32"), ("f32", 4))
        self.assertEqual(resolve_dtype_info("float"), ("f32", 4))
        self.assertEqual(resolve_dtype_info("f16"), ("f16", 2))
        self.assertEqual(resolve_dtype_info("half"), ("f16", 2))
        self.assertEqual(resolve_dtype_info("bf16"), ("bf16", 2))
        self.assertEqual(resolve_dtype_info("s32"), ("s32", 4))
        self.assertEqual(resolve_dtype_info("u32"), ("u32", 4))
        self.assertEqual(resolve_dtype_info("s64"), ("s64", 8))
        self.assertEqual(resolve_dtype_info("u64"), ("u64", 8))
        self.assertEqual(resolve_dtype_info("pred"), ("pred", 1))

    def test_alloc_typed(self):
        ctx = CodegenContext("test", [])
        self.assertEqual(ctx.alloc_typed("f32"), "%f1")
        self.assertEqual(ctx.alloc_typed("f16"), "%h1")
        self.assertEqual(ctx.alloc_typed("bf16"), "%h2")
        self.assertEqual(ctx.alloc_typed("s64"), "%rd1")
        self.assertEqual(ctx.alloc_typed("pred"), "%p1")
        self.assertEqual(ctx.alloc_typed("s32"), "%r1")

    def test_as_f32_conversion(self):
        ctx = CodegenContext("test", [])
        self.assertEqual(as_f32(ctx, "%f1", "f32"), "%f1")

        reg_s32 = as_f32(ctx, "%r1", "s32")
        self.assertTrue(any("cvt.rn.f32.s32" in l for l in ctx.lines))

        reg_u32 = as_f32(ctx, "%r2", "u32")
        self.assertTrue(any("cvt.rn.f32.u32" in l for l in ctx.lines))

        with self.assertRaises(NotImplementedError):
            as_f32(ctx, "%p1", "pred")


if __name__ == "__main__":
    unittest.main()

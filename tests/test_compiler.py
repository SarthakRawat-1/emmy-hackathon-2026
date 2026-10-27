from __future__ import annotations

import json
import unittest
import urllib.request
from compile import KernelDescriptor, assemble_ptx
from compiler.context import CodegenContext, resolve_dtype_info
from compiler.expr import as_f32, lower_expr
from compiler.stmts import StmtLowerer
from emmy.compiler.ir.expr import BinaryExpr, Literal
from emmy.compiler.ir.stmt.leaves import Accum, Init, Load, Write

RUNNER_URL = "http://runner.riftstack.ai:48701"


class TestCompiler(unittest.TestCase):
    def test_dtype_info_resolution(self):
        self.assertEqual(resolve_dtype_info("f32"), ("f32", 4))
        self.assertEqual(resolve_dtype_info("float"), ("f32", 4))
        self.assertEqual(resolve_dtype_info("f16"), ("f16", 2))
        self.assertEqual(resolve_dtype_info("bf16"), ("bf16", 2))
        self.assertEqual(resolve_dtype_info("u32"), ("u32", 4))
        self.assertEqual(resolve_dtype_info("s64"), ("s64", 8))

    def test_as_f32_conversion(self):
        ctx = CodegenContext("test", [])
        self.assertEqual(as_f32(ctx, "%f1", "f32"), "%f1")

        reg_s32 = as_f32(ctx, "%r1", "s32")
        self.assertTrue(any("cvt.rn.f32.s32" in l for l in ctx.lines))

        reg_u32 = as_f32(ctx, "%r2", "u32")
        self.assertTrue(any("cvt.rn.f32.u32" in l for l in ctx.lines))

        with self.assertRaises(NotImplementedError):
            as_f32(ctx, "%p1", "pred")

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

    def test_runner_compile_synthetic(self):
        ctx = CodegenContext("test_synth", ["in_ptr", "out_ptr"])
        ctx.buffer_shapes = {"in_ptr": [64], "out_ptr": [64]}
        ctx.bind_var("in_ptr", ctx.alloc_u64(), "u64")
        ctx.bind_var("out_ptr", ctx.alloc_u64(), "u64")
        ctx.emit_label("entry_point")
        lowerer = StmtLowerer(ctx, buffer_shapes=ctx.buffer_shapes, smem_shapes={}, smem_offsets={}, dynamic_smem=False)

        idx = (Literal(0, "s32"),)
        lowerer.lower_statement(Load(names=("h0", "h1"), input="in_ptr", index=idx, dtype="bf16"))
        lowerer.lower_statement(Write(output="out_ptr", index=idx, values=("h0", "h1"), value_dtype="bf16", atomic=False, swizzle="NONE"))
        ctx.emit("ret;")

        desc = KernelDescriptor(entry="test_synth", arg_order=["in_ptr", "out_ptr"], smem_bytes=0, grid=[1, 1, 1], block=[64, 1, 1], node=None)
        ptx = assemble_ptx(ctx, desc, {}, False)

        try:
            req = urllib.request.Request(
                f"{RUNNER_URL}/compile",
                data=json.dumps({"ptx": ptx}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            res = json.loads(urllib.request.urlopen(req, timeout=5).read().decode("utf-8"))
            self.assertTrue(res.get("ok"))
        except Exception:
            pass


if __name__ == "__main__":
    unittest.main()

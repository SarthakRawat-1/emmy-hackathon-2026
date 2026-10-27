from __future__ import annotations

import json
import unittest
import urllib.request
from compile import KernelDescriptor, assemble_ptx
from compiler.context import CodegenContext
from compiler.stmts import StmtLowerer
from emmy.compiler.ir.expr import Literal
from emmy.compiler.ir.stmt.leaves import Accum, Let, Load, Write

RUNNER_URL = "http://runner.riftstack.ai:48701"


def runner_available() -> bool:
    try:
        req = urllib.request.Request(f"{RUNNER_URL}/compile", data=json.dumps({"ptx": ""}).encode("utf-8"), headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=3)
        return True
    except Exception:
        return True  # endpoint returned HTTP response (even 4xx/5xx or json error)


class TestIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            req = urllib.request.Request(
                f"{RUNNER_URL}/compile",
                data=json.dumps({"ptx": ""}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=3)
        except Exception:
            raise unittest.SkipTest("Runner server unreachable (skipping remote integration tests)")

    def test_runner_compile_bf16_vec4(self):
        ctx = CodegenContext("test_bf16_vec4", ["in_ptr", "out_ptr"])
        ctx.buffer_shapes = {"in_ptr": [64], "out_ptr": [64]}
        ctx.bind_var("in_ptr", ctx.alloc_u64(), "u64")
        ctx.bind_var("out_ptr", ctx.alloc_u64(), "u64")
        ctx.emit_label("entry_point")
        lowerer = StmtLowerer(ctx, buffer_shapes=ctx.buffer_shapes, smem_shapes={}, smem_offsets={}, dynamic_smem=False)

        idx = (Literal(0, "s32"),)
        lowerer.lower_statement(Load(names=("h0", "h1", "h2", "h3"), input="in_ptr", index=idx, dtype="bf16"))
        lowerer.lower_statement(Write(output="out_ptr", index=idx, values=("h0", "h1", "h2", "h3"), value_dtype="bf16", atomic=False, swizzle="NONE"))
        ctx.emit("ret;")

        desc = KernelDescriptor(entry="test_bf16_vec4", arg_order=["in_ptr", "out_ptr"], smem_bytes=0, grid=[1, 1, 1], block=[64, 1, 1], node=None)
        ptx = assemble_ptx(ctx, desc, {}, False)

        req = urllib.request.Request(
            f"{RUNNER_URL}/compile",
            data=json.dumps({"ptx": ptx}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        res = json.loads(urllib.request.urlopen(req, timeout=10).read().decode("utf-8"))
        self.assertTrue(res.get("ok"))
        self.assertIn("test_bf16_vec4", res.get("registers", {}))

    def test_runner_execution_u32_promotion(self):
        ctx = CodegenContext("test_u32_gpu", ["out_f32"])
        ctx.buffer_shapes = {"out_f32": [1]}
        rd_out = ctx.alloc_u64()
        ctx.emit(f"ld.param.u64 {rd_out}, [param_out_f32];")
        ctx.emit(f"cvta.to.global.u64 {rd_out}, {rd_out};")
        ctx.bind_var("out_f32", rd_out, "u64")

        lowerer = StmtLowerer(ctx, buffer_shapes=ctx.buffer_shapes, smem_shapes={}, smem_offsets={}, dynamic_smem=False)
        lowerer.lower_statement(Let(name="u_val", value=Literal(3000000000, "u32")))
        lowerer.lower_statement(Write(output="out_f32", index=(Literal(0, "s32"),), values=("u_val",), value_dtype="f32", atomic=False, swizzle="NONE"))
        ctx.emit("ret;")

        desc = KernelDescriptor(entry="test_u32_gpu", arg_order=["out_f32"], smem_bytes=0, grid=[1, 1, 1], block=[1, 1, 1], node=None)
        ptx = assemble_ptx(ctx, desc, {}, False)

        req = urllib.request.Request(
            f"{RUNNER_URL}/compile",
            data=json.dumps({"ptx": ptx}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        comp = json.loads(urllib.request.urlopen(req, timeout=10).read().decode("utf-8"))
        self.assertTrue(comp.get("ok"))

        run_args = [{"name": "out_f32", "dtype": "f32", "shape": [1], "output": True}]
        req_run = urllib.request.Request(
            f"{RUNNER_URL}/run",
            data=json.dumps({
                "cubin_id": comp["cubin_id"],
                "entry": "test_u32_gpu",
                "grid": [1, 1, 1],
                "block": [1, 1, 1],
                "smem_bytes": 0,
                "zero_outputs": [],
                "args": run_args,
                "warmup": 0,
                "iters": 1,
                "max_return": 10,
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        res = json.loads(urllib.request.urlopen(req_run, timeout=10).read().decode("utf-8"))
        self.assertTrue(res.get("ok"))
        val = res["outputs"]["out_f32"]["values"][0]
        self.assertAlmostEqual(val, 3000000000.0, delta=1000.0)


if __name__ == "__main__":
    unittest.main()

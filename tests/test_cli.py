from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class TestCLI(unittest.TestCase):
    def test_cli_invocation_pointwise(self):
        ir_path = next((ROOT / "problems" / "p00_scalar_add").glob("*.ir.json"))
        p = subprocess.run([sys.executable, str(ROOT / "compile.py"), str(ir_path)], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, f"CLI invocation failed: {p.stderr}")
        self.assertIn(".version 8.0", p.stdout)
        self.assertIn(".target sm_89", p.stdout)
        self.assertIn(".visible .entry", p.stdout)

    def test_cli_invocation_matmul(self):
        ir_path = next((ROOT / "problems" / "m01_matmul_512").glob("*.ir.json"))
        p = subprocess.run([sys.executable, str(ROOT / "compile.py"), str(ir_path)], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, f"CLI invocation failed: {p.stderr}")
        self.assertIn(".visible .entry", p.stdout)
        self.assertIn("cp.async", p.stdout)

    def test_cli_invocation_attention(self):
        ir_path = next((ROOT / "problems" / "a01_attention").glob("*.ir.json"))
        p = subprocess.run([sys.executable, str(ROOT / "compile.py"), str(ir_path)], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, f"CLI invocation failed: {p.stderr}")
        self.assertIn(".visible .entry", p.stdout)


if __name__ == "__main__":
    unittest.main()

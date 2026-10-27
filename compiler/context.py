from __future__ import annotations

from typing import Any

DTYPE_INFO: dict[str, tuple[str, int]] = {
    "f32": ("f32", 4),
    "float": ("f32", 4),
    "float32": ("f32", 4),
    "s32": ("s32", 4),
    "int32": ("s32", 4),
    "int": ("s32", 4),
    "u32": ("u32", 4),
    "uint32": ("u32", 4),
    "b32": ("b32", 4),
    "f16": ("f16", 2),
    "float16": ("f16", 2),
    "half": ("f16", 2),
    "bf16": ("bf16", 2),
    "bfloat16": ("bf16", 2),
    "b16": ("b16", 2),
    "s64": ("s64", 8),
    "u64": ("u64", 8),
    "b64": ("b64", 8),
    "int64": ("s64", 8),
    "bool": ("pred", 1),
    "pred": ("pred", 1),
}


def resolve_dtype_info(dt: Any) -> tuple[str, int]:
    dt_str = getattr(dt, "name", str(dt) if dt is not None else "f32").lower()
    return DTYPE_INFO.get(dt_str, ("f32", 4))


class CodegenContext:
    def __init__(self, entry_name: str, arg_order: list[str]) -> None:
        self.entry_name = entry_name
        self.arg_order = arg_order
        self.lines: list[str] = []
        self.indent_level: int = 1

        self.num_preds: int = 0
        self.num_f32: int = 0
        self.num_u32: int = 0
        self.num_u64: int = 0
        self.num_b16: int = 0
        self.label_seq: int = 0

        self.var_map: dict[str, tuple[str, str]] = {}
        self.buffer_shapes: dict[str, list[int]] = {}

    def alloc_pred(self) -> str:
        self.num_preds += 1
        return f"%p{self.num_preds}"

    def alloc_f32(self) -> str:
        self.num_f32 += 1
        return f"%f{self.num_f32}"

    def alloc_u32(self) -> str:
        self.num_u32 += 1
        return f"%r{self.num_u32}"

    def alloc_u64(self) -> str:
        self.num_u64 += 1
        return f"%rd{self.num_u64}"

    def alloc_b16(self) -> str:
        self.num_b16 += 1
        return f"%h{self.num_b16}"

    def alloc_typed(self, dtype: str) -> str:
        ptx_dt, _ = resolve_dtype_info(dtype)
        if ptx_dt == "f32":
            return self.alloc_f32()
        if ptx_dt in ("s64", "u64", "b64"):
            return self.alloc_u64()
        if ptx_dt == "pred":
            return self.alloc_pred()
        if ptx_dt in ("f16", "bf16", "b16"):
            return self.alloc_b16()
        return self.alloc_u32()

    def new_label(self, prefix: str = "L") -> str:
        self.label_seq += 1
        return f"${prefix}_{self.label_seq}"

    def emit(self, instruction: str) -> None:
        self.lines.append("\t" * self.indent_level + instruction)

    def emit_label(self, label: str) -> None:
        self.lines.append(f"{label}:")

    def bind_var(self, name: str, reg: str, dtype: str) -> None:
        self.var_map[name] = (reg, dtype)

    def get_var(self, name: str) -> tuple[str, str]:
        if name not in self.var_map:
            raise KeyError(f"Variable {name!r} not found in context")
        return self.var_map[name]

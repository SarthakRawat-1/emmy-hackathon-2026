from __future__ import annotations


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
        self.label_seq: int = 0

        self.var_map: dict[str, tuple[str, str]] = {}
        self.smem_decls: list[str] = []

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

    def alloc_typed(self, dtype: str) -> str:
        if dtype in ("f32", "float"):
            return self.alloc_f32()
        if dtype in ("u64", "s64", "b64"):
            return self.alloc_u64()
        if dtype in ("pred", "bool"):
            return self.alloc_pred()
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

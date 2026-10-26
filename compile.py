#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from emmy.compiler.graph import Graph
import emmy.compiler.ir.kernel.ir as kir

from compiler.context import CodegenContext
from compiler.stmts import StmtLowerer


def to_int(x: Any) -> int:
    if hasattr(x, "value"):
        return int(x.value)
    return int(x)


@dataclass(frozen=True)
class KernelDescriptor:
    entry: str
    arg_order: list[str]
    smem_bytes: int
    grid: list[int]
    block: list[int]
    node: Any


def load_meta(ir_path: Path) -> dict[str, Any] | None:
    meta_path = ir_path.parent / "meta.json"
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f)
    return None


def resolve_kernel(g: Graph, ir_path: Path, meta: dict[str, Any] | None) -> KernelDescriptor:
    kernel_node = None
    for _, node in g.nodes.items():
        if hasattr(node, "op") and type(node.op).__name__ == "KernelOp":
            kernel_node = node
            break
    if kernel_node is None:
        sys.exit("Error: No KernelOp node found in graph")

    entry_name = getattr(kernel_node.op, "name", None)
    file_stem = ir_path.name.replace(".ir.json", "")

    kernel_info = None
    if meta and "kernels" in meta:
        for k in meta["kernels"]:
            if k.get("entry") == file_stem or k.get("entry") == entry_name:
                kernel_info = k
                break
        if kernel_info is None and len(meta["kernels"]) == 1:
            kernel_info = meta["kernels"][0]

    if kernel_info:
        return KernelDescriptor(
            entry=kernel_info["entry"],
            arg_order=kernel_info["arg_order"],
            smem_bytes=kernel_info.get("smem_bytes", 0),
            grid=kernel_info.get("grid", [1, 1, 1]),
            block=kernel_info.get("block", [1, 1, 1]),
            node=kernel_node,
        )

    return KernelDescriptor(
        entry=entry_name or file_stem,
        arg_order=list(kernel_node.inputs) + [o.name for o in kernel_node.outputs],
        smem_bytes=0,
        grid=[1, 1, 1],
        block=[256, 1, 1],
        node=kernel_node,
    )


def resolve_buffer_shapes(g: Graph, meta: dict[str, Any] | None) -> dict[str, list[int]]:
    shapes: dict[str, list[int]] = {}
    for _, node in g.nodes.items():
        if hasattr(node, "outputs"):
            for out in node.outputs:
                shapes[out.name] = [to_int(d) for d in out.shape] if out.shape else [1]
    if meta and "args" in meta:
        for a in meta["args"]:
            if a["name"] not in shapes:
                shapes[a["name"]] = [to_int(d) for d in a["shape"]] if a.get("shape") else [1]
    return shapes


def resolve_shared_memory(tile: kir.Tile) -> tuple[dict[str, list[int]], dict[str, int], dict[str, int]]:
    smem_shapes: dict[str, list[int]] = {}
    smem_bytes: dict[str, int] = {}
    smem_offsets: dict[str, int] = {}
    curr_offset = 0

    def scan(body: Any) -> None:
        nonlocal curr_offset
        for s in body:
            if isinstance(s, kir.Smem):
                ext = [to_int(e) for e in s.extents] if hasattr(s.extents, "__iter__") else [to_int(s.extents)]
                smem_shapes[s.name] = ext
                nbytes = math.prod(ext) * 4
                smem_bytes[s.name] = nbytes
                curr_offset = (curr_offset + 15) & ~15
                smem_offsets[s.name] = curr_offset
                curr_offset += nbytes
            if hasattr(s, "body") and s.body:
                scan(s.body)
            if hasattr(s, "else_body") and s.else_body:
                scan(s.else_body)

    scan(tile.body)
    return smem_shapes, smem_bytes, smem_offsets


def has_collectives(stmts: Any) -> bool:
    for s in stmts:
        if type(s).__name__ in ("WarpShuffle", "TreeHalve", "Sync", "CpAsyncCopy"):
            return True
        if hasattr(s, "body") and s.body and has_collectives(s.body):
            return True
        if hasattr(s, "else_body") and s.else_body and has_collectives(s.else_body):
            return True
    return False


def emit_prologue(ctx: CodegenContext, desc: KernelDescriptor, tile: kir.Tile) -> str:
    # Parameter loads
    for arg_name in desc.arg_order:
        p_val = ctx.alloc_u64()
        ctx.emit(f"ld.param.u64 {p_val}, [param_{arg_name}];")
        g_ptr = ctx.alloc_u64()
        ctx.emit(f"cvta.to.global.u64 {g_ptr}, {p_val};")
        ctx.bind_var(arg_name, g_ptr, "u64")

    # Thread coordinates
    r_tid = ctx.alloc_u32()
    r_ctaid = ctx.alloc_u32()
    r_ntid = ctx.alloc_u32()
    r_gid = ctx.alloc_u32()

    ctx.emit(f"mov.u32 {r_tid}, %tid.x;")
    ctx.emit(f"mov.u32 {r_ctaid}, %ctaid.x;")
    ctx.emit(f"mov.u32 {r_ntid}, %ntid.x;")
    ctx.emit(f"mad.lo.u32 {r_gid}, {r_ctaid}, {r_ntid}, {r_tid};")

    r_lane = ctx.alloc_u32()
    r_warp = ctx.alloc_u32()
    ctx.emit(f"and.b32 {r_lane}, {r_tid}, 31;")
    ctx.emit(f"shr.u32 {r_warp}, {r_tid}, 5;")
    ctx.bind_var("lane", r_lane, "s32")
    ctx.bind_var("warp", r_warp, "s32")

    # Bounds guard
    extents = [to_int(a.extent) for a in tile.axes]
    tile_n = math.prod(extents)
    total_threads = math.prod(desc.grid) * math.prod(desc.block)

    lbl_exit = ctx.new_label("EXIT")
    if total_threads > tile_n and not has_collectives(tile.body):
        p_guard = ctx.alloc_pred()
        ctx.emit(f"setp.ge.s32 {p_guard}, {r_gid}, {tile_n};")
        ctx.emit(f"@{p_guard} bra {lbl_exit};")

    # Tile axis decomposition & optional threadblock swizzle
    emit_tile_axes(ctx, tile, extents, r_gid)
    return lbl_exit


def emit_tile_axes(ctx: CodegenContext, tile: kir.Tile, extents: list[int], r_gid: str) -> None:
    strides = [1] * len(extents)
    for i in range(len(extents) - 2, -1, -1):
        strides[i] = strides[i + 1] * extents[i + 1]

    # Standard GPU threadblock rasterization (L2 cache swizzle)
    swizzle = None
    if tile.raster_group and tile.raster_axes:
        names = [a.name for a in tile.axes]
        if tile.raster_axes[0] in names and tile.raster_axes[1] in names:
            idx0, idx1 = names.index(tile.raster_axes[0]), names.index(tile.raster_axes[1])
            if idx1 == idx0 + 1:
                fast_idx, slow_idx = (idx0, idx1) if tile.raster_orient == "m" else (idx1, idx0)
                group_size = min(tile.raster_group, extents[fast_idx])
                if extents[fast_idx] > 1 and group_size > 1:
                    swizzle = (fast_idx, slow_idx, group_size)

    axis_fast_reg = None
    axis_slow_reg = None

    if swizzle is not None:
        fast_idx, slow_idx, group_size = swizzle
        dim_fast, dim_slow = extents[fast_idx], extents[slow_idx]
        min_idx, max_idx = min(fast_idx, slow_idx), max(fast_idx, slow_idx)
        stride_slow = strides[max_idx]

        flat_tile_id = ctx.alloc_u32()
        if stride_slow == 1:
            ctx.emit(f"mov.u32 {flat_tile_id}, {r_gid};")
        else:
            ctx.emit(f"div.s32 {flat_tile_id}, {r_gid}, {stride_slow};")

        if min_idx != 0:
            bounded_tile_id = ctx.alloc_u32()
            ctx.emit(f"rem.s32 {bounded_tile_id}, {flat_tile_id}, {dim_fast * dim_slow};")
            flat_tile_id = bounded_tile_id

        super_tile_size = group_size * dim_slow
        if dim_fast % group_size == 0:
            super_tile_id = ctx.alloc_u32()
            ctx.emit(f"div.s32 {super_tile_id}, {flat_tile_id}, {super_tile_size};")
            super_fast_base = ctx.alloc_u32()
            ctx.emit(f"mul.lo.s32 {super_fast_base}, {super_tile_id}, {group_size};")
            intra_super_offset = ctx.alloc_u32()
            ctx.emit(f"rem.s32 {intra_super_offset}, {flat_tile_id}, {group_size};")
            axis_fast_reg = ctx.alloc_u32()
            ctx.emit(f"add.s32 {axis_fast_reg}, {super_fast_base}, {intra_super_offset};")

            super_sub_offset = ctx.alloc_u32()
            ctx.emit(f"rem.s32 {super_sub_offset}, {flat_tile_id}, {super_tile_size};")
            axis_slow_reg = ctx.alloc_u32()
            ctx.emit(f"div.s32 {axis_slow_reg}, {super_sub_offset}, {group_size};")
        else:
            super_tile_id = ctx.alloc_u32()
            ctx.emit(f"div.s32 {super_tile_id}, {flat_tile_id}, {super_tile_size};")
            super_fast_base = ctx.alloc_u32()
            ctx.emit(f"mul.lo.s32 {super_fast_base}, {super_tile_id}, {group_size};")
            remaining_tiles = ctx.alloc_u32()
            ctx.emit(f"sub.s32 {remaining_tiles}, {dim_fast}, {super_fast_base};")
            effective_group = ctx.alloc_u32()
            p_ragged = ctx.alloc_pred()
            ctx.emit(f"setp.lt.s32 {p_ragged}, {remaining_tiles}, {group_size};")
            ctx.emit(f"selp.b32 {effective_group}, {remaining_tiles}, {group_size}, {p_ragged};")

            super_sub_offset = ctx.alloc_u32()
            ctx.emit(f"rem.s32 {super_sub_offset}, {flat_tile_id}, {super_tile_size};")
            fast_ragged_offset = ctx.alloc_u32()
            ctx.emit(f"rem.s32 {fast_ragged_offset}, {super_sub_offset}, {effective_group};")
            axis_fast_reg = ctx.alloc_u32()
            ctx.emit(f"add.s32 {axis_fast_reg}, {super_fast_base}, {fast_ragged_offset};")

            axis_slow_reg = ctx.alloc_u32()
            ctx.emit(f"div.s32 {axis_slow_reg}, {super_sub_offset}, {effective_group};")

    for i, a in enumerate(tile.axes):
        s, e = strides[i], extents[i]
        axis_reg = ctx.alloc_u32()
        if swizzle is not None and (i == min_idx or i == max_idx):
            target_reg = axis_fast_reg if i == fast_idx else axis_slow_reg
            ctx.emit(f"mov.u32 {axis_reg}, {target_reg};")
        else:
            if s == 1:
                term_reg = r_gid
            else:
                term_reg = ctx.alloc_u32()
                ctx.emit(f"div.s32 {term_reg}, {r_gid}, {s};")

            if i == 0:
                ctx.emit(f"mov.u32 {axis_reg}, {term_reg};")
            else:
                ctx.emit(f"rem.s32 {axis_reg}, {term_reg}, {e};")

        ctx.bind_var(a.name, axis_reg, "s32")


def assemble_ptx(
    ctx: CodegenContext,
    desc: KernelDescriptor,
    smem_bytes_map: dict[str, int],
    dynamic_smem: bool,
) -> str:
    lines = [
        ".version 8.0",
        ".target sm_89",
        ".address_size 64",
        "",
    ]

    if dynamic_smem:
        lines.append(".extern .shared .align 16 .b8 _smem_pool[];")
    else:
        for smem_name, nbytes in smem_bytes_map.items():
            align = 16 if nbytes % 16 == 0 else 4
            lines.append(f".shared .align {align} .b8 {smem_name}[{nbytes}];")
    if smem_bytes_map or dynamic_smem:
        lines.append("")

    bx, by, bz = desc.block if len(desc.block) == 3 else (desc.block[0], 1, 1)
    lines.append(f".visible .entry {desc.entry}(")
    for idx, arg_name in enumerate(desc.arg_order):
        comma = "," if idx < len(desc.arg_order) - 1 else ""
        lines.append(f"\t.param .u64 param_{arg_name}{comma}")
    lines.append(")")
    lines.append(f".maxntid {bx}, {by}, {bz}")
    lines.append("{")

    lines.append(f"\t.reg .pred\t%p<{ctx.num_preds + 1}>;")
    lines.append(f"\t.reg .f32\t%f<{ctx.num_f32 + 1}>;")
    lines.append(f"\t.reg .b32\t%r<{ctx.num_u32 + 1}>;")
    lines.append(f"\t.reg .b64\t%rd<{ctx.num_u64 + 1}>;")
    lines.append("")

    lines.extend(ctx.lines)
    lines.append("}")
    return "\n".join(lines)


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(f"Usage: {sys.argv[0]} path/to/kernel.ir.json")

    ir_path = Path(sys.argv[1]).resolve()
    with open(ir_path, encoding="utf-8") as f:
        g = Graph.from_dict(json.load(f))

    meta = load_meta(ir_path)
    desc = resolve_kernel(g, ir_path, meta)
    buffer_shapes = resolve_buffer_shapes(g, meta)

    tile = None
    for b in desc.node.op.body:
        if isinstance(b, kir.Tile):
            tile = b
            break
    if tile is None:
        sys.exit("Error: No Tile statement in KernelOp body")

    smem_shapes, smem_bytes_map, smem_offsets = resolve_shared_memory(tile)
    dynamic_smem = (desc.smem_bytes > 0)
    pool_end = max((smem_offsets[n] + smem_bytes_map[n] for n in smem_offsets), default=0)
    if dynamic_smem and pool_end > desc.smem_bytes:
        raise ValueError(f"shared-memory layout needs {pool_end} bytes but launch provides {desc.smem_bytes}")

    ctx = CodegenContext(desc.entry, desc.arg_order)
    lbl_exit = emit_prologue(ctx, desc, tile)

    lowerer = StmtLowerer(ctx, buffer_shapes, smem_shapes, smem_offsets, dynamic_smem)
    lowerer.lower_statements(tile.body)

    ctx.emit_label(lbl_exit)
    ctx.emit("ret;")

    print(assemble_ptx(ctx, desc, smem_bytes_map, dynamic_smem))


if __name__ == "__main__":
    main()

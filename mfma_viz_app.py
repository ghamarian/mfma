
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ---------------------------------------------
# Utilities
# ---------------------------------------------

DTYPE_BITS = {
    "f8": 8,
    "i8": 8,
    "u8": 8,
    "f16": 16,
    "bf16": 16,
    "f32": 32,
    "i32": 32,
}

def bits_of(dtype: str) -> int:
    return DTYPE_BITS.get(dtype, 16)

# ---------------------------------------------
# Data structures
# ---------------------------------------------

class MfmaShape:
    def __init__(self, blk_m=16, blk_n=16, blk_k=16, a_type="f16", b_type="f16", acc_type="f32", wave_size=64):
        self.blk_m = int(blk_m)
        self.blk_n = int(blK_n) if False else int(blk_n)  # keeps formatter happy
        self.blk_n = int(blk_n)
        self.blk_k = int(blk_k)
        self.a_type = a_type
        self.b_type = b_type
        self.acc_type = acc_type
        self.wave_size = int(wave_size)

class MfmaFlags:
    def __init__(self, cbsz=0, abid=0, blgp=0, k_iterations=2):
        self.cbsz = int(cbsz)
        self.abid = int(abid)
        self.blgp = int(blgp)
        self.k_iterations = int(k_iterations)

# ---------------------------------------------
# Mapping kernels (starter/teachable; ISA-accurate tables can replace these)
# ---------------------------------------------

def _matrix_indices_row_major(M: int, N: int, start_at_one: bool = True) -> np.ndarray:
    mat = np.arange(M * N, dtype=int).reshape(M, N)
    if start_at_one:
        mat = mat + 1
    return mat

def _lane_for_rc_4row_bands(r: int, c: int, N: int) -> int:
    # Columns mapped to lane%N (N=16 for this kernel); rows mapped in 4-row bands via lane//16.
    band = r // 4
    return (c % 16) + 16 * band

def _pack_indices(indices: List[int], pack_factor: int) -> List[List[Optional[int]]]:
    """Slice linear indices into VGPR-sized chunks, each chunk has up to pack_factor elements."""
    out = []
    for i in range(0, len(indices), pack_factor):
        chunk = indices[i:i+pack_factor]
        if len(chunk) < pack_factor:
            chunk = chunk + [None] * (pack_factor - len(chunk))  # pad for table
        out.append(chunk)
    return out

def _fpX_mapping_16x16_generic(dtype: str) -> Tuple[Dict[int, List[List[Optional[int]]]], int]:
    """
    Generic per-lane mapping for a 16x16 tile where each lane owns one column
    within a 4-row band. Returns (mapping, pack_factor). The mapping value is
    a list of VGPRs; each VGPR is a list of slots [s0, s1, ...] with length=pack_factor.
    """
    M=N=16
    pack_factor = max(1, 32 // bits_of(dtype))  # e.g., f16/bf16=2, i8=4, f32=1
    mapping: Dict[int, List[List[Optional[int]]]] = {}
    for lane in range(64):
        col = lane % 16
        row_band = lane // 16  # 0..3
        rows = [4*row_band + r for r in range(4)]  # four rows in this band
        # linear indices (row-major, 1-based) for this lane's column
        idxs = [r * N + col + 1 for r in rows]
        # pack them into VGPR-sized chunks
        vgprs = _pack_indices(idxs, pack_factor)
        mapping[lane] = vgprs
    return mapping, pack_factor

def _accum_mapping_16x16_generic(acc_dtype: str) -> Tuple[Dict[int, List[List[Optional[int]]]], int]:
    # Same structure but indices refer to the 16x16 accumulator tile.
    return _fpX_mapping_16x16_generic(acc_dtype)

# Orientation helper
def transpose_map(mapping: Dict[int, List[List[Optional[int]]]], M: int = 16, N: int = 16) -> Dict[int, List[List[Optional[int]]]]:
    def tp(idx_1b: Optional[int]) -> Optional[int]:
        if idx_1b is None: return None
        i0 = idx_1b - 1
        r, c = divmod(i0, N)
        rt, ct = c, r
        return (rt * M + ct) + 1
    out: Dict[int, List[List[Optional[int]]]] = {}
    for lane, vgprs in mapping.items():
        new_vgprs = []
        for vg in vgprs:
            new_vgprs.append([tp(x) for x in vg])
        out[lane] = new_vgprs
    return out

# ---------------------------------------------
# Visualization helpers
# ---------------------------------------------

def plot_matrix_numbering(M: int, N: int, title: str):
    mat = _matrix_indices_row_major(M, N, start_at_one=True)
    fig, ax = plt.subplots()
    ax.imshow(mat, aspect='equal')
    for i in range(M):
        for j in range(N):
            ax.text(j, i, str(mat[i, j]), ha='center', va='center', fontsize=8)
    ax.set_xticks(range(N))
    ax.set_yticks(range(M))
    ax.set_title(title)
    st.pyplot(fig)

def dynamic_df_from_mapping(mapping: Dict[int, List[List[Optional[int]]]], pack_factor: int) -> pd.DataFrame:
    rows = []
    for lane, vgprs in mapping.items():
        row = {"LaneID": lane}
        for i, slots in enumerate(vgprs):
            for s, val in enumerate(slots):
                row[f"Vgpr[{i}] s{s}"] = val
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("LaneID").reset_index(drop=True)
    return df

def plot_lane_ownership_heatmap(owner_func, M: int, N: int, title: str):
    owners = np.zeros((M, N), dtype=int)
    for r in range(M):
        for c in range(N):
            owners[r, c] = owner_func(r, c)
    fig, ax = plt.subplots()
    ax.imshow(owners, aspect='equal')
    ax.set_xticks(range(N))
    ax.set_yticks(range(M))
    ax.set_title(title)
    st.pyplot(fig)

# ---------------------------------------------
# Streamlit UI
# ---------------------------------------------

st.set_page_config(page_title="MFMA VGPR Mapping Visualizer", layout="wide")
st.title("MFMA VGPR Mapping Visualizer")
st.caption("Starter kernels for 16×16 tiles with parametric dtypes & packing. Swap kernels for ISA-accurate variants.")

with st.sidebar:
    st.header("Shape")
    blk_m = st.number_input("BlkM", min_value=1, max_value=64, value=16, step=1)
    blk_n = st.number_input("BlkN", min_value=1, max_value=64, value=16, step=1)
    blk_k = st.number_input("BlkK", min_value=1, max_value=64, value=16, step=1)

    st.header("Data types")
    a_type = st.selectbox("A type", ["f16","bf16","i8","f32"], index=0)
    b_type = st.selectbox("B type", ["f16","bf16","i8","f32"], index=0)
    acc_type = st.selectbox("Accum type", ["f32","f16","bf16","i32"], index=0)

    st.header("Wave/Flags")
    wave_size = st.selectbox("Wave size", [64], index=0)
    cbsz = st.selectbox("cbsz", [0,1,2,3], index=0)
    abid = st.selectbox("abid", [0,1,2,3], index=0)
    blgp = st.selectbox("blgp", [0,1], index=0)
    k_iters = st.number_input("K-iterations", min_value=1, value=2, step=1)

    st.header("Orientation")
    orientation_base = st.selectbox("Choose base mapping", ["A is base, B = A^T", "B is base, A = B^T"], index=0)

shape = MfmaShape(blk_m=blk_m, blk_n=blk_n, blk_k=blk_k, a_type=a_type, b_type=b_type, acc_type=acc_type, wave_size=wave_size)
flags = MfmaFlags(cbsz=cbsz, abid=abid, blgp=blgp, k_iterations=k_iters)

# Guardrail for mapping kernel support
if shape.blk_m != 16 or shape.blk_n != 16:
    st.error("This starter mapping currently supports BlkM=16 and BlkN=16 only. "
             "Expose your own mapping kernels to support other shapes (e.g., 32×32).")
    st.stop()

# Numbering
c1, c2 = st.columns([1.0, 1.4])
with c1:
    st.subheader("Tile numbering (row‑major, 1‑based)")
    plot_matrix_numbering(shape.blk_m, shape.blk_n, f"{shape.blk_m}×{shape.blk_n} numbering")
    st.caption("Indices used in the mapping tables.")

# ACC mapping
with c2:
    st.subheader(f"ACC mapping ({shape.acc_type})")
    acc_map, acc_pack = _accum_mapping_16x16_generic(shape.acc_type)
    acc_df = dynamic_df_from_mapping(acc_map, acc_pack)
    st.dataframe(acc_df, use_container_width=True)
    st.download_button("Download ACC CSV", acc_df.to_csv(index=False).encode("utf-8"),
                       file_name=f"acc_{shape.blk_m}x{shape.blk_n}_{shape.acc_type}.csv", mime="text/csv")

st.divider()

# A/B mappings with orientation
if "A is base" in orientation_base:
    base_map, base_pack = _fpX_mapping_16x16_generic(shape.a_type)
    a_map = base_map
    a_pack = base_pack
    b_map = transpose_map(base_map, M=16, N=16)
    b_pack = base_pack
else:
    base_map, base_pack = _fpX_mapping_16x16_generic(shape.b_type)
    b_map = base_map
    b_pack = base_pack
    a_map = transpose_map(base_map, M=16, N=16)
    a_pack = base_pack

c3, c4 = st.columns([1.0, 1.0])
with c3:
    st.subheader(f"A mapping ({shape.a_type})")
    a_df = dynamic_df_from_mapping(a_map, a_pack)
    st.dataframe(a_df, use_container_width=True)
    st.download_button("Download A CSV", a_df.to_csv(index=False).encode("utf-8"),
                       file_name=f"A_{shape.blk_m}x{shape.blk_n}_{shape.a_type}.csv", mime="text/csv")

with c4:
    st.subheader(f"B mapping ({shape.b_type})")
    b_df = dynamic_df_from_mapping(b_map, b_pack)
    st.dataframe(b_df, use_container_width=True)
    st.download_button("Download B CSV", b_df.to_csv(index=False).encode("utf-8"),
                       file_name=f"B_{shape.blk_m}x{shape.blk_n}_{shape.b_type}.csv", mime="text/csv")

st.divider()
st.subheader("Lane ownership heatmaps (starter heuristic)")

def owner_from_map(mapping):
    # Pick first slot of first VGPR to color who "owns" (ties don't happen in this scheme).
    def owner(r, c):
        # reconstruct lane by inverse of the banded rule
        band = r // 4
        lane = (c % 16) + 16 * band
        return lane
    return owner

c5, c6, c7 = st.columns(3)
with c5:
    plot_lane_ownership_heatmap(owner_from_map(acc_map), shape.blk_m, shape.blk_n, "ACC ownership")
with c6:
    plot_lane_ownership_heatmap(owner_from_map(a_map), shape.blk_m, shape.blk_n, "A ownership")
with c7:
    plot_lane_ownership_heatmap(owner_from_map(b_map), shape.blk_m, shape.blk_n, "B ownership")

st.divider()
st.markdown(
    """
**Notes**  
- A/B orientation is selectable. If you choose "A is base", B is generated as **transpose(A)** and vice‑versa—matching the slide note that A is the transpose of B for the 16×16 f16 case.  
- Dtype packing is generic: `pack = 32 / bits(dtype)`, so `f16/bf16→2`, `i8→4`, `f32→1`. The tables adapt slot counts automatically.  
- `BlkM/BlkN/BlkK` are parameters; this starter kernel supports 16×16 tiles. To add 32×32, etc., plug in a new mapping kernel and route by shape.
"""
)

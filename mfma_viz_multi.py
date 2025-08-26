
import io
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ---------------------------------------------
# Palette (closer to slide look)
# ---------------------------------------------
RED = "#e74c3c"   # vivid red
BLUE = "#2e86c1"  # saturated blue
BG = "#f3f6fa"    # light bluish background for tiles
GRID = "#5f6a6a"  # grid/outline

DTYPE_BITS = {"f8": 8, "i8": 8, "u8": 8, "f16": 16, "bf16": 16, "f32": 32, "i32": 32}

def bits_of(dtype: str) -> int:
    return DTYPE_BITS.get(dtype, 16)

def pack_factor_from_dtype(dtype: str) -> int:
    return max(1, 32 // bits_of(dtype))

def matrix_indices_row_major(M: int, N: int, start_at_one: bool = True) -> np.ndarray:
    mat = np.arange(M * N, dtype=int).reshape(M, N)
    if start_at_one:
        mat = mat + 1
    return mat

def tpose_idx(idx_1b: Optional[int], M: int, N: int) -> Optional[int]:
    if idx_1b is None: return None
    i0 = idx_1b - 1
    r, c = divmod(i0, N)
    rt, ct = c, r
    return (rt * M + ct) + 1

def pack_into_vgprs(indices: List[int], pack_factor: int) -> List[List[Optional[int]]]:
    out: List[List[Optional[int]]] = []
    for i in range(0, len(indices), pack_factor):
        chunk = indices[i:i+pack_factor]
        if len(chunk) < pack_factor:
            chunk = chunk + [None] * (pack_factor - len(chunk))
        out.append(chunk)
    return out

# ---------------------------------------------
# Mapping kernels (starter/teachable)
# ---------------------------------------------

def b_mapping_16x16(dtype: str, blgp: int) -> Tuple[Dict[int, List[List[Optional[int]]]], int]:
    M = N = 16
    pack = pack_factor_from_dtype(dtype)
    mapping: Dict[int, List[List[Optional[int]]]] = {}
    for lane in range(64):
        col = lane % 16
        if blgp == 1:
            col = (col + 8) % 16
        row_band = lane // 16  # 0..3
        rows = [4*row_band + r for r in range(4)]
        linear = [r * N + col + 1 for r in rows]
        mapping[lane] = pack_into_vgprs(linear, pack)
    return mapping, pack

def b_mapping_32x32(dtype: str, blgp: int) -> Tuple[Dict[int, List[List[Optional[int]]]], int]:
    M = N = 32
    pack = pack_factor_from_dtype(dtype)
    mapping: Dict[int, List[List[Optional[int]]]] = {}
    for lane in range(64):
        col = lane % 32
        if blgp == 1:
            col = (col + 16) % 32
        row_band = lane // 16  # 0..3
        rows = [8*row_band + r for r in range(8)]
        linear = [r * N + col + 1 for r in rows]
        mapping[lane] = pack_into_vgprs(linear, pack)
    return mapping, pack

def a_from_b_transpose(dtype: str, cbsz: int, abid: int, b_map: Dict[int, List[List[Optional[int]]]], M: int, N: int) -> Tuple[Dict[int, List[List[Optional[int]]]], int]:
    # A = B^T (then optional simplified "broadcast A" effect)
    pack = pack_factor_from_dtype(dtype)
    t_map: Dict[int, List[List[Optional[int]]]] = {}
    for lane, vgprs in b_map.items():
        new_vgprs = []
        for vg in vgprs:
            new_vgprs.append([tpose_idx(x, M, N) for x in vg])
        t_map[lane] = new_vgprs

    if cbsz <= 0:
        return t_map, pack

    # Simple broadcast: fix row band by abid%4
    band_rows = M // 4
    band = abid % 4
    out: Dict[int, List[List[Optional[int]]]] = {}
    for lane, _v in t_map.items():
        col = lane % N
        rows = [band_rows*band + r for r in range(band_rows)]
        linear = [r * N + col + 1 for r in rows]
        out[lane] = pack_into_vgprs(linear, pack)
    return out, pack

def accum_mapping(M: int, N: int, dtype: str) -> Tuple[Dict[int, List[List[Optional[int]]]], int]:
    pack = pack_factor_from_dtype(dtype)
    mapping: Dict[int, List[List[Optional[int]]]] = {}
    band_rows = M // 4
    for lane in range(64):
        col = lane % N
        row_band = lane // 16
        rows = [band_rows*row_band + r for r in range(band_rows)]
        linear = [r * N + col + 1 for r in rows]
        mapping[lane] = pack_into_vgprs(linear, pack)
    return mapping, pack

# ---------------------------------------------
# Slide-style "left matrix" visuals
# ---------------------------------------------

def draw_corner_blocks(M: int, N: int, title: str, red_corners: List[str], blue_corners: List[str]):
    """Draw MxN grid and highlight 4 corner sub-blocks with slide-like coloring."""
    mat = matrix_indices_row_major(M, N, start_at_one=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(np.ones((M, N)), aspect='equal', cmap='Blues', vmin=0.95, vmax=1.05, origin='upper')

    # Corner sizes (quarter tile)
    hs, ws = M//4, N//4

    def rect_and_labels(r0, c0, color, face_alpha=0.06):
        # semi-transparent fill + outline
        ax.add_patch(plt.Rectangle((c0-0.5, r0-0.5), ws, hs, fill=True, linewidth=2.5,
                                   edgecolor=color, facecolor=color, alpha=face_alpha))
        for i in range(r0, r0+hs):
            for j in range(c0, c0+ws):
                ax.text(j, i, str(mat[i, j]), ha='center', va='center', fontsize=8, color='black')

    corners = {'TL': (0, 0), 'TR': (0, N-ws), 'BL': (M-hs, 0), 'BR': (M-hs, N-ws)}
    for k in red_corners:
        r0, c0 = corners[k]; rect_and_labels(r0, c0, RED)
    for k in blue_corners:
        r0, c0 = corners[k]; rect_and_labels(r0, c0, BLUE)

    ax.set_xticks(range(N))
    ax.set_yticks(range(M))
    ax.set_title(title)
    # thin outer border
    ax.add_patch(plt.Rectangle((-0.5, -0.5), N, M, fill=False, linewidth=2.0, edgecolor=GRID))
    st.pyplot(fig)

    bio = io.BytesIO()
    fig.savefig(bio, format='png', bbox_inches='tight', dpi=200)
    bio.seek(0)
    return bio.getvalue()

# ---------------------------------------------
# Common helpers
# ---------------------------------------------

def df_from_mapping(mapping: Dict[int, List[List[Optional[int]]]]) -> pd.DataFrame:
    rows = []
    for lane, vgprs in mapping.items():
        row = {"LaneID": lane}
        for i, slots in enumerate(vgprs):
            for s, val in enumerate(slots):
                row[f"Vgpr[{i}] s{s}"] = val
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("LaneID").reset_index(drop=True)
    return df

def plot_heatmap(owner_fn, M, N, title):
    owners = np.zeros((M, N), dtype=int)
    for r in range(M):
        for c in range(N):
            owners[r, c] = owner_fn(r, c)
    fig, ax = plt.subplots()
    ax.imshow(owners, aspect='equal', origin='upper')
    ax.set_xticks(range(N))
    ax.set_yticks(range(M))
    ax.set_title(title)
    st.pyplot(fig)

# ---------------------------------------------
# Streamlit UI
# ---------------------------------------------

st.set_page_config(page_title="MFMA VGPR Mapping Visualizer", layout="wide")
st.title("MFMA VGPR Mapping Visualizer")
st.caption("Orientation fixed to match slides (top-left is row 0/col 0). Colors tuned to slide palette. B is base; A = Bᵀ.")

inst = st.sidebar.selectbox(
    "Instruction",
    [
        "mfma_f32_16x16x16_f16/bf16",
        "mfma_f32_32x32x8_f16/bf16",
        "mfma_i32_16x16x32_i8",
    ],
    index=0,
)

if inst == "mfma_f32_16x16x16_f16/bf16":
    M, N, K_step = 16, 16, 16
    st.sidebar.markdown("**Shape:** 16×16×16")
    a_type = st.sidebar.selectbox("A type", ["f16","bf16"], index=0)
    b_type = st.sidebar.selectbox("B type", ["f16","bf16"], index=0)
    acc_type = "f32"; st.sidebar.write("Accum type: **f32**")
    blgp = st.sidebar.selectbox("blgp (B)", [0,1], index=0)
    cbsz = st.sidebar.selectbox("cbsz (A)", [0,1,2,3], index=0)
    abid = st.sidebar.selectbox("abid (A)", [0,1,2,3], index=0)
    blk_k = st.sidebar.number_input("BlkK", min_value=16, max_value=256, value=16, step=16)
    k_iters = max(1, (blk_k + K_step - 1) // K_step)
    st.sidebar.write(f"K-iterations (derived): **{k_iters}**")

    c1, c2 = st.columns([1.0, 1.4])
    with c1:
        st.subheader("Left-matrix visuals (downloadable PNGs)")
        a_png = draw_corner_blocks(M, N, "A — corner blocks", red_corners=['TL','BL'], blue_corners=['TR','BR'])
        st.download_button("Download A visual (PNG)", a_png, file_name="A_left_visual_16x16.png", mime="image/png")
        b_png = draw_corner_blocks(M, N, "B — corner blocks", red_corners=['TL','TR'], blue_corners=['BL','BR'])
        st.download_button("Download B visual (PNG)", b_png, file_name="B_left_visual_16x16.png", mime="image/png")
        acc_png = draw_corner_blocks(M, N, "ACC — corner blocks", red_corners=['TL','TR'], blue_corners=['BL','BR'])
        st.download_button("Download ACC visual (PNG)", acc_png, file_name="ACC_left_visual_16x16.png", mime="image/png")

    b_map, b_pack = b_mapping_16x16(b_type, blgp)
    a_map, a_pack = a_from_b_transpose(a_type, cbsz, abid, b_map, M, N)
    acc_map, acc_pack = accum_mapping(M, N, acc_type)

    with c2:
        st.subheader(f"ACC mapping ({acc_type}; pack={acc_pack})")
        acc_df = df_from_mapping(acc_map)
        st.dataframe(acc_df, use_container_width=True)

    st.divider()
    c3, c4 = st.columns(2)
    with c3:
        st.subheader(f"A mapping ({a_type}; pack={a_pack}; cbsz={cbsz}, abid={abid})")
        a_df = df_from_mapping(a_map)
        st.dataframe(a_df, use_container_width=True)
    with c4:
        st.subheader(f"B mapping ({b_type}; pack={b_pack}; blgp={blgp})")
        b_df = df_from_mapping(b_map)
        st.dataframe(b_df, use_container_width=True)

    st.divider()
    st.subheader("Lane ownership heatmaps")
    def owner_b(r, c):
        band = r // 4; col = c % 16
        if blgp == 1: col = (col + 8) % 16
        return col + 16 * band
    def owner_a(r, c):
        if cbsz <= 0: return owner_b(c, r)
        band = (abid % 4); col = c % 16
        return col + 16 * band
    def owner_acc(r, c):
        band = r // 4; col = c % 16
        return col + 16 * band

    c5, c6, c7 = st.columns(3)
    with c5: plot_heatmap(owner_acc, M, N, "ACC")
    with c6: plot_heatmap(owner_a, M, N, "A (broadcast-aware)")
    with c7: plot_heatmap(owner_b, M, N, "B (blgp-aware)")

elif inst == "mfma_f32_32x32x8_f16/bf16":
    M, N, K_step = 32, 32, 8
    st.sidebar.markdown("**Shape:** 32×32×8")
    a_type = st.sidebar.selectbox("A type", ["f16","bf16"], index=0)
    b_type = st.sidebar.selectbox("B type", ["f16","bf16"], index=0)
    acc_type = "f32"; st.sidebar.write("Accum type: **f32**")
    blgp = st.sidebar.selectbox("blgp (B)", [0,1], index=0)
    cbsz = st.sidebar.selectbox("cbsz (A)", [0,1,2,3], index=0)
    abid = st.sidebar.selectbox("abid (A)", [0,1,2,3], index=0)
    blk_k = st.sidebar.number_input("BlkK", min_value=8, max_value=256, value=8, step=8)
    k_iters = max(1, (blk_k + K_step - 1) // K_step)
    st.sidebar.write(f"K-iterations (derived): **{k_iters}**")

    c1, c2 = st.columns([1.0, 1.4])
    with c1:
        st.subheader("Left-matrix visuals (downloadable PNGs)")
        a_png = draw_corner_blocks(M, N, "A — corner blocks", red_corners=['TL','BL'], blue_corners=['TR','BR'])
        st.download_button("Download A visual (PNG)", a_png, file_name="A_left_visual_32x32.png", mime="image/png")
        b_png = draw_corner_blocks(M, N, "B — corner blocks", red_corners=['TL','TR'], blue_corners=['BL','BR'])
        st.download_button("Download B visual (PNG)", b_png, file_name="B_left_visual_32x32.png", mime="image/png")
        acc_png = draw_corner_blocks(M, N, "ACC — corner blocks", red_corners=['TL','TR'], blue_corners=['BL','BR'])
        st.download_button("Download ACC visual (PNG)", acc_png, file_name="ACC_left_visual_32x32.png", mime="image/png")

    b_map, b_pack = b_mapping_32x32(b_type, blgp)
    a_map, a_pack = a_from_b_transpose(a_type, cbsz, abid, b_map, M, N)
    acc_map, acc_pack = accum_mapping(M, N, acc_type)

    with c2:
        st.subheader(f"ACC mapping ({acc_type}; pack={acc_pack})")
        acc_df = df_from_mapping(acc_map)
        st.dataframe(acc_df, use_container_width=True)

    st.divider()
    c3, c4 = st.columns(2)
    with c3:
        st.subheader(f"A mapping ({a_type}; pack={a_pack}; cbsz={cbsz}, abid={abid})")
        a_df = df_from_mapping(a_map)
        st.dataframe(a_df, use_container_width=True)
    with c4:
        st.subheader(f"B mapping ({b_type}; pack={b_pack}; blgp={blgp})")
        b_df = df_from_mapping(b_map)
        st.dataframe(b_df, use_container_width=True)

    st.divider()
    st.subheader("Lane ownership heatmaps")
    def owner_b(r, c):
        band = r // 8; col = c % 32
        if blgp == 1: col = (col + 16) % 32
        return col + 32 * band
    def owner_a(r, c):
        if cbsz <= 0: return owner_b(c, r)
        band = (abid % 4); col = c % 32
        return col + 32 * band
    def owner_acc(r, c):
        band = r // 8; col = c % 32
        return col + 32 * band

    c5, c6, c7 = st.columns(3)
    with c5: plot_heatmap(owner_acc, M, N, "ACC")
    with c6: plot_heatmap(owner_a, M, N, "A (broadcast-aware)")
    with c7: plot_heatmap(owner_b, M, N, "B (blgp-aware)")

else:  # mfma_i32_16x16x32_i8
    M, N, K_step = 16, 16, 32
    st.sidebar.markdown("**Shape:** 16×16×32")
    a_type = b_type = "i8"; acc_type = "i32"
    st.sidebar.write("A/B type: **i8**"); st.sidebar.write("Accum type: **i32**")
    blgp = st.sidebar.selectbox("blgp (B)", [0,1], index=0)
    cbsz = st.sidebar.selectbox("cbsz (A)", [0,1,2,3], index=0)
    abid = st.sidebar.selectbox("abid (A)", [0,1,2,3], index=0)
    blk_k = st.sidebar.number_input("BlkK", min_value=32, max_value=256, value=32, step=32)
    k_iters = max(1, (blk_k + K_step - 1) // K_step)
    st.sidebar.write(f"K-iterations (derived): **{k_iters}**")

    c1, c2 = st.columns([1.0, 1.4])
    with c1:
        st.subheader("Left-matrix visuals (downloadable PNGs)")
        a_png = draw_corner_blocks(M, N, "A — corner blocks", red_corners=['TL','BL'], blue_corners=['TR','BR'])
        st.download_button("Download A visual (PNG)", a_png, file_name="A_left_visual_16x16_i8.png", mime="image/png")
        b_png = draw_corner_blocks(M, N, "B — corner blocks", red_corners=['TL','TR'], blue_corners=['BL','BR'])
        st.download_button("Download B visual (PNG)", b_png, file_name="B_left_visual_16x16_i8.png", mime="image/png")
        acc_png = draw_corner_blocks(M, N, "ACC — corner blocks", red_corners=['TL','TR'], blue_corners=['BL','BR'])
        st.download_button("Download ACC visual (PNG)", acc_png, file_name="ACC_left_visual_16x16_i32.png", mime="image/png")

    b_map, b_pack = b_mapping_16x16(b_type, blgp)
    a_map, a_pack = a_from_b_transpose(a_type, cbsz, abid, b_map, M, N)
    acc_map, acc_pack = accum_mapping(M, N, acc_type)

    with c2:
        st.subheader(f"ACC mapping ({acc_type}; pack={acc_pack})")
        acc_df = df_from_mapping(acc_map)
        st.dataframe(acc_df, use_container_width=True)

    st.divider()
    c3, c4 = st.columns(2)
    with c3:
        st.subheader(f"A mapping ({a_type}; pack={a_pack}; cbsz={cbsz}, abid={abid})")
        a_df = df_from_mapping(a_map)
        st.dataframe(a_df, use_container_width=True)
    with c4:
        st.subheader(f"B mapping ({b_type}; pack={b_pack}; blgp={blgp})")
        b_df = df_from_mapping(b_map)
        st.dataframe(b_df, use_container_width=True)

    st.divider()
    st.subheader("Lane ownership heatmaps")
    def owner_b(r, c):
        band = r // 4; col = c % 16
        if blgp == 1: col = (col + 8) % 16
        return col + 16 * band
    def owner_a(r, c):
        if cbsz <= 0: return owner_b(c, r)
        band = (abid % 4); col = c % 16
        return col + 16 * band
    def owner_acc(r, c):
        band = r // 4; col = c % 16
        return col + 16 * band

    c5, c6, c7 = st.columns(3)
    with c5: plot_heatmap(owner_acc, M, N, "ACC")
    with c6: plot_heatmap(owner_a, M, N, "A (broadcast-aware)")
    with c7: plot_heatmap(owner_b, M, N, "B (blgp-aware)")

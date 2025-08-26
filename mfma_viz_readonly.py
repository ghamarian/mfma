
import io
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ------------ Palette ------------
RED  = "#e74c3c"
BLUE = "#2e86c1"
GOLD = "#f1c40f"
GREEN = "#27ae60"
GRID = "#2c3e50"
DOT  = "#7f8c8d"

DTYPE_BITS = {"f8":8,"i8":8,"u8":8,"f16":16,"bf16":16,"f32":32,"i32":32}

def bits_of(dtype:str)->int: return DTYPE_BITS.get(dtype,16)
def pack_factor(dtype:str)->int: return max(1, 32 // bits_of(dtype))

def matrix_indices(M:int, N:int, one_based=True)->np.ndarray:
    a = np.arange(M*N, dtype=int).reshape(M,N)
    return a+1 if one_based else a

def tpose_idx(idx_1b: Optional[int], M:int, N:int)->Optional[int]:
    if idx_1b is None: return None
    i0 = idx_1b-1
    r, c = divmod(i0, N)  # 0-based
    return (c * M + r) + 1  # transpose

def pack_chunks(indices: List[int], p:int)->List[List[Optional[int]]]:
    out=[]
    for i in range(0,len(indices),p):
        chunk = indices[i:i+p]
        if len(chunk)<p: chunk += [None]*(p-len(chunk))
        out.append(chunk)
    return out

# ------------ Mapping kernels (B base; A = B^T) ------------
def b_map_16x16(dtype:str)->Tuple[Dict[int,List[List[Optional[int]]]], int]:
    M=N=16; p=pack_factor(dtype); out={}
    for lane in range(64):
        col = lane % 16
        band = lane // 16
        rows = [4*band + r for r in range(4)]
        idxs = [r*N + col + 1 for r in rows]
        out[lane] = pack_chunks(idxs, p)
    return out, p

def b_map_32x32(dtype:str)->Tuple[Dict[int,List[List[Optional[int]]]], int]:
    M=N=32; p=pack_factor(dtype); out={}
    for lane in range(64):
        col = lane % 32
        band = lane // 16
        rows = [8*band + r for r in range(8)]
        idxs = [r*N + col + 1 for r in rows]
        out[lane] = pack_chunks(idxs, p)
    return out, p

def a_from_b(dtype:str, b_map:Dict[int,List[List[Optional[int]]]], M:int, N:int)->Tuple[Dict[int,List[List[Optional[int]]]], int]:
    p=pack_factor(dtype); out={}
    for lane,vgprs in b_map.items():
        out[lane]=[[tpose_idx(x,M,N) for x in slots] for slots in vgprs]
    return out, p

def acc_map_generic(M:int,N:int,dtype:str)->Tuple[Dict[int,List[List[Optional[int]]]], int]:
    p=pack_factor(dtype); out={}; band_rows=M//4
    for lane in range(64):
        col = lane % N
        band = lane // 16
        rows = [band_rows*band + r for r in range(band_rows)]
        idxs = [r*N + col + 1 for r in rows]
        out[lane] = pack_chunks(idxs, p)
    return out, p

# ------------ Left-matrix corner visuals ------------
def draw_corner_blocks(M:int,N:int,title:str, red_corners:List[str], blue_corners:List[str], label_mode:str="row"):
    """
    label_mode: "row"  -> show row-major numbers in the blocks (B/ACC look)
                "col"  -> show column-major numbers (A look)
    """
    mat = matrix_indices(M,N,True)
    # larger cells & font for readability (especially 32x32)
    if max(M,N)==16:
        fig_size=8.5; fs_num=14; dpi=220; dot_size=28
    else:
        fig_size=12.0; fs_num=13; dpi=230; dot_size=36
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=dpi)
    ax.imshow(np.ones((M,N)), origin='upper', aspect='equal', cmap='Greys', vmin=0.96, vmax=1.04)
    hs, ws = M//4, N//4
    corners = {'TL':(0,0),'TR':(0,N-ws),'BL':(M-hs,0),'BR':(M-hs,N-ws)}

    def get_label(i,j):
        return int(mat[i,j]) if label_mode=="row" else int(mat.T[i,j])

    def paint(r0,c0,color):
        ax.add_patch(plt.Rectangle((c0-0.5, r0-0.5), ws, hs, fill=True, linewidth=2.6,
                                   edgecolor=color, facecolor=color, alpha=0.10))
        for i in range(r0, r0+hs):
            for j in range(c0, c0+ws):
                ax.text(j, i, str(get_label(i,j)), ha='center', va='center', fontsize=fs_num, color='#111')

    for k in red_corners:
        r0,c0=corners[k]; paint(r0,c0,RED)
    for k in blue_corners:
        r0,c0=corners[k]; paint(r0,c0,BLUE)

    # dotted region between colored corners (sparse dot grid)
    r1, c1 = hs, ws
    r2, c2 = M-hs, N-ws
    ys = np.linspace(r1, r2-1, 8)
    xs = np.linspace(c1, c2-1, 8)
    X, Y = np.meshgrid(xs, ys)
    ax.scatter(X.flatten(), Y.flatten(), s=dot_size, c=DOT, marker='o', linewidths=0)

    # clean slide look: no ticks
    ax.set_xticks([]); ax.set_yticks([])
    ax.add_patch(plt.Rectangle((-0.5,-0.5), N, M, fill=False, linewidth=2.4, edgecolor=GRID))
    ax.set_title(title, fontsize=16)
    st.pyplot(fig, use_container_width=True)
    bio = io.BytesIO(); fig.savefig(bio, format='png', bbox_inches='tight', dpi=dpi); bio.seek(0)
    return bio.getvalue()

# ------------ SIMD execution schematic ------------
def draw_simd_execution(M:int,N:int, title:str):
    """
    Stylized figure: A strip on the left, B strip on top, four colored tiles
    moving along the diagonal into an accumulator strip.
    """
    fig_w = 13 if max(M,N)==32 else 11
    fig_h = 7.5
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Background central canvas
    ax.add_patch(plt.Rectangle((1.5, 1.0), 8.5, 5.5, fill=False, linestyle='-', linewidth=1.5, edgecolor=GRID))

    # A vertical strip
    ax.add_patch(plt.Rectangle((0.5, 1.0), 0.6, 5.5, fill=False, linewidth=2, edgecolor=GRID))
    ax.text(0.8, 6.8, "A", ha='center', va='center', fontsize=18)
    # B horizontal strip
    ax.add_patch(plt.Rectangle((1.5, 6.0), 8.5, 0.6, fill=False, linewidth=2, edgecolor=GRID))
    ax.text(10.3, 6.3, "B", ha='center', va='center', fontsize=18)
    # Accumulator strip
    ax.add_patch(plt.Rectangle((10.5, 1.0), 0.6, 5.5, fill=False, linewidth=2, edgecolor=GRID))
    ax.text(10.8, 6.8, "accum", ha='center', va='center', fontsize=16)

    # Four colored tiles along diagonal with sigma labels
    colors = [RED, GOLD, GREEN, BLUE]
    starts = [(2.2,5.3), (4.2,4.1), (6.2,2.9), (8.2,1.7)]
    for (x,y), col in zip(starts, colors):
        ax.add_patch(plt.Rectangle((x, y), 1.2, 1.2, fill=False, linewidth=2.5, edgecolor=col))
        ax.text(x+0.6, y+1.35, "Σ", ha='center', va='bottom', fontsize=18)

    ax.axis('off')
    ax.set_title(title, fontsize=18)
    st.pyplot(fig, use_container_width=True)
    bio = io.BytesIO(); fig.savefig(bio, format='png', bbox_inches='tight', dpi=220); bio.seek(0)
    return bio.getvalue()

# ------------ Streamlit ------------
st.set_page_config(page_title="MFMA VGPR Mapping — slide style", layout="wide")
st.title("MFMA VGPR Mapping — slide style")

inst = st.sidebar.selectbox(
    "Instruction",
    ["mfma_f32_16x16x16_f16/bf16","mfma_f32_32x32x8_f16/bf16","mfma_i32_16x16x32_i8"],
    index=0,
)

INFO = {
    "mfma_f32_16x16x16_f16/bf16": {"tile":(16,16,16), "A/B":"f16 | bf16", "acc":"f32", "note":"GEMM example K=32 ⇒ two 16-wide K iterations."},
    "mfma_f32_32x32x8_f16/bf16":  {"tile":(32,32,8),  "A/B":"f16 | bf16", "acc":"f32", "note":"K consumed in steps of 8 per MFMA."},
    "mfma_i32_16x16x32_i8":       {"tile":(16,16,32), "A/B":"i8",         "acc":"i32", "note":"Int8 form; K step is 32."},
}

M,N,K = INFO[inst]["tile"]

with st.sidebar:
    st.header("Instruction details")
    st.markdown(f"""
**BlkM×BlkN×BlkK:** `{M} × {N} × {K}`  
**A/B dtype:** `{INFO[inst]['A/B']}`  
**Accumulator:** `{INFO[inst]['acc']}`  
**Relationship:** `A = Bᵀ` in this training example.  
{INFO[inst]['note']}
""")

# Compute mappings
if inst == "mfma_f32_16x16x16_f16/bf16":
    b_map, b_pack = b_map_16x16("f16")
    a_map, a_pack = a_from_b("f16", b_map, M, N)
    acc_map, acc_pack = acc_map_generic(M,N,"f32")
elif inst == "mfma_f32_32x32x8_f16/bf16":
    b_map, b_pack = b_map_32x32("f16")
    a_map, a_pack = a_from_b("f16", b_map, M, N)
    acc_map, acc_pack = acc_map_generic(M,N,"f32")
else:
    b_map, b_pack = b_map_16x16("i8")
    a_map, a_pack = a_from_b("i8", b_map, M, N)
    acc_map, acc_pack = acc_map_generic(M,N,"i32")

col_left, col_right = st.columns([1.0, 1.6])

with col_left:
    st.subheader("Corner visuals (larger cells + dotted center)")
    a_png = draw_corner_blocks(M,N,"A — corner blocks", red_corners=['TL','BL'], blue_corners=['TR','BR'], label_mode="col")
    st.download_button("Download A (PNG)", a_png, file_name=f"A_left_{M}x{N}.png", mime="image/png")
    b_png = draw_corner_blocks(M,N,"B — corner blocks", red_corners=['TL','TR'], blue_corners=['BL','BR'], label_mode="row")
    st.download_button("Download B (PNG)", b_png, file_name=f"B_left_{M}x{N}.png", mime="image/png")
    acc_png = draw_corner_blocks(M,N,"ACC — corner blocks", red_corners=['TL','TR'], blue_corners=['BL','BR'], label_mode="row")
    st.download_button("Download ACC (PNG)", acc_png, file_name=f"ACC_left_{M}x{N}.png", mime="image/png")

def df_from_mapping(mapping: Dict[int, List[List[Optional[int]]]])->pd.DataFrame:
    rows=[]
    for lane,vgprs in mapping.items():
        row={"LaneID":lane}
        for i,slots in enumerate(vgprs):
            for s,val in enumerate(slots):
                row[f"Vgpr[{i}] s{s}"]=val
        rows.append(row)
    return pd.DataFrame(rows).sort_values("LaneID").reset_index(drop=True)

with col_right:
    st.subheader(f"A mapping table (pack={a_pack})")
    a_df = df_from_mapping(a_map)
    st.dataframe(a_df, use_container_width=True)
    st.download_button("Download A mapping CSV", a_df.to_csv(index=False).encode("utf-8"),
                       file_name=f"A_map_{M}x{N}.csv", mime="text/csv")

    st.subheader(f"B mapping table (pack={b_pack})")
    b_df = df_from_mapping(b_map)
    st.dataframe(b_df, use_container_width=True)
    st.download_button("Download B mapping CSV", b_df.to_csv(index=False).encode("utf-8"),
                       file_name=f"B_map_{M}x{N}.csv", mime="text/csv")

    st.subheader(f"ACC mapping table (pack={acc_pack})")
    acc_df = df_from_mapping(acc_map)
    st.dataframe(acc_df, use_container_width=True)
    st.download_button("Download ACC mapping CSV", acc_df.to_csv(index=False).encode("utf-8"),
                       file_name=f"ACC_map_{M}x{N}.csv", mime="text/csv")

st.divider()
st.subheader("SIMD execution (schematic)")
_ = draw_simd_execution(M,N, "SIMD execution — conceptual flow")

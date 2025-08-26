
import io, os, json, subprocess, shlex
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ---------------- Palette ----------------
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

# ---------------- Heuristic kernels (fallback) ----------------
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

# ---------------- Calculator wrapper (optional ISA-accurate) ----------------
def run_calculator(path:str, args:List[str])->Tuple[bool, str]:
    try:
        cmd = [path] + args
        # Use python to run script to avoid exec perms issues
        if not path.endswith(".py"):
            # assume it's directly executable
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, text=True)
        else:
            res = subprocess.run(["python3", path] + args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, text=True)
        ok = (res.returncode == 0)
        return ok, res.stdout if ok else res.stderr
    except Exception as e:
        return False, str(e)

def parse_register_layout(text:str)->pd.DataFrame:
    """
    The calculator prints tabulated rows with lanes/registers. We'll do a simple parse:
    Expect lines like: "register lane value" OR a table. We'll try to detect CSV-like segments.
    For robustness we return raw text if parsing fails.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    # try to find header
    try:
        # find lines starting with "Lane"
        header_idx = None
        for i,ln in enumerate(lines):
            if ln.lower().startswith("lane") or ln.lower().startswith("laneid"):
                header_idx = i; break
        if header_idx is None: raise ValueError("no header")
        # split by whitespace
        cols = lines[header_idx].split()
        rows = []
        for ln in lines[header_idx+1:]:
            parts = ln.split()
            if len(parts) != len(cols): break
            rows.append(parts)
        df = pd.DataFrame(rows, columns=cols)
        return df
    except Exception:
        return pd.DataFrame({"raw": lines})

# ---------------- Visuals ----------------
def draw_corner_blocks(M:int,N:int,title:str, red_corners:List[str], blue_corners:List[str], label_mode:str="row", zoom:float=1.0, dot_density:int=8):
    mat = matrix_indices(M,N,True)
    base_size = 8.5 if max(M,N)==16 else 12.0
    fig_size = base_size * zoom
    fs_num = (14 if max(M,N)==16 else 13) * zoom
    dot_size = (28 if max(M,N)==16 else 36) * zoom

    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
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

    # dotted center
    r1, c1 = hs, ws
    r2, c2 = M-hs, N-ws
    ys = np.linspace(r1, r2-1, dot_density)
    xs = np.linspace(c1, c2-1, dot_density)
    X, Y = np.meshgrid(xs, ys)
    ax.scatter(X.flatten(), Y.flatten(), s=dot_size, c=DOT, marker='o', linewidths=0)

    ax.set_xticks([]); ax.set_yticks([])
    ax.add_patch(plt.Rectangle((-0.5,-0.5), N, M, fill=False, linewidth=2.4, edgecolor=GRID))
    ax.set_title(title, fontsize=16*zoom)
    bio = io.BytesIO(); fig.savefig(bio, format='png', bbox_inches='tight', dpi=220); bio.seek(0)
    st.image(bio, caption=title, use_column_width=True)
    return bio.getvalue()

def draw_corner_single(M:int,N:int,title:str, which:str, label_mode:str="row", zoom:float=1.2):
    """Render a single corner (bigger numbers)."""
    mat = matrix_indices(M,N,True)
    hs, ws = M//4, N//4
    corners = {'TL':(0,0),'TR':(0,N-ws),'BL':(M-hs,0),'BR':(M-hs,N-ws)}
    r0,c0 = corners[which]
    fig_size = (7.5 if max(M,N)==16 else 9.0) * zoom
    fs_num = (16 if max(M,N)==16 else 15) * zoom
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    ax.imshow(np.ones((hs,ws)), origin='upper', aspect='equal', cmap='Greys', vmin=0.96, vmax=1.04)
    def get_label(i,j):
        return int(mat[r0+i, c0+j]) if label_mode=="row" else int(mat.T[r0+i, c0+j])
    for i in range(hs):
        for j in range(ws):
            ax.text(j, i, str(get_label(i,j)), ha='center', va='center', fontsize=fs_num, color='#111')
    ax.set_xticks([]); ax.set_yticks([])
    ax.add_patch(plt.Rectangle((-0.5,-0.5), ws, hs, fill=False, linewidth=2.4, edgecolor=GRID))
    ax.set_title(f"{title} — {which}", fontsize=18*zoom)
    bio = io.BytesIO(); fig.savefig(bio, format='png', bbox_inches='tight', dpi=240); bio.seek(0)
    st.image(bio, caption=f"{title} — {which}", use_column_width=True)
    return bio.getvalue()

# ---------------- Streamlit ----------------
st.set_page_config(page_title="AMD Matrix Instruction Visualizer (ISA-backed when available)", layout="wide")
st.title("AMD Matrix Instruction Visualizer")

with st.sidebar:
    st.subheader("Instruction source")
    calc_path = st.text_input("Path to AMD matrix_calculator.py (optional for full coverage)", value="matrix_calculator.py")
    use_calc = st.checkbox("Use AMD calculator if available", value=True)
    st.caption("Get it from: https://github.com/RadeonOpenCompute/amd_matrix_instruction_calculator")

    arch = st.selectbox("Architecture", ["CDNA1/MI100", "CDNA2/MI200 (gfx90a)", "CDNA3/MI300 (gfx94x)", "RDNA3 (gfx11xx)"], index=1)

    # lazy-map to tool names
    arch_token = {
        "CDNA1/MI100": "MI100",
        "CDNA2/MI200 (gfx90a)": "MI200",
        "CDNA3/MI300 (gfx94x)": "MI300",
        "RDNA3 (gfx11xx)": "RDNA3",
    }[arch]

    inst_list = []
    calc_ok = False
    if use_calc and os.path.exists(calc_path):
        ok, out = run_calculator(calc_path, ["-a", arch_token, "-L"])
        calc_ok = ok
        if ok:
            # expect one mnemonic per line
            inst_list = [ln.strip() for ln in out.splitlines() if ln.strip() and not ln.lower().startswith(("available","legal"))]
        else:
            st.warning("Calculator error while listing instructions:\n" + out[:400])
    else:
        st.info("Calculator not found. Falling back to three common MFMA kernels.")

    if inst_list:
        inst = st.selectbox("Instruction", inst_list, index=0)
    else:
        inst = st.selectbox("Instruction (fallback)", ["v_mfma_f32_16x16x16_f16", "v_mfma_f32_32x32x8_f16", "v_mfma_i32_16x16x32_i8"], index=0)

    # Common modifiers (only used when supported)
    st.subheader("Modifiers")
    cbsz = st.number_input("cbsz (A broadcast size, log2)", min_value=0, max_value=3, value=0, step=1)
    abid = st.number_input("abid (A broadcast id)", min_value=0, max_value=7, value=0, step=1)
    blgp = st.number_input("blgp (B lane group pattern)", min_value=0, max_value=7, value=0, step=1)

    st.subheader("Visual options")
    zoom = st.slider("Zoom", 0.8, 2.0, 1.2, 0.1)
    dot_density = st.slider("Dot density", 4, 16, 8, 1)
    corner_mode = st.radio("Corner layout", ["Full tile", "One corner at a time"], index=1)

# Attempt to fetch layouts from calculator when available
def fetch_layouts_via_calc():
    info = {"tile": None, "A": None, "B": None, "C": None, "D": None, "note": ""}
    if not (use_calc and calc_ok): return None
    # We try to query register layout tables for A/B/C/D
    base_args = ["-a", arch_token, "-i", inst]
    def q(mat_flag):
        args = base_args + [mat_flag, "--register-layout"]
        if cbsz: args += ["--cbsz", str(cbsz)]
        if abid: args += ["--abid", str(abid)]
        if blgp: args += ["--blgp", str(blgp)]
        ok, out = run_calculator(calc_path, args); return ok, out
    okA, outA = q("-A"); okB, outB = q("-B"); okC, outC = q("-C"); okD, outD = q("-D")
    if not (okA and okB and okC and okD):
        return None
    # We don't try to parse complex ASCII tables fully; display raw text & offer download
    info["A_raw"] = outA; info["B_raw"] = outB; info["C_raw"] = outC; info["D_raw"] = outD
    # Try to detect tile MxNxK from mnemonic
    import re
    m = re.search(r"(\d+)x(\d+)x(\d+)", inst)
    if m: info["tile"] = tuple(map(int, m.groups()))
    return info

isa_info = fetch_layouts_via_calc()

# Page layout
c1, c2 = st.columns([1.0, 1.6])

with c1:
    st.subheader("Corner visuals")
    # Decide M,N from instruction name (best-effort)
    if isa_info and isa_info.get("tile"):
        M,N,_ = isa_info["tile"]
    elif "32x32" in inst: M=N=32
    else: M=N=16

    if corner_mode == "Full tile":
        a_png = draw_corner_blocks(M,N,"A — corner blocks", red_corners=['TL','BL'], blue_corners=['TR','BR'], label_mode="col", zoom=zoom, dot_density=dot_density)
        st.download_button("Download A (PNG)", a_png, file_name=f"A_left_{M}x{N}.png", mime="image/png")
        b_png = draw_corner_blocks(M,N,"B — corner blocks", red_corners=['TL','TR'], blue_corners=['BL','BR'], label_mode="row", zoom=zoom, dot_density=dot_density)
        st.download_button("Download B (PNG)", b_png, file_name=f"B_left_{M}x{N}.png", mime="image/png")
        acc_png = draw_corner_blocks(M,N,"ACC — corner blocks", red_corners=['TL','TR'], blue_corners=['BL','BR'], label_mode="row", zoom=zoom, dot_density=dot_density)
        st.download_button("Download ACC (PNG)", acc_png, file_name=f"ACC_left_{M}x{N}.png", mime="image/png")
    else:
        st.write("**A block**")
        for corner in ["TL","TR","BL","BR"]:
            _ = draw_corner_single(M,N,"A", which=corner, label_mode="col", zoom=zoom)
        st.write("**B block**")
        for corner in ["TL","TR","BL","BR"]:
            _ = draw_corner_single(M,N,"B", which=corner, label_mode="row", zoom=zoom)
        st.write("**ACC block**")
        for corner in ["TL","TR","BL","BR"]:
            _ = draw_corner_single(M,N,"ACC", which=corner, label_mode="row", zoom=zoom)

with c2:
    st.subheader("Register layout tables")
    if isa_info:
        st.caption("ISA-accurate register layouts sourced from AMD matrix calculator.")
        st.download_button("Download A layout (raw)", isa_info["A_raw"].encode("utf-8"), file_name="A_register_layout.txt")
        st.download_button("Download B layout (raw)", isa_info["B_raw"].encode("utf-8"), file_name="B_register_layout.txt")
        st.download_button("Download C layout (raw)", isa_info["C_raw"].encode("utf-8"), file_name="C_register_layout.txt")
        st.download_button("Download D layout (raw)", isa_info["D_raw"].encode("utf-8"), file_name="D_register_layout.txt")
        st.text_area("A register layout", isa_info["A_raw"], height=240)
        st.text_area("B register layout", isa_info["B_raw"], height=240)
        st.text_area("C register layout", isa_info["C_raw"], height=240)
        st.text_area("D register layout", isa_info["D_raw"], height=240)
    else:
        st.info("Calculator not available or failed. Showing fallback tables for a few common MFMA kernels.")
        # Fallback: pick kernel by name
        if "32x32" in inst:
            b_map, b_pack = b_map_32x32("f16"); M=N=32
            a_map, a_pack = a_from_b("f16", b_map, M, N)
            acc_map, acc_pack = acc_map_generic(M,N,"f32")
        elif "i32" in inst or "i8" in inst:
            b_map, b_pack = b_map_16x16("i8"); M=N=16
            a_map, a_pack = a_from_b("i8", b_map, M, N)
            acc_map, acc_pack = acc_map_generic(M,N,"i32")
        else:
            b_map, b_pack = b_map_16x16("f16"); M=N=16
            a_map, a_pack = a_from_b("f16", b_map, M, N)
            acc_map, acc_pack = acc_map_generic(M,N,"f32")

        def df_from_mapping(mapping: Dict[int, List[List[Optional[int]]]])->pd.DataFrame:
            rows=[]
            for lane,vgprs in mapping.items():
                row={"LaneID":lane}
                for i,slots in enumerate(vgprs):
                    for s,val in enumerate(slots):
                        row[f"Vgpr[{i}] s{s}"]=val
                rows.append(row)
            return pd.DataFrame(rows).sort_values("LaneID").reset_index(drop=True)

        st.write(f"**A mapping (pack={pack_factor('f16') if 'i8' not in inst else pack_factor('i8')})**")
        st.dataframe(df_from_mapping(a_map), use_container_width=True)
        st.write(f"**B mapping (pack={pack_factor('f16') if 'i8' not in inst else pack_factor('i8')})**")
        st.dataframe(df_from_mapping(b_map), use_container_width=True)
        st.write(f"**ACC mapping (pack={pack_factor('f32') if 'i8' not in inst else pack_factor('i32')})**")
        st.dataframe(df_from_mapping(acc_map), use_container_width=True)

st.divider()
st.subheader("SIMD execution (schematic) — conceptual flow")
def draw_simd_execution(M:int,N:int, title:str):
    fig_w = 13 if max(M,N)==32 else 11
    fig_h = 7.5
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.add_patch(plt.Rectangle((1.5, 1.0), 8.5, 5.5, fill=False, linestyle='-', linewidth=1.5, edgecolor=GRID))
    ax.add_patch(plt.Rectangle((0.5, 1.0), 0.6, 5.5, fill=False, linewidth=2, edgecolor=GRID))
    ax.text(0.8, 6.8, "A", ha='center', va='center', fontsize=18)
    ax.add_patch(plt.Rectangle((1.5, 6.0), 8.5, 0.6, fill=False, linewidth=2, edgecolor=GRID))
    ax.text(10.3, 6.3, "B", ha='center', va='center', fontsize=18)
    ax.add_patch(plt.Rectangle((10.5, 1.0), 0.6, 5.5, fill=False, linewidth=2, edgecolor=GRID))
    ax.text(10.8, 6.8, "accum", ha='center', va='center', fontsize=16)
    colors = [RED, GOLD, GREEN, BLUE]; starts = [(2.2,5.3), (4.2,4.1), (6.2,2.9), (8.2,1.7)]
    for (x,y), col in zip(starts, colors):
        ax.add_patch(plt.Rectangle((x, y), 1.2, 1.2, fill=False, linewidth=2.5, edgecolor=col))
        ax.text(x+0.6, y+1.35, "Σ", ha='center', va='bottom', fontsize=18)
    ax.axis('off'); ax.set_title(title, fontsize=18)
    bio = io.BytesIO(); fig.savefig(bio, format='png', bbox_inches='tight', dpi=220); bio.seek(0)
    st.image(bio, use_column_width=True)
    return bio.getvalue()

_ = draw_simd_execution(16,16,"SIMD execution — conceptual flow")

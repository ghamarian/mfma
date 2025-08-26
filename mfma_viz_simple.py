#!/usr/bin/env python3
"""
AMD Matrix Instruction Visualizer - Simplified Version
Focuses only on actual data from matrix_calculator.py
"""

import io
import os
import re
import subprocess
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# Configure page
st.set_page_config(
    page_title="AMD Matrix Instruction Visualizer",
    page_icon="🎯",
    layout="wide"
)

class MatrixCalculator:
    """Simple wrapper for matrix_calculator.py"""
    
    def __init__(self, path: str):
        self.path = path
        self.available = os.path.exists(path)
    
    def run_command(self, args: List[str]) -> Tuple[bool, str]:
        """Run calculator with arguments"""
        if not self.available:
            return False, "Calculator not found"
        
        try:
            cmd = ["python3", self.path] + args
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            return result.returncode == 0, result.stdout
        except Exception as e:
            return False, str(e)
    
    def get_register_layout(self, arch: str, inst: str, matrix: str,
                           cbsz: int = 0, abid: int = 0, blgp: int = 0,
                           opsel: int = 0, wavefront: int = 64) -> Optional[List[pd.DataFrame]]:
        """Get register layout for a matrix - returns list of blocks"""
        args = [
            "-a", arch,
            "-i", inst,
            f"-{matrix}",
            "-R",
            "--cbsz", str(cbsz),
            "--abid", str(abid),
            "--blgp", str(blgp),
            "--opsel", str(opsel),
            "-w", str(wavefront),
            "--csv"
        ]
        
        success, output = self.run_command(args)
        if not success:
            return None
        
        # Parse CSV output with block structure
        lines = output.strip().split('\n')
        blocks = []
        current_block_data = []
        current_block_header = None
        current_block_num = None
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Skip empty lines
            if not line:
                i += 1
                continue
                
            # Skip architecture/instruction info
            if line.startswith('Architecture:') or line.startswith('Instruction:'):
                i += 1
                continue
                
            # Check for block header
            if line.startswith('Block '):
                # Save previous block if we have one
                if current_block_data and current_block_header:
                    try:
                        csv_data = current_block_header + '\n' + '\n'.join(current_block_data)
                        df = pd.read_csv(io.StringIO(csv_data))
                        df['Block'] = current_block_num
                        blocks.append(df)
                    except Exception as e:
                        st.warning(f"Failed to parse block {current_block_num}: {e}")
                
                # Start new block
                current_block_num = line.split('Block ')[1].strip()
                current_block_data = []
                current_block_header = None
                i += 1
                continue
            
            # Check for column header (contains matrix name like A[M][K])
            if '[' in line and ']' in line and ',' in line and not line[0].isdigit():
                current_block_header = line
                i += 1
                continue
            
            # Regular data line (starts with a number)
            if ',' in line and current_block_header and (line[0].isdigit() or line.startswith('-')):
                current_block_data.append(line)
            
            i += 1
        
        # Don't forget the last block
        if current_block_data and current_block_header:
            try:
                csv_data = current_block_header + '\n' + '\n'.join(current_block_data)
                df = pd.read_csv(io.StringIO(csv_data))
                df['Block'] = current_block_num
                blocks.append(df)
            except Exception as e:
                st.warning(f"Failed to parse final block {current_block_num}: {e}")
        
        return blocks if blocks else None
    
    def get_matrix_layout(self, arch: str, inst: str, matrix: str,
                         cbsz: int = 0, abid: int = 0, blgp: int = 0,
                         opsel: int = 0, wavefront: int = 64) -> Optional[str]:
        """Get matrix layout (shows which matrix elements are in which registers)"""
        args = [
            "-a", arch,
            "-i", inst,
            f"-{matrix}",
            "-M",  # Matrix layout
            "--cbsz", str(cbsz),
            "--abid", str(abid),
            "--blgp", str(blgp),
            "--opsel", str(opsel),
            "-w", str(wavefront)
        ]
        
        success, output = self.run_command(args)
        return output if success else None
    
    def get_single_register(self, arch: str, inst: str, matrix: str, i: int, j: int, k: int = 0, block: int = 0,
                           cbsz: int = 0, abid: int = 0, blgp: int = 0, opsel: int = 0, wavefront: int = 64) -> Optional[str]:
        """Get register/lane for specific matrix element"""
        args = [
            "-a", arch, "-i", inst, f"-{matrix}", "-g",
            "-I", str(i), "-J", str(j), "-K", str(k), "-b", str(block),
            "--cbsz", str(cbsz), "--abid", str(abid), "--blgp", str(blgp), "--opsel", str(opsel),
            "-w", str(wavefront)
        ]
        success, output = self.run_command(args)
        return output if success else None
    
    def get_matrix_element(self, arch: str, inst: str, matrix: str, register: int, lane: int,
                          cbsz: int = 0, abid: int = 0, blgp: int = 0, opsel: int = 0, wavefront: int = 64) -> Optional[str]:
        """Get matrix element for specific register/lane"""
        args = [
            "-a", arch, "-i", inst, f"-{matrix}", "-m",
            "-r", str(register), "-l", str(lane),
            "--cbsz", str(cbsz), "--abid", str(abid), "--blgp", str(blgp), "--opsel", str(opsel),
            "-w", str(wavefront)
        ]
        success, output = self.run_command(args)
        return output if success else None
    
    def get_mapping_formulas(self, arch: str, inst: str) -> Optional[Dict[str, str]]:
        """Extract the mapping formulas from detailed instruction output"""
        args = ["-a", arch, "-i", inst, "-d"]
        success, output = self.run_command(args)
        if not success:
            return None
        
        formulas = {}
        lines = output.split('\n')
        in_element_mapping = False
        in_register_mapping = False
        
        for line in lines:
            line = line.strip()
            
            if 'Matrix element to register mapping' in line:
                in_element_mapping = True
                in_register_mapping = False
                continue
            elif 'Register to matrix element mapping' in line:
                in_element_mapping = False
                in_register_mapping = True
                continue
            elif line.startswith('Architecture:') or line.startswith('Instruction:') or not line:
                continue
            
            if in_element_mapping and ':' in line:
                if 'A[i][k]' in line and 'GPR' in line:
                    formulas['A_element_to_gpr'] = line.split('GPR:')[1].strip()
                elif 'A[i][k]' in line and 'Lane' in line:
                    formulas['A_element_to_lane'] = line.split('Lane:')[1].strip()
                elif 'B[k][j]' in line and 'GPR' in line:
                    formulas['B_element_to_gpr'] = line.split('GPR:')[1].strip()
                elif 'B[k][j]' in line and 'Lane' in line:
                    formulas['B_element_to_lane'] = line.split('Lane:')[1].strip()
                elif ('C or D[i][j]' in line or 'C[i][j]' in line or 'D[i][j]' in line) and 'GPR' in line:
                    formulas['CD_element_to_gpr'] = line.split('GPR:')[1].strip()
                elif ('C or D[i][j]' in line or 'C[i][j]' in line or 'D[i][j]' in line) and 'Lane' in line:
                    formulas['CD_element_to_lane'] = line.split('Lane:')[1].strip()
            
            if in_register_mapping and ':' in line:
                if 'A i:' in line:
                    formulas['gpr_to_A_i'] = line.split('A i:')[1].strip()
                elif 'A k:' in line:
                    formulas['gpr_to_A_k'] = line.split('A k:')[1].strip()
                elif 'A block:' in line:
                    formulas['gpr_to_A_block'] = line.split('A block:')[1].strip()
                elif 'B j:' in line:
                    formulas['gpr_to_B_j'] = line.split('B j:')[1].strip()
                elif 'B k:' in line:
                    formulas['gpr_to_B_k'] = line.split('B k:')[1].strip()
                elif 'B block:' in line:
                    formulas['gpr_to_B_block'] = line.split('B block:')[1].strip()
                elif 'C or D i:' in line:
                    formulas['gpr_to_CD_i'] = line.split('C or D i:')[1].strip()
                elif 'C or D j:' in line:
                    formulas['gpr_to_CD_j'] = line.split('C or D j:')[1].strip()
                elif 'C or D block:' in line:
                    formulas['gpr_to_CD_block'] = line.split('C or D block:')[1].strip()
        
        return formulas if formulas else None
    
    def get_instruction_list(self, arch: str) -> List[str]:
        """Get list of instructions for architecture"""
        args = ["-a", arch, "-L"]
        success, output = self.run_command(args)
        if not success:
            return []
        
        instructions = []
        for line in output.split('\n'):
            line = line.strip()
            if line and line.startswith('v_'):
                instructions.append(line)
        return instructions
    
    def get_instruction_details(self, arch: str, inst: str) -> Optional[Dict]:
        """Get comprehensive instruction details"""
        args = ["-a", arch, "-i", inst, "-d"]
        success, output = self.run_command(args)
        if not success:
            return None
        
        details = {}
        lines = output.split('\n')
        
        for line in lines:
            line = line.strip()
            if ':' not in line:
                continue
                
            # Matrix dimensions
            if 'M:' in line and 'Matrix' not in line:
                details['m'] = int(line.split(':')[1].strip())
            elif 'N:' in line and 'Matrix' not in line:
                details['n'] = int(line.split(':')[1].strip())
            elif 'K:' in line and 'Matrix' not in line:
                details['k'] = int(line.split(':')[1].strip())
            elif 'blocks:' in line:
                details['blocks'] = int(line.split(':')[1].strip())
            
            # Execution stats
            elif 'FLOPs:' in line:
                details['flops'] = int(line.split(':')[1].strip())
            elif 'Execution cycles:' in line:
                details['cycles'] = int(line.split(':')[1].strip())
            elif 'FLOPs/CU/cycle:' in line:
                details['flops_per_cu_cycle'] = int(line.split(':')[1].strip())
            elif 'Can co-execute with VALU:' in line:
                details['valu_coexec'] = 'True' in line.split(':')[1]
            elif 'VALU co-execution cycles possible:' in line:
                details['valu_coexec_cycles'] = int(line.split(':')[1].strip())
                
            # Register usage  
            elif 'GPRs required for A:' in line:
                details['gpr_a'] = int(line.split(':')[1].strip())
            elif 'GPRs required for B:' in line:
                details['gpr_b'] = int(line.split(':')[1].strip())
            elif 'GPRs required for C:' in line:
                details['gpr_c'] = int(line.split(':')[1].strip())
            elif 'GPRs required for D:' in line:
                details['gpr_d'] = int(line.split(':')[1].strip())
            elif 'GPR alignment requirement:' in line:
                details['gpr_align'] = line.split(':')[1].strip()
                
            # Data types
            elif 'Src0:' in line:
                details['src0_type'] = line.split(':')[1].strip()
            elif 'Src1:' in line:
                details['src1_type'] = line.split(':')[1].strip()
            elif 'Src2:' in line:
                details['src2_type'] = line.split(':')[1].strip()
            elif 'Vdst:' in line:
                details['vdst_type'] = line.split(':')[1].strip()
                
            # Modifiers support
            elif 'CBSZ and ABID bits supported:' in line:
                details['cbsz_abid_support'] = 'True' in line.split(':')[1]
            elif 'BLGP bits supported:' in line:
                details['blgp_support'] = 'True' in line.split(':')[1]
            elif 'Sparse A matrix:' in line:
                details['sparse'] = 'True' in line.split(':')[1]
        
        # Calculate K iterations (K dimension represents how many K iterations)
        if 'k' in details:
            details['k_iterations'] = details['k']
        
        return details if details else None

def parse_register_notation(reg_str: str) -> Tuple[int, int, str]:
    """Parse register notations like v0{32}.[15:0] to (register, lane, bits)"""
    if 'v' not in reg_str:
        return -1, -1, ""
    
    # Parse v0{32}.[15:0] format (with bit range)
    match = re.match(r'v(\d+)\{(\d+)\}\.?\[?([^\]]*)\]?', reg_str)
    if match:
        reg = int(match.group(1))
        lane = int(match.group(2))
        bits = match.group(3) if match.group(3) else ""
        return reg, lane, bits
    
    # Parse v0{32} format (without bit range)
    match = re.match(r'v(\d+)\{(\d+)\}', reg_str)
    if match:
        return int(match.group(1)), int(match.group(2)), ""
    
    # Parse v0[32] format
    match = re.match(r'v(\d+)\[(\d+)\]', reg_str)
    if match:
        return int(match.group(1)), int(match.group(2)), ""
    
    # Parse simple v0 format
    match = re.match(r'v(\d+)', reg_str)
    if match:
        return int(match.group(1)), 0, ""
    
    return -1, -1, ""

def create_block_tables(blocks: List[pd.DataFrame], matrix_name: str) -> List[go.Figure]:
    """Create tables for each block showing the exact register mapping from calculator"""
    if not blocks:
        fig = go.Figure().add_annotation(text="No data available", showarrow=False)
        return [fig]
    
    figures = []
    
    for block_df in blocks:
        if block_df is None or block_df.empty:
            continue
            
        # Get block number
        block_num = block_df['Block'].iloc[0] if 'Block' in block_df.columns else "Unknown"
        
        # Remove Block column for display
        display_df = block_df.drop('Block', axis=1) if 'Block' in block_df.columns else block_df
        cols = display_df.columns.tolist()
        
        # Create a table figure
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=cols,
                fill_color='lightblue',
                align='center',
                font=dict(size=11, color='black')
            ),
            cells=dict(
                values=[display_df[col].tolist() for col in cols],
                fill_color='white',
                align='center',
                font=dict(size=9, color='black')
            )
        )])
        
        fig.update_layout(
            title=f"{matrix_name} Matrix - Block {block_num}",
            height=min(400, max(200, len(display_df) * 20 + 80)),
            margin=dict(l=10, r=10, t=40, b=10)
        )
        
        figures.append(fig)
    
    return figures

def main():
    st.title("🎯 AMD Matrix Instruction Visualizer")
    st.markdown("### Visualizing actual register layouts from AMD matrix_calculator.py")
    
    # Find calculator
    calc_path = None
    for path in [
        "/Users/amir/projects/github/mfma/amd_matrix_instruction_calculator/matrix_calculator.py",
        "amd_matrix_instruction_calculator/matrix_calculator.py",
        "matrix_calculator.py"
    ]:
        if os.path.exists(path):
            calc_path = path
            break
    
    if not calc_path:
        st.error("❌ matrix_calculator.py not found!")
        st.stop()
    
    calc = MatrixCalculator(calc_path)
    st.sidebar.success(f"✅ Calculator found at: {calc_path}")
    
    # Architecture selection
    arch_map = {
        'cdna1': 'CDNA1 (MI100)',
        'cdna2': 'CDNA2 (MI200)',
        'cdna3': 'CDNA3 (MI300)',
        'rdna3': 'RDNA3',
        'rdna4': 'RDNA4'
    }
    
    arch = st.sidebar.selectbox(
        "Architecture",
        options=list(arch_map.keys()),
        format_func=lambda x: arch_map[x]
    )
    
    # Get instruction list
    instructions = calc.get_instruction_list(arch)
    if not instructions:
        st.error(f"Could not get instructions for {arch}")
        st.stop()
    
    # Sort instructions alphabetically
    instructions.sort()
    
    # Add search functionality
    st.sidebar.subheader("🔍 Instruction Selection")
    search_term = st.sidebar.text_input("Search instructions:", placeholder="e.g., f32_16x16, bf16, i8")
    
    # Filter instructions based on search
    if search_term:
        filtered_instructions = [inst for inst in instructions if search_term.lower() in inst.lower()]
        if not filtered_instructions:
            st.sidebar.warning(f"No instructions found matching '{search_term}'")
            filtered_instructions = instructions
    else:
        filtered_instructions = instructions
    
    # Keep alphabetical sorting
    filtered_instructions.sort()
    
    # Show count
    st.sidebar.caption(f"Showing {len(filtered_instructions)} of {len(instructions)} instructions")
    
    inst = st.sidebar.selectbox("Instruction", filtered_instructions)
    
    # Get instruction details
    details = calc.get_instruction_details(arch, inst)
    if details:
        m = details.get('m', 16)
        n = details.get('n', 16)
        k = details.get('k', 16)
        blocks = details.get('blocks', 1)
        cycles = details.get('cycles', 32)
    else:
        # Parse from instruction name
        match = re.search(r'(\d+)x(\d+)x(\d+)', inst)
        if match:
            m, n, k = map(int, match.groups())
        else:
            m = n = k = 16
        blocks = 1
        cycles = 32
        details = {}
    
    # Wavefront
    wavefront = st.sidebar.selectbox(
        "Wavefront",
        [32, 64],
        index=1 if arch in ['cdna1', 'cdna2', 'cdna3'] else 0
    )
    
    # Modifiers
    st.sidebar.divider()
    st.sidebar.subheader("Modifiers")
    cbsz = st.sidebar.number_input("CBSZ", 0, 3, 0)
    abid = st.sidebar.number_input("ABID", 0, 7, 0)
    blgp = st.sidebar.number_input("BLGP", 0, 7, 0)
    opsel = st.sidebar.number_input("OPSEL", 0, 15, 0)
    
    # === MFMA PARAMETERS SECTION ===
    st.header("📊 MFMA Parameters")
    
    # GEMM parameters
    st.subheader("GEMM parameters:")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("**M**", m)
    with col2:
        st.metric("**N**", n) 
    with col3:
        st.metric("**K**", k)
    
    # MFMA parameters
    st.subheader("MFMA parameters:")
    
    # Parse data types from instruction name
    inst_lower = inst.lower()
    if 'f32' in inst_lower:
        if 'x1f32' in inst_lower:
            ab_type = "f32"
            accum_type = "f32"
        elif 'f16' in inst_lower:
            ab_type = "f16"
            accum_type = "f32"
        elif 'bf16' in inst_lower:
            ab_type = "bf16"
            accum_type = "f32"
        elif 'i8' in inst_lower:
            ab_type = "i8"
            accum_type = "i32"
        else:
            ab_type = details.get('src0_type', 'unknown').split('(')[0].strip()
            accum_type = details.get('vdst_type', 'unknown').split('(')[0].strip()
    elif 'i32' in inst_lower:
        ab_type = "i8"
        accum_type = "i32"
    elif 'f64' in inst_lower:
        ab_type = "f64"
        accum_type = "f64"
    else:
        ab_type = details.get('src0_type', 'unknown').split('(')[0].strip()
        accum_type = details.get('vdst_type', 'unknown').split('(')[0].strip()
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("**BlkM**", m)
    with col2:
        st.metric("**BlkN**", n)
    with col3:
        st.metric("**BlkK**", k)
    with col4:
        st.metric("**A / B type**", ab_type)
    with col5:
        st.metric("**accum type**", accum_type)
    
    # K-iterations
    k_iterations = details.get('k_iterations', k)
    st.metric("**K-iterations**", k_iterations)
    
    # Additional performance info if available
    if details:
        st.divider()
        st.subheader("📈 Performance Details")
        
        perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
        with perf_col1:
            st.metric("Execution Cycles", details.get('cycles', 'N/A'))
        with perf_col2:
            st.metric("FLOPs", details.get('flops', 'N/A'))
        with perf_col3:
            st.metric("FLOPs/CU/cycle", details.get('flops_per_cu_cycle', 'N/A'))
        with perf_col4:
            coexec = "✅" if details.get('valu_coexec', False) else "❌"
            st.metric("VALU Co-exec", coexec)
        
        # Register usage
        st.subheader("🗃️ Register Usage")
        reg_col1, reg_col2, reg_col3, reg_col4 = st.columns(4)
        with reg_col1:
            st.metric("A matrix GPRs", details.get('gpr_a', 'N/A'))
        with reg_col2:
            st.metric("B matrix GPRs", details.get('gpr_b', 'N/A'))
        with reg_col3:
            st.metric("C matrix GPRs", details.get('gpr_c', 'N/A'))
        with reg_col4:
            st.metric("D matrix GPRs", details.get('gpr_d', 'N/A'))
    
    # Show all matrices
    st.subheader("🔍 All Matrix Register Layouts")
    
    matrices = ['A', 'B', 'C', 'D']
    matrix_data = {}
    
    # Fetch all matrix layouts
    for matrix in matrices:
        with st.spinner(f"Fetching {matrix} matrix register layout..."):
            blocks = calc.get_register_layout(arch, inst, matrix, cbsz, abid, blgp, opsel, wavefront)
            matrix_data[matrix] = blocks
    
    # Display all matrices
    for matrix in matrices:
        blocks = matrix_data[matrix]
        
        st.divider()
        st.subheader(f"📋 Matrix {matrix} Register Layout")
        
        if blocks and len(blocks) > 0:
            total_entries = sum(len(block) for block in blocks if block is not None)
            st.success(f"✅ Retrieved {len(blocks)} blocks with {total_entries} total entries for matrix {matrix}")
            
            # Create block tables
            figures = create_block_tables(blocks, matrix)
            
            # Display each block
            if len(blocks) == 1:
                # Single block - show directly
                st.plotly_chart(figures[0], use_container_width=True)
            else:
                # Multiple blocks - show in columns or tabs
                st.markdown(f"**Matrix {matrix} has {len(blocks)} blocks:**")
                
                # Show first few blocks directly, rest in expander
                max_direct_blocks = 3
                for i, fig in enumerate(figures[:max_direct_blocks]):
                    st.plotly_chart(fig, use_container_width=True)
                
                if len(figures) > max_direct_blocks:
                    with st.expander(f"Show remaining {len(figures) - max_direct_blocks} blocks"):
                        for fig in figures[max_direct_blocks:]:
                            st.plotly_chart(fig, use_container_width=True)
            
            # Show matrix layout (matrix elements to register mapping)
            with st.expander(f"🗺️ Matrix {matrix} Layout (Elements → Registers)"):
                matrix_layout = calc.get_matrix_layout(arch, inst, matrix, cbsz, abid, blgp, opsel, wavefront)
                if matrix_layout:
                    st.text(matrix_layout)
                else:
                    st.error("Could not retrieve matrix layout")
            
            # Show raw data in expander
            with st.expander(f"📋 Raw Calculator Output for Matrix {matrix}"):
                for i, block in enumerate(blocks):
                    if block is not None and not block.empty:
                        st.markdown(f"**Block {i}:**")
                        st.dataframe(block, use_container_width=True)
                        st.caption(f"Block {i}: {len(block)} entries")
                
                # Show the command that was run
                cmd = [
                    "python3", calc_path.split('/')[-1],
                    "-a", arch,
                    "-i", inst,
                    f"-{matrix}",
                    "-R",
                    "--cbsz", str(cbsz),
                    "--abid", str(abid),
                    "--blgp", str(blgp),
                    "--opsel", str(opsel),
                    "-w", str(wavefront),
                    "--csv"
                ]
                st.code(' '.join(cmd))
        else:
            st.error(f"❌ Could not retrieve register layout for matrix {matrix}")
            st.info(f"Matrix {matrix} may not be supported for this instruction/modifier combination")
    
    # === ADDITIONAL FEATURES ===
    st.divider()
    st.header("🔍 Additional Calculator Features")
    
    # Matrix element lookup
    with st.expander("🎯 Matrix Element ↔ Register Lookup"):
        st.markdown("**Find register/lane for specific matrix element:**")
        
        lookup_cols = st.columns(5)
        with lookup_cols[0]:
            lookup_matrix = st.selectbox("Matrix", ['A', 'B', 'C', 'D'], key="lookup_matrix")
        with lookup_cols[1]:
            lookup_i = st.number_input("I coordinate", 0, max(m-1, 0), 0, key="lookup_i")
        with lookup_cols[2]:
            lookup_j = st.number_input("J coordinate", 0, max(n-1, 0), 0, key="lookup_j")
        with lookup_cols[3]:
            lookup_k = st.number_input("K coordinate", 0, max(k-1, 0), 0, key="lookup_k")
        with lookup_cols[4]:
            max_blocks = details.get('blocks', 1) if isinstance(details.get('blocks'), int) else 1
            lookup_block = st.number_input("Block", 0, max(max_blocks-1, 0), 0, key="lookup_block")
        
        if st.button("🔍 Find Register", key="find_register"):
            result = calc.get_single_register(arch, inst, lookup_matrix, lookup_i, lookup_j, lookup_k, lookup_block, cbsz, abid, blgp, opsel, wavefront)
            if result:
                st.success(f"**Result:**")
                st.code(result)
            else:
                st.error("Could not find register mapping")
        
        st.markdown("**Find matrix element for specific register/lane:**")
        reverse_cols = st.columns(3)
        with reverse_cols[0]:
            reverse_matrix = st.selectbox("Matrix", ['A', 'B', 'C', 'D'], key="reverse_matrix")
        with reverse_cols[1]:
            reverse_reg = st.number_input("Register", 0, 255, 0, key="reverse_reg")
        with reverse_cols[2]:
            reverse_lane = st.number_input("Lane", 0, 63, 0, key="reverse_lane")
        
        if st.button("🔍 Find Matrix Element", key="find_element"):
            result = calc.get_matrix_element(arch, inst, reverse_matrix, reverse_reg, reverse_lane, cbsz, abid, blgp, opsel, wavefront)
            if result:
                st.success(f"**Result:**")
                st.code(result)
            else:
                st.error("Could not find matrix element")
    
    # === MAPPING FORMULAS ===
    with st.expander("📐 Matrix Element ↔ Register Mapping Formulas"):
        st.markdown("**Mathematical formulas used by the instruction for mapping between matrix elements and registers:**")
        
        formulas = calc.get_mapping_formulas(arch, inst)
        if formulas:
            st.subheader("🔗 Matrix Element → Register/Lane")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Matrix A:**")
                if 'A_element_to_gpr' in formulas:
                    st.code(f"GPR = {formulas['A_element_to_gpr']}")
                if 'A_element_to_lane' in formulas:
                    st.code(f"Lane = {formulas['A_element_to_lane']}")
                
                st.markdown("**Matrix B:**")
                if 'B_element_to_gpr' in formulas:
                    st.code(f"GPR = {formulas['B_element_to_gpr']}")
                if 'B_element_to_lane' in formulas:
                    st.code(f"Lane = {formulas['B_element_to_lane']}")
            
            with col2:
                st.markdown("**Matrix C/D:**")
                if 'CD_element_to_gpr' in formulas:
                    st.code(f"GPR = {formulas['CD_element_to_gpr']}")
                if 'CD_element_to_lane' in formulas:
                    st.code(f"Lane = {formulas['CD_element_to_lane']}")
            
            st.subheader("🔙 Register/Lane → Matrix Element")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Matrix A:**")
                if 'gpr_to_A_i' in formulas:
                    st.code(f"i = {formulas['gpr_to_A_i']}")
                if 'gpr_to_A_k' in formulas:
                    st.code(f"k = {formulas['gpr_to_A_k']}")
                if 'gpr_to_A_block' in formulas:
                    st.code(f"block = {formulas['gpr_to_A_block']}")
                
                st.markdown("**Matrix B:**")
                if 'gpr_to_B_j' in formulas:
                    st.code(f"j = {formulas['gpr_to_B_j']}")
                if 'gpr_to_B_k' in formulas:
                    st.code(f"k = {formulas['gpr_to_B_k']}")
                if 'gpr_to_B_block' in formulas:
                    st.code(f"block = {formulas['gpr_to_B_block']}")
            
            with col2:
                st.markdown("**Matrix C/D:**")
                if 'gpr_to_CD_i' in formulas:
                    st.code(f"i = {formulas['gpr_to_CD_i']}")
                if 'gpr_to_CD_j' in formulas:
                    st.code(f"j = {formulas['gpr_to_CD_j']}")
                if 'gpr_to_CD_block' in formulas:
                    st.code(f"block = {formulas['gpr_to_CD_block']}")
        else:
            st.warning("Could not extract mapping formulas for this instruction")

if __name__ == "__main__":
    main()

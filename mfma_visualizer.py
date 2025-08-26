#!/usr/bin/env python3
"""
AMD Matrix Instruction Visualizer - Final Version
Shows both Register Layout and Matrix Layout in nice tabular formats
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

# Import the code generator
try:
    from generate_mfma_code import generate_mfma_cpp
    CODE_GEN_AVAILABLE = True
except ImportError:
    CODE_GEN_AVAILABLE = False

# Copy the MatrixCalculator class and helper functions from mfma_viz_simple.py
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
            # Get the directory of the calculator
            calc_dir = os.path.dirname(self.path)
            calc_filename = os.path.basename(self.path)
            
            # If running from calculator directory, use just the filename
            if calc_dir:
                cmd = ["python3", calc_filename] + args
                cwd = calc_dir
            else:
                cmd = ["python3", self.path] + args
                cwd = None
            
            # Run the command
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=10,
                cwd=cwd
            )
            
            if result.returncode != 0:
                error_msg = f"Command failed with return code {result.returncode}\n"
                error_msg += f"stderr: {result.stderr}\n"
                error_msg += f"stdout: {result.stdout}"
                return False, error_msg
            
            # Also check if stdout is empty but stderr has content
            if not result.stdout and result.stderr:
                # Some programs write to stderr even on success
                return True, result.stderr
            elif not result.stdout:
                return False, f"No output from command. stderr: {result.stderr}"
                
            return True, result.stdout
        except Exception as e:
            return False, f"Exception: {str(e)}"
    
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
    
    def parse_matrix_layout_table(self, output: str) -> Optional[pd.DataFrame]:
        """Parse the ASCII table from matrix layout output into a DataFrame"""
        if not output:
            return None
        
        lines = output.strip().split('\n')
        
        # Find the table header
        header_idx = -1
        for i, line in enumerate(lines):
            if 'lane' in line.lower() and '|' in line:
                header_idx = i
                break
        
        if header_idx < 0:
            return None
        
        # Extract headers
        header_line = lines[header_idx]
        headers = [h.strip() for h in header_line.split('|') if h.strip()]
        
        # Extract data rows
        data_rows = []
        i = header_idx + 2  # Skip separator line
        while i < len(lines):
            line = lines[i].strip()
            if not line or '+' in line:
                i += 1
                continue
            if '|' not in line:
                break
                
            # Parse data row
            values = [v.strip() for v in line.split('|') if v.strip()]
            if len(values) == len(headers):
                data_rows.append(values)
            i += 1
        
        if not data_rows:
            return None
        
        # Create DataFrame
        df = pd.DataFrame(data_rows, columns=headers)
        return df
    
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
        
        return details if details else None
    
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
                elif 'B j:' in line:
                    formulas['gpr_to_B_j'] = line.split('B j:')[1].strip()
                elif 'B k:' in line:
                    formulas['gpr_to_B_k'] = line.split('B k:')[1].strip()
                elif 'C or D i:' in line:
                    formulas['gpr_to_CD_i'] = line.split('C or D i:')[1].strip()
                elif 'C or D j:' in line:
                    formulas['gpr_to_CD_j'] = line.split('C or D j:')[1].strip()
        
        return formulas if formulas else None

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
    
    # Parse simple v0 format
    match = re.match(r'v(\d+)', reg_str)
    if match:
        return int(match.group(1)), 0, ""
    
    return -1, -1, ""

def create_register_layout_table(blocks: List[pd.DataFrame], matrix_name: str) -> List[go.Figure]:
    """Create tables for register layout with improved headers showing VGPR grouping"""
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
        
        # Process data
        table_data = []
        cell_colors = []
        combined_headers = []
        
        # First column is the row label
        combined_headers.append(display_df.columns[0].split('[')[0] + ' Row')
        table_data.append(display_df.iloc[:, 0].tolist())
        cell_colors.append(['white'] * len(display_df))
        
        # Check if this is f16/bf16 data
        has_bit_ranges = False
        if len(display_df.columns) > 1:
            first_val = str(display_df.iloc[0, 1])
            has_bit_ranges = '[15:0]' in first_val or '[31:16]' in first_val
        
        # Track VGPRs for visual grouping
        prev_vgpr = -1
        
        # Process each column
        for i in range(1, len(display_df.columns)):
            col = display_df.columns[i]
            k_val = col if col.isdigit() else str(i - 1)
            
            # Get first value to determine VGPR and bit range
            first_val = str(display_df.iloc[0, i])
            parsed = parse_register_notation(first_val)
            
            if parsed[0] != -1:
                vgpr = parsed[0]
                
                # Create header with visual grouping indicator
                if has_bit_ranges:
                    if '[15:0]' in first_val:
                        bit_text = "[0:15]"
                    elif '[31:16]' in first_val:
                        bit_text = "[16:31]"
                    else:
                        bit_text = "[0:31]"
                    
                    # Add grouping indicator if same VGPR as previous
                    if vgpr == prev_vgpr:
                        header = f"├─ VGPR{vgpr}<br>{bit_text}<br>K={k_val}"
                    else:
                        header = f"┌─ VGPR{vgpr}<br>{bit_text}<br>K={k_val}"
                else:
                    # No bit ranges (full 32-bit registers)
                    header = f"VGPR{vgpr}<br>K={k_val}"
                
                prev_vgpr = vgpr
            else:
                header = f"K={k_val}"
                prev_vgpr = -1
            
            combined_headers.append(header)
            
            # Process column data and colors
            col_data = []
            col_colors = []
            for val in display_df.iloc[:, i]:
                parsed = parse_register_notation(str(val))
                if parsed[0] != -1:
                    reg, lane, _ = parsed
                    col_data.append(f"v{reg}{{{lane}}}")
                    # Color based on register
                    if reg == 0:
                        col_colors.append('#E6F3FF')  # Light blue for v0
                    elif reg == 1:
                        col_colors.append('#FFE6CC')  # Light orange for v1
                    else:
                        col_colors.append('#E6FFE6')  # Light green for others
                else:
                    col_data.append(str(val))
                    col_colors.append('white')
            
            table_data.append(col_data)
            cell_colors.append(col_colors)
        
        # Create the table figure
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=combined_headers,
                fill_color='lightblue',
                align='center',
                font=dict(size=11, color='black', family='monospace'),
                line_color='darkslategray',
                height=70
            ),
            cells=dict(
                values=table_data,
                fill_color=cell_colors,
                align='center',
                font=dict(size=9, color='black'),
                line_color='darkslategray',
                height=25
            )
        )])
        
        fig.update_layout(
            title=f"{matrix_name} Register Layout - Block {block_num}",
            height=min(500, max(250, len(display_df) * 30 + 130)),
            margin=dict(l=10, r=10, t=40, b=10)
        )
        
        figures.append(fig)
    
    return figures

def create_matrix_layout_table(df: pd.DataFrame, matrix_name: str) -> go.Figure:
    """Create table for matrix layout (Elements → Register mapping)"""
    if df is None or df.empty:
        fig = go.Figure().add_annotation(text="No data available", showarrow=False)
        return fig
    
    # Create a color-coded table
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=list(df.columns),
            fill_color='lightcoral',
            align='center',
            font=dict(size=12, color='white'),
            line_color='darkslategray'
        ),
        cells=dict(
            values=[df[col] for col in df.columns],
            fill_color='lavender',
            align='center',
            font=dict(size=10, color='black'),
            line_color='darkslategray'
        )
    )])
    
    fig.update_layout(
        title=f"{matrix_name} Matrix Layout (Elements → Registers)",
        height=min(600, max(300, len(df) * 25 + 100)),
        margin=dict(l=10, r=10, t=40, b=10)
    )
    
    return fig

def main():
    st.set_page_config(
        page_title="AMD Matrix Instruction Visualizer - Final",
        page_icon="🎯",
        layout="wide"
    )
    
    st.title("🎯 AMD Matrix Instruction Visualizer")
    st.markdown("### Shows both Register Layout and Matrix Layout in tabular formats")
    
    # Find calculator
    calc_path = None
    for path in [
        "/home/aghamari/github/mfma/amd_matrix_instruction_calculator/matrix_calculator.py",
        "amd_matrix_instruction_calculator/matrix_calculator.py",
        "./amd_matrix_instruction_calculator/matrix_calculator.py",
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
    
    # Show count
    st.sidebar.caption(f"Showing {len(filtered_instructions)} of {len(instructions)} instructions")
    
    inst = st.sidebar.selectbox("Instruction", sorted(filtered_instructions))
    
    # Get instruction details
    details = calc.get_instruction_details(arch, inst)
    if details:
        m = details.get('m', 16)
        n = details.get('n', 16)
        k = details.get('k', 16)
    else:
        # Parse from instruction name
        match = re.search(r'(\d+)x(\d+)x(\d+)', inst)
        if match:
            m, n, k = map(int, match.groups())
        else:
            m = n = k = 16
    
    # Wavefront and modifiers
    st.sidebar.divider()
    st.sidebar.subheader("Settings")
    wavefront = st.sidebar.selectbox(
        "Wavefront",
        [32, 64],
        index=1 if arch in ['cdna1', 'cdna2', 'cdna3'] else 0
    )
    
    # Modifiers
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
            ab_type = details.get('src0_type', 'unknown').split('(')[0].strip() if details else 'unknown'
            accum_type = details.get('vdst_type', 'unknown').split('(')[0].strip() if details else 'unknown'
    elif 'i32' in inst_lower:
        ab_type = "i8"
        accum_type = "i32"
    elif 'f64' in inst_lower:
        ab_type = "f64"
        accum_type = "f64"
    else:
        ab_type = details.get('src0_type', 'unknown').split('(')[0].strip() if details else 'unknown'
        accum_type = details.get('vdst_type', 'unknown').split('(')[0].strip() if details else 'unknown'
    
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
    k_iterations = details.get('k_iterations', k) if details else k
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
    
    # Display all matrices
    st.header("🔍 All Matrix Layouts - Dual View")
    
    # Note about VGPR grouping
    st.info("💡 **Tip**: In the Register Layout tables, columns with '┌─' and '├─' prefixes show VGPR grouping. For f16/bf16 data, each VGPR holds two K values in different bit ranges.")
    
    matrices = ['A', 'B', 'C', 'D']
    
    for matrix in matrices:
        st.divider()
        st.subheader(f"📋 Matrix {matrix}")
        
        # Get both layouts
        register_blocks = calc.get_register_layout(arch, inst, matrix, cbsz, abid, blgp, opsel, wavefront)
        matrix_layout_raw = calc.get_matrix_layout(arch, inst, matrix, cbsz, abid, blgp, opsel, wavefront)
        
        # Show both in columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🗂️ Register Layout** *(full matrix view)*")
            
            if register_blocks:
                figures = create_register_layout_table(register_blocks, matrix)
                for fig in figures:
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Could not retrieve register layout")
                st.info(f"Matrix {matrix} may not be supported for this instruction/modifier combination")
        
        with col2:
            st.markdown("**🔄 Matrix Layout** *(grouped by register)*")
            
            if matrix_layout_raw:
                # Parse the ASCII table
                df = calc.parse_matrix_layout_table(matrix_layout_raw)
                if df is not None:
                    fig = create_matrix_layout_table(df, matrix)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show element count
                    st.caption(f"Total lanes: {len(df)}")
                else:
                    # If parsing fails, show raw output
                    with st.expander("Show raw output"):
                        st.code(matrix_layout_raw[:1000] + "..." if len(matrix_layout_raw) > 1000 else matrix_layout_raw)
            else:
                st.error("Could not retrieve matrix layout")
                st.info(f"Matrix {matrix} may not be supported for this instruction/modifier combination")
        
        # Show raw outputs
        with st.expander(f"📋 Raw Calculator Output for Matrix {matrix}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Register Layout Raw Output (-R):**")
                if register_blocks:
                    # Show the raw command
                    cmd = f"python3 {calc_path} -a {arch} -i {inst} -{matrix} -R --csv"
                    if cbsz or abid or blgp or opsel or wavefront != 64:
                        cmd += f" --cbsz {cbsz} --abid {abid} --blgp {blgp} --opsel {opsel} -w {wavefront}"
                    st.code(cmd)
                    
                    # Get raw output
                    args = ["-a", arch, "-i", inst, f"-{matrix}", "-R", "--csv",
                           "--cbsz", str(cbsz), "--abid", str(abid), "--blgp", str(blgp),
                           "--opsel", str(opsel), "-w", str(wavefront)]
                    success, raw_output = calc.run_command(args)
                    if success:
                        st.code(raw_output[:2000] + "..." if len(raw_output) > 2000 else raw_output)
                else:
                    st.info("No register layout data available")
            
            with col2:
                st.markdown("**Matrix Layout Raw Output (-M):**")
                if matrix_layout_raw:
                    # Show the raw command
                    cmd = f"python3 {calc_path} -a {arch} -i {inst} -{matrix} -M"
                    if cbsz or abid or blgp or opsel or wavefront != 64:
                        cmd += f" --cbsz {cbsz} --abid {abid} --blgp {blgp} --opsel {opsel} -w {wavefront}"
                    st.code(cmd)
                    
                    # Show raw output
                    st.code(matrix_layout_raw[:2000] + "..." if len(matrix_layout_raw) > 2000 else matrix_layout_raw)
                else:
                    st.info("No matrix layout data available")
    
    # === MAPPING FORMULAS ===
    st.divider()
    st.header("📐 Matrix Element ↔ Register Mapping Formulas")
    
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
            
            st.markdown("**Matrix B:**")
            if 'gpr_to_B_j' in formulas:
                st.code(f"j = {formulas['gpr_to_B_j']}")
            if 'gpr_to_B_k' in formulas:
                st.code(f"k = {formulas['gpr_to_B_k']}")
        
        with col2:
            st.markdown("**Matrix C/D:**")
            if 'gpr_to_CD_i' in formulas:
                st.code(f"i = {formulas['gpr_to_CD_i']}")
            if 'gpr_to_CD_j' in formulas:
                st.code(f"j = {formulas['gpr_to_CD_j']}")
    else:
        st.warning("Could not extract mapping formulas for this instruction")
    
    # === C++ CODE GENERATION ===
    st.divider()
    st.header("🚀 C++ Kernel Code Generation")
    
    if CODE_GEN_AVAILABLE:
        st.markdown("Generate a complete C++ MFMA kernel based on the selected instruction and parameters.")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.info(f"""
            **Generated code will include:**
            - Complete HIP/ROCm kernel for `{inst}`
            - Fragment types sized correctly ({details.get('gpr_a', 'N/A')} VGPRs for A, {details.get('gpr_b', 'N/A')} for B, {details.get('gpr_c', 'N/A')} for C)
            - Load/store functions with register mapping comments
            - MFMA builtin wrapper with modifiers (CBSZ={cbsz}, ABID={abid}, BLGP={blgp}, OPSEL={opsel})
            - Test harness structure
            """)
        
        with col2:
            if st.button("🔨 Generate C++ Code", type="primary", use_container_width=True):
                with st.spinner("Generating C++ code..."):
                    try:
                        cpp_code = generate_mfma_cpp(
                            arch=arch,
                            instruction=inst,
                            cbsz=cbsz,
                            abid=abid,
                            blgp=blgp,
                            opsel=opsel,
                            wavefront=wavefront
                        )
                        
                        # Store in session state
                        st.session_state['generated_cpp'] = cpp_code
                        st.session_state['generated_filename'] = f"{inst}_kernel.cpp"
                        st.success("✅ C++ code generated successfully!")
                    except Exception as e:
                        st.error(f"❌ Error generating code: {str(e)}")
        
        # Show generated code if available
        if 'generated_cpp' in st.session_state:
            st.subheader("📄 Generated C++ Code")
            
            # Download button
            st.download_button(
                label="⬇️ Download C++ File",
                data=st.session_state['generated_cpp'],
                file_name=st.session_state['generated_filename'],
                mime="text/x-c++",
                help="Download the generated C++ kernel code"
            )
            
            # Show code preview with syntax highlighting
            with st.expander("👁️ Preview Generated Code", expanded=True):
                st.code(st.session_state['generated_cpp'], language='cpp')
    else:
        st.warning("⚠️ Code generation not available. Make sure `generate_mfma_code.py` is in the same directory.")
    
    # Show commands used
    with st.expander("🔧 All Commands Reference"):
        st.code(f"""
# Register Layout (-R flag) - Shows full matrix structure
python3 {calc_path} -a {arch} -i {inst} -{{matrix}} -R --csv \\
    --cbsz {cbsz} --abid {abid} --blgp {blgp} --opsel {opsel} -w {wavefront}

# Matrix Layout (-M flag) - Shows register grouping
python3 {calc_path} -a {arch} -i {inst} -{{matrix}} -M \\
    --cbsz {cbsz} --abid {abid} --blgp {blgp} --opsel {opsel} -w {wavefront}

# Detailed instruction info (-d flag)
python3 {calc_path} -a {arch} -i {inst} -d

# List instructions (-L flag)
python3 {calc_path} -a {arch} -L
""")

if __name__ == "__main__":
    main()

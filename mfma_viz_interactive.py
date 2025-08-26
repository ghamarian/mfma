#!/usr/bin/env python3
"""
AMD Matrix Instruction Interactive Visualizer
A modern Streamlit app that provides graphical visualizations of AMD matrix instructions
using the AMD matrix_calculator.py backend for accurate ISA information.
"""

import io
import os
import re
import json
import subprocess
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Configure page
st.set_page_config(
    page_title="AMD Matrix Instruction Interactive Visualizer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
    div[data-testid="metric-container"] {
        background-color: rgba(28, 131, 225, 0.1);
        border: 1px solid rgba(28, 131, 225, 0.2);
        padding: 5px 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# Color palette for visualizations
COLORS = {
    'primary': '#1c83e1',
    'secondary': '#ff6b6b',
    'accent': '#4ecdc4',
    'warning': '#ffd93d',
    'success': '#6bcf7f',
    'dark': '#2c3e50',
    'light': '#ecf0f1',
    'grid': '#95a5a6',
    'highlight': '#e74c3c'
}

@dataclass
class InstructionInfo:
    """Container for instruction information"""
    name: str
    arch: str
    m: int
    n: int
    k: int
    blocks: int
    cycles: int
    in_type: str
    out_type: str
    sparse: bool
    blgp: bool
    cbsz_abid: bool
    cd_opsel: bool
    neg: bool
    coexec: bool
    coexec_delay: int

class MatrixCalculatorWrapper:
    """Wrapper for the AMD matrix_calculator.py tool"""
    
    def __init__(self, calculator_path: str):
        self.calculator_path = calculator_path
        self.available = self._check_availability()
        
    def _check_availability(self) -> bool:
        """Check if the calculator is available"""
        if not os.path.exists(self.calculator_path):
            return False
        try:
            result = self._run_command(["-v"])
            return result[0]
        except:
            return False
    
    def _run_command(self, args: List[str]) -> Tuple[bool, str]:
        """Run the calculator with given arguments"""
        try:
            cmd = ["python3", self.calculator_path] + args
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=10
            )
            success = result.returncode == 0
            output = result.stdout if success else result.stderr
            return success, output
        except Exception as e:
            return False, str(e)
    
    def get_architectures(self) -> List[str]:
        """Get list of supported architectures"""
        # Based on the dict_isas from matrix_calculator.py
        return ['cdna1', 'cdna2', 'cdna3', 'rdna3', 'rdna4']
    
    def get_instructions(self, arch: str) -> List[str]:
        """Get list of instructions for an architecture"""
        success, output = self._run_command(["-a", arch, "-L"])
        if not success:
            return []
        
        instructions = []
        for line in output.splitlines():
            line = line.strip()
            if line and not line.lower().startswith(('available', 'legal', 'instruction')):
                instructions.append(line)
        return instructions
    
    def get_instruction_details(self, arch: str, inst: str) -> Optional[Dict]:
        """Get detailed information about an instruction"""
        success, output = self._run_command(["-a", arch, "-i", inst, "-d"])
        if not success:
            return None
        
        details = {}
        for line in output.splitlines():
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower().replace(' ', '_')
                value = value.strip()
                details[key] = value
        return details
    
    def get_register_layout(self, arch: str, inst: str, matrix: str, 
                          cbsz: int = 0, abid: int = 0, blgp: int = 0, 
                          opsel: int = 0, wavefront: int = 0) -> Optional[pd.DataFrame]:
        """Get register layout for a matrix"""
        args = ["-a", arch, "-i", inst, f"-{matrix}", "-R"]
        # Always include modifiers even if 0
        args.extend(["--cbsz", str(cbsz)])
        args.extend(["--abid", str(abid)])
        args.extend(["--blgp", str(blgp)])
        args.extend(["--opsel", str(opsel)])
        if wavefront > 0:
            args.extend(["-w", str(wavefront)])
        
        args.append("--csv")  # Request CSV format for easier parsing
        
        success, output = self._run_command(args)
        if not success:
            # Try without CSV flag as fallback
            args.remove("--csv")
            success, output = self._run_command(args)
            if not success:
                return None
            # Parse table format
            return self._parse_table_output(output)
        
        try:
            # Parse CSV output
            df = pd.read_csv(io.StringIO(output))
            return df
        except:
            # Try to parse as table
            return self._parse_table_output(output)
    
    def _parse_table_output(self, output: str) -> Optional[pd.DataFrame]:
        """Parse table format output from calculator"""
        try:
            lines = output.strip().split('\n')
            data = []
            for line in lines:
                if 'Lane' in line or 'Register' in line or 'v' in line:
                    # This might be a data line
                    parts = line.split()
                    if len(parts) >= 2:
                        data.append(parts)
            if data:
                return pd.DataFrame(data[1:], columns=data[0] if data else [])
            return None
        except:
            return None
    
    def get_matrix_layout(self, arch: str, inst: str, matrix: str,
                         cbsz: int = 0, abid: int = 0, blgp: int = 0,
                         opsel: int = 0, wavefront: int = 0) -> Optional[pd.DataFrame]:
        """Get matrix layout showing values in registers"""
        args = ["-a", arch, "-i", inst, f"-{matrix}", "-M"]
        # Always include modifiers
        args.extend(["--cbsz", str(cbsz)])
        args.extend(["--abid", str(abid)])
        args.extend(["--blgp", str(blgp)])
        args.extend(["--opsel", str(opsel)])
        if wavefront > 0:
            args.extend(["-w", str(wavefront)])
        
        args.append("--csv")
        
        success, output = self._run_command(args)
        if not success:
            # Try without CSV
            args.remove("--csv")
            success, output = self._run_command(args)
            if not success:
                return None
            return self._parse_table_output(output)
        
        try:
            df = pd.read_csv(io.StringIO(output))
            return df
        except:
            return self._parse_table_output(output)
    
    def get_single_register(self, arch: str, inst: str, matrix: str,
                           i: int, j: int, k: int, block: int = 0,
                           cbsz: int = 0, abid: int = 0, blgp: int = 0,
                           opsel: int = 0, wavefront: int = 0) -> Optional[Dict]:
        """Get register and lane for a specific matrix element"""
        args = ["-a", arch, "-i", inst, f"-{matrix}", "-g",
                "-I", str(i), "-J", str(j), "-K", str(k), "-b", str(block)]
        # Always include all parameters
        args.extend(["--cbsz", str(cbsz)])
        args.extend(["--abid", str(abid)])
        args.extend(["--blgp", str(blgp)])
        args.extend(["--opsel", str(opsel)])
        if wavefront > 0:
            args.extend(["-w", str(wavefront)])
        
        success, output = self._run_command(args)
        if not success:
            return None
        
        # Parse output to extract register and lane
        result = {}
        for line in output.splitlines():
            # Look for patterns like "v123" for register and "lane 45"
            if 'v' in line.lower() or 'register' in line.lower() or 'lane' in line.lower():
                # Extract register number
                import re
                reg_match = re.search(r'v(\d+)', line)
                if reg_match:
                    result['register'] = f'v{reg_match.group(1)}'
                lane_match = re.search(r'lane[\s:]+(\d+)', line, re.IGNORECASE)
                if lane_match:
                    result['lane'] = lane_match.group(1)
        return result if result else None

def create_matrix_heatmap(data: np.ndarray, title: str, colorscale: str = 'Viridis') -> go.Figure:
    """Create an interactive heatmap for matrix visualization"""
    fig = go.Figure(data=go.Heatmap(
        z=data,
        colorscale=colorscale,
        showscale=True,
        hovertemplate='Row: %{y}<br>Col: %{x}<br>Value: %{z}<extra></extra>',
        colorbar=dict(
            title="Value",
            titleside="right",
            tickmode="linear",
            tick0=0,
            dtick=1
        )
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Column",
        yaxis_title="Row",
        height=500,
        hovermode='closest',
        xaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)', autorange='reversed')
    )
    
    return fig

def create_register_visualization(register_data: pd.DataFrame, matrix_name: str) -> go.Figure:
    """Create visualization of register layout"""
    # Create a 3D surface plot showing register distribution
    if register_data is None or register_data.empty:
        return go.Figure().add_annotation(text="No data available", showarrow=False)
    
    # Extract lane and register information
    lanes = register_data.index if 'Lane' not in register_data.columns else register_data['Lane'].values
    
    # Create a grid for visualization
    max_lanes = 64
    max_regs = 16  # Typical max registers per matrix
    
    grid = np.zeros((max_lanes, max_regs))
    
    # Fill grid with data (this is a simplified visualization)
    for i, lane in enumerate(lanes[:max_lanes]):
        for j in range(min(max_regs, len(register_data.columns))):
            if j < len(register_data.columns):
                try:
                    val = register_data.iloc[i, j]
                    if pd.notna(val):
                        grid[i, j] = float(val) if isinstance(val, (int, float)) else 1
                except:
                    grid[i, j] = 0
    
    fig = go.Figure(data=go.Heatmap(
        z=grid,
        colorscale='Viridis',
        showscale=True,
        hovertemplate='Lane: %{y}<br>Register: %{x}<br>Value: %{z}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f"{matrix_name} Matrix Register Layout",
        xaxis_title="Register Index",
        yaxis_title="Lane ID",
        height=600,
        hovermode='closest'
    )
    
    return fig

def create_matrix_info_card(m: int, n: int, k: int, matrix_type: str, data_type: str) -> Dict[str, Any]:
    """Create matrix information card data"""
    colors = {
        'A': COLORS['primary'],
        'B': COLORS['secondary'],
        'C': COLORS['accent'],
        'D': COLORS['warning']
    }
    
    if matrix_type == 'A':
        dims = f"{m} × {k}"
        elements = m * k
        role = "Left multiplicand"
    elif matrix_type == 'B':
        dims = f"{k} × {n}"
        elements = k * n
        role = "Right multiplicand"
    else:
        dims = f"{m} × {n}"
        elements = m * n
        role = "Accumulator/Result"
    
    return {
        'dimensions': dims,
        'elements': elements,
        'data_type': data_type,
        'role': role,
        'color': colors.get(matrix_type, COLORS['dark'])
    }

def create_performance_gauge(cycles: int, max_cycles: int = 128) -> go.Figure:
    """Create a gauge chart for performance metrics"""
    efficiency = (max_cycles - cycles) / max_cycles * 100
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=efficiency,
        title={'text': "Efficiency %"},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': COLORS['success'] if efficiency > 70 else COLORS['warning'] if efficiency > 40 else COLORS['secondary']},
            'steps': [
                {'range': [0, 40], 'color': "lightgray"},
                {'range': [40, 70], 'color': "gray"},
                {'range': [70, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=250)
    return fig

def create_block_visualization(blocks: int, m: int, n: int) -> go.Figure:
    """Visualize how blocks are arranged in the matrix"""
    fig = make_subplots(
        rows=1, cols=blocks,
        subplot_titles=[f"Block {i}" for i in range(blocks)],
        horizontal_spacing=0.05
    )
    
    for block in range(blocks):
        # Create a simple pattern for each block
        block_data = np.ones((m//blocks, n)) * (block + 1)
        
        fig.add_trace(
            go.Heatmap(
                z=block_data,
                colorscale='Viridis',
                showscale=False,
                hovertemplate=f'Block {block}<br>Row: %{{y}}<br>Col: %{{x}}<extra></extra>'
            ),
            row=1, col=block+1
        )
    
    fig.update_layout(
        title="Matrix Block Distribution",
        height=300,
        showlegend=False
    )
    
    return fig

def create_wave_visualization(wavefront: int) -> go.Figure:
    """Visualize wavefront execution pattern"""
    # Create a circular visualization for wavefront
    theta = np.linspace(0, 2*np.pi, wavefront, endpoint=False)
    r = np.ones(wavefront)
    
    fig = go.Figure()
    
    # Add wave lanes as a polar scatter plot
    fig.add_trace(go.Scatterpolar(
        r=r,
        theta=theta * 180/np.pi,
        mode='markers+text',
        marker=dict(size=15, color=COLORS['primary']),
        text=[str(i) for i in range(wavefront)],
        textposition='top center',
        name='Wave Lanes'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=False),
            angularaxis=dict(visible=True)
        ),
        title=f"Wavefront {wavefront} Execution Pattern",
        height=400,
        showlegend=False
    )
    
    return fig

def main():
    """Main application"""
    st.title("🎯 AMD Matrix Instruction Interactive Visualizer")
    st.markdown("### Graphical visualization of AMD matrix instructions with ISA-accurate data")
    
    # Initialize calculator wrapper - check multiple possible paths
    possible_paths = [
        "/Users/amir/projects/github/mfma/amd_matrix_instruction_calculator/matrix_calculator.py",
        "amd_matrix_instruction_calculator/matrix_calculator.py",
        "./amd_matrix_instruction_calculator/matrix_calculator.py",
        "matrix_calculator.py"
    ]
    
    calculator_path = None
    for path in possible_paths:
        if os.path.exists(path):
            calculator_path = path
            break
    
    if calculator_path is None:
        calculator_path = possible_paths[0]  # Use first as default
    
    calc = MatrixCalculatorWrapper(calculator_path)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        if calc.available:
            st.success("✅ Calculator detected")
        else:
            st.warning("⚠️ Calculator not found - using fallback data")
            st.info(f"Looking for: {calculator_path}")
        
        # Architecture selection
        architectures = {
            'cdna1': 'CDNA1 (MI100)',
            'cdna2': 'CDNA2 (MI200)',
            'cdna3': 'CDNA3 (MI300)',
            'rdna3': 'RDNA3 (RX 7000)',
            'rdna4': 'RDNA4'
        }
        
        arch = st.selectbox(
            "🏗️ Architecture",
            options=list(architectures.keys()),
            format_func=lambda x: architectures[x],
            index=1
        )
        
        # Get instructions for selected architecture
        if calc.available:
            instructions = calc.get_instructions(arch)
        else:
            # Fallback instructions
            instructions = [
                'v_mfma_f32_32x32x1f32',
                'v_mfma_f32_16x16x16f16',
                'v_mfma_f32_32x32x8f16',
                'v_mfma_i32_16x16x32i8'
            ]
        
        inst = st.selectbox(
            "📋 Instruction",
            options=instructions,
            index=0 if instructions else None
        )
        
        # Wavefront configuration
        wavefront = st.selectbox(
            "🌊 Wavefront",
            options=[32, 64],
            index=1 if arch in ['cdna1', 'cdna2', 'cdna3'] else 0
        )
        
        st.divider()
        st.subheader("🎛️ Modifiers")
        
        # Instruction modifiers
        cbsz = st.slider("CBSZ (A broadcast)", 0, 3, 0)
        abid = st.slider("ABID (A broadcast ID)", 0, 7, 0)
        blgp = st.slider("BLGP (B lane group)", 0, 7, 0)
        opsel = st.slider("OPSEL (C/D select)", 0, 15, 0)
        
        st.divider()
        st.subheader("🎨 Visualization")
        
        color_scheme = st.selectbox(
            "Color Scheme",
            ["Viridis", "Plasma", "Inferno", "Turbo", "Rainbow", "Portland"]
        )
        
        show_perf = st.checkbox("Show performance metrics", value=True)
    
    # Extract instruction dimensions from name
    m = n = k = 16  # defaults
    blocks = 1
    cycles = 32
    
    if inst:
        # Try to parse dimensions from instruction name
        match = re.search(r'(\d+)x(\d+)x(\d+)', inst)
        if match:
            m, n, k = map(int, match.groups())
        
        # Get detailed instruction info if available
        if calc.available:
            details = calc.get_instruction_details(arch, inst)
            if details:
                try:
                    m = int(details.get('m', m))
                    n = int(details.get('n', n))
                    k = int(details.get('k', k))
                    blocks = int(details.get('blocks', blocks))
                    cycles = int(details.get('cycles', cycles))
                except:
                    pass
    
    # Main content area with tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Matrix Layouts", 
        "🔲 Register Mapping", 
        "⚡ Performance", 
        "🎯 Block Distribution"
    ])
    
    with tab1:
        st.header("Matrix Layout Visualization")
        
        # Determine data types
        in_type = "fp16"  # default
        out_type = "fp32"  # default
        if inst:
            if "f32" in inst and "f32" in inst.split("x")[-1]:
                in_type = "fp32"
            elif "f16" in inst or "bf16" in inst:
                in_type = "fp16"
            elif "i8" in inst:
                in_type = "int8"
            elif "i32" in inst:
                out_type = "int32"
        
        # Matrix information cards
        st.subheader("📐 Matrix Dimensions")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            info_a = create_matrix_info_card(m, n, k, 'A', in_type)
            st.info(f"**Matrix A**\n\n"
                   f"📏 {info_a['dimensions']}\n\n"
                   f"🔢 {info_a['elements']} elements\n\n"
                   f"📊 {info_a['data_type']}")
        
        with col2:
            info_b = create_matrix_info_card(m, n, k, 'B', in_type)
            st.info(f"**Matrix B**\n\n"
                   f"📏 {info_b['dimensions']}\n\n"
                   f"🔢 {info_b['elements']} elements\n\n"
                   f"📊 {info_b['data_type']}")
        
        with col3:
            info_c = create_matrix_info_card(m, n, k, 'C', out_type)
            st.warning(f"**Matrix C**\n\n"
                      f"📏 {info_c['dimensions']}\n\n"
                      f"🔢 {info_c['elements']} elements\n\n"
                      f"📊 {info_c['data_type']}")
        
        with col4:
            info_d = create_matrix_info_card(m, n, k, 'D', out_type)
            st.success(f"**Matrix D**\n\n"
                      f"📏 {info_d['dimensions']}\n\n"
                      f"🔢 {info_d['elements']} elements\n\n"
                      f"📊 {info_d['data_type']}")
        
        st.divider()
        
        # Matrix heatmaps
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔵 Input Matrices")
            
            # Create heatmap for Matrix A with element indices
            a_data = np.arange(m * k).reshape(m, k)
            fig_a_heat = create_matrix_heatmap(a_data, f"Matrix A [{m}×{k}] - Element Indices", color_scheme)
            st.plotly_chart(fig_a_heat, use_container_width=True)
            
            # Matrix B visualization
            b_data = np.arange(k * n).reshape(k, n)
            fig_b_heat = create_matrix_heatmap(b_data, f"Matrix B [{k}×{n}] - Element Indices", color_scheme)
            st.plotly_chart(fig_b_heat, use_container_width=True)
        
        with col2:
            st.subheader("🟢 Output Matrices")
            
            # Matrix C visualization (accumulator)
            c_data = np.ones((m, n)) * 0.5  # Initialize with neutral values
            fig_c_heat = create_matrix_heatmap(c_data, f"Matrix C [{m}×{n}] - Accumulator", color_scheme)
            st.plotly_chart(fig_c_heat, use_container_width=True)
            
            # Matrix D visualization (result)
            d_data = np.arange(m * n).reshape(m, n)
            fig_d_heat = create_matrix_heatmap(d_data, f"Matrix D [{m}×{n}] - Result Indices", color_scheme)
            st.plotly_chart(fig_d_heat, use_container_width=True)
    
    with tab2:
        st.header("Register Mapping Visualization")
        st.markdown("""
        This tab shows how matrix elements are mapped to GPU registers and lanes.
        Each matrix element is stored in a specific register (v0-v255) at a specific lane (0-63).
        """)
        
        matrices = ['A', 'B', 'C', 'D']
        selected_matrix = st.selectbox("Select Matrix", matrices)
        
        if calc.available:
            with st.spinner(f"Fetching register layout for matrix {selected_matrix}..."):
                # Get register layout from calculator
                reg_data = calc.get_register_layout(
                    arch, inst, selected_matrix,
                    cbsz=cbsz, abid=abid, blgp=blgp, 
                    opsel=opsel, wavefront=wavefront
                )
            
            if reg_data is not None and not reg_data.empty:
                st.success(f"✅ Register layout retrieved for matrix {selected_matrix}")
                
                # Create visualization
                fig_reg = create_register_visualization(reg_data, selected_matrix)
                st.plotly_chart(fig_reg, use_container_width=True)
                
                # Show raw data in expander
                with st.expander("📋 View Raw Register Data"):
                    st.dataframe(reg_data)
                    st.caption(f"Showing {len(reg_data)} entries")
            else:
                st.warning(f"⚠️ Could not retrieve register layout for matrix {selected_matrix}")
                st.info("""
                Possible reasons:
                - The calculator may not support this specific instruction/matrix combination
                - The modifiers (CBSZ, ABID, BLGP, OPSEL) may not be valid for this instruction
                - Try selecting a different matrix or adjusting the modifiers
                """)
        else:
            # Create synthetic register visualization
            st.warning("⚠️ Calculator not available - showing example data")
            st.info(f"To get accurate data, ensure matrix_calculator.py is at: {calculator_path}")
            
            # Create example data that looks realistic
            synthetic_data = pd.DataFrame({
                'Lane': range(64),
                'Register': [f'v{i//4}' for i in range(64)],
                'Element': [f'[{i//8},{i%8}]' for i in range(64)]
            })
            fig_reg = create_register_visualization(synthetic_data, selected_matrix)
            st.plotly_chart(fig_reg, use_container_width=True)
        
        # Interactive element picker
        st.divider()
        st.subheader("🎯 Element to Register Mapping")
        st.markdown("""
        Find which register and lane stores a specific matrix element.
        Enter the coordinates of the element you want to locate.
        """)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            elem_i = st.number_input("I coordinate (row)", 0, m-1, 0, help="Row index (0-based)")
        with col2:
            elem_j = st.number_input("J coordinate (column)", 0, n-1, 0, help="Column index (0-based)")
        with col3:
            elem_k = st.number_input("K coordinate (depth)", 0, k-1, 0, help="K dimension index (0-based)")
        
        if st.button("🔍 Find Register Location", type="primary"):
            if calc.available:
                with st.spinner("Querying calculator..."):
                    result = calc.get_single_register(
                        arch, inst, selected_matrix,
                        elem_i, elem_j, elem_k, 0,
                        cbsz=cbsz, abid=abid, blgp=blgp,
                        opsel=opsel, wavefront=wavefront
                    )
                if result:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.success(f"✅ Found mapping for element [{elem_i},{elem_j},{elem_k}]")
                    with col2:
                        st.info(f"**Register:** {result.get('register', 'Unknown')}\n\n**Lane:** {result.get('lane', 'Unknown')}")
                else:
                    st.warning("⚠️ Could not determine register mapping for this element")
                    st.caption("This might be because the element coordinates are out of bounds for this matrix")
            else:
                # Synthetic result for demonstration
                reg = (elem_i * 8 + elem_j) % 256
                lane = (elem_i + elem_j * 2) % wavefront
                st.info(f"📍 Example mapping (not from calculator):\n\nElement [{elem_i},{elem_j},{elem_k}] → Register: v{reg}, Lane: {lane}")
    
    with tab3:
        st.header("Performance Metrics")
        
        if show_perf:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Cycles", cycles)
                st.metric("Blocks", blocks)
                st.metric("Operations", f"{m*n*k*2} FLOPs")
            
            with col2:
                throughput = (m * n * k * 2) / cycles if cycles > 0 else 0
                st.metric("Throughput", f"{throughput:.1f} FLOPs/cycle")
                st.metric("Matrix Size", f"{m}×{n}×{k}")
                st.metric("Wavefront", wavefront)
            
            with col3:
                # Performance gauge
                fig_gauge = create_performance_gauge(cycles)
                st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Execution timeline
        st.subheader("Execution Timeline")
        
        timeline_data = []
        for i in range(blocks):
            timeline_data.append({
                'Block': f'Block {i}',
                'Start': i * (cycles // blocks) if blocks > 0 else 0,
                'End': (i + 1) * (cycles // blocks) if blocks > 0 else cycles,
                'Duration': cycles // blocks if blocks > 0 else cycles
            })
        
        if timeline_data:
            df_timeline = pd.DataFrame(timeline_data)
            
            # Use a Gantt-style bar chart instead of timeline
            fig_timeline = go.Figure()
            for i, row in df_timeline.iterrows():
                fig_timeline.add_trace(go.Bar(
                    name=row['Block'],
                    x=[row['Duration']],
                    y=[row['Block']],
                    orientation='h',
                    marker=dict(color=COLORS['primary'] if i % 2 == 0 else COLORS['accent']),
                    hovertemplate=f"{row['Block']}<br>Start: {row['Start']}<br>End: {row['End']}<br>Duration: {row['Duration']} cycles<extra></extra>"
                ))
            
            fig_timeline.update_layout(
                title="Block Execution Timeline",
                xaxis_title="Cycles",
                yaxis_title="Block",
                height=300,
                showlegend=False,
                barmode='stack'
            )
            st.plotly_chart(fig_timeline, use_container_width=True)
    
    with tab4:
        st.header("Block Distribution")
        
        if blocks > 1:
            fig_blocks = create_block_visualization(blocks, m, n)
            st.plotly_chart(fig_blocks, use_container_width=True)
            
            # Block details
            st.subheader("Block Configuration")
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**Total Blocks:** {blocks}")
                st.info(f"**Block Size:** {m//blocks if blocks > 0 else m}×{n}")
            with col2:
                st.info(f"**Cycles per Block:** {cycles//blocks if blocks > 0 else cycles}")
                st.info(f"**Parallel Execution:** {'Yes' if blocks > 1 else 'No'}")
        else:
            st.info("This instruction uses a single block")
    
    # Removed tab5 - Execution Pattern was confusing
    if False:  # Keep code for potential future use
        st.header("Execution Pattern")
        
        # Wavefront visualization
        fig_wave = create_wave_visualization(wavefront)
        st.plotly_chart(fig_wave, use_container_width=True)
        
        # SIMD lane mapping
        st.subheader("SIMD Lane Mapping")
        
        lane_data = []
        for lane in range(min(wavefront, 16)):  # Show first 16 lanes
            lane_data.append({
                'Lane': lane,
                'Thread': f'T{lane}',
                'Active': 'Yes' if lane < wavefront else 'No',
                'Group': lane // 16
            })
        
        df_lanes = pd.DataFrame(lane_data)
        
        fig_lanes = px.scatter(
            df_lanes,
            x='Lane',
            y='Group',
            color='Active',
            size=[10]*len(df_lanes),
            title="SIMD Lane Organization",
            hover_data=['Thread']
        )
        fig_lanes.update_layout(height=300)
        st.plotly_chart(fig_lanes, use_container_width=True)
        
        # Modifier effects visualization
        if any([cbsz, abid, blgp, opsel]):
            st.subheader("Modifier Effects")
            
            modifier_effects = {
                'CBSZ': cbsz,
                'ABID': abid,
                'BLGP': blgp,
                'OPSEL': opsel
            }
            
            fig_mods = go.Figure(data=[
                go.Bar(
                    x=list(modifier_effects.keys()),
                    y=list(modifier_effects.values()),
                    marker_color=[COLORS['primary'] if v > 0 else COLORS['light'] 
                                 for v in modifier_effects.values()]
                )
            ])
            fig_mods.update_layout(
                title="Active Modifiers",
                xaxis_title="Modifier",
                yaxis_title="Value",
                height=300
            )
            st.plotly_chart(fig_mods, use_container_width=True)
    
    # Footer with information
    st.divider()
    st.markdown("""
    ### 📚 About This Visualizer
    
    This interactive tool provides graphical visualizations of AMD matrix instructions, converting
    the text-based output from the AMD Matrix Instruction Calculator into modern, interactive charts.
    
    **Features:**
    - 🎯 ISA-accurate data when calculator is available
    - 📊 Interactive Plotly visualizations
    - 🔲 Register mapping heatmaps
    - ⚡ Performance metrics and timelines
    - 🎨 Customizable color schemes
    - 📈 3D matrix views and execution patterns
    
    **Supported Architectures:**
    - CDNA1 (MI100), CDNA2 (MI200), CDNA3 (MI300)
    - RDNA3, RDNA4
    
    ---
    *Powered by AMD Matrix Instruction Calculator*
    """)

if __name__ == "__main__":
    main()

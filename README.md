# AMD MFMA Visualizer

A comprehensive tool for visualizing AMD Matrix Fused Multiply-Add (MFMA) instructions and generating optimized C++ kernel code.

## Credits

This project builds upon the excellent work from AMD's [Matrix Instruction Calculator](https://github.com/ROCm/amd_matrix_instruction_calculator) repository. The calculator provides the core functionality for computing register layouts and matrix element mappings for AMD's matrix instructions.

## Features

- **🔍 Matrix Layout Visualization**: View both Register Layout and Matrix Layout for all matrix operands (A, B, C, D)
- **📊 Performance Metrics**: See execution cycles, FLOPs, register usage, and co-execution capabilities
- **🚀 C++ Code Generation**: Generate complete HIP/ROCm kernel code for any MFMA instruction
- **🎯 Multi-Architecture Support**: CDNA1, CDNA2, CDNA3, RDNA3, RDNA4
- **⚙️ Modifier Support**: Configure CBSZ, ABID, BLGP, OPSEL modifiers

## Quick Start

```bash
# Run the main visualizer app
streamlit run mfma_visualizer.py
```

## Main Components

### 1. `mfma_visualizer.py` - Main Streamlit Application
The primary tool that provides:
- Interactive instruction selection with search
- Dual-view matrix visualization (Register Layout + Matrix Layout)
- Performance metrics display
- C++ kernel code generation with download
- Mapping formula display

### 2. `generate_mfma_code.py` - C++ Code Generator
Standalone module for generating MFMA kernel code:
```python
from generate_mfma_code import generate_mfma_cpp

code = generate_mfma_cpp(
    arch="cdna1",
    instruction="v_mfma_f32_16x16x16f16",
    cbsz=0, abid=0, blgp=0, opsel=0,
    wavefront=64
)
```

Generated code includes:
- Complete HIP kernel implementation
- Fragment types with correct VGPR sizing
- Load/store functions with detailed register mapping comments
- MFMA builtin wrapper
- Test harness structure

### 3. `amd_matrix_instruction_calculator/` - Core Calculator
The underlying matrix calculator from AMD that provides:
- Register layout calculations
- Matrix element to register/lane mappings
- Instruction metadata

## Project Structure

```
mfma/
├── mfma_visualizer.py      # Main Streamlit app with all features
├── generate_mfma_code.py   # C++ kernel code generator
├── README.md               # This file
├── amd_matrix_instruction_calculator/  # AMD's matrix calculator
├── examples/               # Generated C++ examples
├── tests/                  # Test scripts
├── docs/                   # Documentation and analysis
└── old_versions/           # Previous app versions (archived)
```

## Example Usage

1. **Visualize MFMA Instructions**:
   - Select architecture (e.g., CDNA1)
   - Choose instruction (e.g., v_mfma_f32_16x16x16f16)
   - View register layouts and performance metrics

2. **Generate C++ Code**:
   - Configure modifiers if needed
   - Click "Generate C++ Code"
   - Download the complete kernel implementation

3. **Understand Register Mappings**:
   - View both Register Layout (full matrix view)
   - See Matrix Layout (grouped by register)
   - Check mapping formulas for manual calculations

## Requirements

- Python 3.8+
- Streamlit
- Pandas
- NumPy
- Plotly

## Installation

```bash
pip install streamlit pandas numpy plotly
```

## Key Insights

- **VGPR Grouping**: For f16/bf16 data types, each VGPR holds two K values in different bit ranges ([0:15] and [16:31])
- **Memory Access**: The visualizer shows which loads are contiguous (can be vectorized) vs non-contiguous
- **Register Efficiency**: Matrix C/D loads/stores are typically contiguous for optimal performance

## Support

For issues or questions about the AMD Matrix Instruction Calculator, refer to the original documentation in `amd_matrix_instruction_calculator/README.md`.

## License

The AMD Matrix Instruction Calculator is used under its original license. Please see `amd_matrix_instruction_calculator/LICENSE` for details.

## Acknowledgments

Special thanks to AMD for providing the Matrix Instruction Calculator tool, which makes this visualization and code generation possible.

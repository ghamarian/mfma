# Matrix A Layout Differences: Raw Calculator vs App Display

## Overview
This document explains the differences between the raw matrix calculator output and how it's displayed in the mfma_viz_simple.py app for Matrix A.

## 1. Raw Calculator Output
When you run the matrix calculator directly:
```bash
python3 amd_matrix_instruction_calculator/matrix_calculator.py -a cdna1 -i v_mfma_f32_16x16x16f16 -A -R --csv
```

**Output format:**
```
Architecture: CDNA1
Instruction: V_MFMA_F32_16X16X16F16
Block 0
A[M][K],0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
0,v0{0}.[15:0],v0{0}.[31:16],v1{0}.[15:0],v1{0}.[31:16],...
1,v0{1}.[15:0],v0{1}.[31:16],v1{1}.[15:0],v1{1}.[31:16],...
```

**Characteristics:**
- Plain CSV format
- Simple headers: just K indices (0-15)
- No visual formatting
- No block column in the data

## 2. App Processed Data (DataFrame)
The app parses the raw output into a pandas DataFrame:

**Structure:**
- Shape: (16, 18) for 16x16x16 instruction
- Columns: ['A[M][K]', '0', '1', ..., '15', 'Block']
- Added 'Block' column (always 0 for single-block instructions)

**Example row:**
```python
{'A[M][K]': 0, 
 '0': 'v0{0}.[15:0]', 
 '1': 'v0{0}.[31:16]',
 '2': 'v1{0}.[15:0]',
 '3': 'v1{0}.[31:16]',
 ...
 'Block': '0'}
```

## 3. Visual Display in App (Plotly Table)

**Enhanced Headers:**
- For f16/bf16 instructions, headers show both K index and bit range
- Example: "K=0 Low [15:0]", "K=1 High [31:16]"
- First column shows "A Row" instead of "A[M][K]"

**Color Coding:**
- v0 register cells: Light blue (#E6F3FF)
- v1 register cells: Light orange (#FFE6CC)
- Makes it easy to visualize register usage patterns

**Interactive Features:**
- HTML table with hover effects
- Scrollable for large matrices
- Clean, professional appearance

## 4. Key Differences Summary

| Aspect | Raw Calculator | App Display |
|--------|---------------|-------------|
| Format | Plain CSV text | Interactive HTML table |
| Headers | Simple K indices | K indices + bit ranges |
| Colors | None | Blue for v0, Orange for v1 |
| Block info | Not included | Added as last column |
| Readability | Basic | Enhanced with visual cues |

## 5. Data Integrity
**Important:** The actual register mapping data remains unchanged. The app only adds visual enhancements and structure - it doesn't modify the core information from the calculator.

## Example: CDNA1 f16 Instruction Pattern
- Uses 2 GPRs (v0 and v1)
- Pattern: v0, v0, v1, v1, v0, v0, v1, v1... across K dimension
- Each register holds 2 f16 values (low/high bits)
- App makes this pattern visually obvious through color coding

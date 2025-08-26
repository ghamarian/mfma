# CDNA1 v_mfma_f32_16x16x8bf16 Investigation Summary

## User Concern
The user mentioned that for CDNA1, the instruction should show "two columns" (v0 low and v0 high) for matrix A.

## Investigation Results

### For v_mfma_f32_16x16x8bf16 on CDNA1:
- **Matrix dimensions**: M=16, N=16, K=8
- **GPRs required**: 1 (only v0 register)
- **Actual output**: Shows 8 K columns (K=0 through K=7)

### Register Mapping Pattern:
```
K=0: v0{0}.[15:0]    (v0, lane 0, low bits)
K=1: v0{0}.[31:16]   (v0, lane 0, high bits)
K=2: v0{16}.[15:0]   (v0, lane 16, low bits)
K=3: v0{16}.[31:16]  (v0, lane 16, high bits)
K=4: v0{32}.[15:0]   (v0, lane 32, low bits)
K=5: v0{32}.[31:16]  (v0, lane 32, high bits)
K=6: v0{48}.[15:0]   (v0, lane 48, low bits)
K=7: v0{48}.[31:16]  (v0, lane 48, high bits)
```

### Key Differences Between CDNA1 and CDNA2:
1. **CDNA1**: Uses only v0 register (1 GPR)
2. **CDNA2**: Uses v0 register (1 GPR for this specific bf16 instruction)
3. Both show the same layout with 8 K columns

### For Comparison - v_mfma_f32_16x16x2bf16 on CDNA1:
- Shows only 2 K columns as expected:
  - K=0: v0{0}.[15:0]
  - K=1: v0{0}.[31:16]

## Conclusion
The visualization is working correctly:
- The number of K columns displayed matches the K dimension of the instruction
- CDNA1 correctly uses only the v0 register for all values
- The "two columns" the user mentioned might be referring to the K=2 instruction, not K=8

## Test Coverage Added
1. **test_matrix_a_layout_bf16**: Tests CDNA2 bf16 instruction layout
2. **test_cdna1_bf16_single_register**: Tests CDNA1 uses only v0 register

Both tests are passing, confirming the implementation is correct.

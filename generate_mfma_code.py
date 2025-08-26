#!/usr/bin/env python3
"""
Generate C++ MFMA kernel code based on instruction parameters
"""

import os
import re
from typing import Dict, Optional, Tuple

def generate_mfma_cpp(
    arch: str,
    instruction: str,
    cbsz: int = 0,
    abid: int = 0,
    blgp: int = 0,
    opsel: int = 0,
    wavefront: int = 64
) -> str:
    """Generate a complete C++ file for the given MFMA instruction"""
    
    # Parse instruction to extract parameters
    inst_parts = parse_instruction(instruction)
    if not inst_parts:
        return "// Error: Could not parse instruction"
    
    m, n, k = inst_parts['m'], inst_parts['n'], inst_parts['k']
    accum_type = inst_parts['accum_type']
    ab_type = inst_parts['ab_type']
    
    # Generate the complete C++ file
    code = f"""/*******************************************************************************
 * Auto-generated MFMA kernel for {instruction}
 * Architecture: {arch}
 * Wavefront: {wavefront}
 * Modifiers: CBSZ={cbsz}, ABID={abid}, BLGP={blgp}, OPSEL={opsel}
 *******************************************************************************/

#include <iostream>
#include <vector>
#include <hip/hip_ext.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

// Helper types
template<typename T, uint32_t Rank>
using VecT = T __attribute__((ext_vector_type(Rank)));

// Constants
const int WAVE_SIZE = {wavefront};
const int T_BLOCK_X = 1 * WAVE_SIZE;
const int T_BLOCK_Y = 1;

// Block dimensions from instruction
const int BLOCK_M = {m};
const int BLOCK_N = {n};
const int BLOCK_K = {k};

// Type definitions
{generate_type_definitions(ab_type, accum_type)}

// Fragment types
using AFragT = VecT<{get_cpp_type(ab_type)}, BLOCK_M * BLOCK_K / WAVE_SIZE>;
using BFragT = VecT<{get_cpp_type(ab_type)}, BLOCK_N * BLOCK_K / WAVE_SIZE>;
using AccumFragT = VecT<{get_cpp_type(accum_type)}, BLOCK_M * BLOCK_N / WAVE_SIZE>;
using CFragT = AccumFragT;

// MFMA builtin for this instruction
__device__ AccumFragT {generate_mfma_builtin(instruction, ab_type)}(AFragT aFrag, BFragT bFrag, AccumFragT accumFrag)
{{
    return __builtin_amdgcn_{generate_mfma_builtin(instruction, ab_type)}(aFrag, bFrag, accumFrag, {cbsz}, {abid}, {blgp});
}}

{generate_load_functions(m, n, k, ab_type, accum_type, arch, instruction)}

{generate_store_function(m, n, accum_type)}

__global__ void mfma_kernel_{m}x{n}x{k}(
    uint32_t m,
    uint32_t n,
    uint32_t k,
    {get_cpp_type(ab_type)} const* a,
    {get_cpp_type(ab_type)} const* b,
    {get_cpp_type(accum_type)} const* c,
    {get_cpp_type(accum_type)}* d,
    uint32_t lda,
    uint32_t ldb,
    uint32_t ldc,
    uint32_t ldd,
    {get_cpp_type(accum_type)} alpha,
    {get_cpp_type(accum_type)} beta)
{{
    // Create fragments
    auto fragA = AFragT{{}};
    auto fragB = BFragT{{}};
    auto fragAcc = AccumFragT{{}};
    
    // Initialize accumulator
    for(int i = 0; i < vectorSize(fragAcc); i++)
    {{
        fragAcc[i] = 0;
    }}
    
    // Get wave coordinates
    auto waveGridX = (blockIdx.x * blockDim.x + threadIdx.x) / WAVE_SIZE;
    auto waveGridY = (blockIdx.y * blockDim.y + threadIdx.y);
    
    // Scale to target C block coords
    auto cRow = waveGridX * BLOCK_M;
    auto cCol = waveGridY * BLOCK_N;
    
    // Bounds check
    if(cRow < m && cCol < n)
    {{
        // Step 1: Accumulate A x B
        for(int i = 0; i < k; i += BLOCK_K)
        {{
            fragA = load_A_{m}x{k}_col_major(a + (cRow + i * lda), lda);
            fragB = load_B_{k}x{n}_row_major(b + (i * ldb + cCol), ldb);
            fragAcc = {generate_mfma_builtin(instruction, ab_type)}(fragA, fragB, fragAcc);
        }}
        
        // Step 2: D = alpha * accumulator + beta * C
        auto fragC = load_C_{m}x{n}_col_major(c + (cRow + cCol * ldc), ldc);
        
        for(int i = 0; i < vectorSize(fragC); ++i)
        {{
            fragC[i] = alpha * fragAcc[i] + beta * fragC[i];
        }}
        
        // Step 3: Store result
        store_C_{m}x{n}_col_major(d + (cRow + cCol * ldd), fragC, ldd);
    }}
}}

// Helper function template
template<typename T, uint32_t Rank>
static constexpr int32_t vectorSize(VecT<T, Rank> const& v)
{{
    return Rank;
}}

// Host test function
__host__ void test_mfma_{m}x{n}x{k}(uint32_t m, uint32_t n, uint32_t k, 
                                    {get_cpp_type(accum_type)} alpha, 
                                    {get_cpp_type(accum_type)} beta)
{{
    // Size validation
    if((m < (BLOCK_M * T_BLOCK_X / WAVE_SIZE) || n < (BLOCK_N * T_BLOCK_Y) || k < BLOCK_K)
       || (m % BLOCK_M || n % BLOCK_N || k % BLOCK_K))
    {{
        std::cout << "Unsupported size!" << std::endl;
        return;
    }}
    
    // Leading dimensions
    int lda = m;  // col_major
    int ldb = n;  // row_major  
    int ldc = m;  // col_major
    int ldd = ldc;
    
    std::cout << "Testing {instruction} with M=" << m << ", N=" << n << ", K=" << k << std::endl;
    
    // Initialize host matrices
    std::vector<{get_cpp_type(ab_type)}> matrixA(m * k);
    std::vector<{get_cpp_type(ab_type)}> matrixB(k * n);
    std::vector<{get_cpp_type(accum_type)}> matrixC(m * n);
    std::vector<{get_cpp_type(accum_type)}> matrixD(m * n);
    
    // TODO: Initialize with test data
    // TODO: Allocate device memory
    // TODO: Launch kernel
    // TODO: Verify results
    
    std::cout << "Test completed!" << std::endl;
}}

int main()
{{
    // Test with default problem size
    test_mfma_{m}x{n}x{k}({m}, {n}, {k * 2}, 1.0, 0.0);
    return 0;
}}
"""
    
    return code

def parse_instruction(instruction: str) -> Optional[Dict]:
    """Parse MFMA instruction name to extract parameters"""
    # Pattern: v_mfma_<accum_type>_MxNxK<ab_type>
    match = re.match(r'v_mfma_(\w+)_(\d+)x(\d+)x(\d+)(\w+)', instruction)
    if not match:
        return None
    
    accum_type = match.group(1)
    m = int(match.group(2))
    n = int(match.group(3))
    k = int(match.group(4))
    ab_suffix = match.group(5)
    
    # Determine AB type from suffix
    if ab_suffix == 'f32':
        ab_type = 'f32'
    elif ab_suffix == 'f16':
        ab_type = 'f16'
    elif ab_suffix == 'bf16' or ab_suffix == 'bf16_1k':
        ab_type = 'bf16'
    elif ab_suffix == 'i8':
        ab_type = 'i8'
    elif ab_suffix == 'f64':
        ab_type = 'f64'
    else:
        ab_type = 'unknown'
    
    return {
        'accum_type': accum_type,
        'ab_type': ab_type,
        'm': m,
        'n': n,
        'k': k
    }

def get_cpp_type(type_str: str) -> str:
    """Convert type string to C++ type"""
    type_map = {
        'f16': '_Float16',
        'bf16': '__bf16',
        'f32': 'float',
        'f64': 'double',
        'i8': 'int8_t',
        'i32': 'int32_t'
    }
    return type_map.get(type_str, 'float')

def generate_type_definitions(ab_type: str, accum_type: str) -> str:
    """Generate type definitions based on data types"""
    defs = []
    if ab_type == 'f16':
        defs.append("using float16_t = _Float16;")
    if ab_type == 'bf16':
        defs.append("using bfloat16_t = __bf16;")
    if accum_type == 'f32' or ab_type == 'f32':
        defs.append("using float32_t = float;")
    if accum_type == 'f64' or ab_type == 'f64':
        defs.append("using float64_t = double;")
    return '\n'.join(defs)

def generate_mfma_builtin(instruction: str, ab_type: str) -> str:
    """Generate the MFMA builtin function name"""
    # Extract the base function name from instruction
    # v_mfma_f32_16x16x16f16 -> mfma_f32_16x16x16f16
    return instruction.replace('v_', '')

def generate_load_functions(m: int, n: int, k: int, ab_type: str, accum_type: str, arch: str, instruction: str) -> str:
    """Generate load functions for A, B, and C matrices"""
    
    # This is a simplified version - in reality, you'd use the matrix calculator
    # to get the exact register mappings
    cpp_ab_type = get_cpp_type(ab_type)
    cpp_accum_type = get_cpp_type(accum_type)
    
    # Determine if we need bit range comments for f16/bf16
    show_bit_ranges = ab_type in ['f16', 'bf16']
    
    load_a = f"""
// Load A matrix block (col-major)
// Size: (BLOCK_M x BLOCK_K)
// ASSUMPTION:
// - We want contiguous BLOCK_M sized column neighbors in register.
// - Data is in col_major format
// This means:
// - From A we will load K columns of size BLOCK_M to satisfy our input data
__device__ AFragT load_A_{m}x{k}_col_major({cpp_ab_type} const* input, int ld)
{{
    // Register Mapping:
    // Size              |   BLOCK_M  |   BLOCK_M   |   BLOCK_M   |   BLOCK_M    |  Vector
    // Register Element  | 0  ... {m-1:2} | {m}  ... {2*m-1:2} | {2*m}  ... {3*m-1:2} | {3*m}  ... {4*m-1:2} |  Element
    //                    ____________ _____________ _____________ ______________"""
    
    if show_bit_ranges:
        load_a += f"""
    // Reg 0 [0 :15]     |     K0     |     K4      |     K8      |     K12      |  v[0]
    // Reg 0 [16:31]     |     K1     |     K5      |     K9      |     K13      |  v[1]
    // Reg 1 [0 :15]     |     K2     |     K6      |     K10     |     K14      |  v[2]
    // Reg 1 [16:31]     |     K3     |     K7      |     K11     |     K15      |  v[3]"""
    else:
        load_a += f"""
    // Reg 0             |     K0     |     K1      |     K2      |     K3       |  v[0]
    // Reg 1             |     K4     |     K5      |     K6      |     K7       |  v[1]
    // Reg 2             |     K8     |     K9      |     K10     |     K11      |  v[2]
    // Reg 3             |     K12    |     K13     |     K14     |     K15      |  v[3]"""
    
    load_a += f"""

    static constexpr uint32_t VW = vectorSize(AFragT{{}});
    static constexpr uint32_t Dim = BLOCK_M;
    
    // To start the loading process, let's visualize in 2D coords.
    // Each thread will load {m * k // 64} elements.
    // We need to know where they start, and where the next elements are.
    auto startCoord2D = std::make_pair(threadIdx.x % Dim,         // Row
                                       (threadIdx.x / Dim) * VW); // Col
    auto stepCoord2D = std::make_pair(0u, 1u);
    
    // Flatten to 1D col_major offsets.
    auto col_major = [](auto const& coord, auto ld) {{ return coord.first + coord.second * ld; }};
    
    auto startOffset = col_major(startCoord2D, ld);
    auto kOffset = col_major(stepCoord2D, ld);
    
    // If you notice carefully, kOffset != 1.
    // This means the following is vector is loaded with {m * k // 64} non-contiguous offsets,
    // which the compiler will separate into {m * k // 64} different global_load instructions.
    auto fragA = AFragT
    {{"""
    
    for i in range(m * k // 64):
        if show_bit_ranges and i < 4:
            reg_desc = f"Reg {i//2} [{0 if i%2==0 else 16}:{15 if i%2==0 else 31}]"
        else:
            reg_desc = f"Reg {i}"
        offset = f"startOffset{' + ' + str(i) + ' * kOffset' if i > 0 else ''}"
        load_a += f"\n        input[{offset}],{' '*(25-len(offset))}// v[{i}] = {reg_desc}"
    
    load_a += f"""
    }};

    return fragA;
}}"""
    
    load_b = f"""
// Load B matrix block (row-major)
// Size: (BLOCK_K x BLOCK_N)
// ASSUMPTION:
// - We want contiguous BLOCK_N sized row neighbors in register.
// - Data is in row_major format
// This means:
// - From B we will load K rows of size BLOCK_N to satisfy our input data
__device__ BFragT load_B_{k}x{n}_row_major({cpp_ab_type} const* input, int ld)
{{
    // Register Mapping:
    // Size              |   BLOCK_N  |   BLOCK_N   |   BLOCK_N   |   BLOCK_N    |  Vector
    // Register Element  | 0  ... {n-1:2} | {n}  ... {2*n-1:2} | {2*n}  ... {3*n-1:2} | {3*n}  ... {4*n-1:2} |  Element
    //                    ____________ _____________ _____________ ______________"""
    
    if show_bit_ranges:
        load_b += f"""
    // Reg 0 [0 :15]     |     K0     |     K4      |     K8      |     K12      |  v[0]
    // Reg 0 [16:31]     |     K1     |     K5      |     K9      |     K13      |  v[1]
    // Reg 1 [0 :15]     |     K2     |     K6      |     K10     |     K14      |  v[2]
    // Reg 1 [16:31]     |     K3     |     K7      |     K11     |     K15      |  v[3]"""
    else:
        load_b += f"""
    // Reg 0             |     K0     |     K1      |     K2      |     K3       |  v[0]
    // Reg 1             |     K4     |     K5      |     K6      |     K7       |  v[1]
    // Reg 2             |     K8     |     K9      |     K10     |     K11      |  v[2]
    // Reg 3             |     K12    |     K13     |     K14     |     K15      |  v[3]"""
    
    load_b += f"""

    static constexpr uint32_t VW = vectorSize(BFragT{{}});
    static constexpr uint32_t Dim = BLOCK_N;
    
    // To start the loading process, let's visualize in 2D coords.
    // Each thread will load {n * k // 64} elements.
    // We need to know where they start, and where the next elements are.
    auto startCoord2D = std::make_pair((threadIdx.x / Dim) * VW, // Row
                                        threadIdx.x % Dim);      // Col
    auto stepCoord2D = std::make_pair(1u, 0u);
    
    // Flatten to 1D row_major offsets.
    auto row_major = [](auto const& coord, auto ld) {{ return coord.first * ld + coord.second; }};
    
    auto startOffset = row_major(startCoord2D, ld);
    auto kOffset = row_major(stepCoord2D, ld);
    
    // If you notice carefully, kOffset != 1.
    // This means the following is vector is loaded with {n * k // 64} non-contiguous offsets,
    // which the compiler will separate into {n * k // 64} different global_load instructions.
    auto fragB = BFragT
    {{"""
    
    for i in range(n * k // 64):
        if show_bit_ranges and i < 4:
            reg_desc = f"Reg {i//2} [{0 if i%2==0 else 16}:{15 if i%2==0 else 31}]"
        else:
            reg_desc = f"Reg {i}"
        offset = f"startOffset{' + ' + str(i) + ' * kOffset' if i > 0 else ''}"
        load_b += f"\n        input[{offset}],{' '*(25-len(offset))}// v[{i}] = {reg_desc}"
    
    load_b += f"""
    }};

    return fragB;
}}"""
    
    load_c = f"""
// Load C matrix block (col-major)
// Size: (BLOCK_M x BLOCK_N)
// ASSUMPTION:
// - We want contiguous BLOCK_N sized row neighbors in register.
// - Data is in col_major format
// This means:
// - From C we will load BLOCK_M rows of size BLOCK_N to satisfy our input data
__device__ CFragT load_C_{m}x{n}_col_major({cpp_accum_type} const* input, int ld)
{{
    // Register Mapping:
    // Size              |   BLOCK_N  |   BLOCK_N   |   BLOCK_N   |   BLOCK_N    | Vector
    // Register Element  | 0  ... {n-1:2} | {n}  ... {2*n-1:2} | {2*n}  ... {3*n-1:2} | {3*n}  ... {4*n-1:2} | Element
    //                    ____________ _____________ _____________ ______________
    // Reg0              |     M0     |     M4      |     M8      |     M12      | v[0]
    // Reg1              |     M1     |     M5      |     M9      |     M13      | v[1]
    // Reg2              |     M2     |     M6      |     M10     |     M14      | v[2]
    // Reg3              |     M3     |     M7      |     M11     |     M15      | v[3]

    static constexpr uint32_t VW = vectorSize(CFragT{{}});
    static constexpr uint32_t Dim = BLOCK_N;
    
    // To start the loading process, let's visualize in 2D coords.
    // Each thread will load {m * n // 64} elements.
    // We need to know where they start, and where the next elements are.
    auto startCoord2D = std::make_pair((threadIdx.x / Dim) * VW, // Row
                                        threadIdx.x % Dim);      // Col
    auto stepCoord2D = std::make_pair(1u, 0u);
    
    // Flatten to 1D col_major offsets.
    auto col_major = [](auto const& coord, auto ld) {{ return coord.first + coord.second * ld; }};
    
    auto startOffset = col_major(startCoord2D, ld);
    auto kOffset = col_major(stepCoord2D, ld);
    
    // If you notice carefully, kOffset == 1.
    // This means the following is vector load of {m * n // 64} contiguous elements.
    // When you check out the assembly, the compiler will convert the 
    // following into a single global_load_dwordx{m * n // 64} (woohoo!)
    auto fragC = *((CFragT*)(input + startOffset));

    // Reference:
    // {{"""
    
    for i in range(m * n // 64):
        offset = f"startOffset{' + ' + str(i) + ' * kOffset' if i > 0 else ''}"
        load_c += f"\n    //     input[{offset}],{' '*(25-len(offset))}// v[{i}] = Reg {i}"
    
    load_c += f"""
    // }};

    return fragC;
}}"""
    
    return load_a + '\n' + load_b + '\n' + load_c

def generate_store_function(m: int, n: int, accum_type: str) -> str:
    """Generate store function for result matrix"""
    cpp_type = get_cpp_type(accum_type)
    
    store_func = f"""
// Store C matrix block (col-major)
__device__ void store_C_{m}x{n}_col_major({cpp_type}* output, CFragT cFrag, int ld)
{{
    // Register Mapping:
    // Size              |   BLOCK_N  |   BLOCK_N   |   BLOCK_N   |   BLOCK_N    | Vector
    // Register Element  | 0  ... {n-1:2} | {n}  ... {2*n-1:2} | {2*n}  ... {3*n-1:2} | {3*n}  ... {4*n-1:2} | Element
    //                    ____________ _____________ _____________ ______________
    // Reg0              |     M0     |     M4      |     M8      |     M12      | v[0]
    // Reg1              |     M1     |     M5      |     M9      |     M13      | v[1]
    // Reg2              |     M2     |     M6      |     M10     |     M14      | v[2]
    // Reg3              |     M3     |     M7      |     M11     |     M15      | v[3]

    static constexpr uint32_t VW = vectorSize(CFragT{{}});
    static constexpr uint32_t Dim = BLOCK_N;
    
    // To start the storing process, let's visualize in 2D coords.
    // Each thread will store {m * n // 64} elements.
    // We need to know where they start, and where the next elements are.
    auto startCoord2D = std::make_pair((threadIdx.x / Dim) * VW, // Row
                                        threadIdx.x % Dim);      // Col
    auto stepCoord2D = std::make_pair(1u, 0u);
    
    // Flatten to 1D col_major offsets.
    auto col_major = [](auto const& coord, auto ld) {{ return coord.first + coord.second * ld; }};
    
    auto startOffset = col_major(startCoord2D, ld);
    auto kOffset = col_major(stepCoord2D, ld);
    
    // If you notice carefully, kOffset == 1.
    // This means the following is vector store of {m * n // 64} contiguous elements.
    // When you check out the assembly, the compiler will convert the 
    // following into a single global_store_dwordx{m * n // 64} (woohoo!)
    *((CFragT*)(output + startOffset)) = cFrag;

    // Reference:"""
    
    for i in range(m * n // 64):
        offset = f"startOffset{' + ' + str(i) + ' * kOffset' if i > 0 else ''}"
        store_func += f"\n    // output[{offset}] = cFrag[{i}];{' '*(25-len(offset)-len(str(i))-12)}// v[{i}] = Reg {i}"
    
    store_func += "\n}"
    
    return store_func

if __name__ == "__main__":
    # Example usage
    code = generate_mfma_cpp(
        arch="cdna1",
        instruction="v_mfma_f32_16x16x16f16",
        cbsz=0,
        abid=0,
        blgp=0,
        opsel=0,
        wavefront=64
    )
    
    # Save to file
    with open("generated_mfma_16x16x16_f16.cpp", "w") as f:
        f.write(code)
    
    print("Generated MFMA kernel code saved to generated_mfma_16x16x16_f16.cpp")

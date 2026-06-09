//
// PHASE_LAUNCH_FUSION_SWEEP #195 — bit-identical read-once GEMV family for the
// small-batch (ne11 <= K) Q4_0 decode matmul. Replaces the force-dispatched
// mmq_x=8 GEMM at small ne11 with a memory-bound read-once kernel whose float
// output is byte-identical to production MMQ (mul_mat_q_split_k<Q4_0,...,4>).
//
// Spec: specs/dispatch/m1_quant_matmul.allium + M1QuantMatmul.tla.
// Design derivation: docs/active/PHASE_LAUNCH_FUSION_SWEEP.md
//   "Kernel design — full reduction-order derivation".
//
// SPDX-License-Identifier: MIT
//
#pragma once

#include "mmq.cuh"  // mmq_args, block_q8_1_mmq, MMQ_ITER_K

// Run the faithful Q4_0 GEMV. Consumes the SAME inputs as mul_mat_q_case<Q4_0>:
//   args.x        — block_q4_0 weights, row stride args.stride01 bytes
//   args.y        — block_q8_1_mmq activations (quantize_mmq_q8_1_cuda<DS4>)
//   args.dst      — fp32 output, dst[j*args.ne0 + i]
//   args.ne00=K, args.ne01=rows, args.ne11=M cols, args.stride11=col stride
void ggml_cuda_mul_mat_q4_0_gemv_faithful(const mmq_args & args, cudaStream_t stream);

// Runtime toggle (default from env GGML_CUDA_M1_GEMV) + carve-out threshold K
// (default from env GGML_CUDA_M1_GEMV_K, else 8). The toggle lets the
// byte-identity test A/B MMQ vs the GEMV through the identical dispatcher in
// one process; production caches the env read once.
extern "C" void ggml_cuda_m1_gemv_set_enabled(int enabled);
extern "C" int  ggml_cuda_m1_gemv_enabled(void);
extern "C" void ggml_cuda_m1_gemv_set_threshold(int k);
extern "C" int  ggml_cuda_m1_gemv_threshold(void);

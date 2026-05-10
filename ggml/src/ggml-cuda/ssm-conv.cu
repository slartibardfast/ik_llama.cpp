#include "ssm-conv.cuh"

#define CUDA_SSM_CONV_BLOCK_SIZE 256

template <int split_n_t>
static __global__ void ssm_conv_single_seq_f32(
        const float * src0,
        const float * src1,
        const float * src2,
        float * dst_x,
        int nc,
        int nr,
        int n_t,
        int src0_s0,
        int src0_s1,
        int src1_s1) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nr) {
        return;
    }

    const int t0 = blockIdx.y * split_n_t;
    if (t0 >= n_t) {
        return;
    }

    const float * state_row = src0 + (size_t) row * src0_s1;
    const float * c_row = src2 + (size_t) row * nc;

#pragma unroll
    for (int it = 0; it < split_n_t; ++it) {
        const int t = t0 + it;
        if (t >= n_t) {
            break;
        }

        float sumf = 0.0f;
        for (int j = 0; j < nc; ++j) {
            const int idx = t + j;
            const float x = idx < nc - 1
                ? state_row[(size_t) idx * src0_s0]
                : src1[row + (size_t) (idx - (nc - 1)) * src1_s1];

            sumf += x * c_row[j];
        }

        dst_x[row + (size_t) t * nr] = sumf;
    }
}

template <int split_n_t>
static __global__ void ssm_conv_single_seq_f32_nc4(
        const float * src0,
        const float * src1,
        const float * src2,
        float * dst_x,
        int nr,
        int n_t,
        int src0_s0,
        int src0_s1,
        int src1_s1) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nr) {
        return;
    }

    const int t0 = blockIdx.y * split_n_t;
    if (t0 >= n_t) {
        return;
    }

    const float * state_row = src0 + (size_t) row * src0_s1;
    const float * c_row = src2 + (size_t) row * 4;
    const float c0 = c_row[0];
    const float c1 = c_row[1];
    const float c2 = c_row[2];
    const float c3 = c_row[3];

#pragma unroll
    for (int it = 0; it < split_n_t; ++it) {
        const int t = t0 + it;
        if (t >= n_t) {
            break;
        }

        const int i0 = t;
        const int i1 = t + 1;
        const int i2 = t + 2;
        const int i3 = t + 3;

        const float x0 = i0 < 3 ? state_row[(size_t) i0 * src0_s0] : src1[row + (size_t) (i0 - 3) * src1_s1];
        const float x1 = i1 < 3 ? state_row[(size_t) i1 * src0_s0] : src1[row + (size_t) (i1 - 3) * src1_s1];
        const float x2 = i2 < 3 ? state_row[(size_t) i2 * src0_s0] : src1[row + (size_t) (i2 - 3) * src1_s1];
        const float x3 = i3 < 3 ? state_row[(size_t) i3 * src0_s0] : src1[row + (size_t) (i3 - 3) * src1_s1];

        dst_x[row + (size_t) t * nr] = x0 * c0 + x1 * c1 + x2 * c2 + x3 * c3;
    }
}

static __global__ void ssm_conv_single_seq_final_state_f32(
        const float * src0,
        const float * src1,
        float * dst_state,
        int nc,
        int nr,
        int n_t,
        int src0_s0,
        int src0_s1,
        int src1_s1) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nr) {
        return;
    }

    const float * state_row = src0 + (size_t) row * src0_s1;
    float * dst_row = dst_state + (size_t) row * nc;

    for (int j = 0; j < nc; ++j) {
        const int idx = n_t - 1 + j;
        dst_row[j] = idx < nc - 1
            ? state_row[(size_t) idx * src0_s0]
            : src1[row + (size_t) (idx - (nc - 1)) * src1_s1];
    }
}

static __global__ void ssm_conv_init_states_f32_nc4(
        const float * src0,
        float * state,
        int nr,
        int n_kv) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int seq = blockIdx.y;

    if (row >= nr || seq >= n_kv) {
        return;
    }

    const float * src_row = src0 + (size_t) seq * nr * 3 + (size_t) row * 3;
    float * state_row = state + (size_t) seq * nr * 4 + (size_t) row * 4;

    state_row[1] = src_row[0];
    state_row[2] = src_row[1];
    state_row[3] = src_row[2];
}

static __global__ void ssm_conv_init_states_f32(
        const float * src0,
        float * state,
        int nc,
        int nr,
        int n_kv) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int seq = blockIdx.y;

    if (row >= nr || seq >= n_kv) {
        return;
    }

    const float * src_row = src0 + (size_t) seq * nr * (nc - 1) + (size_t) row * (nc - 1);
    float * state_row = state + (size_t) seq * nr * nc + (size_t) row * nc;

    for (int i0 = 0; i0 < nc - 1; ++i0) {
        state_row[1 + i0] = src_row[i0];
    }
}

// Phase B: runtime single-seq detection.
// Scans src3 once; sets single_ok[0]=1 and single_slot[0]=S iff every
// token's primary seq_id == S and all entries are valid. Otherwise
// single_ok[0]=0. Caller must zero-init single_ok before launch.
static __global__ void ssm_conv_detect_runtime_single_seq(
        const int32_t * src3,
        int32_t * single_ok,
        int32_t * single_slot,
        int n_t,
        int n_kv,
        int src3_nb1) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_t) {
        return;
    }
    const int32_t * sq = src3 + (size_t) t * src3_nb1;
    const int32_t s0 = sq[0];
    if (s0 < 0 || s0 >= n_kv) {
        atomicExch(single_ok, 0);
        return;
    }
    if (n_kv > 1) {
        const int32_t s1 = sq[1];
        if (s1 >= 0 && s1 < n_kv) {
            // multiple seqs targeted by this token => not single-seq
            atomicExch(single_ok, 0);
            return;
        }
    }
    if (t == 0) {
        // Publish slot from token 0; later tokens compare against it.
        atomicExch(single_slot, s0);
    } else {
        // Wait until token 0 has published; spin via volatile read.
        // In practice CUDA serialises in-block; this works because
        // single_slot is written by t==0 then read by others.
        // We accept a small race where t!=0 may read -1 before t=0
        // wrote — handle by invalidating single_ok if mismatch.
        const int32_t pub = single_slot[0];
        if (pub >= 0 && pub != s0) {
            atomicExch(single_ok, 0);
        }
    }
}

// Phase B: runtime single-seq fast path with slot offset.
// Mirrors ssm_conv_single_seq_f32 but reads slot S from device memory
// and applies offsets to src0 / dst_state. Guarded by single_ok flag —
// returns early if the detection kernel rejected the batch.
template <int split_n_t>
static __global__ void ssm_conv_runtime_single_seq_f32(
        const float * src0,
        const float * src1,
        const float * src2,
        const int32_t * single_ok,
        const int32_t * single_slot,
        float * dst_x,
        int nc,
        int nr,
        int n_t,
        int src0_s0,
        int src0_s1,
        int src0_s2,
        int src1_s1) {
    if (single_ok[0] == 0) return;
    const int slot = single_slot[0];
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nr) return;
    const int t0 = blockIdx.y * split_n_t;
    if (t0 >= n_t) return;

    const float * state_row = src0 + (size_t) slot * src0_s2 + (size_t) row * src0_s1;
    const float * c_row = src2 + (size_t) row * nc;

#pragma unroll
    for (int it = 0; it < split_n_t; ++it) {
        const int t = t0 + it;
        if (t >= n_t) break;
        float sumf = 0.0f;
        for (int j = 0; j < nc; ++j) {
            const int idx = t + j;
            const float x = idx < nc - 1
                ? state_row[(size_t) idx * src0_s0]
                : src1[row + (size_t) (idx - (nc - 1)) * src1_s1];
            sumf += x * c_row[j];
        }
        dst_x[row + (size_t) t * nr] = sumf;
    }
}

template <int split_n_t>
static __global__ void ssm_conv_runtime_single_seq_f32_nc4(
        const float * src0,
        const float * src1,
        const float * src2,
        const int32_t * single_ok,
        const int32_t * single_slot,
        float * dst_x,
        int nr,
        int n_t,
        int src0_s0,
        int src0_s1,
        int src0_s2,
        int src1_s1) {
    if (single_ok[0] == 0) return;
    const int slot = single_slot[0];
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nr) return;
    const int t0 = blockIdx.y * split_n_t;
    if (t0 >= n_t) return;

    const float * state_row = src0 + (size_t) slot * src0_s2 + (size_t) row * src0_s1;
    const float * c_row = src2 + (size_t) row * 4;
    const float c0 = c_row[0], c1 = c_row[1], c2 = c_row[2], c3 = c_row[3];

#pragma unroll
    for (int it = 0; it < split_n_t; ++it) {
        const int t = t0 + it;
        if (t >= n_t) break;
        const int i0 = t, i1 = t + 1, i2 = t + 2, i3 = t + 3;
        const float x0 = i0 < 3 ? state_row[(size_t) i0 * src0_s0] : src1[row + (size_t) (i0 - 3) * src1_s1];
        const float x1 = i1 < 3 ? state_row[(size_t) i1 * src0_s0] : src1[row + (size_t) (i1 - 3) * src1_s1];
        const float x2 = i2 < 3 ? state_row[(size_t) i2 * src0_s0] : src1[row + (size_t) (i2 - 3) * src1_s1];
        const float x3 = i3 < 3 ? state_row[(size_t) i3 * src0_s0] : src1[row + (size_t) (i3 - 3) * src1_s1];
        dst_x[row + (size_t) t * nr] = x0 * c0 + x1 * c1 + x2 * c2 + x3 * c3;
    }
}

static __global__ void ssm_conv_runtime_single_seq_final_state_f32(
        const float * src0,
        const float * src1,
        const int32_t * single_ok,
        const int32_t * single_slot,
        float * dst_state,
        int nc,
        int nr,
        int n_t,
        int n_kv,
        int src0_s0,
        int src0_s1,
        int src0_s2,
        int src1_s1) {
    if (single_ok[0] == 0) return;
    const int slot = single_slot[0];
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nr) return;

    const float * state_row = src0 + (size_t) slot * src0_s2 + (size_t) row * src0_s1;
    float * dst_row = dst_state + (size_t) slot * nr * nc + (size_t) row * nc;
    for (int j = 0; j < nc; ++j) {
        const int idx = n_t - 1 + j;
        dst_row[j] = idx < nc - 1
            ? state_row[(size_t) idx * src0_s0]
            : src1[row + (size_t) (idx - (nc - 1)) * src1_s1];
    }
    (void) n_kv;
}

static __global__ void ssm_conv_validate_unique_seq_map(
        const int32_t * src3,
        int32_t * seq_ids,
        int32_t * seq_seen,
        int32_t * fast_path_ok,
        int n_t,
        int n_kv,
        int src3_nb1) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_t) {
        return;
    }

    const int32_t * sq = src3 + (size_t) t * src3_nb1;
    const int32_t seq0 = sq[0];
    if (seq0 < 0 || seq0 >= n_kv) {
        atomicExch(fast_path_ok, 0);
        return;
    }

    // Fast path supports one sequence per token (no copy-to-multiple-sequences routing).
    if (n_kv > 1) {
        const int32_t seq1 = sq[1];
        if (seq1 >= 0 && seq1 < n_kv) {
            atomicExch(fast_path_ok, 0);
            return;
        }
    }

    seq_ids[t] = seq0;
    if (atomicAdd(seq_seen + seq0, 1) != 0) {
        // Sequence is updated by multiple tokens in the same batch => recurrent dependency across t.
        atomicExch(fast_path_ok, 0);
    }
}

static __global__ void ssm_conv_multi_seq_unique_f32_kernel(
        const float * src0,
        const float * src1,
        const float * src2,
        const int32_t * seq_ids,
        const int32_t * fast_path_ok,
        float * dst_x,
        float * dst_state,
        int nc,
        int nr,
        int n_t,
        int src1_nb1) {
    if (fast_path_ok != nullptr && fast_path_ok[0] == 0) {
        return;
    }

    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int t   = blockIdx.y;

    if (row >= nr || t >= n_t) {
        return;
    }

    const int seq = seq_ids[t];
    const float * src_state_row = src0 + (size_t) seq * nr * (nc - 1) + (size_t) row * (nc - 1);
    float * state_row = dst_state + (size_t) seq * nr * nc + (size_t) row * nc;
    const float * c_row = src2 + (size_t) row * nc;

    float sumf = 0.0f;
    for (int i0 = 0; i0 < nc - 1; ++i0) {
        const float v = src_state_row[i0];
        state_row[i0] = v;
        sumf += v * c_row[i0];
    }

    const float x = src1[row + (size_t) t * src1_nb1];
    state_row[nc - 1] = x;
    sumf += x * c_row[nc - 1];
    dst_x[row + (size_t) t * nr] = sumf;
}

static __global__ void ssm_conv_multi_seq_unique_f32_kernel_nc4(
        const float * src0,
        const float * src1,
        const float * src2,
        const int32_t * seq_ids,
        const int32_t * fast_path_ok,
        float * dst_x,
        float * dst_state,
        int nr,
        int n_t,
        int src1_nb1) {
    if (fast_path_ok != nullptr && fast_path_ok[0] == 0) {
        return;
    }

    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int t   = blockIdx.y;

    if (row >= nr || t >= n_t) {
        return;
    }

    const int seq = seq_ids[t];
    const float * src_state_row = src0 + (size_t) seq * nr * 3 + (size_t) row * 3;
    float * state_row = dst_state + (size_t) seq * nr * 4 + (size_t) row * 4;
    const float * c_row = src2 + (size_t) row * 4;

    const float s0 = src_state_row[0];
    const float s1 = src_state_row[1];
    const float s2 = src_state_row[2];
    const float x  = src1[row + (size_t) t * src1_nb1];

    state_row[0] = s0;
    state_row[1] = s1;
    state_row[2] = s2;
    state_row[3] = x;

    dst_x[row + (size_t) t * nr] = s0 * c_row[0] + s1 * c_row[1] + s2 * c_row[2] + x * c_row[3];
}

static __global__ void ssm_conv_f32_kernel(
        const float * src0,
        const float * src1,
        const float * src2,
        const int32_t * src3,
        const int32_t * fast_path_ok,
        const int32_t * single_ok,
        float * dst_x,
        float * dst_state,
        int nc,
        int nr,
        int n_t,
        int n_kv,
        int src1_nb1,
        int src3_nb1) {
    if (fast_path_ok != nullptr && fast_path_ok[0] != 0) {
        return;
    }
    // Phase B: skip if the runtime-single-seq path took over.
    if (single_ok != nullptr && single_ok[0] != 0) {
        return;
    }

    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nr) {
        return;
    }

    const float * c_row = src2 + (size_t) row * nc;

    for (int t = 0; t < n_t; ++t) {
        const int32_t * sq = src3 + (size_t) t * src3_nb1;
        const int seq0 = sq[0];

        if (seq0 < 0 || seq0 >= n_kv) {
            continue;
        }

        float * state_row = dst_state + (size_t) seq0 * nr * nc + (size_t) row * nc;
        const float * src_state_row;
        if (t == 0) {
            src_state_row = src0 + (size_t) seq0 * nr * (nc - 1) + (size_t) row * (nc - 1);
        } else {
            src_state_row = state_row + 1;
        }

        for (int i0 = 0; i0 < nc - 1; ++i0) {
            state_row[i0] = src_state_row[i0];
        }
        state_row[nc - 1] = src1[row + (size_t) t * src1_nb1];

        for (int i3 = 1; i3 < n_kv; ++i3) {
            const int seq = sq[i3];
            if (seq < 0 || seq >= n_kv) {
                break;
            }

            float * state_row_copy = dst_state + (size_t) seq * nr * nc + (size_t) row * nc;
            for (int i0 = 0; i0 < nc; ++i0) {
                state_row_copy[i0] = state_row[i0];
            }
        }

        float sumf = 0.0f;
        for (int i0 = 0; i0 < nc; ++i0) {
            sumf += state_row[i0] * c_row[i0];
        }
        dst_x[row + (size_t) t * nr] = sumf;
    }
}

template <bool has_multi_seq>
static __global__ void ssm_conv_f32_kernel_nc4(
        const float * src0,
        const float * src1,
        const float * src2,
        const int32_t * src3,
        const int32_t * fast_path_ok,
        const int32_t * single_ok,
        float * dst_x,
        float * dst_state,
        int nr,
        int n_t,
        int n_kv,
        int src1_nb1,
        int src3_nb1) {
    if (fast_path_ok != nullptr && fast_path_ok[0] != 0) {
        return;
    }
    if (single_ok != nullptr && single_ok[0] != 0) {
        return;
    }

    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nr) {
        return;
    }

    const float * c_row = src2 + (size_t) row * 4;
    const float c0 = c_row[0];
    const float c1 = c_row[1];
    const float c2 = c_row[2];
    const float c3 = c_row[3];

    for (int t = 0; t < n_t; ++t) {
        const int32_t * sq = src3 + (size_t) t * src3_nb1;
        const int seq0 = sq[0];

        if (seq0 < 0 || seq0 >= n_kv) {
            continue;
        }

        float * state_row = dst_state + (size_t) seq0 * nr * 4 + (size_t) row * 4;

        const float * src_state_row;
        if (t == 0) {
            src_state_row = src0 + (size_t) seq0 * nr * 3 + (size_t) row * 3;
        } else {
            src_state_row = state_row + 1;
        }

        const float s0 = src_state_row[0];
        const float s1 = src_state_row[1];
        const float s2 = src_state_row[2];
        const float x = src1[row + (size_t) t * src1_nb1];

        state_row[0] = s0;
        state_row[1] = s1;
        state_row[2] = s2;
        state_row[3] = x;

        if constexpr (has_multi_seq) {
            for (int i3 = 1; i3 < n_kv; ++i3) {
                const int seq = sq[i3];
                if (seq < 0 || seq >= n_kv) {
                    break;
                }

                float * state_row_copy = dst_state + (size_t) seq * nr * 4 + (size_t) row * 4;
                state_row_copy[0] = s0;
                state_row_copy[1] = s1;
                state_row_copy[2] = s2;
                state_row_copy[3] = x;
            }
        }

        dst_x[row + (size_t) t * nr] = s0 * c0 + s1 * c1 + s2 * c2 + x * c3;
    }
}

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0]; // conv_state: [d_conv - 1, d_inner, n_kv]
    const ggml_tensor * src1 = dst->src[1]; // x: [d_inner, n_tokens]
    const ggml_tensor * src2 = dst->src[2]; // conv1d.weight: [d_conv, d_inner]
    const ggml_tensor * src3 = dst->src[3]; // state_seq: [n_kv, n_tokens]

    const int nc   = src2->ne[0];
    const int nr   = src0->ne[1];
    const int n_t  = src1->ne[1];
    const int n_kv = src0->ne[2];

    GGML_ASSERT((int64_t) nr * n_t + (int64_t) nc * nr * n_kv == ggml_nelements(dst));

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(src2->type == GGML_TYPE_F32);
    GGML_ASSERT(src3->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(src1->nb[0] == sizeof(float));
    GGML_ASSERT(src2->nb[0] == sizeof(float));
    GGML_ASSERT(src3->nb[0] == sizeof(int32_t));
    GGML_ASSERT(src0->nb[1] == src0->ne[0] * sizeof(float));
    GGML_ASSERT(src2->nb[1] == src2->ne[0] * sizeof(float));
    GGML_ASSERT(src2->nb[2] == src2->ne[1] * src2->ne[0] * sizeof(float));

    GGML_ASSERT(src2->ne[0] == src0->ne[0] + 1);
    GGML_ASSERT(src2->ne[1] == src0->ne[1]);
    GGML_ASSERT(src1->ne[0] == src0->ne[1]);
    GGML_ASSERT(src3->ne[0] == src0->ne[2]);
    GGML_ASSERT(src3->ne[1] == src1->ne[1]);

    float * dst_data = (float *) dst->data;
    float * dst_x = dst_data;
    float * dst_state = dst_data + (size_t) nr * n_t;

    const dim3 block_dims(CUDA_SSM_CONV_BLOCK_SIZE, 1, 1);
    const dim3 row_grid((nr + CUDA_SSM_CONV_BLOCK_SIZE - 1) / CUDA_SSM_CONV_BLOCK_SIZE, 1, 1);
    ggml_cuda_pool_alloc<int32_t> fast_path_ok_d(ctx.pool());
    ggml_cuda_pool_alloc<int32_t> single_ok_d(ctx.pool());
    ggml_cuda_pool_alloc<int32_t> single_slot_d(ctx.pool());
    const int32_t * multi_seq_fast_path_ok = nullptr;
    const int32_t * single_seq_ok          = nullptr;

    // Fast path for single-sequence recurrent updates (Qwen3Next prompt/decode path).
    // In this case, outputs are independent given the initial conv state, so we parallelize over token blocks.
    if (n_kv == 1 && src3->ne[0] == 1) {
        GGML_ASSERT(n_t > 0);

        const int src0_s0 = src0->nb[0] / sizeof(float);
        const int src0_s1 = src0->nb[1] / sizeof(float);
        const int src1_s1 = src1->nb[1] / sizeof(float);

        constexpr int split_n_t = 32;
        const dim3 token_grid(row_grid.x, (n_t + split_n_t - 1) / split_n_t, 1);

        if (nc == 4) {
            ssm_conv_single_seq_f32_nc4<split_n_t><<<token_grid, block_dims, 0, ctx.stream()>>>(
                (const float *) src0->data,
                (const float *) src1->data,
                (const float *) src2->data,
                dst_x,
                nr, n_t,
                src0_s0, src0_s1, src1_s1);
        } else {
            ssm_conv_single_seq_f32<split_n_t><<<token_grid, block_dims, 0, ctx.stream()>>>(
                (const float *) src0->data,
                (const float *) src1->data,
                (const float *) src2->data,
                dst_x,
                nc, nr, n_t,
                src0_s0, src0_s1, src1_s1);
        }

        ssm_conv_single_seq_final_state_f32<<<row_grid, block_dims, 0, ctx.stream()>>>(
            (const float *) src0->data,
            (const float *) src1->data,
            dst_state,
            nc, nr, n_t,
            src0_s0, src0_s1, src1_s1);
        return;
    }

    if (n_kv > 1) {
        const dim3 init_grid(row_grid.x, n_kv, 1);
        if (nc == 4) {
            ssm_conv_init_states_f32_nc4<<<init_grid, block_dims, 0, ctx.stream()>>>(
                (const float *) src0->data,
                dst_state,
                nr, n_kv);
        } else {
            ssm_conv_init_states_f32<<<init_grid, block_dims, 0, ctx.stream()>>>(
                (const float *) src0->data,
                dst_state,
                nc, nr, n_kv);
        }

        // PHASE46 C.1 L.8.f fix: the three function-scope RAII guards
        // (fast_path_ok_d, single_ok_d, single_slot_d declared in that order
        // at lines 631-633) destruct at function end in REVERSE declaration
        // order: single_slot_d → single_ok_d → fast_path_ok_d. The CUDA pool
        // is strict LIFO (ggml-cuda.cu:563). For destruction to be valid,
        // allocation must happen in DECLARATION order so that the reverse-
        // declaration destruction pops the top of the pool stack correctly.
        // Allocate fast_path_ok_d first, single_ok_d second, single_slot_d
        // third (matching declaration order).
        int32_t fast_path_ok_init = 1;
        fast_path_ok_d.alloc(1);
        CUDA_CHECK(cudaMemcpyAsync(fast_path_ok_d.get(), &fast_path_ok_init, sizeof(int32_t), cudaMemcpyHostToDevice, ctx.stream()));

        // Phase B: runtime-single-seq detection.
        // Initialise single_ok=1 / single_slot=-1 then scan src3.
        single_ok_d.alloc(1);
        single_slot_d.alloc(1);
        int32_t single_ok_init = 1;
        int32_t single_slot_init = -1;
        CUDA_CHECK(cudaMemcpyAsync(single_ok_d.get(),   &single_ok_init,   sizeof(int32_t), cudaMemcpyHostToDevice, ctx.stream()));
        CUDA_CHECK(cudaMemcpyAsync(single_slot_d.get(), &single_slot_init, sizeof(int32_t), cudaMemcpyHostToDevice, ctx.stream()));
        constexpr int detect_block_size = 256;
        const dim3 detect_grid((n_t + detect_block_size - 1) / detect_block_size, 1, 1);
        ssm_conv_detect_runtime_single_seq<<<detect_grid, detect_block_size, 0, ctx.stream()>>>(
            (const int32_t *) src3->data,
            single_ok_d.get(),
            single_slot_d.get(),
            n_t,
            n_kv,
            src3->nb[1] / sizeof(int32_t));
        CUDA_CHECK(cudaGetLastError());
        single_seq_ok = single_ok_d.get();

        // Fast path for multi-sequence decode-like batches:
        // one token per unique sequence, no copy-to-multiple-sequences routing.
        // (fast_path_ok_d was allocated above to match RAII destruction order.)
        ggml_cuda_pool_alloc<int32_t> seq_ids(ctx.pool(), n_t);
        ggml_cuda_pool_alloc<int32_t> seq_seen(ctx.pool(), n_kv);

        CUDA_CHECK(cudaMemsetAsync(seq_seen.get(), 0, n_kv * sizeof(int32_t), ctx.stream()));

        constexpr int seq_map_block_size = 256;
        const dim3 seq_map_grid((n_t + seq_map_block_size - 1) / seq_map_block_size, 1, 1);
        ssm_conv_validate_unique_seq_map<<<seq_map_grid, seq_map_block_size, 0, ctx.stream()>>>(
            (const int32_t *) src3->data,
            seq_ids.get(),
            seq_seen.get(),
            fast_path_ok_d.get(),
            n_t,
            n_kv,
            src3->nb[1] / sizeof(int32_t));
        CUDA_CHECK(cudaGetLastError());
        multi_seq_fast_path_ok = fast_path_ok_d.get();

        const dim3 token_grid(row_grid.x, n_t, 1);
        if (nc == 4) {
            ssm_conv_multi_seq_unique_f32_kernel_nc4<<<token_grid, block_dims, 0, ctx.stream()>>>(
                (const float *) src0->data,
                (const float *) src1->data,
                (const float *) src2->data,
                seq_ids.get(),
                multi_seq_fast_path_ok,
                dst_x,
                dst_state,
                nr, n_t,
                src1->nb[1] / sizeof(float));
        } else {
            ssm_conv_multi_seq_unique_f32_kernel<<<token_grid, block_dims, 0, ctx.stream()>>>(
                (const float *) src0->data,
                (const float *) src1->data,
                (const float *) src2->data,
                seq_ids.get(),
                multi_seq_fast_path_ok,
                dst_x,
                dst_state,
                nc, nr, n_t,
                src1->nb[1] / sizeof(float));
        }

        // Phase B: runtime single-seq fast path. Guarded by single_ok;
        // returns early if detection rejected. Uses the same arithmetic
        // as the existing single_seq_f32 kernels but reads slot index
        // from device memory and offsets src0/dst_state accordingly.
        {
            const int src0_s0 = src0->nb[0] / sizeof(float);
            const int src0_s1 = src0->nb[1] / sizeof(float);
            const int src0_s2 = src0->nb[2] / sizeof(float);
            const int src1_s1 = src1->nb[1] / sizeof(float);
            constexpr int split_n_t = 32;
            const dim3 split_token_grid(row_grid.x, (n_t + split_n_t - 1) / split_n_t, 1);
            if (nc == 4) {
                ssm_conv_runtime_single_seq_f32_nc4<split_n_t><<<split_token_grid, block_dims, 0, ctx.stream()>>>(
                    (const float *) src0->data,
                    (const float *) src1->data,
                    (const float *) src2->data,
                    single_ok_d.get(),
                    single_slot_d.get(),
                    dst_x,
                    nr, n_t,
                    src0_s0, src0_s1, src0_s2, src1_s1);
            } else {
                ssm_conv_runtime_single_seq_f32<split_n_t><<<split_token_grid, block_dims, 0, ctx.stream()>>>(
                    (const float *) src0->data,
                    (const float *) src1->data,
                    (const float *) src2->data,
                    single_ok_d.get(),
                    single_slot_d.get(),
                    dst_x,
                    nc, nr, n_t,
                    src0_s0, src0_s1, src0_s2, src1_s1);
            }
            ssm_conv_runtime_single_seq_final_state_f32<<<row_grid, block_dims, 0, ctx.stream()>>>(
                (const float *) src0->data,
                (const float *) src1->data,
                single_ok_d.get(),
                single_slot_d.get(),
                dst_state,
                nc, nr, n_t, n_kv,
                src0_s0, src0_s1, src0_s2, src1_s1);
        }
    }

    if (nc == 4) {
        if (n_kv > 1) {
            ssm_conv_f32_kernel_nc4<true><<<row_grid, block_dims, 0, ctx.stream()>>>(
                (const float *) src0->data,
                (const float *) src1->data,
                (const float *) src2->data,
                (const int32_t *) src3->data,
                multi_seq_fast_path_ok,
                single_seq_ok,
                dst_x,
                dst_state,
                nr, n_t, n_kv,
                src1->nb[1] / sizeof(float),
                src3->nb[1] / sizeof(int32_t));
        } else {
            ssm_conv_f32_kernel_nc4<false><<<row_grid, block_dims, 0, ctx.stream()>>>(
                (const float *) src0->data,
                (const float *) src1->data,
                (const float *) src2->data,
                (const int32_t *) src3->data,
                nullptr,
                nullptr,
                dst_x,
                dst_state,
                nr, n_t, n_kv,
                src1->nb[1] / sizeof(float),
                src3->nb[1] / sizeof(int32_t));
        }
    } else {
        ssm_conv_f32_kernel<<<row_grid, block_dims, 0, ctx.stream()>>>(
            (const float *) src0->data,
            (const float *) src1->data,
            (const float *) src2->data,
            (const int32_t *) src3->data,
            multi_seq_fast_path_ok,
            single_seq_ok,
            dst_x,
            dst_state,
            nc, nr, n_t, n_kv,
            src1->nb[1] / sizeof(float),
            src3->nb[1] / sizeof(int32_t));
    }
}

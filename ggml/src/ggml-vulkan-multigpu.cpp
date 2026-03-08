// Vulkan multi-GPU split buffer type implementation
// Uses only public ggml-backend APIs — no Vulkan headers needed.

#include "ggml-vulkan.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include <cstring>
#include <map>
#include <vector>

#define VK_MATRIX_ROW_PADDING 512

// split buffer type

struct ggml_backend_vk_split_buffer_type_context {
    // placeholder for future tensor_split ratios
};

struct ggml_backend_vk_split_buffer_context {
    ~ggml_backend_vk_split_buffer_context() {
        // per-device buffers are freed by their own buffer objects
        // which are attached to split->buffer in init_tensor
    }
};

GGML_CALL static const char * ggml_backend_vk_split_buffer_get_name(ggml_backend_buffer_t buffer) {
    return "VK_Split";

    GGML_UNUSED(buffer);
}

bool ggml_backend_buft_is_vk_split(ggml_backend_buffer_type_t buft);

static bool ggml_backend_buffer_is_vk_split(ggml_backend_buffer_t buffer) {
    return buffer->iface.get_name == ggml_backend_vk_split_buffer_get_name;
    GGML_UNUSED(ggml_backend_buffer_is_vk_split);
}

GGML_CALL static void ggml_backend_vk_split_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_vk_split_buffer_context * ctx = (ggml_backend_vk_split_buffer_context *)buffer->context;
    delete ctx;
}

GGML_CALL static void * ggml_backend_vk_split_buffer_get_base(ggml_backend_buffer_t buffer) {
    // the pointers are stored in the tensor extras, this is just a dummy address and never dereferenced
    return (void *)0x1000;

    GGML_UNUSED(buffer);
}

GGML_CALL static void ggml_backend_vk_split_buffer_init_tensor([[maybe_unused]] ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    if (!tensor->extra) return;
    auto extra = (ggml_split_tensor_t *)tensor->extra;
    GGML_ASSERT(extra->n_device <= ggml_backend_vk_get_device_count());
    for (int i = 0; i < extra->n_device; ++i) {
        if (!extra->splits[i]) continue;
        auto split = extra->splits[i];
        auto ne0 = split->ne[0];
        auto size = ggml_nbytes(split);
        auto padded_size = size;
        if (ne0 % VK_MATRIX_ROW_PADDING != 0) {
            int nblock = (ne0 + VK_MATRIX_ROW_PADDING - 1) / VK_MATRIX_ROW_PADDING;
            auto padded_row_size = ggml_row_size(split->type, nblock * VK_MATRIX_ROW_PADDING);
            auto row_size = ggml_row_size(split->type, ne0);
            padded_size += padded_row_size - row_size;
        }
        auto buft = ggml_backend_vk_buffer_type(i);
        auto buf = ggml_backend_buft_alloc_buffer(buft, padded_size);
        if (!buf) {
            fprintf(stderr, "%s: failed to allocate %zu bytes on Vulkan device %d\n", __func__, padded_size, i);
            GGML_ABORT("fatal error");
        }
        split->data = ggml_backend_buffer_get_base(buf);
        split->buffer = buf;
        ggml_backend_buffer_set_usage(split->buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }
}

GGML_CALL static void ggml_backend_vk_split_buffer_set_tensor([[maybe_unused]] ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    if (!tensor->extra) return;
    static std::map<ggml_type, int> k_map = {
        { GGML_TYPE_Q4_0_R8   , 8},
        { GGML_TYPE_Q5_0_R4   , 4},
        { GGML_TYPE_Q8_0_R8   , 8},
        { GGML_TYPE_Q2_K_R4   , 4},
        { GGML_TYPE_Q3_K_R4   , 4},
        { GGML_TYPE_Q4_K_R4   , 4},
        { GGML_TYPE_Q5_K_R4   , 4},
        { GGML_TYPE_Q6_K_R4   , 4},
        { GGML_TYPE_IQ2_XXS_R4, 4},
        { GGML_TYPE_IQ2_XS_R4 , 4},
        { GGML_TYPE_IQ3_XXS_R4, 4},
        { GGML_TYPE_IQ1_S_R4  , 4},
        { GGML_TYPE_IQ4_NL_R4 , 4},
        { GGML_TYPE_IQ3_S_R4  , 4},
        { GGML_TYPE_IQ2_S_R4  , 4},
        { GGML_TYPE_IQ4_XS_R8 , 8},
        { GGML_TYPE_IQ1_M_R4  , 4},
        { GGML_TYPE_BF16_R16  , 16},
        { GGML_TYPE_Q6_0_R4   , 4},
        { GGML_TYPE_IQ2_BN_R4 , 4},
        { GGML_TYPE_IQ2_K_R4  , 4},
        { GGML_TYPE_IQ3_K_R4  , 4},
        { GGML_TYPE_IQ4_K_R4  , 4},
        { GGML_TYPE_IQ5_K_R4  , 4},
        { GGML_TYPE_IQ4_KS_R4 , 4},
        { GGML_TYPE_IQ5_KS_R4 , 4},
        { GGML_TYPE_Q8_K_R16  , 4},
        { GGML_TYPE_Q8_KV_R8  , 4},
        { GGML_TYPE_Q8_K_R8   , 8},
    };

    // split tensors must always be set in their entirety at once
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    auto extra = (ggml_split_tensor_t *)tensor->extra;
    GGML_ASSERT(extra->n_device <= ggml_backend_vk_get_device_count());

    if (extra->split_dim < 0) {
        // replicated across all devices
        GGML_ASSERT(ggml_is_contiguous(tensor));
        auto nbytes = ggml_nbytes(tensor);
        for (int i = 0; i < extra->n_device; ++i) {
            auto split = extra->splits[i];
            if (!split) continue;
            GGML_ASSERT(split->type == tensor->type);
            GGML_ASSERT(ggml_are_same_shape(tensor, split));
            GGML_ASSERT(ggml_nbytes(split) == nbytes);
            split->buffer->iface.set_tensor(split->buffer, split, data, 0, nbytes);
        }
    }
    else if (extra->split_dim == 0) {
        int n_interleave = 1;
        if (auto it = k_map.find(tensor->type); it != k_map.end()) n_interleave = it->second;
        auto tt = ggml_internal_get_type_traits(tensor->type);
        std::vector<char> host_buffer;
        GGML_ASSERT(ggml_is_contiguous(tensor));
        int nrows = ggml_nrows(tensor);
        auto bs = tt.blck_size;
        auto ts = tt.type_size;
        int ne = 0;
        for (int i = 0; i < extra->n_device; ++i) {
            auto split = extra->splits[i];
            if (!split) continue;
            GGML_ASSERT(split->ne[1] % n_interleave == 0);
            GGML_ASSERT(split->type == tensor->type);
            GGML_ASSERT((int)ggml_nrows(split) == nrows);
            GGML_ASSERT(split->ne[0] % bs == 0);
            auto source_offset = n_interleave * (tt.row_meta_size + (ne / bs) * ts);
            auto split_row_size = ggml_row_size(split->type, split->ne[0]);
            if (host_buffer.size() < (size_t)(nrows * split_row_size)) host_buffer.resize(nrows * split_row_size);
            for (int64_t i02 = 0; i02 < split->ne[2]; ++i02) {
                for (int64_t i01 = 0; i01 < split->ne[1]; i01 += n_interleave) {
                    auto dst = host_buffer.data() + (i02 * split->ne[1] + i01) * split_row_size;
                    auto src = (const char *)data + i02 * tensor->nb[2] + i01 * tensor->nb[1];
                    if (tt.row_meta_size > 0) {
                        memcpy(dst, src, tt.row_meta_size * n_interleave);
                    }
                    memcpy(dst + tt.row_meta_size * n_interleave, src + source_offset, n_interleave * (split_row_size - tt.row_meta_size));
                }
            }
            split->buffer->iface.set_tensor(split->buffer, split, host_buffer.data(), 0, nrows * split_row_size);
            ne += split->ne[0];
        }
    }
    else if (extra->split_dim == 1) {
        if (tensor->ne[2] > 1) {
            auto row_size = ggml_row_size(tensor->type, tensor->ne[0]);
            std::vector<char> host_buffer;
            int ne1 = 0;
            for (int i = 0; i < extra->n_device; ++i) {
                auto split = extra->splits[i];
                if (!split) continue;
                auto split_size = ggml_nbytes(split);
                if (host_buffer.size() < split_size) host_buffer.resize(split_size);
                for (int64_t i02 = 0; i02 < split->ne[2]; ++i02) {
                    auto dst = host_buffer.data() + i02 * split->ne[1] * row_size;
                    auto src = (const char *)data + i02 * tensor->nb[2] + ne1 * tensor->nb[1];
                    memcpy(dst, src, split->ne[1] * row_size);
                }
                split->buffer->iface.set_tensor(split->buffer, split, host_buffer.data(), 0, split_size);
                ne1 += split->ne[1];
            }
        } else {
            int n_interleave = 1;
            if (auto it = k_map.find(tensor->type); it != k_map.end()) n_interleave = it->second;
            size_t cur_offset = 0;
            for (int i = 0; i < extra->n_device; ++i) {
                auto split = extra->splits[i];
                if (!split) continue;
                GGML_ASSERT(split->ne[1] % n_interleave == 0);
                auto split_size = ggml_nbytes(split);
                const char * buf_host = (const char *)data + cur_offset;
                split->buffer->iface.set_tensor(split->buffer, split, buf_host, 0, split_size);
                cur_offset += split_size;
            }
        }
    }
    else {
        fprintf(stderr, "%s: not implemented for split dim %d\n", __func__, extra->split_dim);
        GGML_ABORT("fatal error");
    }
}

GGML_CALL static void ggml_backend_vk_split_buffer_get_tensor([[maybe_unused]] ggml_backend_buffer_t buffer, const ggml_tensor * tensor,
        [[maybe_unused]] void * data, size_t offset, size_t size) {
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    GGML_ABORT("not implemented");
}

GGML_CALL static void ggml_backend_vk_split_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_UNUSED(buffer);
    GGML_UNUSED(value);
}

static struct ggml_backend_buffer_i ggml_backend_vk_split_buffer_interface = {
    /* .get_name        = */ ggml_backend_vk_split_buffer_get_name,
    /* .free_buffer     = */ ggml_backend_vk_split_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_vk_split_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_vk_split_buffer_init_tensor,
    /* .memset_tensor   = */ NULL,
    /* .set_tensor      = */ ggml_backend_vk_split_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_vk_split_buffer_get_tensor,
    /* .cpy_tensor      = */ NULL,
    /* .clear           = */ ggml_backend_vk_split_buffer_clear,
    /* .reset           = */ NULL,
};

// split buffer type

GGML_CALL static const char * ggml_backend_vk_split_buffer_type_name(ggml_backend_buffer_type_t buft) {
    return "VK_Split";

    GGML_UNUSED(buft);
}

bool ggml_backend_buft_is_vk_split(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_vk_split_buffer_type_name;
}

GGML_CALL static ggml_backend_buffer_t ggml_backend_vk_split_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_vk_split_buffer_context * ctx = new ggml_backend_vk_split_buffer_context();
    return ggml_backend_buffer_init(buft, ggml_backend_vk_split_buffer_interface, ctx, size);
}

GGML_CALL static size_t ggml_backend_vk_split_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return 128;

    GGML_UNUSED(buft);
}

GGML_CALL static size_t ggml_backend_vk_split_buffer_type_get_alloc_size([[maybe_unused]] ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    if (!tensor->extra) return 0;
    auto extra = (ggml_split_tensor_t *)tensor->extra;
    GGML_ASSERT(extra->n_device <= ggml_backend_vk_get_device_count());

    size_t total_size = 0;
    for (int i = 0; i < extra->n_device; ++i) {
        auto split = extra->splits[i];
        if (!split) continue;
        total_size += ggml_nbytes(split);
        auto ne0 = split->ne[0];
        if (ne0 % VK_MATRIX_ROW_PADDING != 0) {
            auto nblock = (ne0 + VK_MATRIX_ROW_PADDING - 1) / VK_MATRIX_ROW_PADDING;
            auto row_size = ggml_row_size(split->type, ne0);
            auto padded_row_size = ggml_row_size(split->type, nblock * VK_MATRIX_ROW_PADDING);
            total_size += padded_row_size - row_size;
        }
    }
    return total_size;
}

GGML_CALL static bool ggml_backend_vk_split_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return false;

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_type_i ggml_backend_vk_split_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_vk_split_buffer_type_name,
    /* .alloc_buffer     = */ ggml_backend_vk_split_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_vk_split_buffer_type_get_alignment,
    /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
    /* .get_alloc_size   = */ ggml_backend_vk_split_buffer_type_get_alloc_size,
    /* .is_host          = */ ggml_backend_vk_split_buffer_type_is_host,
};

GGML_CALL ggml_backend_buffer_type_t ggml_backend_vk_split_buffer_type(const float * /*tensor_split*/) {
    static ggml_backend_buffer_type buft {
        /* .iface   = */ ggml_backend_vk_split_buffer_type_interface,
        /* .context = */ new ggml_backend_vk_split_buffer_type_context{},
    };
    return &buft;
}

#include "../llama-build-context.h"
#include "../llama-model.h"
#include "../llama-context.h"
#include "../llama-delta-net.h"

ggml_cgraph * llm_build_context::build_qwen35moe() {

    if (cparams.mtp_op_type == MTP_OP_DRAFT_GEN_FUSED) {
        return build_qwen35_mtp_fused(cparams.mtp_fused_n_steps, cparams.mtp_fused_n_extend, /*is_moe=*/true);
    }

    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes(n_tokens), false);

    const int64_t n_embd_head = hparams.n_embd_head_v(0);
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k(0));

    ggml_tensor * cur = nullptr;
    ggml_tensor * inp_pos = build_inp_pos();

    if (cparams.mtp_op_type != MTP_OP_NONE) {
        // MTP tail-only graph (mirrors build_qwen35 dense path).
        // inp_mtp_states is 2D (n_embd, n_tokens) for ALL MTP op types so
        // DRAFT_GEN can carry one hidden state per batch row when N slots
        // draft together. n_tokens=1 reduces to the original single-row
        // case (a 2D shape with second dim 1 is mathematically identical
        // to 1D for downstream concat + GEMM).
        ggml_tensor * hidden_states_from_main_model =
            ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd, n_tokens);
        ggml_set_name(hidden_states_from_main_model, "inp_mtp_states");
        ggml_set_input(hidden_states_from_main_model);
        lctx.default_decoder.inp_mtp_states = hidden_states_from_main_model;

        const int il_mtp = hparams.n_layer - 1;
        const auto & mtp_layer = model.layers[il_mtp];

        cur = build_qwen35_mtp(mtp_layer, hidden_states_from_main_model, n_embd_head, gf, inp_pos);
    } else {
        delta_net delta(lctx, batch);

        ggml_tensor * inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);
        ggml_tensor * inp_out_ids = (n_tokens > 1 && !lctx.cparams.mtp) ? build_inp_out_ids() : nullptr;
        ggml_tensor * KQ_mask = build_inp_KQ_mask();

        lctx.default_decoder.inp_s_seq_qnext = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, 1, n_tokens);
        cb(lctx.default_decoder.inp_s_seq_qnext, "inp_s_seq_qnext", -1);
        ggml_set_input(lctx.default_decoder.inp_s_seq_qnext);

        float KQ_scale = hparams.f_attention_scale == 0.0f ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

        const int n_transformer_layers = n_layer - hparams.nextn_predict_layers;
        for (int il = 0; il < n_transformer_layers; ++il) {

            if (hparams.is_recurrent(il)) {
                cur = delta.build_layer_attn_linear(ctx0, gf, inpL, il == n_transformer_layers - 1 ? inp_out_ids : nullptr, il, cb);
            } else {
                cur = build_std_attention(gf, model.layers[il].attn_norm, inpL, inp_pos, il == n_transformer_layers - 1 ? inp_out_ids : nullptr, nullptr,
                        KQ_mask, nullptr, nullptr, KQ_scale, 0.0f, 0, il, true, false, true, false, true);
            }

            cur = llm_build_std_moe_ffn(ctx0, lctx, model.layers[il].ffn_norm, cur,
                    model.layers[il].ffn_gate_inp,  nullptr,
                    model.layers[il].ffn_up_exps,   nullptr,
                    model.layers[il].ffn_gate_exps, nullptr,
                    model.layers[il].ffn_down_exps, nullptr,
                    nullptr,
                    model.layers[il].ffn_up_shexp,    nullptr, // we don't have shared expert biases?
                    model.layers[il].ffn_gate_shexp,  nullptr,
                    model.layers[il].ffn_down_shexp,  nullptr,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, true, false, 0.0f,
                    LLM_EXPERT_GATING_FUNC_SOFTMAX,
                    LLM_FFN_SILU, cb, il, gf, true, model.layers[il].ffn_up_gate_exps, nullptr, model.layers[il].ffn_gate_inp_shexp);

            cur = lctx.cvec.apply_to(ctx0, cur, il);
            cb(cur, "l_out", il);

            inpL = cur;
        }

        if (lctx.cparams.mtp) {
            // Tag pre-final-norm residual ("h_pre_norm") so the per-ubatch
            // MTP KV hook (Phase 36 Step 3) can read it without a name
            // search. Same tensor still feeds the embd-extraction path
            // below — the rename matches what the residual actually is
            // (input to the final norm + lm_head, NOT the MTP layer's
            // output). Stash on the context so callers that already had
            // ctx (rather than the cgraph) can find it.
            struct ggml_tensor * embd_copy = ggml_dup(ctx0, inpL);
            cb(embd_copy, "h_pre_norm", -1);
            ggml_set_output(embd_copy);
            lctx.default_decoder.t_h_pre_norm = embd_copy;
        }

        cur = build_output(lctx, ctx0, inpL, model.output, model.output_norm, cb);
        cb(cur, "result_output", -1);

        // Phase 36 Step 3.2: per-ubatch kv-only MTP hook. Fold MTP KV
        // writes into the verify forward so the separate
        // MTP_OP_UPDATE_ACCEPTED decode can be eliminated. Gated by
        // cparams.mtp_inline_kv_hook to allow A/B comparison.
        if (lctx.cparams.mtp && lctx.cparams.mtp_inline_kv_hook) {
            const int il_mtp = hparams.n_layer - 1;
            const auto & mtp_layer = model.layers[il_mtp];
            ggml_tensor * mtp_kv = build_qwen35_mtp_kv_only(
                    mtp_layer, inpL, lctx.default_decoder.inp_tokens,
                    n_embd_head, gf, inp_pos, KQ_mask);
            cb(mtp_kv, "mtp_kv_inline_moe", il_mtp);
            ggml_build_forward_expand(gf, mtp_kv);
            ++lctx.default_decoder.mtp_hook_fire_count;
            ++lctx.default_decoder.mtp_inline_decode_count;
        }
    }

    ggml_build_forward_expand(gf, cur);

    return gf;
}

ggml_cgraph * llm_build_context::build_qwen35() {

    if (cparams.mtp_op_type == MTP_OP_DRAFT_GEN_FUSED) {
        return build_qwen35_mtp_fused(cparams.mtp_fused_n_steps, cparams.mtp_fused_n_extend, /*is_moe=*/false);
    }

    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes(n_tokens), false);

    const int64_t n_embd_head = hparams.n_embd_head_v(0);
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k(0));

    ggml_tensor * cur;

    ggml_tensor * inp_pos = build_inp_pos();

    if (cparams.mtp_op_type != MTP_OP_NONE) {
        // MTP tail-only graph.
        // PHASE45 D10.b: inp_mtp_states is 2D (n_embd, n_tokens) for ALL
        // MTP op types so DRAFT_GEN can carry one hidden state per batch
        // row when N slots draft together. n_tokens=1 reduces to the
        // original single-row case (a 2D shape with second dim 1 is
        // mathematically identical to 1D for downstream concat + GEMM).
        // Matches the MoE build (build_qwen35moe).
        ggml_tensor * hidden_states_from_main_model =
            ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd, n_tokens);
        ggml_set_name(hidden_states_from_main_model, "inp_mtp_states");
        ggml_set_input(hidden_states_from_main_model);
        lctx.default_decoder.inp_mtp_states = hidden_states_from_main_model;

        const int il_mtp = hparams.n_layer - 1;
        const auto & mtp_layer = model.layers[il_mtp];

        cur = build_qwen35_mtp(mtp_layer, hidden_states_from_main_model, n_embd_head, gf, inp_pos);
    } else {
        delta_net delta(lctx, batch);

        ggml_tensor * inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);
        // C.1 diagnostic gate: force inp_out_ids to be built even when mtp is
        // on. Tests whether the inp_out_ids=nullptr graph topology when
        // cparams.mtp is the source of np>1 slot divergence.
        static const bool force_out_ids = []() {
            const char * v = getenv("LLAMA_FORCE_OUT_IDS_WITH_MTP");
            return v && *v && *v != '0';
        }();
        ggml_tensor * inp_out_ids = (n_tokens > 1 && (!lctx.cparams.mtp || force_out_ids)) ? build_inp_out_ids() : nullptr;
        ggml_tensor * KQ_mask = build_inp_KQ_mask();

        lctx.default_decoder.inp_s_seq_qnext = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, 1, n_tokens);
        cb(lctx.default_decoder.inp_s_seq_qnext, "inp_s_seq_qnext", -1);
        ggml_set_input(lctx.default_decoder.inp_s_seq_qnext);

        float KQ_scale = hparams.f_attention_scale == 0.0f ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

        cur = nullptr;

        const int n_transformer_layers = n_layer - hparams.nextn_predict_layers;
        for (int il = 0; il < n_transformer_layers; ++il) {

            if (hparams.is_recurrent(il)) {
                cur = delta.build_layer_attn_linear(ctx0, gf, inpL, il == n_transformer_layers - 1 ? inp_out_ids : nullptr, il, cb);
            } else {
                cur = build_std_attention(gf, model.layers[il].attn_norm, inpL, inp_pos, il == n_transformer_layers - 1 ? inp_out_ids : nullptr, nullptr,
                        KQ_mask, nullptr, nullptr, KQ_scale, 0.0f, 0, il, true, false, true, false, true);
            }

            cur = llm_build_ffn(ctx0, lctx, model.layers[il].ffn_norm, cur,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il, gf, true, false);

            cur = lctx.cvec.apply_to(ctx0, cur, il);
            cb(cur, "l_out", il);

            inpL = cur;
        }

        // C.1 diagnostic gate: env-disable the h_pre_norm extra-output side
        // effect that fires when -mtp is on. Tests whether the dup+set_output
        // is the source of np>1 slot divergence.
        static const bool disable_h_pre_norm = []() {
            const char * v = getenv("LLAMA_DISABLE_H_PRE_NORM");
            return v && *v && *v != '0';
        }();
        if (lctx.cparams.mtp && !disable_h_pre_norm) {
            // See build_qwen35moe() above for the rationale on the
            // "h_pre_norm" tag + lctx.default_decoder.t_h_pre_norm stash.
            struct ggml_tensor * embd_copy = ggml_dup(ctx0, inpL);
            cb(embd_copy, "h_pre_norm", -1);
            ggml_set_output(embd_copy);
            lctx.default_decoder.t_h_pre_norm = embd_copy;
        }

        cur = build_output(lctx, ctx0, inpL, model.output, model.output_norm, cb);
        cb(cur, "result_output", -1);

        // Phase 36 Step 3.2: per-ubatch kv-only MTP hook. See
        // build_qwen35moe() for the rationale.
        if (lctx.cparams.mtp && lctx.cparams.mtp_inline_kv_hook) {
            const int il_mtp = hparams.n_layer - 1;
            const auto & mtp_layer = model.layers[il_mtp];
            ggml_tensor * mtp_kv = build_qwen35_mtp_kv_only(
                    mtp_layer, inpL, lctx.default_decoder.inp_tokens,
                    n_embd_head, gf, inp_pos, KQ_mask);
            cb(mtp_kv, "mtp_kv_inline", il_mtp);
            ggml_build_forward_expand(gf, mtp_kv);
            ++lctx.default_decoder.mtp_hook_fire_count;
            ++lctx.default_decoder.mtp_inline_decode_count;
        }
    }

    ggml_build_forward_expand(gf, cur);

    return gf;
}

struct ggml_tensor * llm_build_context::build_qwen35_mtp(
    const llama_layer & mtp_layer,
    struct ggml_tensor * prev_embeddings,
    int64_t n_embd_head,
    struct ggml_cgraph * gf,
    struct ggml_tensor * inp_pos
) {
    const int il = hparams.n_layer - 1;

    struct ggml_tensor * KQ_mask = build_inp_KQ_mask();

    struct ggml_tensor * inp_out_ids = (n_outputs < n_tokens) ? build_inp_out_ids() : nullptr;

    ggml_tensor * token_emb = build_inp_embd_mtp(model.tok_embd);

    ggml_tensor * token_emb_norm = llm_build_norm(ctx0, token_emb, hparams, mtp_layer.nextn.enorm, NULL, LLM_NORM_RMS, cb, il);
    ggml_tensor * hidden_state_norm = llm_build_norm(ctx0, prev_embeddings, hparams, mtp_layer.nextn.hnorm, NULL, LLM_NORM_RMS, cb, il);

    ggml_tensor * cur;
    if (mtp_layer.nextn.eh_proj != nullptr) {
        // Full fusion: concat + project (27B, 4B, 2B, 0.8B)
        ggml_tensor * combined = ggml_concat(ctx0, token_emb_norm, hidden_state_norm, 0);
        cb(combined, "mtp_concat", il);
        cur = llm_build_lora_mm(lctx, ctx0, mtp_layer.nextn.eh_proj, combined);
    } else {
        // 9B — no fc/eh_proj
        cur = ggml_add(ctx0, token_emb_norm, hidden_state_norm);
    }
    cb(cur, "mtp_fused", il);

    // Self-Attention (wq may be shared from main model's last layer)
    GGML_ASSERT(il < (int)kv_self.k_l.size() && il < (int)kv_self.v_l.size());
    if (!kv_self.k_l[il] || !kv_self.v_l[il]) {
        LLAMA_LOG_ERROR("%s: KV cache not allocated for MTP layer %d (k=%p, v=%p)\n",
                __func__, il, (void*)kv_self.k_l[il], (void*)kv_self.v_l[il]);
        GGML_ABORT("KV cache not allocated for MTP layer");
    }
    if (!model.layers[il].wq || !model.layers[il].wk || !model.layers[il].wv || !model.layers[il].wo) {
        LLAMA_LOG_ERROR("%s: Missing attention weights for MTP layer %d (wq=%p, wk=%p, wv=%p, wo=%p)\n",
                __func__, il, (void*)model.layers[il].wq, (void*)model.layers[il].wk,
                (void*)model.layers[il].wv, (void*)model.layers[il].wo);
        GGML_ABORT("Missing attention weights for MTP layer");
    }

    const float kq_scale = 1.0f / sqrtf(float(n_embd_head));

    cur = build_std_attention(gf, mtp_layer.attn_norm, cur,
            inp_pos, nullptr, nullptr,
            KQ_mask, nullptr, nullptr,
            kq_scale, 0.0f, 0, il, true, false, true, false, true, nullptr);

    if (inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    // Dense FFN — optional (9B and 4B don't have FFN in MTP layer)
    if (mtp_layer.ffn_gate != nullptr) {
        cur = llm_build_ffn(ctx0, lctx, mtp_layer.ffn_norm, cur,
                mtp_layer.ffn_up,   NULL, NULL,
                mtp_layer.ffn_gate, NULL, NULL,
                mtp_layer.ffn_down, NULL, NULL,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, cb, il, gf, true, false);
    }

    cur = lctx.cvec.apply_to(ctx0, cur, il);
    cb(cur, "ffn_out", il);

    cur = llm_build_norm(ctx0, cur, hparams, mtp_layer.nextn.shared_head_norm, NULL, LLM_NORM_RMS, cb, il);
    cb(cur, "result_norm", -1);

    cur = build_output(lctx, ctx0, cur, model.output, nullptr, cb);
    cb(cur, "result_output", -1);

    return cur;
}

// Phase 36 #3: shared MTP chain-residual primitive. See header comment.
// Returns the post-shared_head_norm tensor; the caller invokes lm_head
// + argmax separately. Both per-step (build_qwen35_mtp) and fused
// (build_qwen35_mtp_fused chain step) call this so the graph optimizer
// sees identical op sequences and picks identical kernels.
struct ggml_tensor * llm_build_context::build_qwen35_mtp_chain_residual(
    const llama_layer & mtp_layer,
    struct ggml_tensor * prev_embeddings,
    struct ggml_tensor * tokens_input,
    int64_t n_embd_head,
    struct ggml_cgraph * gf,
    struct ggml_tensor * inp_pos,
    struct ggml_tensor * KQ_mask,
    int kv_head_offset) {

    const int il = hparams.n_layer - 1;

    GGML_ASSERT(il < (int)kv_self.k_l.size() && il < (int)kv_self.v_l.size());
    if (!kv_self.k_l[il] || !kv_self.v_l[il]) {
        LLAMA_LOG_ERROR("%s: KV cache not allocated for MTP layer %d (k=%p, v=%p)\n",
                __func__, il, (void*)kv_self.k_l[il], (void*)kv_self.v_l[il]);
        GGML_ABORT("KV cache not allocated for MTP layer");
    }
    if (!model.layers[il].wq || !model.layers[il].wk ||
        !model.layers[il].wv || !model.layers[il].wo) {
        LLAMA_LOG_ERROR("%s: Missing attention weights for MTP layer %d\n",
                __func__, il);
        GGML_ABORT("Missing attention weights for MTP layer");
    }

    ggml_tensor * token_emb = ggml_get_rows(ctx0, model.tok_embd, tokens_input);
    ggml_tensor * token_emb_norm = llm_build_norm(
            ctx0, token_emb, hparams, mtp_layer.nextn.enorm,
            NULL, LLM_NORM_RMS, cb, il);
    ggml_tensor * hidden_state_norm = llm_build_norm(
            ctx0, prev_embeddings, hparams, mtp_layer.nextn.hnorm,
            NULL, LLM_NORM_RMS, cb, il);

    ggml_tensor * cur;
    if (mtp_layer.nextn.eh_proj != nullptr) {
        ggml_tensor * combined = ggml_concat(ctx0, token_emb_norm, hidden_state_norm, 0);
        cb(combined, "mtp_concat", il);
        cur = llm_build_lora_mm(lctx, ctx0, mtp_layer.nextn.eh_proj, combined);
    } else {
        cur = ggml_add(ctx0, token_emb_norm, hidden_state_norm);
    }
    cb(cur, "mtp_fused", il);

    const float kq_scale = 1.0f / sqrtf(float(n_embd_head));

    cur = build_std_attention(gf, mtp_layer.attn_norm, cur,
            inp_pos, nullptr, nullptr,
            KQ_mask, nullptr, nullptr,
            kq_scale, 0.0f, 0, il,
            /*do_rope=*/true, /*add_graph_split=*/false,
            /*add_input=*/true, /*is_norm=*/false,
            /*is_multi=*/true, /*post_norm=*/nullptr,
            kv_head_offset, /*fa_prec_f32=*/true);

    if (mtp_layer.ffn_gate != nullptr) {
        cur = llm_build_ffn(ctx0, lctx, mtp_layer.ffn_norm, cur,
                mtp_layer.ffn_up,   NULL, NULL,
                mtp_layer.ffn_gate, NULL, NULL,
                mtp_layer.ffn_down, NULL, NULL,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, cb, il, gf, true, false);
    }

    cur = lctx.cvec.apply_to(ctx0, cur, il);
    cb(cur, "ffn_out", il);

    cur = llm_build_norm(ctx0, cur, hparams,
            mtp_layer.nextn.shared_head_norm,
            NULL, LLM_NORM_RMS, cb, il);
    cb(cur, "result_norm", -1);

    return cur;
}

// Phase 36 Step 3: MTP layer compute that stops after attention. Used by
// the per-ubatch hook when integrating MTP KV writes into verify forward.
// Skips FFN, final norm, and lm_head — KV write is the only side effect
// we want.
struct ggml_tensor * llm_build_context::build_qwen35_mtp_kv_only(
    const llama_layer & mtp_layer,
    struct ggml_tensor * prev_embeddings,
    struct ggml_tensor * tokens_input,
    int64_t n_embd_head,
    struct ggml_cgraph * gf,
    struct ggml_tensor * inp_pos,
    struct ggml_tensor * KQ_mask,
    int kv_head_offset) {

    const int il = hparams.n_layer - 1;

    ggml_tensor * token_emb = ggml_get_rows(ctx0, model.tok_embd, tokens_input);
    ggml_tensor * token_emb_norm = llm_build_norm(ctx0, token_emb, hparams, mtp_layer.nextn.enorm, NULL, LLM_NORM_RMS, cb, il);
    ggml_tensor * hidden_state_norm = llm_build_norm(ctx0, prev_embeddings, hparams, mtp_layer.nextn.hnorm, NULL, LLM_NORM_RMS, cb, il);

    ggml_tensor * cur;
    if (mtp_layer.nextn.eh_proj != nullptr) {
        ggml_tensor * combined = ggml_concat(ctx0, token_emb_norm, hidden_state_norm, 0);
        cb(combined, "mtp_kv_concat", il);
        cur = llm_build_lora_mm(lctx, ctx0, mtp_layer.nextn.eh_proj, combined);
    } else {
        cur = ggml_add(ctx0, token_emb_norm, hidden_state_norm);
    }
    cb(cur, "mtp_kv_fused", il);

    GGML_ASSERT(il < (int)kv_self.k_l.size() && il < (int)kv_self.v_l.size());

    const float kq_scale = 1.0f / sqrtf(float(n_embd_head));

    cur = build_std_attention(gf, mtp_layer.attn_norm, cur,
            inp_pos, nullptr, nullptr,
            KQ_mask, nullptr, nullptr,
            kq_scale, 0.0f, 0, il, true, false, true, false, true, nullptr,
            kv_head_offset);

    cb(cur, "mtp_kv_attn_out", il);
    return cur;
}

// Phase 36 Step 1: fused multi-draft cgraph. Chains N MTP draft steps in
// a single graph. Each step: token_emb→enorm/hnorm→concat→eh_proj→attn
// (writes KV at pos_base+k)→[ffn]→norm→lm_head→argmax→softmax→prob.
// Step 0 token = inp_tokens[0] (seed); step k>0 token = argmax_{k-1}.
// Outputs: N argmax tensors named "mtp_argmax_<k>" and N prob tensors
// named "mtp_prob_<k>", all set_output.
//
// Caller contract: the batch passed into llama_decode must have
// batch.n_tokens == n_draft so that n_tokens (the const member of
// llm_build_context) matches the chain length and the slot allocator
// reserves n_draft consecutive KV cells.
ggml_cgraph * llm_build_context::build_qwen35_mtp_fused(int n_draft, int n_extend, bool is_moe) {
    GGML_ASSERT(n_draft >= 1 && n_draft <= LLAMA_MTP_FUSED_MAX);
    // Phase 38 C: extended-chain — run n_draft + n_extend internal
    // chain steps but emit only n_draft drafts. n_extend > 0 populates
    // chain_residuals at indices [n_draft, n_draft+n_extend) — used as
    // seed for the all-accept case in Phase 38 E speculative dispatch.
    // The batch passed to llama_decode must have n_tokens = n_chain so
    // the slot allocator reserves n_chain consecutive KV cells (each
    // chain step writes its own cell).
    GGML_ASSERT(n_extend >= 0);
    GGML_ASSERT(n_draft + n_extend <= LLAMA_MTP_FUSED_MAX);
    const int n_chain = n_draft + n_extend;
    GGML_ASSERT(n_chain == n_tokens);

    struct ggml_cgraph * gf = ggml_new_graph_custom(
        ctx0, model.max_nodes(n_tokens) * (n_chain + 2), false);

    const int64_t n_embd_head = hparams.n_embd_head_v(0);
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k(0));

    const int il_mtp = hparams.n_layer - 1;
    const auto & mtp_layer = model.layers[il_mtp];

    // Initial hidden state input (h_pre_norm from verify, 1 row).
    // Filled by prepare_mtp_graph_inputs from lctx.default_decoder.draft_input_hidden_state.
    ggml_tensor * inp_states = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd, 1);
    ggml_set_name(inp_states, "inp_mtp_states");
    ggml_set_input(inp_states);
    lctx.default_decoder.inp_mtp_states = inp_states;

    // Token tensor sized for the batch (n_tokens = n_draft). Step 0
    // reads index 0 (seed_token); steps k>0 use argmax_{k-1} in-graph.
    lctx.default_decoder.inp_tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    cb(lctx.default_decoder.inp_tokens, "inp_tokens", -1);
    ggml_set_input(lctx.default_decoder.inp_tokens);

    // Position tensor: matches build_inp_pos() — for MROPE/IMROPE,
    // 4 entries per token; otherwise 1 per token. Step k slices its
    // n_pos_per_embd-element segment via ggml_view_1d below.
    const int n_pos_per_embd = (hparams.rope_type == LLAMA_ROPE_TYPE_MROPE ||
                                hparams.rope_type == LLAMA_ROPE_TYPE_IMROPE) ? 4 : 1;
    lctx.default_decoder.inp_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, (int64_t)n_tokens * n_pos_per_embd);
    cb(lctx.default_decoder.inp_pos, "inp_pos", -1);
    ggml_set_input(lctx.default_decoder.inp_pos);

    // KQ_mask: (n_kv, n_tokens * GGML_KQ_MASK_PAD). Each step k uses a
    // (n_kv, GGML_KQ_MASK_PAD) view at row offset k*GGML_KQ_MASK_PAD —
    // satisfies the flash-attention requirement that mask->ne[1] >=
    // GGML_PAD(q->ne[1], GGML_KQ_MASK_PAD) for single-token queries.
    // Per-step visibility filled by llama_set_inputs.
    lctx.default_decoder.inp_KQ_mask = ggml_new_tensor_2d(
        ctx0, flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32,
        n_kv, (int64_t)n_tokens * GGML_KQ_MASK_PAD);
    cb(lctx.default_decoder.inp_KQ_mask, "KQ_mask", -1);
    ggml_set_input(lctx.default_decoder.inp_KQ_mask);

    ggml_tensor * prev_residual = inp_states;
    ggml_tensor * argmaxes[LLAMA_MTP_FUSED_MAX] = {};
    ggml_tensor * probs[LLAMA_MTP_FUSED_MAX] = {};

    for (int k = 0; k < n_chain; ++k) {
        // Token id for this step. Step 0 reads inp_tokens[0] via a
        // 1-element view; step k>0 uses argmax_{k-1} (already shape [1]).
        // The view path keeps each step's compute single-token even
        // though the outer build context has n_tokens = n_draft.
        ggml_tensor * tok_id_k;
        if (k == 0) {
            tok_id_k = ggml_view_1d(ctx0, lctx.default_decoder.inp_tokens, 1, 0);
        } else {
            tok_id_k = argmaxes[k-1];
        }

        // Per-step inp_pos view: n_pos_per_embd elements at offset
        // k * n_pos_per_embd. For MROPE/IMROPE this is 4 elements per
        // step; otherwise 1.
        ggml_tensor * pos_k = ggml_view_1d(ctx0, lctx.default_decoder.inp_pos,
            n_pos_per_embd,
            (size_t) k * n_pos_per_embd * lctx.default_decoder.inp_pos->nb[0]);

        // Per-step KQ_mask view: (n_kv, GGML_KQ_MASK_PAD) at row offset
        // k * GGML_KQ_MASK_PAD. The mask's first row is the actual
        // visibility for step k's query; the remaining KQ_MASK_PAD-1
        // rows are padding (FA reads them but they don't affect output
        // for n_queries=1).
        ggml_tensor * mask_k = ggml_view_2d(ctx0, lctx.default_decoder.inp_KQ_mask,
            lctx.default_decoder.inp_KQ_mask->ne[0], GGML_KQ_MASK_PAD,
            lctx.default_decoder.inp_KQ_mask->nb[1],
            (size_t) k * GGML_KQ_MASK_PAD * lctx.default_decoder.inp_KQ_mask->nb[1]);

        // Phase 36 #3: route through the shared chain-residual primitive
        // (the SAME primitive build_qwen35_mtp uses).
        ggml_tensor * normed = build_qwen35_mtp_chain_residual(
            mtp_layer, prev_residual, tok_id_k,
            n_embd_head, gf, pos_k, mask_k,
            /*kv_head_offset=*/k);
        (void) is_moe;

        // Phase 37 fix: force materialization of the chain residual at
        // each step boundary. ggml_dup is identity in F32; its purpose
        // is to break ggml's graph optimizer's ability to fuse step
        // k+1's compute into kernels of step k. Per-step has this
        // materialization implicitly via the D2H boundary between
        // separate decodes; fused needs it explicitly. Closes the
        // cumulative drift identified by the d=2 probe (d=2 ratio 0.93
        // vs d=3 ratio 0.67 — drift compounds at chain step ≥ 2).
        //
        // Phase 37 #2 extension: also expose normed at EVERY step (not
        // just k+1<n_draft) as set_output. The exposed tensor is the
        // hidden state at position S_k+(k+1) — exactly what the next
        // cycle's fused decode needs as its seed when verify accepts
        // k drafts. Indexed via lctx.default_decoder.mtp_fused_chain_residuals[k].
        normed = ggml_dup(ctx0, normed);
        char nm_residual[40];
        snprintf(nm_residual, sizeof(nm_residual), "mtp_chain_residual_%d", k);
        ggml_set_name(normed, nm_residual);
        ggml_set_output(normed);
        lctx.default_decoder.mtp_fused_chain_residuals[k] = normed;
        cb(normed, "mtp_chain_residual", il_mtp);

        if (k + 1 < n_chain) {
            // Phase 37 #3b: dependency anchor on step k's KV cpy(s).
            // cache_copies's MTP-layer slot is overwritten by step k+1's
            // build_std_attention call (il_mtp is fixed across the chain),
            // so capture the cpy tensor handle here while it's still
            // step k's. Anchoring via ggml_dup adds a leaf node that
            // depends on the cpy, forcing any scheduler — single-stream
            // FIFO, multi-stream with reorder, or graph optimizer — to
            // sequence step k's cpy before step k+1's compute.
            const size_t splits_eff =
                model.splits.empty() ? 1 : model.splits.size();
            const size_t kv_idx_base = 2 * splits_eff * (size_t) il_mtp;
            const size_t n_slots = 2 * splits_eff;
            for (size_t s = 0; s < n_slots; ++s) {
                if (kv_idx_base + s >= lctx.cache_copies.size()) break;
                ggml_tensor * cpy = lctx.cache_copies[kv_idx_base + s].cpy;
                if (cpy == nullptr) continue;
                ggml_tensor * dep_anchor = ggml_dup(ctx0, cpy);
                cb(dep_anchor,
                   (s % 2 == 0) ? "mtp_kv_dep_anchor_k"
                                : "mtp_kv_dep_anchor_v",
                   il_mtp);
                ggml_build_forward_expand(gf, dep_anchor);
            }
        }
        prev_residual = normed;

        // Head: per-device lm_head matmul, per-device argmax + max-val,
        // then in-graph reduction across devices to pick the global
        // winner. This avoids the Issue F bug where ggml_argmax on the
        // ggml_concat of split-mode lm_head outputs reads past one
        // device's slice. Per-device argmax stays on its own device;
        // the small (n_dev,) reduction tensors sched-route to one
        // device for the final argmax.
        //
        // Single-device case (output->extra == nullptr) falls through
        // to a one-shot lm_head + argmax — identical to the previous
        // code path, just routed via the same helper.
        ggml_tensor * argmax = nullptr;
        ggml_tensor * prob   = nullptr;
        if (model.output->extra != nullptr) {
            // Multi-device split path: per-device argmax + reduction.
            auto split_output = (ggml_split_tensor_t *)model.output->extra;
            const int n_dev = split_output->n_device;

            std::vector<ggml_tensor *> dev_amaxes;
            std::vector<ggml_tensor *> dev_max_vals;
            int64_t device_offset_const[16 /*MAX_DEVICES*/] = {0};
            int64_t cum_offset = 0;
            for (int d = 0; d < n_dev; ++d) {
                auto split = split_output->splits[d];
                if (!split) continue;
                ggml_tensor * dev_logits = llm_build_lora_mm(lctx, ctx0, split, normed); // (n_vocab/n_dev, 1)
                cb(dev_logits, "mtp_dev_logits", il_mtp);

                // Local argmax (in [0, n_vocab/n_dev)).
                ggml_tensor * local_amax = ggml_argmax(ctx0, dev_logits);
                cb(local_amax, "mtp_dev_argmax_local", il_mtp);

                // Max value at local_amax: get_rows on contiguous
                // transpose so dim-0 contiguity is satisfied for CUDA.
                ggml_tensor * dev_logits_t = ggml_cont(ctx0, ggml_transpose(ctx0, dev_logits));
                ggml_tensor * local_max_val = ggml_get_rows(ctx0, dev_logits_t, local_amax);
                cb(local_max_val, "mtp_dev_max_val", il_mtp);

                device_offset_const[d] = cum_offset;
                cum_offset += dev_logits->ne[0];

                dev_amaxes.push_back(local_amax);
                dev_max_vals.push_back(local_max_val);
            }

            // In-graph reduction: stack max_vals (n_dev,) and argmax to
            // pick the winning device. Stack global_amaxes (with offset
            // pre-added in graph via ggml_add of a per-device offset
            // input) and get_rows by winning device idx.
            //
            // Allocate one (n_dev,) input tensor for offsets, filled by
            // set_inputs from device_offset_const[].
            char nm_off[40]; snprintf(nm_off, sizeof(nm_off), "mtp_dev_offsets_%d", k);
            ggml_tensor * offsets = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, (int)dev_amaxes.size());
            ggml_set_name(offsets, nm_off);
            ggml_set_input(offsets);
            // Stash the tensor pointer + values; set_inputs fills.
            lctx.default_decoder.mtp_fused_offset_t[k] = offsets;
            lctx.default_decoder.mtp_fused_offset_n_dev[k] = (int32_t) dev_amaxes.size();
            for (int d = 0; d < (int)dev_amaxes.size(); ++d) {
                lctx.default_decoder.mtp_fused_offset_buf[k * 16 /*MAX_DEVICES*/ + d] = (int32_t) device_offset_const[d];
            }

            // Concat per-device argmaxes (each shape [1]) into shape (n_dev,).
            ggml_tensor * stacked_amax = dev_amaxes[0];
            for (int d = 1; d < (int)dev_amaxes.size(); ++d) {
                stacked_amax = ggml_concat(ctx0, stacked_amax, dev_amaxes[d], 0);
            }
            // Add per-device offsets → global indices (n_dev,) I32.
            ggml_tensor * global_amaxes = ggml_add(ctx0, stacked_amax, offsets);
            cb(global_amaxes, "mtp_global_amaxes", il_mtp);

            // Concat per-device max_vals (each shape [1, 1]) → (n_dev,) F32 view.
            ggml_tensor * stacked_max_val = dev_max_vals[0];
            for (int d = 1; d < (int)dev_max_vals.size(); ++d) {
                stacked_max_val = ggml_concat(ctx0, stacked_max_val, dev_max_vals[d], 0);
            }
            // argmax over the (n_dev, 1) shape → (1) I32 = winning device idx.
            ggml_tensor * winner_dev = ggml_argmax(ctx0, stacked_max_val);
            cb(winner_dev, "mtp_winner_dev", il_mtp);

            // get_rows(global_amaxes, winner_dev): we need dim-0
            // contiguity for CUDA. global_amaxes is shape (n_dev,);
            // make it (n_dev, 1) and transpose to (1, n_dev) cont, then
            // index by winner_dev to get scalar.
            ggml_tensor * global_amaxes_2d = ggml_reshape_2d(ctx0, global_amaxes, (int64_t)dev_amaxes.size(), 1);
            ggml_tensor * global_amaxes_t  = ggml_cont(ctx0, ggml_transpose(ctx0, global_amaxes_2d));
            argmax = ggml_get_rows(ctx0, global_amaxes_t, winner_dev);
            // For prob: take the winning device's max value.
            ggml_tensor * stacked_max_val_2d = ggml_reshape_2d(ctx0, stacked_max_val, (int64_t)dev_max_vals.size(), 1);
            ggml_tensor * stacked_max_val_t  = ggml_cont(ctx0, ggml_transpose(ctx0, stacked_max_val_2d));
            prob = ggml_get_rows(ctx0, stacked_max_val_t, winner_dev);
        } else {
            // Single-device fast path.
            ggml_tensor * logits = llm_build_lora_mm(lctx, ctx0, model.output, normed);
            cb(logits, "result_output", -1);

            // Phase 36 Issue G fix: lm_head is (n_embd, hparams.n_vocab)
            // where hparams.n_vocab is the matmul-aligned padded width
            // (Qwen 3.6 27B: 248320). The real tokenizer vocab is
            // model.vocab.n_tokens() (152064). The padded tail is junk
            // — argmax over the full output picks fake tokens (e.g.
            // 248045) and the server rejects with "Invalid token".
            // Per-step works because host-side sampling reads only
            // [0, n_vocab_real). Slice the same way for fused.
            // Issue G note: model.vocab.n_tokens() equals
            // hparams.n_vocab for this AutoRound GGUF (248320 both),
            // so this slice is currently a no-op for Qwen 3.6 27B.
            // The padded vocab range [152064, 248320) still wins
            // argmax on some cycles (giving fused 13-18% accept vs
            // per-step's 58%). Per-step survives because its sampler
            // chain (top_k, top_p) filters; fused on-device argmax
            // does not. Future fix: track the real tokenizer
            // boundary separately or push the per-step sampler ops
            // into the graph.
            const int64_t n_vocab_real = (int64_t) lctx.model.vocab.n_tokens();
            const int64_t n_vocab_pad  = logits->ne[0];
            if (n_vocab_real > 0 && n_vocab_real < n_vocab_pad) {
                ggml_tensor * logits_real = ggml_view_2d(
                        ctx0, logits, n_vocab_real, logits->ne[1],
                        logits->nb[1], 0);
                argmax = ggml_argmax(ctx0, logits_real);
                ggml_tensor * sm = ggml_soft_max(ctx0, logits_real);
                ggml_tensor * sm_t = ggml_cont(ctx0, ggml_transpose(ctx0, sm));
                prob = ggml_get_rows(ctx0, sm_t, argmax);
            } else {
                argmax = ggml_argmax(ctx0, logits);
                ggml_tensor * sm = ggml_soft_max(ctx0, logits);
                ggml_tensor * sm_t = ggml_cont(ctx0, ggml_transpose(ctx0, sm));
                prob = ggml_get_rows(ctx0, sm_t, argmax);
            }
        }

        char nm[32]; snprintf(nm, sizeof(nm), "mtp_argmax_%d", k);
        ggml_set_name(argmax, nm);
        ggml_set_output(argmax);
        argmaxes[k] = argmax;

        snprintf(nm, sizeof(nm), "mtp_prob_%d", k);
        ggml_set_name(prob, nm);
        ggml_set_output(prob);
        probs[k] = prob;
    }

    // Phase 38 E: expand argmax/prob for ALL chain steps including
    // EXTEND. Speculative dispatch needs fr.tokens[n_emitted] (the
    // extend step's argmax) as the predicted bonus token for the
    // all-accept seed. Without expansion, the tensors are set_output
    // but ggml's graph optimizer can DCE the upstream compute since
    // no graph node references them — leading to zero values being
    // read at extract time.
    for (int k = 0; k < n_chain; ++k) {
        ggml_build_forward_expand(gf, argmaxes[k]);
        ggml_build_forward_expand(gf, probs[k]);
    }
    // Phase 38 C: ensure all chain residuals (including extended ones)
    // are wired into the graph so post-compute extraction can capture
    // them. The chain_residual tensors for k in [0, n_chain) are
    // already set_output and referenced via prev_residual chain;
    // ggml_build_forward_expand on the LAST one ensures no DCE.
    if (lctx.default_decoder.mtp_fused_chain_residuals[n_chain - 1] != nullptr) {
        ggml_build_forward_expand(gf, lctx.default_decoder.mtp_fused_chain_residuals[n_chain - 1]);
    }

    return gf;
}
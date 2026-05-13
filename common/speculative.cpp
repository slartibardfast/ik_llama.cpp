#include "speculative.h"

#include "common.h"
#include "ggml.h"
#include "llama.h"
#include "llama-session.h"
#include "llama-decoder.h"
#include "llama-spec-loop.h"
#include "log.h"

extern "C" void probe_top2_push(int32_t t1, int32_t t2);
#include "ngram-cache.h"
#include "ngram-map.h"
#include "ngram-mod.h"
#include "sampling.h"
#include "suffix-tree.h"

#include <algorithm>
#include <cstring>
#include <iomanip>
#include <map>

// PHASE45 D9.x sweep: env-gate truthiness check (treats VAR=0 / VAR= as unset).
// See feedback_verify_test_mechanism_before_trusting auto-memory entry.
static inline bool env_truthy(const char * name) {
    const char * v = std::getenv(name);
    return v != nullptr && v[0] != '\0' && std::strcmp(v, "0") != 0;
}


#define SPEC_VOCAB_MAX_SIZE_DIFFERENCE  128
#define SPEC_VOCAB_CHECK_START_TOKEN_ID 5

const std::vector<enum common_speculative_type> common_speculative_types = {
    COMMON_SPECULATIVE_TYPE_NONE,
    COMMON_SPECULATIVE_TYPE_DRAFT,
    COMMON_SPECULATIVE_TYPE_MTP,
    COMMON_SPECULATIVE_TYPE_EAGLE3,
    COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE,
    COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K,
    COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V,
    COMMON_SPECULATIVE_TYPE_NGRAM_MOD,
    COMMON_SPECULATIVE_TYPE_NGRAM_CACHE,
    COMMON_SPECULATIVE_TYPE_SUFFIX,
    COMMON_SPECULATIVE_TYPE_DFLASH
};

const std::map<std::string, enum common_speculative_type> common_speculative_type_from_name_map = {
    {"none",          COMMON_SPECULATIVE_TYPE_NONE},
    {"draft",         COMMON_SPECULATIVE_TYPE_DRAFT},
    {"mtp",           COMMON_SPECULATIVE_TYPE_MTP},
    {"eagle3",        COMMON_SPECULATIVE_TYPE_EAGLE3},
    {"ngram_simple",  COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE},
    {"ngram_map_k",   COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K},
    {"ngram_map_k4v", COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V},
    {"ngram_mod",     COMMON_SPECULATIVE_TYPE_NGRAM_MOD},
    {"ngram_cache",   COMMON_SPECULATIVE_TYPE_NGRAM_CACHE},
    {"suffix",        COMMON_SPECULATIVE_TYPE_SUFFIX},
    {"dflash",        COMMON_SPECULATIVE_TYPE_DFLASH}
};

struct common_speculative_config {
    common_speculative_type type;
    common_params_speculative params;

    common_speculative_config(common_speculative_type t,
            const common_params_speculative & p = common_params_speculative{}) : type(t), params(p) {}
};

static bool common_speculative_are_compatible(
    const llama_model * model_tgt,
    const llama_model * model_dft) {
    const llama_vocab * vocab_tgt = llama_model_get_vocab(model_tgt);
    const llama_vocab * vocab_dft = llama_model_get_vocab(model_dft);

    const bool vocab_type_tgt = llama_vocab_type(vocab_tgt);
    LOG_DBG("%s: vocab_type tgt: %d\n", __func__, vocab_type_tgt);

    const bool vocab_type_dft = llama_vocab_type(vocab_dft);
    LOG_DBG("%s: vocab_type dft: %d\n", __func__, vocab_type_dft);

    if (vocab_type_tgt != vocab_type_dft) {
        LOG_DBG("%s: draft model vocab type must match target model to use speculation but ", __func__);
        LOG_DBG("vocab_type_dft = %d while vocab_type_tgt = %d\n", vocab_type_dft, vocab_type_tgt);
        return false;
    }

    if (
        llama_vocab_get_add_bos(vocab_tgt) != llama_vocab_get_add_bos(vocab_dft) ||
        llama_vocab_get_add_eos(vocab_tgt) != llama_vocab_get_add_eos(vocab_dft) ||
        llama_vocab_bos(vocab_tgt) != llama_vocab_bos(vocab_dft) ||
        llama_vocab_eos(vocab_tgt) != llama_vocab_eos(vocab_dft)
    ) {
        LOG_DBG("%s: draft model special tokens must match target model to use speculation\n", __func__);
        return false;
    }

    {
        const int n_vocab_tgt = llama_vocab_n_tokens(vocab_tgt);
        const int n_vocab_dft = llama_vocab_n_tokens(vocab_dft);
        const int vocab_diff  = n_vocab_tgt > n_vocab_dft
            ? n_vocab_tgt - n_vocab_dft
            : n_vocab_dft - n_vocab_tgt;

        if (vocab_diff > SPEC_VOCAB_MAX_SIZE_DIFFERENCE) {
            LOG_DBG("%s: draft model vocab must closely match target model to use speculation but ", __func__);
            LOG_DBG("target vocab size %d does not match draft vocab size %d - difference %d, max allowed %d\n",
                    n_vocab_tgt, llama_vocab_n_tokens(vocab_dft), vocab_diff, SPEC_VOCAB_MAX_SIZE_DIFFERENCE);
            return false;
        }

        for (int i = SPEC_VOCAB_CHECK_START_TOKEN_ID; i < std::min(n_vocab_tgt, n_vocab_dft); ++i) {
            const char * token_text_tgt = llama_vocab_get_text(vocab_tgt, i);
            const char * token_text_dft = llama_vocab_get_text(vocab_dft, i);

            if (std::strcmp(token_text_tgt, token_text_dft) != 0) {
                LOG_DBG("%s: draft model vocab must match target model to use speculation but ", __func__);
                LOG_DBG("token %d content differs - target '%s', draft '%s'\n", i,
                        common_token_to_piece(vocab_tgt, i).c_str(),
                        common_token_to_piece(vocab_dft, i).c_str());
                return false;
            }
        }
    }

    return true;
}

// state of an implementation of speculative decoding
//
// each implementation has a unique type and a state that is implementation-specific
// in a subclass of common_speculative_state
struct common_speculative_state {
    const enum common_speculative_type type;

    size_t n_call_begin  = 0; // number of times this implementation was called for refresh.
    size_t n_call_draft  = 0; // number of times this implementation was called for generation.
    size_t n_call_accept = 0; // number of times this implementation was called for accumulation.

    size_t n_gen_drafts = 0; // number of times a draft or part was generated by this implementation.
    size_t n_acc_drafts = 0; // number of times a draft or part was accepted by the target model.
    size_t n_gen_tokens = 0; // number of tokens generated by this implementation.
    size_t n_acc_tokens = 0; // number of tokens accepted by the target model.

    // TODO: track performance of most recent calls
    const bool gen_perf = true; // whether to generate performance stats.

    int64_t t_begin_us  = 0; // total time spent in refresh of this implementation in microseconds.
    int64_t t_draft_us  = 0; // total time spent in generating drafts in this implementation in microseconds.
    int64_t t_accept_us = 0; // total time spent in accumulation of this implementation in microseconds.

    common_speculative_state(enum common_speculative_type type) : type(type) {}

    virtual ~common_speculative_state() = default;

    virtual void begin(const llama_tokens & prompt) = 0;

    virtual void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) = 0;

    virtual void accept(uint16_t n_accepted) = 0;
};

struct common_speculative_state_mtp : public common_speculative_state {
    llama_context * ctx_tgt;
    llama_context * ctx_mtp = nullptr;  // PHASE45 D9.5: stays nullptr (no separate ctx)

    // PHASE45 D9.5: collapsed-context MTP. The MTP draft now writes layer
    // N-1 of the SHARED ctx_tgt's KV cache (per the architectural decision
    // in PHASE45.md "PHASE39 collapsed-context MTP wrapping"). No more
    // per-slot ctx_mtp. Per-slot wrappers stay (session_mtp, decoders,
    // loop) but adopt ctx_tgt rather than allocating a separate ctx.
    // Server's get_mtp_ctx callers fall back to ctx_tgt when nullptr is
    // returned (server-context.cpp:3831, 3879).
    //
    // seq_id is per-slot — passed in at construction (was hardcoded 0 in
    // D8.3 because each slot had its own ctx_mtp). The shared ctx
    // requires correct seq_id partitioning.
    llama_session   * session_mtp = nullptr;
    llama_decoder   * verify_dec  = nullptr;
    llama_decoder   * draft_dec   = nullptr;
    llama_spec_loop * loop        = nullptr;
    llama_seq_id      seq_id      = 0;

    common_speculative_state_mtp(
            enum common_speculative_type type,
            llama_context * ctx_tgt,
            const llama_context_params & /*mtp_cparams*/,
            llama_seq_id seq_id_)
        : common_speculative_state(type)
        , ctx_tgt(ctx_tgt)
        , seq_id(seq_id_)
    {
        // No more ctx_mtp allocation. Adopt the shared ctx_tgt directly.
        // mtp_cparams is unused (kept in signature for ABI stability of
        // the old common_speculative_init call sites until D9.6).
        session_mtp = llama_session_adopt(ctx_tgt);

        llama_decoder_params verify_p = llama_decoder_default_params(LLAMA_DECODER_VERIFY);
        llama_decoder_params draft_p  = llama_decoder_default_params(LLAMA_DECODER_DRAFT_MTP);
        verify_dec = llama_decoder_create(session_mtp, verify_p);
        draft_dec  = llama_decoder_create(session_mtp, draft_p);

        llama_spec_loop_params lp = llama_spec_loop_default_params();
        loop = llama_spec_loop_create(verify_dec, &draft_dec, 1, lp);

        LOG_INF("%s: collapsed-context MTP on shared ctx (seq_id=%d)\n",
                __func__, (int) seq_id);
    }

    ~common_speculative_state_mtp() override {
        if (loop)        llama_spec_loop_free(loop);
        if (draft_dec)   llama_decoder_free(draft_dec);
        if (verify_dec)  llama_decoder_free(verify_dec);
        if (session_mtp) llama_session_free(session_mtp);  // non-owning; doesn't touch ctx_tgt
        // ctx_mtp stays nullptr; nothing to free.
    }

    void begin(const llama_tokens & prompt) override {
        GGML_UNUSED(prompt);
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) override {

        const int32_t n_past = (int32_t)prompt_tgt.size();

        // Pre-clean any leftover draft cells from prior cycles. The
        // shared ctx_tgt's KV stores per-slot draft cells under
        // seq_id=this->seq_id (slot-specific).
        const llama_pos mtp_pos_max = llama_kv_cache_seq_pos_max(ctx_tgt, seq_id);
        if (mtp_pos_max >= n_past) {
            llama_kv_cache_seq_rm(ctx_tgt, seq_id, n_past, -1);
        }

        if (!loop) { result.clear(); return; }

        std::vector<llama_token> buf((size_t) std::max(0, params.n_max));
        const int32_t n = llama_spec_loop_gen_drafts(
                loop,
                id_last,
                params.p_min,
                params.n_max,
                seq_id,
                n_past,
                buf.data());

        if (n > 0) {
            buf.resize((size_t) n);
            result = std::move(buf);
        } else {
            result.clear();
        }
    }

    void accept(uint16_t n_accepted) override {
        GGML_UNUSED(n_accepted);
    }
};


struct common_speculative_state_draft : public common_speculative_state {
    llama_context * ctx_tgt; // only used for retokenizing from ctx_dft
    llama_context * ctx_dft;

    common_sampler * smpl;

    llama_batch  batch;
    llama_tokens prompt_dft;

    bool vocab_cmpt = true; // whether retokenization is needed
    std::unordered_map<std::string, std::string> vocab_map;

    common_speculative_state_draft(
            enum common_speculative_type type,
            llama_context * ctx_tgt,
            llama_context * ctx_dft,
            const std::vector<std::pair<std::string, std::string>> & replacements)
        : common_speculative_state(type)
        , ctx_tgt(ctx_tgt)
        , ctx_dft(ctx_dft)
    {
        batch = llama_batch_init(llama_n_batch(ctx_dft), 0, 1);
        smpl = nullptr;
        {
            struct common_params_sampling params;
            params.top_k = 10;
            params.samplers_sequence = {
                llama_sampler_type::TOP_K,
                llama_sampler_type::DIST, // needed to get probabilities
            };
            smpl = common_sampler_init(llama_get_model(ctx_dft), params);
        }

        vocab_cmpt = common_speculative_are_compatible(llama_get_model(ctx_tgt), llama_get_model(ctx_dft));
        LOG_DBG("vocab_cmpt = %d\n", vocab_cmpt);

        if (!vocab_cmpt) {
            LOG_WRN("the target and draft vocabs are not compatible - tokens will be translated between the two\n");

            for (const auto & pair : replacements) {
                vocab_map[pair.first] = pair.second;
            }
        }
    }

    ~common_speculative_state_draft() override {
        llama_free(ctx_dft);

        common_sampler_free(smpl);

        llama_batch_free(batch);
    }

    void begin(const llama_tokens & prompt) override {
        GGML_UNUSED(prompt);
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) override {
        auto * spec = this;

        auto & batch      = spec->batch;
        auto & ctx_tgt    = spec->ctx_tgt;
        auto & ctx_dft    = spec->ctx_dft;
        auto & smpl       = spec->smpl;
        auto & prompt_dft = spec->prompt_dft;

        int reuse_i = 0;
        int reuse_n = 0;

        const int n_ctx = llama_n_ctx(ctx_dft) - params.n_max;

        llama_tokens prompt_cnv;
        if (!spec->vocab_cmpt) {
            // convert id_last to draft vocab. llama_detokenize is called directly to avoid an allocation
            const auto * model_tgt = llama_get_model(ctx_tgt);
            const auto * vocab_tgt = llama_model_get_vocab(model_tgt);

            std::string text;

            text = common_detokenize(ctx_tgt, prompt_tgt, true);
            text = replace_to_dft(text);

            LOG_DBG("%s: main->draft detokenized string: '%s'\n", __func__, text.c_str());

            prompt_cnv = common_tokenize(ctx_dft, text, false, true);



            int32_t n_chars = llama_detokenize(vocab_tgt, &id_last, 1, nullptr, 0, false, false);
            GGML_ASSERT(n_chars < 0 && "failed to detokenize id_last");

            text.resize(-n_chars);
            llama_detokenize(vocab_tgt, &id_last, 1, text.data(), text.size(), false, false);
            text = replace_to_dft(text);

            LOG_DBG("main->draft detokenized id_last(%d): '%s'\n", id_last, text.c_str());
            id_last = common_tokenize(ctx_dft, text, false, true)[0];
        }

        const llama_tokens & prompt_cur = spec->vocab_cmpt ? prompt_tgt : prompt_cnv;

        const int i_start = std::max<int>(0, (int) prompt_cur.size() - n_ctx);

        // reuse as much as possible from the old draft context
        // ideally, the draft context should be as big as the target context and we will always reuse the entire prompt
        for (int i = 0; i < (int) prompt_dft.size(); ++i) {
            int cur = 0;
            while (i_start + cur < (int) prompt_cur.size() &&
                    i       + cur < (int) prompt_dft.size() &&
                    prompt_cur[i_start + cur] == prompt_dft[i + cur]) {
                cur++;
            }

            if ((cur >= 256 || n_ctx >= (int) prompt_cur.size()) && cur > reuse_n) {
                reuse_i = i;
                reuse_n = cur;
            }
        }

        LOG_DBG("%s: reuse_i = %d, reuse_n = %d, prompt = %d\n", __func__, reuse_i, reuse_n, (int) prompt_dft.size());

        result.clear();
        result.reserve(params.n_max);

        if (reuse_n == 0) {
            llama_kv_cache_clear(ctx_dft);
            prompt_dft.clear();
        } else {
            // this happens when a previous draft has been discarded (for example, due to being too small), but the
            // target model agreed with it. in this case, we simply pass back the previous results to save compute
            if (reuse_i + reuse_n < (int) prompt_dft.size() && prompt_dft[reuse_i + reuse_n] == id_last) {
                for (int i = reuse_i + reuse_n + 1; i < (int) prompt_dft.size(); ++i) {
                    result.push_back(prompt_dft[i]);

                    if (params.n_max <= (int) result.size()) {
                        break;
                    }
                }

                return;
            }

            if (reuse_i > 0) {
                llama_kv_cache_seq_rm (ctx_dft, 0, 0, reuse_i);
                llama_kv_cache_seq_add(ctx_dft, 0, reuse_i, -1, -reuse_i);

                prompt_dft.erase(prompt_dft.begin(), prompt_dft.begin() + reuse_i);
            }

            if (reuse_n < (int) prompt_dft.size()) {
                llama_kv_cache_seq_rm (ctx_dft, 0, reuse_n, -1);
                prompt_dft.erase(prompt_dft.begin() + reuse_n, prompt_dft.end());
            }
        }

        // prepare a batch to evaluate any new tokens in the prompt
        common_batch_clear(batch);

        for (size_t i = i_start + reuse_n; i < prompt_cur.size(); ++i) {
            //LOG_DBG("i = %d, i_start = %d, reuse_n = %d, i - i_start = %d, id = %6d\n", i, i_start, reuse_n, i - i_start, prompt_cur[i]);
            common_batch_add(batch, prompt_cur[i], i - i_start, { 0 }, false);

            prompt_dft.push_back(prompt_cur[i]);
        }

        // we should rarely end-up here during normal decoding
        if (batch.n_tokens > 0) {
            //LOG_DBG("%s: draft prompt batch: %s\n", __func__, string_from(ctx, batch).c_str());

            llama_decode(ctx_dft, batch);
        }

        const llama_pos n_past = prompt_dft.size();

        LOG_DBG("%s: n_past = %d\n", __func__, n_past);

        common_batch_clear(batch);
        common_batch_add  (batch, id_last, n_past, { 0 }, true);

        prompt_dft.push_back(id_last);

        //LOG_DBG("%s: draft prompt: %s\n", __func__, string_from(ctx_dft, prompt_dft).c_str());

        llama_decode(ctx_dft, batch);

        common_sampler_reset(smpl);

        // sample n_draft tokens from the draft model
        for (int i = 0; i < params.n_max; ++i) {
            common_batch_clear(batch);

            common_sampler_sample(smpl, ctx_dft, 0, true);

            const auto * cur_p = common_sampler_get_candidates(smpl, true);

            for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                LOG_DBG(" - draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                        k, i, cur_p->data[k].id, cur_p->data[k].p, common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
            }

            // add drafted token for each sequence
            const llama_token id = cur_p->data[0].id;

            common_sampler_accept(smpl, nullptr, id, true);

            result.push_back(id);

            if (params.n_max <= (int) result.size()) {
                break;
            }

            // only collect very high-confidence draft tokens
            if (cur_p->data[0].p < params.p_min) {
                break;
            }

            common_batch_add(batch, id, n_past + i + 1, { 0 }, true);

            // evaluate the drafted tokens on the draft model
            llama_decode(ctx_dft, batch);

            prompt_dft.push_back(id);
        }

        if (!spec->vocab_cmpt) {
            std::string detokenized = common_detokenize(ctx_dft, result, true);
            detokenized = replace_to_tgt(detokenized);
            LOG_DBG("draft->main detokenized string: '%s'\n", detokenized.c_str());
            result = common_tokenize(ctx_tgt, detokenized, false, true);
            if (result.size() > (size_t)params.n_max) {
                result.resize(params.n_max);
            }
        }
    }

    void accept(uint16_t n_accepted) override {
        // noop
        GGML_UNUSED(n_accepted);
    }

    std::string replace_to_dft(const std::string & input) const {
        std::string result = input;

        for (const auto & pair : this->vocab_map) {
            size_t pos = result.find(pair.first);
            while (pos != std::string::npos) {
                result.replace(pos, pair.first.length(), pair.second);
                pos = result.find(pair.first, pos + pair.second.length());
            }
        }

        return result;
    }

    std::string replace_to_tgt(const std::string & input) const {
        std::string result = input;

        for (const auto & pair : this->vocab_map) {
            size_t pos = result.find(pair.second);
            while (pos != std::string::npos) {
                result.replace(pos, pair.second.length(), pair.first);
                pos = result.find(pair.second, pos + pair.first.length());
            }
        }

        return result;
    }
};

struct common_speculative_state_eagle3 : public common_speculative_state {
    common_speculative_state_eagle3(enum common_speculative_type type) : common_speculative_state(type) {}

    void begin(const llama_tokens & prompt) override {
        GGML_UNUSED(prompt);
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & draft_tokens) override {
        // TODO: implement
        GGML_UNUSED(params);
        GGML_UNUSED(prompt_tgt);
        GGML_UNUSED(id_last);
        GGML_UNUSED(draft_tokens);
    }

    void accept(uint16_t n_accepted) override {
        // noop
        GGML_UNUSED(n_accepted);
    }
};

// state of self-speculation (simple implementation, not ngram-map)
struct common_speculative_state_ngram_simple : public common_speculative_state {
    common_ngram_simple_config config;

    common_speculative_state_ngram_simple(
            enum common_speculative_type type,
            common_ngram_simple_config config)
        : common_speculative_state(type), config(config) {}

    void begin(const llama_tokens & prompt) override {
        GGML_UNUSED(prompt);
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) override {

        result = common_ngram_simple_draft(config, prompt_tgt, id_last);
        GGML_UNUSED(params);
    }

    void accept(uint16_t n_accepted) override {
        // noop
        GGML_UNUSED(n_accepted);
    }
};

struct common_speculative_state_ngram_map_k : public common_speculative_state {
    // draft ngram map for speculative decoding without draft model
    common_ngram_map map;

    common_speculative_state_ngram_map_k(
            enum common_speculative_type type,
            common_ngram_map map)
        : common_speculative_state(type), map(std::move(map)) {}

    void begin(const llama_tokens & prompt) override {
        common_ngram_map_begin(map, prompt);
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) override {
        common_ngram_map_draft(map, prompt_tgt, id_last, result);
        GGML_UNUSED(params);
    }

    void accept(uint16_t n_accepted) override {
        common_ngram_map_accept(map, n_accepted);
    }
};

struct common_speculative_state_ngram_mod : public common_speculative_state {
    common_ngram_mod & mod;

    // the last position in the prompt that was added to the ngram container
    size_t i_last = 0;

    // length of the last drafted n‑gram (number of tokens returned by draft)
    size_t n_draft_last = 0;

    // consecutive accept rounds with low acceptance fraction (< 0.5)
    int n_low = 0;

    // enable trace logging if LLAMA_TRACE is set
    const bool verbose;

    common_speculative_state_ngram_mod(enum common_speculative_type type, common_ngram_mod & mod)
        : common_speculative_state(type), mod(mod), verbose(env_truthy("LLAMA_TRACE")) {
        static_assert(sizeof(llama_token) == sizeof(common_ngram_mod::entry_t));
    }

    void begin(const llama_tokens & prompt) override {
        i_last = 0;

        n_draft_last = 0;
        n_low = 0;

        const size_t n = mod.get_n();

        if (prompt.size() < n) {
            return;
        }

        for (size_t i = 0; i < prompt.size() - n; ++i) {
            mod.add(prompt.data() + i);
        }

        i_last = prompt.size() - n;

        const double f = (double)mod.get_used() / (double)mod.size();
        LOG_INF("%s: ngram_mod occupancy = %zu/%zu (%.2f)\n", __func__, mod.get_used(), mod.size(), f);

        constexpr double f_thold = 0.25;
        if (f > f_thold) {
            LOG_WRN("%s: ngram_mod occupancy %.2f exceeds threshold (%.2f) - resetting\n", __func__, f, f_thold);

            mod.reset();
        }
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) override {
        GGML_UNUSED(params);

        n_draft_last = 0;

        const size_t cur_len = prompt_tgt.size();
        if (cur_len < mod.get_n()) {
            return;
        }

        const size_t n = mod.get_n();

        // add new ngrams in chunks
        if (i_last + 32 < cur_len) {
            for (size_t i = i_last; i < cur_len - n; ++i) {
                mod.add(prompt_tgt.data() + i);
            }

            i_last = cur_len - n;
        }

        result.resize(n + params.n_max);
        for (size_t i = 0; i < n - 1; ++i) {
            result[i] = prompt_tgt[cur_len - n + 1 + i];
        }
        result[n - 1] = id_last;

        for (int i = 0; i < params.n_max; ++i) {
            const llama_token token = mod.get(result.data() + i);
            if (token == common_ngram_mod::EMPTY) {
                if (i < params.n_min) {
                    result.clear();
                    return;
                }

                result.resize(n + i);
                break;
            }
            result[n + i] = token;
        }

        // only return the m tokens that were drafted
        for (size_t i = 0; n + i < result.size(); ++i) {
            result[i] = result[n + i];
        }
        result.resize(result.size() - n);

        // store length of drafted n‑gram for later acceptance analysis
        n_draft_last = result.size();
    }

    void accept(uint16_t n_accepted) override {
        if (verbose) {
            LOG_INF("%s: accepted %d tokens from %zu drafted tokens\n", __func__, n_accepted, n_draft_last);
        }

        // compute acceptance fraction if we have a recorded draft length
        if (n_draft_last > 0) {
            const double f_acc = (double)n_accepted / (double)n_draft_last;
            if (f_acc < 0.5) {
                n_low++;
                if (n_low >= 3) {
                    LOG_WRN("%s: low acceptance streak (%d) – resetting ngram_mod\n", __func__, n_low);

                    mod.reset();
                    n_low = 0;
                    i_last = 0;
                }
            } else {
                n_low = 0;
            }
        }
    }
};

struct common_speculative_state_ngram_cache : public common_speculative_state {
    uint16_t n_draft;
    bool save_dynamic;
    bool save_static;

    common_ngram_cache ngram_cache_context;
    common_ngram_cache ngram_cache_dynamic;
    common_ngram_cache ngram_cache_static;

    size_t cache_size = 0; // number of tokens in n-gram cache

    common_speculative_state_ngram_cache(
            const enum common_speculative_type type,
            const std::string & path_static,
            const std::string & path_dynamic,
            uint16_t            n_draft,
            bool                save_dynamic,
            bool                save_static)
        : common_speculative_state(type)
        , n_draft(n_draft)
        , save_dynamic(save_dynamic)
        , save_static(save_static)
    {
        if (!path_static.empty()) {
            try {
                ngram_cache_static = common_ngram_cache_load(path_static);
            } catch (...) {
                LOG_ERR("failed to open static lookup cache: %s", path_static.c_str());
                GGML_ABORT("Couldn't read static lookup cache");
            }
        }

        if (!path_dynamic.empty()) {
            try {
                ngram_cache_dynamic = common_ngram_cache_load(path_dynamic);
            } catch (...) {
                LOG_ERR("failed to open dynamic lookup cache: %s", path_dynamic.c_str());
                GGML_ABORT("Couldn't read dynamic lookup cache");
            }
        }
    }

    void begin(const llama_tokens & prompt) override {
        GGML_UNUSED(prompt);
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) override {
        GGML_UNUSED(params);

        if (cache_size < prompt_tgt.size() + 1) {
            llama_tokens tokens_new;
            tokens_new.reserve(prompt_tgt.size() + 1 - cache_size);
            for (size_t j = cache_size; j < prompt_tgt.size(); ++j) {
                tokens_new.push_back(prompt_tgt[j]);
            }
            tokens_new.push_back(id_last); // add the last token

            // Update context ngram cache with new prompt_tgt:
            common_ngram_cache_update(ngram_cache_context, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX,
                    tokens_new, tokens_new.size(), false);
            cache_size = prompt_tgt.size() + 1;
        }

        llama_tokens inp;
        inp.reserve(prompt_tgt.size() + 1);
        for (size_t j = 0; j < prompt_tgt.size(); ++j) {
            inp.push_back(prompt_tgt[j]);
        }
        inp.push_back(id_last);

        result.push_back(id_last);

        common_ngram_cache_draft(inp, result, n_draft, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX,
                ngram_cache_context,
                ngram_cache_dynamic,
                ngram_cache_static);

        if (result.size() > 0) {
            // delete first token in result (which is the id_last token)
            result.erase(result.begin());
        }
    }

    void accept(uint16_t n_accepted) override {
        // TODO: noop
        GGML_UNUSED(n_accepted);
    }
};

struct common_speculative_state_suffix : public common_speculative_state {
    common_suffix_tree tree;
    common_suffix_tree corpus_tree;
    bool has_corpus = false;
    size_t cache_size   = 0;

    // Acceptance feedback
    size_t n_draft_last  = 0;
    bool   had_accept    = false;
    int    n_low         = 0;
    float  base_p_min    = 0.1f;
    float  eff_p_min     = 0.1f;

    common_speculative_state_suffix(
            enum common_speculative_type type,
            int max_depth,
            const std::string & corpus_path,
            const llama_model * model)
        : common_speculative_state(type)
        , tree(max_depth)
        , corpus_tree(max_depth)
    {
        if (!corpus_path.empty()) {
            std::function<std::vector<llama_token>(const std::string &)> tokenize_fn;
            if (model) {
                tokenize_fn = [model](const std::string & text) -> std::vector<llama_token> {
                    return common_tokenize(model, text, false, true);
                };
            }
            has_corpus = corpus_tree.load_corpus(corpus_path, tokenize_fn);
        }
    }

    void begin(const llama_tokens & prompt) override {
        cache_size   = 0;
        n_draft_last = 0;
        had_accept   = false;
        n_low        = 0;
        GGML_UNUSED(prompt);
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) override {

        base_p_min = params.p_min;
        if (n_draft_last > 0 && !had_accept) {
            if (++n_low >= 3) {
                eff_p_min = std::min(eff_p_min + 0.1f, 0.5f);
                n_low     = 0;
            }
        }
        had_accept = false;

        if (cache_size < prompt_tgt.size() + 1) {
            llama_tokens tokens_new;
            tokens_new.reserve(prompt_tgt.size() + 1 - cache_size);
            for (size_t j = cache_size; j < prompt_tgt.size(); ++j) {
                tokens_new.push_back(prompt_tgt[j]);
            }
            tokens_new.push_back(id_last);

            tree.extend(tokens_new.data(), (int)tokens_new.size());
            cache_size = prompt_tgt.size() + 1;
        }

        const int ctx_len = std::min((int)(prompt_tgt.size() + 1), tree.max_depth());
        llama_tokens context;
        context.reserve(ctx_len);
        const int ctx_start = (int)prompt_tgt.size() + 1 - ctx_len;
        for (int j = ctx_start; j < (int)prompt_tgt.size(); ++j) {
            context.push_back(prompt_tgt[j]);
        }
        context.push_back(id_last);
        const int min_match_len = std::max(1, params.suffix_min_match_len);

        result = tree.speculate(
            context.data(), (int)context.size(),
            params.n_max,
            eff_p_min,
            1,
            min_match_len);

        if (has_corpus) {
            auto corpus_result = corpus_tree.speculate(
                context.data(), (int)context.size(),
                params.n_max,
                eff_p_min,
                1,
                min_match_len);
            if (corpus_result.size() > result.size()) {
                result = std::move(corpus_result);
            }
        }

        n_draft_last = result.size();
    }

    void accept(uint16_t n_accepted) override {
        if (n_draft_last == 0) {
            return;
        }
        had_accept = true;
        const double f_acc = (double)n_accepted / (double)n_draft_last;
        if (f_acc < 0.5) {
            if (++n_low >= 3) {
                eff_p_min = std::min(eff_p_min + 0.1f, 0.5f);
                n_low     = 0;
            }
        } else {
            n_low = 0;
            if (eff_p_min > base_p_min) {
                eff_p_min = std::max(eff_p_min - 0.05f, base_p_min);
            }
        }
    }
};

// DFlash sidecar drafter state. Wraps llama_dflash_drafter (owned) and
// dispatches into llama_dflash_draft per draft call. The C API in
// include/llama.h handles all kernel orchestration; this state subclass
// is just a thin adapter onto the common_speculative_state contract.
//
// Spec: specs/dflash/dflash.allium
struct common_speculative_state_dflash : public common_speculative_state {
    llama_context             * ctx_tgt = nullptr;
    llama_dflash_drafter      * drafter = nullptr;
    int32_t                     block_size = 0;

    common_speculative_state_dflash(
            enum common_speculative_type type,
            llama_context              * ctx_tgt_,
            const std::string          & drafter_path)
        : common_speculative_state(type)
        , ctx_tgt(ctx_tgt_)
    {
        drafter = llama_dflash_drafter_load(drafter_path.c_str());
        if (!drafter) {
            LOG_ERR("%s: failed to load DFlash drafter from '%s'\n",
                    __func__, drafter_path.c_str());
            return;
        }
        block_size = llama_dflash_block_size(drafter);
        int32_t rc = llama_set_dflash(ctx_tgt, drafter);
        if (rc != LLAMA_DFLASH_OK) {
            LOG_ERR("%s: llama_set_dflash failed: rc=%d\n", __func__, rc);
            llama_dflash_drafter_free(drafter);
            drafter = nullptr;
            return;
        }
        LOG_INF("%s: DFlash sidecar drafter bound to ctx_tgt (block_size=%d)\n",
                __func__, block_size);
    }

    ~common_speculative_state_dflash() override {
        // ctx_tgt's destructor releases the per-context DFlash state
        // (llama_dflash_release_ctx_state). We free only the drafter
        // we loaded here.
        if (drafter) llama_dflash_drafter_free(drafter);
    }

    void begin(const llama_tokens & prompt) override {
        // Target prefill drives the cb_eval extract hook installed by
        // llama_set_dflash; nothing extra to do here.
        GGML_UNUSED(prompt);
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) override {
        if (!drafter) {
            result.clear();
            return;
        }
        const int32_t n_max = std::min<int32_t>(params.n_max, block_size);
        if (n_max <= 0) {
            result.clear();
            return;
        }

        // Sized to the drafter's max block size so llama_dflash_draft's
        // bounds check passes; the kernel's operating BS is set in
        // llama-dflash.cpp and may be less.
        result.assign(block_size, 0);
        // anchor_pos = prompt_tgt.size() — the seq position where id_last
        // sits when fed into the target verify decode for this cycle.
        const int32_t rc = llama_dflash_draft(
                ctx_tgt, id_last, (int32_t) prompt_tgt.size(),
                result.data(), (int32_t) result.size());
        if (rc < 0) {
            LOG_WRN("%s: llama_dflash_draft failed: rc=%d (giving up this cycle)\n", __func__, rc);
            result.clear();
            return;
        }
        // rc is the number of candidates actually written.
        result.resize((size_t) rc);
        // Caller's n_max may be less than the kernel's operating BS;
        // truncate the returned candidate list rather than over-drafting.
        if ((int32_t) result.size() > n_max) {
            result.resize(n_max);
        }
        n_gen_drafts += 1;
        n_gen_tokens += result.size();
    }

    void accept(uint16_t n_accepted) override {
        n_call_accept += 1;
        n_acc_drafts  += (n_accepted > 0) ? 1 : 0;
        n_acc_tokens  += n_accepted;
    }
};

struct common_speculative {
    std::vector<std::unique_ptr<common_speculative_state>> impls; // list of implementations to use and their states
    common_speculative_state * curr_impl = nullptr; // current implementation in use (for stats)
    std::unique_ptr<spec_tuner> tuner;
    int last_n_drafted = 0;
    int64_t t_step_start_us = 0;
};

static common_ngram_map get_common_ngram_map(const common_speculative_config & config) {
    uint16_t size_key   = config.params.ngram_size_n;
    uint16_t size_value = config.params.ngram_size_m;
    bool     key_only   = (config.type == COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K);
    uint16_t min_hits   = config.params.ngram_min_hits;

    return common_ngram_map(size_key, size_value, key_only, min_hits);
}

static common_speculative_state_ngram_cache create_state_ngram_cache(
        const std::string & path_static, const std::string & path_dynamic,
        const common_speculative_config & config) {
    uint16_t n_draft = 8; // TODO get from config?

    // TODO bool param in common/common.h to set save_static/save_dynamic?
    bool save_static = false;
    bool save_dynamic = false;

    common_speculative_state_ngram_cache state(config.type, path_static, path_dynamic, n_draft, save_static, save_dynamic);

    return state;
}

std::string common_speculative_type_name_str() {
    std::string result;
    for (size_t i = 0; i < common_speculative_types.size(); i++) {
        if (i > 0) {
            result += ", ";
        }
        result += common_speculative_type_to_str(common_speculative_types[i]);
    }
    return result;
}

std::string common_speculative_type_to_str(enum common_speculative_type type) {
    switch (type) {
        case COMMON_SPECULATIVE_TYPE_NONE:          return "none";
        case COMMON_SPECULATIVE_TYPE_DRAFT:         return "draft";
        case COMMON_SPECULATIVE_TYPE_MTP:           return "mtp";
        case COMMON_SPECULATIVE_TYPE_EAGLE3:        return "eagle3";
        case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE:  return "ngram_simple";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K:   return "ngram_map_k";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V: return "ngram_map_k4v";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MOD:     return "ngram_mod";
        case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE:   return "ngram_cache";
        case COMMON_SPECULATIVE_TYPE_SUFFIX:        return "suffix";
        case COMMON_SPECULATIVE_TYPE_DFLASH:        return "dflash";
        default:                                    return "unknown";
    }
}

enum common_speculative_type common_speculative_type_from_name(const std::string & name) {
    const auto it = common_speculative_type_from_name_map.find(name);
    if (it == common_speculative_type_from_name_map.end()) {
        return COMMON_SPECULATIVE_TYPE_COUNT;
    }
    return it->second;
}

bool common_speculative_is_compat(llama_context * ctx_tgt) {
    bool res = true;

    llama_kv_cache_clear(ctx_tgt);

    // eval 2 tokens to check if the context is compatible
    std::vector<llama_token> tmp;
    tmp.push_back(0);
    tmp.push_back(0);

    int ret = llama_decode(ctx_tgt, llama_batch_get_one(tmp.data(), tmp.size(), 0, 0));
    if (ret != 0) {
        LOG_ERR("%s: llama_decode() failed: %d\n", __func__, ret);
        res = false;
        goto done;
    }

    // try to remove the last tokens
    if (!llama_kv_cache_seq_rm(ctx_tgt, 0, 1, -1)) {
        LOG_WRN("%s: the target context does not support partial sequence removal\n", __func__);
        res = false;
        goto done;
    }

done:
    llama_kv_cache_clear(ctx_tgt);
    llama_synchronize(ctx_tgt);

    return res;
}

// initialization of the speculative decoding system
//
common_speculative * common_speculative_init(
        common_params_speculative & params,
        llama_context             * ctx_tgt,
        llama_seq_id                seq_id) {
    llama_context * ctx_dft = nullptr;
    if (params.model_dft) {
        ctx_dft = llama_init_from_model(params.model_dft, params.cparams_dft);
        if (ctx_dft == nullptr) {
            LOG_ERR("%s", "failed to create draft context\n");
            return nullptr;
        }
    }

    // Compute the implementations to use based on the config and their order of preference
    std::vector<common_speculative_config> configs = {}; // list of speculative configs to try
    {
        bool has_draft = !params.mparams_dft.path.empty();
        bool has_draft_eagle3 = false; // TODO PR-18039: if params.speculative.eagle3
        bool has_mtp = (params.type == COMMON_SPECULATIVE_TYPE_MTP);

        bool has_ngram_cache   = (params.type == COMMON_SPECULATIVE_TYPE_NGRAM_CACHE);
        bool has_ngram_simple  = (params.type == COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE);
        bool has_ngram_map_k   = (params.type == COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K);
        bool has_ngram_map_k4v = (params.type == COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V);
        bool has_ngram_mod     = (params.type == COMMON_SPECULATIVE_TYPE_NGRAM_MOD);
        bool has_suffix        = (params.type == COMMON_SPECULATIVE_TYPE_SUFFIX);
        bool has_dflash        = (params.type == COMMON_SPECULATIVE_TYPE_DFLASH);
        if (has_dflash) {
            // Prefer mparams_dft.path; fall back to legacy params.model string.
            if (params.mparams_dft.path.empty() && params.model.empty()) {
                LOG_ERR("%s: DFlash requires a drafter GGUF path (--model-draft or speculative.model)\n", __func__);
                return nullptr;
            }
            // DFlash supersedes the generic "draft model" path — both flags
            // are mutually exclusive at the loader level.
            has_draft = false;
        }

        // In a more complex implementation we could use the same implementation but with different parameters.
        // This was initially used in PR-18471 but removed to simplify the code.
        if (has_ngram_simple) {
            // This implementation can guess a lot of tokens without any draft model.
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE, params));
        }
        if (has_ngram_map_k) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K, params));
        }
        if (has_ngram_map_k4v) {
            // This implementation can guess tokens with high acceptance rate but is more expensive.
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V, params));
        }
        if (has_ngram_mod) {
            // shared instance for all speculative decoding contexts
            if (!params.ngram_mod) {
                params.ngram_mod = std::make_shared<common_ngram_mod>(params.ngram_size_n, 4*1024*1024);

                LOG_INF("%s: initialized ngram_mod with n=%d, size=%zu (%.3f MB)\n", __func__,
                        params.ngram_size_n, params.ngram_mod->size(),
                        (float)(params.ngram_mod->size_bytes())/1024/1024);

                if (params.ngram_size_n < 16) {
                    LOG_WRN("%s: ngram_mod n=%d is too small - poor quality is possible, see: https://github.com/ggml-org/llama.cpp/pull/19164\n", __func__, params.ngram_size_n);
                }
            }

            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_MOD, params));
        }
        if (has_ngram_cache) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_CACHE, params));
        }
        if (has_suffix) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_SUFFIX, params));
        }
        if (has_mtp) {
             configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_MTP, params));
        }
        if (has_dflash) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_DFLASH, params));
        }
        if (has_draft) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_DRAFT, params));
        }
        if (has_draft_eagle3) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_EAGLE3, params));
        }
    }

    std::vector<std::unique_ptr<common_speculative_state>> impls = {};

    for (const common_speculative_config & config : configs) {
        LOG_DBG("%s: adding implementation %s\n", __func__, common_speculative_type_to_str(config.type).c_str());
        switch (config.type) {
            case COMMON_SPECULATIVE_TYPE_NONE:
                break;
            case COMMON_SPECULATIVE_TYPE_DRAFT: {
                impls.push_back(std::make_unique<common_speculative_state_draft>(config.type,
                    /* .ctx_tgt      = */ ctx_tgt,
                    /* .ctx_dft      = */ ctx_dft,
                    /* .replacements = */ params.replacements
                ));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_MTP: {
                auto mtp_state = std::make_unique<common_speculative_state_mtp>(config.type,
                    /* .ctx_tgt      = */ ctx_tgt,
                    /* .mtp_cparams  = */ params.cparams_dft,
                    /* .seq_id       = */ seq_id
                );
                // PHASE45 D9.5: ctx_mtp is intentionally nullptr (no
                // separate MTP context; the shared ctx_tgt is used).
                // Verify the spec_loop wrapper succeeded instead.
                if (!mtp_state->loop) {
                    LOG_ERR("%s: failed to create spec_loop on shared ctx\n", __func__);
                    return nullptr;
                }
                impls.push_back(std::move(mtp_state));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_EAGLE3: {
                impls.push_back(std::make_unique<common_speculative_state_eagle3>(config.type));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_DFLASH: {
                const std::string & drafter_path = !params.mparams_dft.path.empty()
                    ? params.mparams_dft.path
                    : params.model;
                auto state = std::make_unique<common_speculative_state_dflash>(
                    config.type, ctx_tgt, drafter_path);
                if (!state->drafter) {
                    LOG_ERR("%s: failed to construct DFlash sidecar state\n", __func__);
                    return nullptr;
                }
                impls.push_back(std::move(state));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE: {
                common_ngram_map ngram_map = get_common_ngram_map(config);

                uint16_t ngram_size_key   = ngram_map.size_key;
                uint16_t mgram_size_value = ngram_map.size_value;

                auto config_simple = common_ngram_simple_config {
                    /* .size_ngram      = */ ngram_size_key,
                    /* .size_mgram      = */ mgram_size_value
                };
                auto state = std::make_unique<common_speculative_state_ngram_simple>(
                    /* .type            = */ config.type,
                    /* .state           = */ config_simple
                );
                impls.push_back(std::move(state));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K:
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V: {
                impls.push_back(std::make_unique<common_speculative_state_ngram_map_k>(
                    (config.type),
                    get_common_ngram_map(config)
                ));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MOD: {
                GGML_ASSERT(config.params.ngram_mod);
                impls.push_back(std::make_unique<common_speculative_state_ngram_mod>(config.type, *config.params.ngram_mod));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE: {
                auto state = create_state_ngram_cache(
                        params.lookup_cache_static, params.lookup_cache_dynamic, config);
                impls.push_back(std::make_unique<common_speculative_state_ngram_cache>(state));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_SUFFIX: {
                int depth = config.params.suffix_max_depth > 0 ? config.params.suffix_max_depth : 64;
                const llama_model * model = llama_get_model(ctx_tgt);
                impls.push_back(std::make_unique<common_speculative_state_suffix>(
                    config.type, depth, config.params.suffix_corpus, model));
                break;
            }
            default:
                break;
        }
    }

    if (impls.empty()) {
        LOG_WRN("%s", "no implementations specified for speculative decoding\n");
        return nullptr;
    }

    auto * result = new common_speculative {
        /* .impls = */ std::move(impls)
    };

    // initialize autotune if requested
    if (params.autotune && !result->impls.empty()) {
        auto actual_type = result->impls[0]->type;
        if (actual_type != COMMON_SPECULATIVE_TYPE_NONE &&
            actual_type != COMMON_SPECULATIVE_TYPE_EAGLE3) {
            result->tuner = std::make_unique<spec_tuner>();
            result->tuner->init(actual_type, params);
            LOG_DBG("Autotune initialized for %s, tuning %zu parameters\n",
                    common_speculative_type_to_str(actual_type).c_str(),
                    result->tuner->coords.size());
        } else {
            LOG_WRN("Autotune disabled — speculative type %s is not supported for autotuning\n",
                    common_speculative_type_to_str(actual_type).c_str());
        }
    }

    return result;
}

void common_speculative_free(common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    delete spec;
}

void common_speculative_begin(common_speculative * spec, const llama_tokens & prompt) {
    if (spec == nullptr) {
        return;
    }

    for (auto & impl : spec->impls) {
        common_time_meas tm(impl->t_begin_us, !impl->gen_perf);
        impl->begin(prompt);
        impl->n_call_begin++;
    }
}

llama_tokens common_speculative_draft(
        common_speculative * spec,
        common_params_speculative & params,
        const llama_tokens & prompt_tgt, // specified in target model vocab
        llama_token id_last) {
    llama_tokens result;

    spec->t_step_start_us = ggml_time_us();

    // apply autotune proposal if enabled
    if (spec->tuner && spec->tuner->enabled) {
        spec->tuner->propose(params);
    }

    spec->curr_impl = nullptr; // reset current implementation

    for (auto & impl : spec->impls) {
        {
            common_time_meas tm(impl->t_draft_us, !impl->gen_perf);
            impl->draft(params, prompt_tgt, id_last, result);
            impl->n_call_draft++;
        }

        if (!result.empty()) {
            LOG_DBG("%s: called impl %s, hist size = %zu, call_count = %zu, gen = %zu\n", __func__,
                    common_speculative_type_to_str(impl.get()->type).c_str(), prompt_tgt.size(),
                    impl.get()->n_call_draft, result.size());

            spec->curr_impl = impl.get(); // set current implementation for stats
            impl->n_gen_drafts++;
            impl->n_gen_tokens += result.size();

            break; // We have a draft, so break out of the loop and return it.
        }
    }

    // store draft count for tuner feedback
    if (spec->tuner && spec->tuner->enabled) {
        spec->last_n_drafted = (int)result.size();
    }

    return result;
}

// PHASE45 D10.b: batched-draft entry point.
//
// All slots share the post-D9.5 collapsed-context invariant: each
// per-slot common_speculative_state_mtp wraps the same shared ctx_tgt
// (and therefore the same spec_loop verify+draft decoder pair). When
// every input is MTP-typed and the shared-decoder invariant holds,
// run a single batched forward per draft step. Otherwise fall back to
// the per-slot serial path so that mixed deployments keep working.
std::vector<llama_tokens> common_speculative_draft_batched(
        const std::vector<common_speculative_batched_in> & inputs,
        const common_params_speculative & params) {
    std::vector<llama_tokens> out(inputs.size());
    if (inputs.empty()) return out;

    // First, drive each spec's per-slot begin() bookkeeping (matches
    // common_speculative_draft's per-spec impl->begin/draft semantics).
    // We also collect the MTP states + their loops, and bail to fallback
    // if any spec is non-MTP.
    std::vector<common_speculative_state_mtp *> mtp_states(inputs.size(), nullptr);
    std::vector<llama_spec_loop *>              loops     (inputs.size(), nullptr);
    bool all_mtp = true;
    for (size_t i = 0; i < inputs.size(); ++i) {
        common_speculative * spec = inputs[i].spec;
        if (spec == nullptr || spec->impls.empty()) { all_mtp = false; break; }
        // Reset curr_impl (matches common_speculative_draft semantics).
        spec->t_step_start_us = ggml_time_us();
        if (spec->tuner && spec->tuner->enabled) {
            // Per-slot autotune proposal. We pass the same params object;
            // for batched mode the proposal is a per-slot mutation.
            // (Behaves identically to per-slot draft.)
            common_params_speculative tmp = params;
            spec->tuner->propose(tmp);
        }
        spec->curr_impl = nullptr;

        // Find the MTP impl (single-impl assumption matches the existing
        // per-slot draft path which breaks on the first non-empty impl).
        common_speculative_state_mtp * mtp = nullptr;
        for (auto & impl : spec->impls) {
            if (impl->type == COMMON_SPECULATIVE_TYPE_MTP) {
                mtp = dynamic_cast<common_speculative_state_mtp *>(impl.get());
                break;
            }
        }
        if (mtp == nullptr || mtp->loop == nullptr) { all_mtp = false; break; }
        mtp_states[i] = mtp;
        loops[i]      = mtp->loop;
    }

    // All slots must share the underlying llama_context (post-D9.5
    // collapsed-context invariant: every common_speculative_state_mtp
    // captures ctx_tgt; each per-slot session_adopt creates its own
    // session/decoders that wrap the SAME ctx). The batched primitive
    // forwards through the chosen decoder pair into that one ctx.
    if (all_mtp && mtp_states.size() > 1) {
        llama_context * ctx0 = mtp_states[0]->ctx_tgt;
        for (size_t i = 1; i < mtp_states.size() && all_mtp; ++i) {
            if (mtp_states[i]->ctx_tgt != ctx0) all_mtp = false;
        }
    }

    if (!all_mtp) {
        // Fallback: per-slot serial via existing common_speculative_draft.
        for (size_t i = 0; i < inputs.size(); ++i) {
            // common_speculative_draft takes params by reference; we use
            // a local copy so the autotune proposal applied above doesn't
            // leak into other slots.
            common_params_speculative tmp = params;
            out[i] = common_speculative_draft(inputs[i].spec, tmp,
                    inputs[i].prompt_tgt, inputs[i].id_last);
        }
        return out;
    }

    // All-MTP batched path. Pre-clean per-slot draft cells (matches
    // common_speculative_state_mtp::draft).
    llama_context * ctx_tgt = mtp_states[0]->ctx_tgt;
    std::vector<llama_spec_mtp_slot_in> slots_in(inputs.size());
    int32_t stride = 0;
    for (size_t i = 0; i < inputs.size(); ++i) {
        const llama_seq_id seq_id = mtp_states[i]->seq_id;
        const int32_t      n_past = (int32_t) inputs[i].prompt_tgt.size();

        const llama_pos mtp_pos_max = llama_kv_cache_seq_pos_max(ctx_tgt, seq_id);
        if (mtp_pos_max >= n_past) {
            llama_kv_cache_seq_rm(ctx_tgt, seq_id, n_past, -1);
        }

        slots_in[i].seq_id      = seq_id;
        slots_in[i].id_last     = inputs[i].id_last;
        slots_in[i].n_past      = n_past;
        slots_in[i].n_draft_max = std::max(0, params.n_max);
        if (slots_in[i].n_draft_max > stride) stride = slots_in[i].n_draft_max;
    }
    if (stride == 0) return out;

    std::vector<llama_token>             buf(inputs.size() * (size_t) stride);
    std::vector<llama_spec_mtp_slot_out> outs(inputs.size());

    const int32_t total = llama_spec_loop_gen_drafts_batched(
            loops.data(), (int32_t) loops.size(),
            slots_in.data(),
            params.p_min,
            buf.data(), stride,
            outs.data());

    if (total <= 0) {
        return out;
    }

    // Pack per-slot vectors and update curr_impl + stats so that the
    // accept/print paths remain correct (matches per-slot draft path).
    for (size_t i = 0; i < inputs.size(); ++i) {
        const int32_t k = outs[i].n_drafted;
        if (k <= 0) {
            out[i].clear();
            continue;
        }
        out[i].assign(buf.begin() + (ptrdiff_t) (i * (size_t) stride),
                      buf.begin() + (ptrdiff_t) (i * (size_t) stride + (size_t) k));

        common_speculative * spec = inputs[i].spec;
        // Locate the MTP impl in spec->impls and stamp it as curr_impl
        // so common_speculative_accept / print_stats find the right one.
        for (auto & impl : spec->impls) {
            if (impl->type == COMMON_SPECULATIVE_TYPE_MTP) {
                spec->curr_impl = impl.get();
                impl->n_call_draft++;
                impl->n_gen_drafts++;
                impl->n_gen_tokens += (size_t) k;
                break;
            }
        }
        if (spec->tuner && spec->tuner->enabled) {
            spec->last_n_drafted = k;
        }
    }
    return out;
}

void common_speculative_accept(common_speculative * spec, uint16_t n_accepted) {
    if (spec->tuner && spec->tuner->enabled && spec->t_step_start_us > 0) {
        int64_t step_time_us = ggml_time_us() - spec->t_step_start_us;
        double step_tps = (step_time_us > 100)
            ? (n_accepted + 1.0) * 1e6 / (double)step_time_us
            : 0.0;
        spec->tuner->accept_feedback(n_accepted, spec->last_n_drafted, step_tps);
        spec->t_step_start_us = 0;
    }

    common_speculative_state * impl = spec->curr_impl;

    if (!impl) {
        return;
    }

    {
        common_time_meas tm(impl->t_accept_us, !impl->gen_perf);
        if (n_accepted > 0) {
            impl->n_acc_drafts++;
            impl->n_acc_tokens += n_accepted;
        }

        impl->accept(n_accepted);
        impl->n_call_accept++;
    }
}

void common_speculative_print_stats(const common_speculative * spec, double slot_tps, int n_decoded, int n_past, common_params_speculative * active_params) {
    if (spec == nullptr) {
        return;
    }

    for (const auto & impl : spec->impls) {
        std::string str_perf;
        if (impl->gen_perf) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(3) << impl->t_begin_us / 1000.0 << ", ";
            oss << std::fixed << std::setprecision(3) << impl->t_draft_us / 1000.0 << ", ";
            oss << std::fixed << std::setprecision(3) << impl->t_accept_us / 1000.0;
            str_perf = ", dur(b,g,a) = " + oss.str() + " ms";
        } else {
            str_perf = "";
        }

        LOG_INF("statistics %s: #calls(b,g,a) = %zu %zu %zu, #gen drafts = %zu, #acc drafts = %zu, #gen tokens = %zu, #acc tokens = %zu%s\n",
                common_speculative_type_to_str(impl->type).c_str(),
                impl->n_call_begin, impl->n_call_draft, impl->n_call_accept,
                impl->n_gen_drafts,
                impl->n_acc_drafts,
                impl->n_gen_tokens,
                impl->n_acc_tokens,
                str_perf.c_str());
    }

    if (spec->tuner && spec->tuner->enabled && slot_tps > 0.0 && n_decoded > 0) {
        auto * mutable_spec = const_cast<common_speculative *>(spec);
        if (active_params) {
            mutable_spec->tuner->end_of_request(slot_tps, n_past, *active_params);
        } else {
            common_params_speculative tmp_params;
            mutable_spec->tuner->end_of_request(slot_tps, n_past, tmp_params);
        }
    }
}

// ----------------------------------------------------------------------------
// MTP
// ----------------------------------------------------------------------------

llama_context * common_speculative_get_mtp_ctx(common_speculative * spec) {
    if (!spec) return nullptr;

    for (auto & impl : spec->impls) {
        if (impl->type == COMMON_SPECULATIVE_TYPE_MTP) {
            auto * mtp_state = dynamic_cast<common_speculative_state_mtp *>(impl.get());
            if (mtp_state) {
                return mtp_state->ctx_mtp;
            }
        }
    }
    return nullptr;
}

void common_speculative_context_shift(
        common_speculative * spec,
        llama_seq_id         seq_id,
        llama_pos            kv_keep,
        llama_pos            kv_discard,
        llama_pos            kv_past) {
    if (auto * ctx_mtp = common_speculative_get_mtp_ctx(spec); ctx_mtp != nullptr) {
        llama_kv_cache_seq_rm (ctx_mtp, seq_id, kv_keep, kv_keep + kv_discard);
        llama_kv_cache_seq_add(ctx_mtp, seq_id, kv_keep + kv_discard, kv_past, -kv_discard);
    }
}


void mtp_update_kv_cache(struct llama_context * ctx, const llama_batch& batch, bool is_prompt_warmup) {
    if (batch.n_tokens == 0) {
        return;
    }

    // PHASE45 D10: when LLAMA_MTP_INLINE_KV is on, the per-ubatch hook in
    // the verify forward already wrote the MTP KV layer for every batch
    // position — the WARMUP / UPDATE_ACCEPTED dispatch is redundant. It is
    // also unsafe under multi-slot: this function reads only
    // batch.seq_id[0][0] for the seq_rm and runs a single llama_decode
    // that crashes when the batch contains tokens from multiple seq_ids
    // (multi-slot batched verify). Short-circuit when the hook is on,
    // matching mtp_accept_tokens (line 1450).
    static const bool _hook_on = env_truthy("LLAMA_MTP_INLINE_KV");
    if (_hook_on) {
        return;
    }

    llama_seq_id seq_id    = batch.seq_id[0][0];
    llama_pos    start_pos = batch.pos[0];

    if (llama_kv_cache_seq_pos_max(ctx, seq_id) >= start_pos) {
        llama_kv_cache_seq_rm(ctx, seq_id, start_pos, -1);
    }

    LOG_DBG("[MTP-UPDATE|%s] Updating %d tokens from pos %d...\n",
            is_prompt_warmup ? "PROMPT_WARMUP" : "GEN_ACCEPTED", batch.n_tokens, (int)start_pos);

    llama_batch mtp_batch = batch;
    if (is_prompt_warmup) {
        llama_set_mtp_op_type(ctx, MTP_OP_WARMUP);
    } else {
        llama_set_mtp_op_type(ctx, MTP_OP_UPDATE_ACCEPTED);
    }

    for (int i = 0; i < mtp_batch.n_tokens; ++i) {
        mtp_batch.logits[i] = true;
    }
    llama_decode(ctx, mtp_batch);
    llama_set_mtp_op_type(ctx, MTP_OP_NONE);
}

void mtp_accept_tokens(
    struct llama_context * ctx,
    const std::vector<llama_token> & ids,
    int32_t n_past_base,
    llama_seq_id seq_id
) {
    if (ids.empty()) {
        return;
    }

    // Phase 36 Step 3.3: when the per-ubatch MTP KV hook is on, the
    // verify forward already wrote MTP KV for all batch positions
    // (including the ones now being accepted). The seq_rm at the
    // server's accept tail (server-context.cpp:3968 etc.) trims
    // rejected positions on both main + MTP layers. The separate
    // MTP_OP_UPDATE_ACCEPTED dispatch is therefore redundant — skip it.
    static const bool _hook_on = env_truthy("LLAMA_MTP_INLINE_KV");
    if (_hook_on) {
        // Note: caller is expected to issue llama_kv_cache_seq_rm on
        // its own (ctx, seq_id, slot.n_past, -1). We don't do it here
        // because mtp_accept_tokens has no knowledge of which positions
        // were rejected — only the caller does, via slot.n_past.
        return;
    }

    llama_batch accepted_batch = llama_batch_init(ids.size(), 0, 1);
    for (size_t i = 0; i < ids.size(); ++i) {
        common_batch_add(accepted_batch, ids[i], n_past_base + i, { seq_id }, true);
    }

    mtp_update_kv_cache(ctx, accepted_batch, false);

    llama_batch_free(accepted_batch);
}

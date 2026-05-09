#pragma once

#include "llama.h"
#include "common.h"
#include "spec-tuner.h"

struct common_speculative;

// comma separated list of all types
std::string common_speculative_type_name_str();

// convert string to type
enum common_speculative_type common_speculative_type_from_name(const std::string & name);

// convert type to string
std::string common_speculative_type_to_str(enum common_speculative_type type);

// check if the llama_context is compatible for speculative decoding
// note: clears the memory of the context
bool common_speculative_is_compat(llama_context * ctx_tgt);

common_speculative * common_speculative_init(
        common_params_speculative & params,
        llama_context             * ctx_tgt,
        llama_seq_id                seq_id = 0);  // PHASE45 D9.5: per-slot seq_id for shared ctx

void common_speculative_free(common_speculative * spec);

// optionally call once at the beginning of a new generation
void common_speculative_begin(common_speculative * spec, const llama_tokens & prompt);

// sample up to n_draft tokens and add them to the batch using the draft model
llama_tokens common_speculative_draft(
                     common_speculative * spec,
                     common_params_speculative & params,
                     const llama_tokens & prompt,
                            llama_token   id_last);

// PHASE45 D10.b: batched-draft variant. Inputs describe per-slot work
// (each slot's spec/prompt_tgt/id_last); the batched path drives one
// forward per draft step that processes all slots at once. Falls back
// to per-slot serial drafting if any input fails the MTP cast (only
// MTP supports batched drafting today).
struct common_speculative_batched_in {
    common_speculative * spec;
    llama_tokens         prompt_tgt;
    llama_token          id_last;
};
std::vector<llama_tokens> common_speculative_draft_batched(
        const std::vector<common_speculative_batched_in> & inputs,
        const common_params_speculative & params);

// informs the speculative decoder that n_accepted tokens were accepted by the target model
void common_speculative_accept(common_speculative * spec, uint16_t n_accepted);

// print statistics about the speculative decoding
void common_speculative_print_stats(const common_speculative * spec, double slot_tps = 0.0, int n_decoded = 0, int n_past = 0, common_params_speculative * active_params = nullptr);

// get the MTP context from the speculative object (nullptr if not MTP type)
llama_context * common_speculative_get_mtp_ctx(common_speculative * spec);

// Context shift for MTP to match how server handle main model
void common_speculative_context_shift(
        common_speculative * spec,
        llama_seq_id         seq_id,
        llama_pos            kv_keep,
        llama_pos            kv_discard,
        llama_pos            kv_past);

// PHASE45 D9.9: mtp_speculative_gen_draft removed (~300 LoC dead code).
// Bypassed by D8.3's libcommon shim that forwards to spec_loop, which
// in turn calls libllama-level llama_spec_mtp_draft.

void mtp_update_kv_cache(struct llama_context * ctx, const llama_batch& batch, bool is_prompt_warmup);

void mtp_accept_tokens(
    struct llama_context * ctx,
    const std::vector<llama_token> & ids,
    int32_t n_past_base,
    llama_seq_id seq_id
);

// common/perplexity.h
//
// Shared kernel for negative-log-likelihood + perplexity computation
// over a token sequence under a target's logits. Used by both
// examples/perplexity (corpus PPL) and examples/llama-bench
// (PPL-of-generated-output as a quality bound for speculative
// decoding comparisons — T8 ship-gate work).
//
// Extracted from examples/perplexity/perplexity.cpp (the simple
// overloads at the bottom of the file). The thread-parallel +
// logit_history/prob_history machinery is preserved verbatim from
// the original — both callers benefit. Perplexity-example-only
// helpers (the uint16_t* log_softmax overload + the ostream
// process_logits overload) remain in perplexity.cpp.

#pragma once

#include <thread>
#include <vector>

struct results_log_softmax {
    double log_softmax;
    float  logit;
    float  prob;
};

// Log-softmax for a single position. Returns:
//   .log_softmax = log P(tok | logits[..])  in nats
//   .logit       = logits[tok]
//   .prob        = softmax-normalised probability of tok
results_log_softmax log_softmax(int n_vocab, const float * logits, int tok);

// Compute NLL accumulator over [tokens[1] .. tokens[n_token]]
// using `logits` arranged as [n_token, n_vocab] row-major.
//
//   logits[i*n_vocab .. (i+1)*n_vocab] is the predictive
//   distribution for tokens[i+1] (i.e., the logits emitted AFTER
//   processing tokens[i]).
//
// `nll` and `nll2` are accumulated (the function adds to existing
// values; caller initialises to 0 for fresh measurement).
//   nll   += sum_i -log P(tokens[i+1] | logits[i])
//   nll2  += sum_i (-log P(tokens[i+1] | logits[i]))^2
//
// `logit_history` and `prob_history` must be n_token-element
// buffers; they record per-position .logit and .prob from each
// log_softmax call. Pass throwaway buffers if not needed.
//
// Parallelised across `workers` threads (plus the calling thread).
//
// PPL = exp(nll / n_token).
void process_logits(
    int n_vocab,
    const float * logits,
    const int * tokens,
    int n_token,
    std::vector<std::thread> & workers,
    double & nll,
    double & nll2,
    float * logit_history,
    float * prob_history);

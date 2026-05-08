//
// Copyright (C) 2023-2026 The llama.cpp authors
// Copyright (C) 2024-2026 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//
// PHASE 45 D5: Public API for `llama_kv_txn`.
//
// A KV transaction is an RAII handle representing a reservation of
// position-range capacity in the session's K/V cache. Speculative
// decoders use it to write draft tokens' K/V into a region that can be
// committed (on accept) or rolled back (on reject) atomically.
//
// Without transactions, draft writes alias verify writes in the same
// cells; speculative writes are visible to subsequent reads even if
// the prediction was wrong. The transaction model makes this correct
// by construction.
//
// Generalizes the PHASE36/37/38 ad-hoc seq_rm-on-reject pattern.
//

#ifndef LLAMA_KV_TXN_H
#define LLAMA_KV_TXN_H

#include "llama.h"

#ifdef __cplusplus
extern "C" {
#endif

    struct llama_kv_txn;
    struct llama_session;

    // Begin a transaction reserving cell capacity for a sequence.
    //
    // pos_start: first position the txn will write
    // pos_end:   last position (exclusive)
    //
    // Returns NULL if the session can't reserve the requested range
    // (e.g., insufficient capacity).
    LLAMA_API struct llama_kv_txn * llama_kv_txn_begin(
            struct llama_session * session,
            llama_seq_id           seq,
            llama_pos              pos_start,
            llama_pos              pos_end);

    // Commit the transaction. The reserved range becomes visible as
    // canonical session state. The txn handle is freed.
    LLAMA_API void llama_kv_txn_commit(struct llama_kv_txn * txn);

    // Roll back the transaction. The reserved range is dropped from the
    // session; cells return to their pre-txn state. The txn handle is
    // freed.
    LLAMA_API void llama_kv_txn_rollback(struct llama_kv_txn * txn);

    // Query the txn's reserved range (debugging / introspection).
    LLAMA_API llama_seq_id llama_kv_txn_seq      (const struct llama_kv_txn * txn);
    LLAMA_API llama_pos    llama_kv_txn_pos_start(const struct llama_kv_txn * txn);
    LLAMA_API llama_pos    llama_kv_txn_pos_end  (const struct llama_kv_txn * txn);

#ifdef __cplusplus
}
#endif

#endif // LLAMA_KV_TXN_H

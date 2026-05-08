//
// PHASE 45 D6 stub: llama_kv_txn implementation skeleton.
//
// Bodies are wired through to llama_session_kv_seq_rm on rollback (the
// only operation D6 actually exercises through main.cpp greedy decode);
// commit is a no-op until D8 introduces speculative writes that need
// atomic visibility. Spec-loop will not call into kv_txn until D8.
//

#include "llama-kv-txn.h"
#include "llama-session.h"

struct llama_kv_txn {
    struct llama_session * session;
    llama_seq_id           seq;
    llama_pos              pos_start;
    llama_pos              pos_end;
};

extern "C" {

struct llama_kv_txn * llama_kv_txn_begin(
        struct llama_session * session,
        llama_seq_id           seq,
        llama_pos              pos_start,
        llama_pos              pos_end) {
    if (session == nullptr) return nullptr;
    auto * txn = new llama_kv_txn{session, seq, pos_start, pos_end};
    return txn;
}

void llama_kv_txn_commit(struct llama_kv_txn * txn) {
    // D6: writes are already in place; commit is just lifecycle end.
    // D8 will add deferred-visibility semantics if measurements show a need.
    delete txn;
}

void llama_kv_txn_rollback(struct llama_kv_txn * txn) {
    if (txn == nullptr) return;
    if (txn->session != nullptr && txn->pos_end > txn->pos_start) {
        llama_session_kv_seq_rm(txn->session, txn->seq, txn->pos_start, txn->pos_end);
    }
    delete txn;
}

llama_seq_id llama_kv_txn_seq(const struct llama_kv_txn * txn) {
    return txn ? txn->seq : -1;
}

llama_pos llama_kv_txn_pos_start(const struct llama_kv_txn * txn) {
    return txn ? txn->pos_start : -1;
}

llama_pos llama_kv_txn_pos_end(const struct llama_kv_txn * txn) {
    return txn ? txn->pos_end : -1;
}

} // extern "C"

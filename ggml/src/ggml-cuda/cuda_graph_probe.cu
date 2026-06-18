#include "cuda_graph_probe.cuh"
#include "common.cuh"
#include "graph.cuh"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <random>
#include <set>
#include <sys/stat.h>
#include <thread>
#ifdef _WIN32
#include <direct.h>
#define mkdir(path, mode) _mkdir(path)
#define gmtime_r(t, tm) gmtime_s((tm), (t))
#else
#include <unistd.h>
#endif

namespace {

int          g_active = -1;       // -1 = uncached
std::string  g_dump_dir;
std::string  g_runid;
int          g_flush_sec = 30;

std::atomic<bool> g_initialized{false};
std::once_flag    g_init_once;

std::atomic<bool> g_flush_pending{false};
std::atomic<bool> g_shutdown{false};
std::thread       g_flusher;

// Global registry of contexts that have ever recorded a probe event.
std::mutex g_ctx_mu;
std::set<ggml_backend_cuda_context *> g_ctx_set;

uint64_t now_ns() {
    using namespace std::chrono;
    return duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count();
}

void mkdir_p(const std::string & path) {
    if (path.empty()) return;
    std::string acc;
    for (size_t i = 0; i < path.size(); ++i) {
        char c = path[i];
        acc.push_back(c);
        if (c == '/' || i == path.size() - 1) {
            if (acc != "/" && acc != ".") {
                ::mkdir(acc.c_str(), 0755);  // ignore EEXIST
            }
        }
    }
}

std::string make_runid() {
    char buf[64];
    std::time_t t = std::time(nullptr);
    std::tm tm{};
    gmtime_r(&t, &tm);
    std::snprintf(buf, sizeof(buf), "%04d%02d%02dT%02d%02d%02d",
                  tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
                  tm.tm_hour, tm.tm_min, tm.tm_sec);
    std::random_device rd;
    char rid[8];
    std::snprintf(rid, sizeof(rid), "-%06x", (unsigned) (rd() & 0xffffff));
    return std::string(buf) + rid;
}

void sigusr1_handler(int) {
    g_flush_pending.store(true, std::memory_order_relaxed);
}

void flusher_loop() {
    auto next = std::chrono::steady_clock::now() +
                std::chrono::seconds(g_flush_sec);
    while (!g_shutdown.load(std::memory_order_relaxed)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        bool due = std::chrono::steady_clock::now() >= next;
        bool req = g_flush_pending.exchange(false, std::memory_order_relaxed);
        if (due || req) {
            cuda_graph_probe::flush_all();
            next = std::chrono::steady_clock::now() +
                   std::chrono::seconds(g_flush_sec);
        }
    }
}

void register_ctx(ggml_backend_cuda_context & ctx) {
    std::lock_guard<std::mutex> lk(g_ctx_mu);
    g_ctx_set.insert(&ctx);
}

void unregister_ctx(ggml_backend_cuda_context & ctx) {
    std::lock_guard<std::mutex> lk(g_ctx_mu);
    g_ctx_set.erase(&ctx);
}

inline std::string hex64(uint64_t v) {
    char buf[20];
    std::snprintf(buf, sizeof(buf), "0x%016llx", (unsigned long long) v);
    return buf;
}

void emit_common(std::ofstream & f, const char * probe, ggml_backend_cuda_context & ctx, uint64_t ts_ns) {
    f << "{\"ts_ns\":" << ts_ns
      << ",\"runid\":\"" << g_runid << "\""
      << ",\"backend\":\"cuda" << ctx.device << "\""
      << ",\"probe\":\"" << probe << "\"";
}

} // namespace

namespace cuda_graph_probe {

int active() {
    if (g_active >= 0) return g_active;
    const char * env = std::getenv("GGML_CUDA_GRAPH_PROBE");
    g_active = (env && env[0] == '1') ? 1 : 0;
    return g_active;
}

void ensure_initialized() {
    if (g_initialized.load(std::memory_order_acquire)) return;
    std::call_once(g_init_once, []() {
        if (!active()) return;

        g_runid = make_runid();

        const char * dir_env = std::getenv("GGML_CUDA_GRAPH_PROBE_DIR");
        if (dir_env && *dir_env) {
            g_dump_dir = dir_env;
        } else {
            g_dump_dir = "/mnt/archive/cuda-graph-probe/" + g_runid;
        }
        mkdir_p(g_dump_dir);

        const char * sec_env = std::getenv("GGML_CUDA_GRAPH_PROBE_FLUSH_SEC");
        if (sec_env && *sec_env) {
            int v = std::atoi(sec_env);
            if (v >= 1) g_flush_sec = v;
        }

#ifndef _WIN32
        std::signal(SIGUSR1, &sigusr1_handler);
#endif
        g_flusher = std::thread(&flusher_loop);
        g_flusher.detach();

        g_initialized.store(true, std::memory_order_release);
    });
}

int flush_all() {
    if (!active()) return -1;
    ensure_initialized();
    // Hold g_ctx_mu through the iteration. on_context_destroyed acquires the
    // same lock via unregister_ctx, so this serializes ctx destruction with
    // background flushing — without it, flush_context could be invoked on a
    // ctx whose probe_state mutex is being destroyed.
    std::lock_guard<std::mutex> lk(g_ctx_mu);
    for (auto * c : g_ctx_set) {
        flush_context(*c);
    }
    return 0;
}

int flush_context(ggml_backend_cuda_context & ctx) {
    if (!active()) return -1;
    ensure_initialized();

    char path_buf[1024];
    auto open_for = [&](const char * probe) -> std::ofstream {
        std::snprintf(path_buf, sizeof(path_buf), "%s/cuda%d-%s.jsonl",
                      g_dump_dir.c_str(), ctx.device, probe);
        std::ofstream f(path_buf, std::ios::app);
        return f;
    };

    std::lock_guard<std::mutex> lk(ctx.probe_state.mu);

#ifdef USE_CUDA_GRAPH
    // hit_counter: walk cuda_graphs and emit one record per cached graph.
    if (!ctx.cuda_graphs.empty()) {
        std::ofstream f = open_for("hit_counter");
        if (f) {
            const uint64_t ts = now_ns();
            for (const auto & kv : ctx.cuda_graphs) {
                const auto & g = kv.second;
                if (!g) continue;
                emit_common(f, "hit_counter", ctx, ts);
                f << ",\"topology_key\":\"" << hex64(g->topology_key) << "\""
                  << ",\"shape_key\":\""    << hex64(g->shape_key)    << "\""
                  << ",\"hits_total\":"     << (long long) g->hits_total
                  << ",\"last_use_us\":"    << (long long) g->last_use_us
                  << "}\n";
            }
            f.flush();
            f.close();
        }
    }
#endif

    auto drain_timings = [&]() {
        if (ctx.probe_state.timings.empty()) return;
        std::ofstream f = open_for("timing");
        if (!f) { ctx.probe_state.timings.clear(); return; }
        for (const auto & r : ctx.probe_state.timings) {
            emit_common(f, "timing", ctx, r.ts_ns);
            f << ",\"event\":\""  << r.event << "\""
              << ",\"duration_us\":" << r.duration_us
              << ",\"n_nodes\":"     << r.n_nodes
              << ",\"topology_key\":\"" << hex64(r.topology_key) << "\""
              << "}\n";
        }
        f.flush(); f.close();
        ctx.probe_state.timings.clear();
    };
    auto drain_vram = [&]() {
        if (ctx.probe_state.vram_deltas.empty()) return;
        std::ofstream f = open_for("vram_delta");
        if (!f) { ctx.probe_state.vram_deltas.clear(); return; }
        for (const auto & r : ctx.probe_state.vram_deltas) {
            const long long delta = (long long) r.free_after_bytes - (long long) r.free_before_bytes;
            emit_common(f, "vram_delta", ctx, r.ts_ns);
            f << ",\"event\":\""             << r.event << "\""
              << ",\"topology_key\":\""      << hex64(r.topology_key) << "\""
              << ",\"free_before_bytes\":"   << (long long) r.free_before_bytes
              << ",\"free_after_bytes\":"    << (long long) r.free_after_bytes
              << ",\"delta_bytes\":"         << delta
              << ",\"synced\":true"
              << ",\"note\":\"cudaMemGetInfo reports global free; cuBLAS workspace and pool activity may mask per-entry cost\""
              << "}\n";
        }
        f.flush(); f.close();
        ctx.probe_state.vram_deltas.clear();
    };
    auto drain_updates = [&]() {
        if (ctx.probe_state.update_failures.empty()) return;
        std::ofstream f = open_for("update_failures");
        if (!f) { ctx.probe_state.update_failures.clear(); return; }
        for (const auto & r : ctx.probe_state.update_failures) {
            emit_common(f, "update_failures", ctx, r.ts_ns);
            f << ",\"topology_key\":\""     << hex64(r.topology_key) << "\""
              << ",\"shape_key_old\":\""    << hex64(r.shape_key_old) << "\""
              << ",\"shape_key_new\":\""    << hex64(r.shape_key_new) << "\""
              << ",\"fallback\":\"reinstantiate\""
              << "}\n";
        }
        f.flush(); f.close();
        ctx.probe_state.update_failures.clear();
    };
    auto drain_disable = [&]() {
        if (ctx.probe_state.disable_too_many.empty()) return;
        std::ofstream f = open_for("disable_too_many");
        if (!f) { ctx.probe_state.disable_too_many.clear(); return; }
        for (const auto & r : ctx.probe_state.disable_too_many) {
            emit_common(f, "disable_too_many", ctx, r.ts_ns);
            f << ",\"topology_key\":\""        << hex64(r.topology_key) << "\""
              << ",\"consecutive_updates\":"   << r.consecutive_updates
              << "}\n";
        }
        f.flush(); f.close();
        ctx.probe_state.disable_too_many.clear();
    };
    auto drain_compute_failures = [&]() {
        if (ctx.probe_state.compute_failures.empty()) return;
        std::ofstream f = open_for("compute_failure");
        if (!f) { ctx.probe_state.compute_failures.clear(); return; }
        for (const auto & r : ctx.probe_state.compute_failures) {
            emit_common(f, "compute_failure", ctx, r.ts_ns);
            f << ",\"op\":\""        << (r.op_name        ? r.op_name        : "?") << "\""
              << ",\"dst_type\":\""  << (r.dst_type_name  ? r.dst_type_name  : "?") << "\""
              << ",\"src0_type\":\"" << (r.src0_type_name ? r.src0_type_name : "?") << "\""
              << ",\"src1_type\":\"" << (r.src1_type_name ? r.src1_type_name : "?") << "\""
              << ",\"ne\":[" << (long long) r.ne[0]
              << ","         << (long long) r.ne[1]
              << ","         << (long long) r.ne[2]
              << ","         << (long long) r.ne[3] << "]"
              << "}\n";
        }
        f.flush(); f.close();
        ctx.probe_state.compute_failures.clear();
    };

    drain_timings();
    drain_vram();
    drain_updates();
    drain_disable();
    drain_compute_failures();

    return 0;
}

void on_context_destroyed(ggml_backend_cuda_context & ctx) {
    // Lock order — g_ctx_mu before ctx.probe_state.mu — must match flush_all
    // to avoid deadlock with the background flusher.
    std::lock_guard<std::mutex> lk(g_ctx_mu);
    if (active()) {
        flush_context(ctx);
    }
    // Mark before erase so any record_* fired by ~ggml_cuda_graph (e.g.
    // destroy-time vram_delta from cuda_graphs map teardown) becomes a silent
    // no-op even if the bg flusher races to register_ctx.
    ctx.probe_state.being_destroyed.store(true, std::memory_order_release);
    g_ctx_set.erase(&ctx);
}

void record_timing(ggml_backend_cuda_context & ctx,
                   uint64_t topology_key, const char * event,
                   double duration_us, int n_nodes) {
    if (!active()) return;
    if (ctx.probe_state.being_destroyed.load(std::memory_order_acquire)) return;
    ensure_initialized();
    register_ctx(ctx);
    std::lock_guard<std::mutex> lk(ctx.probe_state.mu);
    ctx.probe_state.timings.push_back({now_ns(), event, duration_us, n_nodes, topology_key});
}

void record_vram(ggml_backend_cuda_context & ctx,
                 uint64_t topology_key, const char * event,
                 int64_t free_before, int64_t free_after) {
    if (!active()) return;
    if (ctx.probe_state.being_destroyed.load(std::memory_order_acquire)) return;
    ensure_initialized();
    register_ctx(ctx);
    std::lock_guard<std::mutex> lk(ctx.probe_state.mu);
    ctx.probe_state.vram_deltas.push_back({now_ns(), event, topology_key, free_before, free_after});
}

void record_update_failure(ggml_backend_cuda_context & ctx,
                           uint64_t topology_key,
                           uint64_t shape_key_old, uint64_t shape_key_new) {
    if (!active()) return;
    if (ctx.probe_state.being_destroyed.load(std::memory_order_acquire)) return;
    ensure_initialized();
    register_ctx(ctx);
    ctx.probe_state.update_failure_count.fetch_add(1, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lk(ctx.probe_state.mu);
    ctx.probe_state.update_failures.push_back({now_ns(), topology_key, shape_key_old, shape_key_new});
}

void record_disable_too_many(ggml_backend_cuda_context & ctx,
                             uint64_t topology_key, int consecutive_updates) {
    if (!active()) return;
    if (ctx.probe_state.being_destroyed.load(std::memory_order_acquire)) return;
    ensure_initialized();
    register_ctx(ctx);
    std::lock_guard<std::mutex> lk(ctx.probe_state.mu);
    ctx.probe_state.disable_too_many.push_back({now_ns(), topology_key, consecutive_updates});
}

void record_compute_failure(ggml_backend_cuda_context & ctx,
                            const char * op_name,
                            const char * dst_type_name,
                            const char * src0_type_name,
                            const char * src1_type_name,
                            const int64_t * ne_out_4) {
    if (!active()) return;
    if (ctx.probe_state.being_destroyed.load(std::memory_order_acquire)) return;
    ensure_initialized();
    register_ctx(ctx);
    cuda_graph_probe_state::compute_failure_rec r{};
    r.ts_ns          = now_ns();
    r.op_name        = op_name;
    r.dst_type_name  = dst_type_name;
    r.src0_type_name = src0_type_name;
    r.src1_type_name = src1_type_name;
    if (ne_out_4) {
        r.ne[0] = ne_out_4[0]; r.ne[1] = ne_out_4[1];
        r.ne[2] = ne_out_4[2]; r.ne[3] = ne_out_4[3];
    }
    {
        std::lock_guard<std::mutex> lk(ctx.probe_state.mu);
        ctx.probe_state.compute_failures.push_back(r);
    }
    // Force an immediate flush so the record lands on disk before the
    // imminent GGML_ASSERT crash takes the process down. flush_context
    // re-acquires probe_state.mu — release first via the scope above.
    flush_context(ctx);
}

} // namespace cuda_graph_probe

// ggml_cuda_graph dtor — defined out-of-line so it can sample the
// per-destroy free-VRAM delta without graph.cuh pulling in the full
// probe header.
#ifdef USE_CUDA_GRAPH
ggml_cuda_graph::~ggml_cuda_graph() {
    int64_t free_before = 0;
    bool sample = (owner_ctx != nullptr) && cuda_graph_probe::active();
    if (sample) {
        size_t f = 0, t = 0;
        cudaDeviceSynchronize();
        cudaMemGetInfo(&f, &t);
        free_before = (int64_t) f;
    }
    if (instance != nullptr) {
        CUDA_CHECK(cudaGraphExecDestroy(instance));
    }
    if (graph != nullptr) {
        CUDA_CHECK(cudaGraphDestroy(graph));
    }
    if (sample) {
        size_t f = 0, t = 0;
        cudaDeviceSynchronize();
        cudaMemGetInfo(&f, &t);
        cuda_graph_probe::record_vram(*owner_ctx, topology_key, "destroy", free_before, (int64_t) f);
    }
}
#endif

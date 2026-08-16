#include "naf.h"
#include "trellis_model.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace trellis {
using T = ggml_tensor;

static constexpr int   DIM       = 256;            // image-encoder output channels
static constexpr int   HEADS     = 4;
static constexpr int   HEAD_DIM  = DIM / HEADS;    // 64
static constexpr int   KERNEL    = 9;              // neighborhood attention window (per axis)
static constexpr int   GROUPS    = 8;              // GroupNorm groups inside EncBlock
static constexpr float GN_EPS    = 1e-5f;
static constexpr int   ROPE_QUARTER = HEAD_DIM / 4;   // 16 periods

// ---------------------------------------------------------------------------
// Image encoder (the only learned part of NAF), built as one ggml graph.
// ---------------------------------------------------------------------------

// torch Conv2d(padding_mode="reflect") on a [W,H,C,1] map. ggml has no 2-D reflect pad, so the
// border rows/columns are mirrored explicitly with views + concat — p is 0 or 1 in every NAF
// conv, so this costs two small copies per axis.
static T* reflect_pad(ggml_context* c, T* x, int p) {
    if (p <= 0) return x;
    const int64_t W = x->ne[0], H = x->ne[1], C = x->ne[2];
    auto row = [&](int64_t r) {
        return ggml_cont(c, ggml_view_4d(c, x, W, 1, C, 1, x->nb[1], x->nb[2], x->nb[3],
                                         (size_t)r * x->nb[1]));
    };
    T* y = x;
    for (int i = 1; i <= p; ++i) y = ggml_concat(c, row(i), y, 1);            // rows p..1 prepended
    for (int i = 1; i <= p; ++i) y = ggml_concat(c, y, row(H - 1 - i), 1);    // rows H-2..H-1-p appended
    const int64_t H2 = y->ne[1];
    auto col = [&](int64_t k) {
        return ggml_cont(c, ggml_view_4d(c, y, 1, H2, C, 1, y->nb[1], y->nb[2], y->nb[3],
                                         (size_t)k * y->nb[0]));
    };
    T* z = y;
    for (int i = 1; i <= p; ++i) z = ggml_concat(c, col(i), z, 0);
    for (int i = 1; i <= p; ++i) z = ggml_concat(c, z, col(W - 1 - i), 0);
    return z;
}

static T* conv2d(ggml_context* c, const Model& m, const std::string& pre, T* x, int k) {
    // im2col (ggml_conv_2d) is the portable path but materializes a [k*k*C, W*H] buffer, which is
    // ~2.4 GB for the 1024-guide EncBlocks. TRELLIS_NAF_CONV_DIRECT=1 switches to the fused kernel
    // where the backend implements it (CUDA), trading portability for that buffer.
    static const bool direct = std::getenv("TRELLIS_NAF_CONV_DIRECT") != nullptr;
    x = reflect_pad(c, x, k / 2);
    T* w = m.get(pre + ".weight");
    T* y = direct ? ggml_conv_2d_direct(c, w, x, 1, 1, 0, 0, 1, 1)
                  : ggml_conv_2d(c, w, x, 1, 1, 0, 0, 1, 1);
    if (T* b = m.try_get(pre + ".bias")) y = ggml_add(c, y, ggml_reshape_4d(c, b, 1, 1, b->ne[0], 1));
    return y;
}

static T* group_norm(ggml_context* c, const Model& m, const std::string& pre, T* x) {
    x = ggml_group_norm(c, x, GROUPS, GN_EPS);
    T* w = m.get(pre + ".weight");
    T* b = m.get(pre + ".bias");
    x = ggml_mul(c, x, ggml_reshape_4d(c, w, 1, 1, w->ne[0], 1));
    return ggml_add(c, x, ggml_reshape_4d(c, b, 1, 1, b->ne[0], 1));
}

// encoder() = Conv2d(3, dim/2, k) followed by two EncBlocks. Both blocks are built with
// residual=False and in_channels == out_channels, so neither the skip nor the 1x1 shortcut
// exists in the checkpoint — the branch is a plain chain.
static T* enc_branch(ggml_context* c, const Model& m, const std::string& pre, T* img, int k) {
    T* x = conv2d(c, m, pre + ".0", img, k);
    for (int i = 1; i <= 2; ++i) {
        const std::string b = pre + "." + std::to_string(i);
        x = group_norm(c, m, b + ".norm1", x);
        x = ggml_silu(c, x);
        x = conv2d(c, m, b + ".conv1", x, k);
        x = group_norm(c, m, b + ".norm2", x);
        x = ggml_silu(c, x);
        x = conv2d(c, m, b + ".conv2", x, k);
    }
    return x;
}

// Runs the guide encoder and the adaptive average pool. Returns the [out*out, DIM] feature map
// in pixel-major order (pixel p = y*out + x, channels contiguous).
static std::vector<float> encode_guide(const Model& m, const std::vector<float>& img01, int S, int out) {
    if (S % out != 0) throw std::runtime_error("naf: guide size must be a multiple of the target size");
    const int f = S / out;

    size_t meta = ggml_tensor_overhead() * 4096 + ggml_graph_overhead_custom(8192, false) + (1 << 20);
    ggml_context* c = ggml_init({ meta, nullptr, true });
    T* img = ggml_new_tensor_4d(c, GGML_TYPE_F32, S, S, 3, 1); ggml_set_input(img);

    T* a = enc_branch(c, m, "image_encoder.encoder",     img, 1);
    T* b = enc_branch(c, m, "image_encoder.sem_encoder", img, 3);
    T* x = ggml_concat(c, a, b, 2);                      // [S, S, DIM, 1]
    if (f > 1) x = ggml_pool_2d(c, x, GGML_OP_POOL_AVG, f, f, f, f, 0.0f, 0.0f);
    ggml_set_output(x);

    ggml_cgraph* g = ggml_new_graph_custom(c, 8192, false);
    ggml_build_forward_expand(g, x);
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend));
    if (!ggml_gallocr_alloc_graph(alloc, g)) throw std::runtime_error("naf: alloc failed");
    ggml_backend_tensor_set(img, img01.data(), 0, img01.size() * 4);
    if (ggml_backend_graph_compute(m.backend, g) != GGML_STATUS_SUCCESS)
        throw std::runtime_error("naf: compute failed");
    std::vector<float> planar = tensor_to_f32(x);        // ggml [out, out, DIM] -> x + out*y + out*out*ch
    ggml_gallocr_free(alloc); ggml_free(c);

    const size_t np = (size_t)out * out;
    std::vector<float> pix(np * DIM);
    for (int ch = 0; ch < DIM; ++ch)
        for (size_t p = 0; p < np; ++p) pix[p * DIM + ch] = planar[(size_t)ch * np + p];
    return pix;
}

// ---------------------------------------------------------------------------
// Axial RoPE (the DINOv3 formulation NAF reuses), applied in place on [np, DIM].
// Angles depend only on the pixel, so all four heads share one cos/sin table.
// ---------------------------------------------------------------------------
static void apply_rope(std::vector<float>& pix, int H, int W, const std::vector<float>& periods) {
    const int half = HEAD_DIM / 2;                       // 32: rotate_half pairs j with j+32
    std::vector<float> cs(half), sn(half);
    for (int y = 0; y < H; ++y) {
        const float ch = ((y + 0.5f) / H) * 2.0f - 1.0f;
        for (int x = 0; x < W; ++x) {
            const float cw = ((x + 0.5f) / W) * 2.0f - 1.0f;
            for (int j = 0; j < half; ++j) {
                const bool wj = j >= ROPE_QUARTER;
                const float ang = 6.283185307179586f * (wj ? cw : ch) / periods[wj ? j - ROPE_QUARTER : j];
                cs[j] = std::cos(ang); sn[j] = std::sin(ang);
            }
            float* v = &pix[((size_t)y * W + x) * DIM];
            for (int h = 0; h < HEADS; ++h) {
                float* q = v + (size_t)h * HEAD_DIM;
                for (int j = 0; j < half; ++j) {
                    const float lo = q[j], hi = q[j + half];
                    q[j]        = lo * cs[j] - hi * sn[j];
                    q[j + half] = hi * cs[j] + lo * sn[j];
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Neighborhood cross-attention.
//
// NATTEN's dilated neighborhood attention splits each axis into `dilation` interleaved
// subsequences and applies a KERNEL-wide window inside one of them. NAF sets dilation to the
// upsampling factor d, and both keys and values are nearest-upsampled from the LR grid, so tap
// t of query i reads index (i%d) + (s+t)*d whose LR cell is exactly s+t, with the window start
// s = clamp(i/d - KERNEL/2, 0, Nlr - KERNEL). The 2-D window therefore degenerates to a plain
// KERNEL x KERNEL block of LR cells, which is what this computes.
// ---------------------------------------------------------------------------
static void attend_pixel(int px, int py, int out_w, int out_h, int Hf, int Wf, int C,
                         const std::vector<float>& q, const std::vector<float>& klr,
                         const float* vlr, float* dst) {
    const int dx = out_w / Wf, dy = out_h / Hf;
    const int sx = std::min(std::max(px / dx - KERNEL / 2, 0), Wf - KERNEL);
    const int sy = std::min(std::max(py / dy - KERNEL / 2, 0), Hf - KERNEL);
    const float scale = 1.0f / std::sqrt((float)HEAD_DIM);
    const int vhead = C / HEADS;

    const float* qv = &q[((size_t)py * out_w + px) * DIM];
    float w[KERNEL * KERNEL];
    for (int h = 0; h < HEADS; ++h) {
        const float* qh = qv + (size_t)h * HEAD_DIM;
        float mx = -INFINITY;
        for (int a = 0; a < KERNEL; ++a) for (int b = 0; b < KERNEL; ++b) {
            const float* kh = &klr[((size_t)(sy + a) * Wf + (sx + b)) * DIM + (size_t)h * HEAD_DIM];
            float s = 0;
            for (int j = 0; j < HEAD_DIM; ++j) s += qh[j] * kh[j];
            s *= scale;
            w[a * KERNEL + b] = s;
            if (s > mx) mx = s;
        }
        float sum = 0;
        for (int t = 0; t < KERNEL * KERNEL; ++t) { w[t] = std::exp(w[t] - mx); sum += w[t]; }
        const float inv = 1.0f / sum;
        float* o = dst + (size_t)h * vhead;
        for (int j = 0; j < vhead; ++j) o[j] = 0.0f;
        for (int a = 0; a < KERNEL; ++a) for (int b = 0; b < KERNEL; ++b) {
            const float ww = w[a * KERNEL + b] * inv;
            const size_t cell = (size_t)(sy + a) * Wf + (sx + b);
            // values keep the DINOv3 channel-major layout: channel c of cell -> vlr[c + C*cell]
            for (int j = 0; j < vhead; ++j) o[j] += ww * vlr[(size_t)h * vhead + j + (size_t)C * cell];
        }
    }
}

std::vector<float> naf_sample(const Model& m,
                              const std::vector<float>& img01, int S,
                              const float* feats_lr, int Hf, int Wf, int C,
                              int out, const std::vector<float>& pts_xy) {
    if (Hf < KERNEL || Wf < KERNEL)
        throw std::runtime_error("naf: low-res feature grid smaller than the attention window");
    if (out % Hf != 0 || out % Wf != 0)
        throw std::runtime_error("naf: target size must be a multiple of the feature grid");
    if (C % HEADS != 0) throw std::runtime_error("naf: channel count must be divisible by 4 heads");

    std::vector<float> q = encode_guide(m, img01, S, out);
    std::vector<float> periods = tensor_to_f32(m.get("image_encoder.rope.periods"));
    apply_rope(q, out, out, periods);

    // keys: the same guide features average-pooled back onto the LR grid (KeyEncoder).
    const int by = out / Hf, bx = out / Wf;
    std::vector<float> klr((size_t)Hf * Wf * DIM, 0.0f);
    for (int y = 0; y < out; ++y) for (int x = 0; x < out; ++x) {
        float* d = &klr[((size_t)(y / by) * Wf + (x / bx)) * DIM];
        const float* s = &q[((size_t)y * out + x) * DIM];
        for (int j = 0; j < DIM; ++j) d[j] += s[j];
    }
    const float navg = 1.0f / (float)(by * bx);
    for (float& v : klr) v *= navg;

    // Only the pixels the projection actually reads are worth attending over: collect the four
    // bilinear taps of every sample point, deduplicate, and compute that set in parallel.
    const size_t NP = pts_xy.size() / 2;
    const float step = (float)out / (float)S;
    std::vector<float> fx(NP), fy(NP);
    std::vector<int> slot((size_t)out * out, -1);
    std::vector<int> needed;
    auto want = [&](int x, int y) {
        int& s = slot[(size_t)y * out + x];
        if (s < 0) { s = (int)needed.size(); needed.push_back(y * out + x); }
    };
    for (size_t p = 0; p < NP; ++p) {
        float gx = (pts_xy[2 * p]     + 0.5f) * step - 0.5f;
        float gy = (pts_xy[2 * p + 1] + 0.5f) * step - 0.5f;
        gx = std::min(std::max(gx, 0.0f), (float)out - 1.0f);   // grid_sample padding_mode="border"
        gy = std::min(std::max(gy, 0.0f), (float)out - 1.0f);
        fx[p] = gx; fy[p] = gy;
        const int x0 = (int)std::floor(gx), y0 = (int)std::floor(gy);
        const int x1 = std::min(x0 + 1, out - 1), y1 = std::min(y0 + 1, out - 1);
        want(x0, y0); want(x1, y0); want(x0, y1); want(x1, y1);
    }

    std::vector<float> hr((size_t)needed.size() * C);
    const unsigned nthreads = std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::thread> pool;
    for (unsigned t = 0; t < nthreads; ++t) {
        pool.emplace_back([&, t] {
            for (size_t i = t; i < needed.size(); i += nthreads)
                attend_pixel(needed[i] % out, needed[i] / out, out, out, Hf, Wf, C,
                             q, klr, feats_lr, &hr[i * C]);
        });
    }
    for (auto& th : pool) th.join();

    std::vector<float> res((size_t)C * NP, 0.0f);
    for (size_t p = 0; p < NP; ++p) {
        const int x0 = (int)std::floor(fx[p]), y0 = (int)std::floor(fy[p]);
        const int x1 = std::min(x0 + 1, out - 1), y1 = std::min(y0 + 1, out - 1);
        const float ax = fx[p] - x0, ay = fy[p] - y0;
        const float wts[4] = { (1 - ax) * (1 - ay), ax * (1 - ay), (1 - ax) * ay, ax * ay };
        const int idx[4] = { slot[(size_t)y0 * out + x0], slot[(size_t)y0 * out + x1],
                             slot[(size_t)y1 * out + x0], slot[(size_t)y1 * out + x1] };
        float* d = &res[(size_t)C * p];
        for (int k = 0; k < 4; ++k) {
            if (wts[k] == 0.0f) continue;
            const float* s = &hr[(size_t)idx[k] * C];
            for (int j = 0; j < C; ++j) d[j] += wts[k] * s[j];
        }
    }
    return res;
}

} // namespace trellis

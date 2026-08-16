// Shared TRELLIS.2 flow-DiT graph builder (dense path, B=1).
#pragma once
#include <map>
#include <string>

struct ggml_context;
struct ggml_tensor;

namespace trellis {
struct Model;

struct DiTParams {
    int n_blocks   = 30;
    int n_heads    = 12;
    int head_dim   = 128;
    int d_model    = 1536;
    int d_mlp      = 8192;     // int(1536 * 5.3334)
    int d_cond     = 1024;
    int in_ch      = 8;
    int out_ch     = 8;
    float ln_eps       = 1e-6f;
    float final_ln_eps = 1e-5f;
    float rms_eps      = 1e-12f;
    bool  cast_f32     = false;   // cast f16 weights to f32 before matmul (precision test)
    // Pixal3D "proj" image-attention mode. The block layout is identical to TRELLIS.2 except
    // that the cross-attention module is wrapped: its weights sit one level deeper, under
    // `cross_attn.cross_attn_block`, and a sibling `cross_attn.proj_linear` maps the per-token
    // view-aligned feature into model space and is added to the cross-attention output.
    bool  proj_mode    = false;
    int   proj_ch      = 0;       // proj_in_channels: 1024 bare, 2048 with the NAF branch
};

// Build the dense SS-flow forward graph (B=1). All input tensors live in `gctx`
// and must be flagged ggml_set_input by the caller; weights come from `m`.
//   h0   : [in_ch, L]          patchified input (channel-major)
//   tfreq: [256]               sinusoidal timestep embedding (host-computed)
//   cond : [d_cond, Lc]        conditioning tokens
//   proj : [proj_ch, L]        per-token view-aligned features (proj mode; else nullptr)
//   cos/sin: [1, head_dim/2, 1, L]  precomputed 3D-RoPE tables
// Returns the [out_ch, L] velocity; `inter` (optional) collects named intermediates.
ggml_tensor* build_dit_dense(ggml_context* gctx, const Model& m, const DiTParams& p,
                             ggml_tensor* h0, ggml_tensor* tfreq, ggml_tensor* cond,
                             ggml_tensor* proj, ggml_tensor* cos, ggml_tensor* sin,
                             std::map<std::string, ggml_tensor*>* inter = nullptr);

} // namespace trellis

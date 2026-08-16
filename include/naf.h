// NAF (Neighborhood Attention Filtering) feature upsampler — the guided upsampler Pixal3D
// runs between DINOv3 and the view-aligned projection for its shape/texture stages.
//
// Reference: valeoai/NAF, as pulled by DinoV3ProjFeatureExtractor._load_naf(). The network is
// image-guided and VFM-agnostic: the only learned part is a two-branch convolutional encoder
// over the RGB guide; the upsampling itself is a parameter-free neighborhood cross-attention
// whose queries come from the guide at high resolution, whose keys are those same guide
// features average-pooled back to the low-resolution grid, and whose values are the VFM
// (DINOv3) features. Nothing in the attention is learned, so a GGUF of the conv encoder plus
// the RoPE period buffer is the whole model.
//
// The neighborhood attention is dilated by exactly the upsampling factor, which makes each
// high-resolution query attend to a KxK window of LOW-resolution cells around its own cell —
// see naf.cpp for the derivation. That collapses NATTEN's dilated 2-D kernel into a plain
// KxK gather over the LR grid and is what makes a CPU implementation practical here.
#pragma once
#include <vector>

namespace trellis {
struct Model;

// Upsample `feats_lr` with `img01` as the guide, then bilinearly sample the result at
// `pts_xy`.
//   img01    : RGB guide in [0,1], torch [3,S,S] memory (== ggml [S,S,3,1]).
//   feats_lr : DINOv3 patch features, channel-major [C, Hf*Wf] (index c + C*(h*Wf + w)).
//   out      : side of the upsampled map; S must be an integer multiple of it, and `out`
//              an integer multiple of Hf/Wf (both hold for every Pixal3D stage).
//   pts_xy   : 2*NP sample positions as (x, y) pixel coordinates in the S-sized image frame.
// Returns [C * NP] channel-major (index c + C*p), matching the projection-conditioning layout.
std::vector<float> naf_sample(const Model& m,
                              const std::vector<float>& img01, int S,
                              const float* feats_lr, int Hf, int Wf, int C,
                              int out, const std::vector<float>& pts_xy);

} // namespace trellis

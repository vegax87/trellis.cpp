// Pixal3D view-aligned projection conditioning.
//
// Pixal3D keeps the TRELLIS.2 denoiser, sampler and decoders unchanged and swaps only how the
// image reaches the DiT. TRELLIS.2 cross-attends over all DINOv3 tokens; Pixal3D cross-attends
// over the 5 global tokens (cls + registers) only, and adds a PIXEL-ALIGNED term: every DiT
// token is a cell of a 3-D grid, that cell is projected into the image with a fixed frontal
// camera, and the DINOv3 feature map is sampled there. A per-block `proj_linear` maps the
// sampled vector into model space and adds it to the cross-attention output.
//
// Stages that set use_naf_upsample additionally sample a NAF-upsampled copy of the same feature
// map and concatenate it, which is why their proj_in_channels is 2048 rather than 1024.
#pragma once
#include <array>
#include <vector>

namespace trellis {
struct Model;

struct CameraParams {
    float camera_angle_x = 0.8575560450553894f;   // horizontal FOV in radians
    float distance       = 2.0f;                  // camera distance along the frontal axis
    float mesh_scale     = 1.0f;
};

// The reference derives `distance` in closed form from the FOV by requiring the grid corner at
// x = -1 to project onto the image border (inference.py: distance_from_fov). Only the FOV is a
// free parameter — upstream estimates it with MoGe-2, we take it from --fov.
CameraParams pixal3d_camera(float camera_angle_x, float mesh_scale = 1.0f,
                            int image_resolution = 512, int extend_pixel = 0);

// Project the centre of one grid cell of an R^3 grid to (x, y) pixel coordinates in an
// `image_resolution`-sized frame. Cells are indexed as the DiT tokenizes a dense grid: value
// along each axis is linspace(-1, 1, R)[i].
void pixal3d_project_cell(int R, int cx, int cy, int cz, const CameraParams& cam,
                          int image_resolution, float& px, float& py);

// The negative branch of classifier-free guidance is zeros_like(proj) — up to 400 MB at the
// texture stage's token budget — so it is not materialized: pass a null proj pointer and the
// runner zeroes the input tensor on the device instead.
struct ProjCond {
    std::vector<float> global;      // [1024 * 5] cls + 4 register tokens, channel-major
    std::vector<float> proj;        // [proj_ch * N] channel-major, one column per DiT token
    int proj_ch = 1024;
    int n_global = 5;
};

// Build the projection conditioning for one stage.
//   dino      : full dinov3_encode output, [1024, 5 + (S/16)^2] channel-major.
//   S         : the image size that produced `dino` (512 or 1024).
//   grid_res  : the projection grid resolution for this stage.
//   proj_ch   : the stage's proj_in_channels, read off its own proj_linear weight — 1024 for the
//               bare feature map, 2048 when the stage concatenates a NAF-upsampled copy.
//   coords    : nullptr for the dense sparse-structure stage (tokens = all grid_res^3 cells in
//               DiT order), otherwise the active voxel list, one token per entry.
//   naf       : NAF model, or nullptr. With proj_ch 2048 and no NAF model the upsampled half is
//               filled with the low-resolution samples so the stage still runs (--no-naf); that
//               is a degraded input, not an equivalent one.
//   img01     : raw [0,1] guide image, [3,S,S] torch CHW — required when `naf` is given.
//   naf_out   : NAF target resolution for this stage.
// Sample the same 3-D points from two DINOv3 feature maps produced at different input sizes and
// report the mean cosine similarity. Both maps encode the same image, so a point's features must
// agree strongly across them; the value only collapses if one map is being read with the wrong
// spatial layout. This isolates "the conditioning is wrong at 1024 but right at 512", which no
// other statistic in the pipeline can distinguish from a bad latent.
double pixal3d_cross_res_agreement(const std::vector<float>& dino_a, int Sa,
                                   const std::vector<float>& dino_b, int Sb,
                                   int grid_res, const CameraParams& cam,
                                   const std::vector<std::array<int, 3>>& coords);

ProjCond pixal3d_proj_cond(const std::vector<float>& dino, int S, int grid_res, int proj_ch,
                           const CameraParams& cam,
                           const std::vector<std::array<int, 3>>* coords,
                           const Model* naf, const std::vector<float>* img01, int naf_out);

} // namespace trellis

#include "pixal3d.h"
#include "naf.h"
#include "trellis_model.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace trellis {

static constexpr int D_DINO   = 1024;   // DINOv3 ViT-L/16 width
static constexpr int N_GLOBAL = 5;      // cls + 4 register tokens
static constexpr int PATCH    = 16;

// Blender's sensor model, as the reference reimplements it: a 32 mm sensor with the focal
// length recovered from the horizontal FOV.
static float focal_pixels(float camera_angle_x, int resolution) {
    const float focal = 16.0f / std::tan(camera_angle_x * 0.5f);
    return focal * (float)resolution / 32.0f;
}

CameraParams pixal3d_camera(float camera_angle_x, float mesh_scale,
                            int image_resolution, int extend_pixel) {
    CameraParams cam;
    cam.camera_angle_x = camera_angle_x;
    cam.mesh_scale     = mesh_scale;
    // Reference: the grid corner (-1, 0, 0) maps through the same rotation as every other grid
    // point, giving world x = -0.5/mesh_scale, and is required to land on the left image border
    // (target x = -extend_pixel). Solving the perspective divide for the camera distance:
    //   distance = f_pixels * x_world / x_ndc  with  x_ndc = -extend_pixel - resolution/2
    const float f_px = focal_pixels(camera_angle_x, image_resolution);
    const float x_world = -0.5f / mesh_scale;
    const float x_ndc   = -(float)extend_pixel - (float)image_resolution * 0.5f;
    cam.distance = f_px * x_world / x_ndc;
    return cam;
}

void pixal3d_project_cell(int R, int cx, int cy, int cz, const CameraParams& cam,
                          int image_resolution, float& px, float& py) {
    const float den = R > 1 ? (float)(R - 1) : 1.0f;
    const float gx = -1.0f + 2.0f * (float)cx / den;
    const float gy = -1.0f + 2.0f * (float)cy / den;
    const float gz = -1.0f + 2.0f * (float)cz / den;
    // ProjGrid rotates the grid into Blender axes with [[1,0,0],[0,0,-1],[0,1,0]] and halves it
    // by mesh_scale; the frontal view matrix then reduces the world->camera transform to
    // x_cam = wx, y_cam = wz, z_cam = -wy - distance.
    const float s = 1.0f / (2.0f * cam.mesh_scale);
    const float wx = gx * s, wy = -gz * s, wz = gy * s;
    const float depth = wy + cam.distance;
    const float f_px  = focal_pixels(cam.camera_angle_x, image_resolution);
    const float inv   = 1.0f / (depth + 1e-8f);
    px = f_px * wx * inv + (float)image_resolution * 0.5f;
    py = -f_px * wz * inv + (float)image_resolution * 0.5f;   // image y grows downward
}

// Bilinear sample of a channel-major [C, Hf*Wf] map at a pixel position expressed in the
// `S`-sized image frame. Mirrors F.grid_sample(align_corners=False, padding_mode="border").
static void sample_bilinear(const float* map, int Hf, int Wf, int C, int S,
                            float px, float py, float* dst) {
    float gx = (px + 0.5f) * (float)Wf / (float)S - 0.5f;
    float gy = (py + 0.5f) * (float)Hf / (float)S - 0.5f;
    gx = std::min(std::max(gx, 0.0f), (float)Wf - 1.0f);
    gy = std::min(std::max(gy, 0.0f), (float)Hf - 1.0f);
    const int x0 = (int)std::floor(gx), y0 = (int)std::floor(gy);
    const int x1 = std::min(x0 + 1, Wf - 1), y1 = std::min(y0 + 1, Hf - 1);
    const float ax = gx - x0, ay = gy - y0;
    const float w00 = (1 - ax) * (1 - ay), w10 = ax * (1 - ay);
    const float w01 = (1 - ax) * ay,       w11 = ax * ay;
    const float* p00 = map + (size_t)C * ((size_t)y0 * Wf + x0);
    const float* p10 = map + (size_t)C * ((size_t)y0 * Wf + x1);
    const float* p01 = map + (size_t)C * ((size_t)y1 * Wf + x0);
    const float* p11 = map + (size_t)C * ((size_t)y1 * Wf + x1);
    for (int j = 0; j < C; ++j)
        dst[j] = w00 * p00[j] + w10 * p10[j] + w01 * p01[j] + w11 * p11[j];
}

ProjCond pixal3d_proj_cond(const std::vector<float>& dino, int S, int grid_res, int proj_ch,
                           const CameraParams& cam,
                           const std::vector<std::array<int, 3>>* coords,
                           const Model* naf, const std::vector<float>* img01, int naf_out) {
    if (proj_ch != D_DINO && proj_ch != 2 * D_DINO)
        throw std::runtime_error("pixal3d: unsupported proj_in_channels");
    const int Hp = S / PATCH;
    const size_t ntok = dino.size() / D_DINO;
    if (ntok != (size_t)N_GLOBAL + (size_t)Hp * Hp)
        throw std::runtime_error("pixal3d: DINOv3 token count does not match the image size");

    ProjCond out;
    out.n_global = N_GLOBAL;
    out.global.assign(dino.begin(), dino.begin() + (size_t)D_DINO * N_GLOBAL);

    // The DINOv3 output is channel-major per token and the patch tokens are row-major over the
    // patch grid, which is exactly the [C, Hf*Wf] layout the samplers want — no repacking, just
    // skip the global tokens.
    const float* patches = dino.data() + (size_t)D_DINO * N_GLOBAL;

    const size_t N = coords ? coords->size() : (size_t)grid_res * grid_res * grid_res;
    std::vector<float> pts(2 * N);
    for (size_t t = 0; t < N; ++t) {
        int cx, cy, cz;
        if (coords) { cx = (*coords)[t][0]; cy = (*coords)[t][1]; cz = (*coords)[t][2]; }
        else {
            const int R = grid_res;
            cx = (int)(t / ((size_t)R * R)); cy = (int)((t / R) % R); cz = (int)(t % R);
        }
        pixal3d_project_cell(grid_res, cx, cy, cz, cam, S, pts[2 * t], pts[2 * t + 1]);
    }

    out.proj_ch = proj_ch;
    out.proj.assign((size_t)proj_ch * N, 0.0f);

    for (size_t t = 0; t < N; ++t)
        sample_bilinear(patches, Hp, Hp, D_DINO, S, pts[2 * t], pts[2 * t + 1],
                        &out.proj[(size_t)proj_ch * t]);

    if (proj_ch == 2 * D_DINO) {
        if (naf) {
            if (!img01) throw std::runtime_error("pixal3d: NAF upsampling needs the raw [0,1] guide");
            std::vector<float> hr = naf_sample(*naf, *img01, S, patches, Hp, Hp, D_DINO, naf_out, pts);
            for (size_t t = 0; t < N; ++t)
                std::copy(hr.begin() + (size_t)D_DINO * t, hr.begin() + (size_t)D_DINO * (t + 1),
                          out.proj.begin() + (size_t)proj_ch * t + D_DINO);
        } else {
            // --no-naf: repeat the low-resolution samples in place of the upsampled branch. The
            // stage runs and stays roughly on-distribution, but every high-frequency cue the
            // second branch was trained to carry is gone.
            for (size_t t = 0; t < N; ++t)
                std::copy(out.proj.begin() + (size_t)proj_ch * t,
                          out.proj.begin() + (size_t)proj_ch * t + D_DINO,
                          out.proj.begin() + (size_t)proj_ch * t + D_DINO);
        }
    }

    return out;
}

} // namespace trellis

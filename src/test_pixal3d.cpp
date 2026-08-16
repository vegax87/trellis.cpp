// Golden-value check for the Pixal3D view-aligned projection camera.
//
// The expected numbers come from the reference implementation transcribed literally out of
// Pixal3D (ProjGrid.forward + project_points_to_image_batch in
// pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py, distance_from_fov in
// inference.py) and evaluated in double precision. Everything downstream — which DINOv3 patch a
// DiT token reads, and therefore whether the generated geometry lands on the silhouette — hangs
// off these two functions, and neither has a runtime signal when it is subtly wrong.
//
//   trellis-test-pixal3d
#include "pixal3d.h"

#include <cmath>
#include <cstdio>

namespace {

struct Golden { int R, res, tok; double x, y; };

// camera_angle_x = Pixal3D's default, mesh_scale 1, distance derived at 512 with extend_pixel 0.
constexpr double CAX      = 0.8575560450553894;
constexpr double DISTANCE = 1.093750014;

const Golden kGolden[] = {
    { 16,  512,      0,   80.313726,  431.686274 },
    { 16,  512,      1,   72.643930,  439.356070 },
    { 16,  512,     16,   80.313726,  408.261438 },
    { 16,  512,    256,  103.738562,  431.686274 },
    { 16,  512,   4095,  727.578934, -215.578934 },
    { 16,  512,   2055,  272.561922,  504.428833 },
    { 32,  512,      0,   80.313726,  431.686274 },
    { 32,  512,      1,   76.684313,  435.315687 },
    { 32,  512,     32,   80.313726,  420.351676 },
    { 32,  512,   1024,   91.648324,  431.686274 },
    { 32,  512,  32767,  727.578934, -215.578934 },
    { 32,  512,  16391,  262.602800,  460.686808 },
    { 64, 1024,      0,  160.627452,  863.372548 },
    { 64, 1024,      1,  157.092739,  866.907261 },
    { 64, 1024,     64,  160.627452,  852.217864 },
    { 64, 1024,   4096,  171.782136,  863.372548 },
    { 64, 1024, 262143, 1455.157869, -431.157869 },
    { 64, 1024, 131079,  517.995316,  889.704917 },
};

} // namespace

int main() {
    int bad = 0;

    trellis::CameraParams cam = trellis::pixal3d_camera((float)CAX, 1.0f, 512, 0);
    const double dd = std::fabs(cam.distance - DISTANCE);
    printf("distance %.9f (expected %.9f, delta %.2e)\n", cam.distance, DISTANCE, dd);
    if (dd > 1e-5) { printf("  FAIL: camera distance\n"); ++bad; }

    for (const Golden& g : kGolden) {
        const int cx = g.tok / (g.R * g.R), cy = (g.tok / g.R) % g.R, cz = g.tok % g.R;
        float px = 0, py = 0;
        trellis::pixal3d_project_cell(g.R, cx, cy, cz, cam, g.res, px, py);
        const double ex = std::fabs(px - g.x), ey = std::fabs(py - g.y);
        // f32 across a perspective divide at ~1e3 pixel magnitudes: 1e-3 px is the noise floor.
        const bool ok = ex < 2e-3 && ey < 2e-3;
        printf("R=%3d res=%4d tok=%7d  x=%12.6f/%12.6f  y=%12.6f/%12.6f  %s\n",
               g.R, g.res, g.tok, (double)px, g.x, (double)py, g.y, ok ? "ok" : "FAIL");
        if (!ok) ++bad;
    }

    // The projection is defined in normalized image space, which is why one camera solved at 512
    // serves the 1024 stages too: doubling the resolution must exactly double the pixel offset
    // from the image centre.
    for (int tok = 0; tok < 4096; tok += 373) {
        const int cx = tok / 256, cy = (tok / 16) % 16, cz = tok % 16;
        float a[2], b[2];
        trellis::pixal3d_project_cell(16, cx, cy, cz, cam,  512, a[0], a[1]);
        trellis::pixal3d_project_cell(16, cx, cy, cz, cam, 1024, b[0], b[1]);
        for (int k = 0; k < 2; ++k) {
            const double lhs = (b[k] - 512.0), rhs = 2.0 * (a[k] - 256.0);
            if (std::fabs(lhs - rhs) > 2e-3) {
                printf("  FAIL: scale invariance at tok=%d axis=%d (%.6f vs %.6f)\n", tok, k, lhs, rhs);
                ++bad;
            }
        }
    }

    printf(bad ? "\n%d check(s) FAILED\n" : "\nall checks passed\n", bad);
    return bad ? 1 : 0;
}

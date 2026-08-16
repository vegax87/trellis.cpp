#include "trellis_args.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace trellis {

const char* model_family_name(ModelFamily f) {
    return f == ModelFamily::Pixal3D ? "pixal3d" : "trellis";
}

void print_usage(const char* argv0, bool server) {
    if (server) {
        fprintf(stderr,
            "usage: %s [--host H] [--port P] [--models DIR] [--gpu N] [generation defaults...]\n",
            argv0);
    } else {
        fprintf(stderr,
            "usage: %s <image.png> <out.glb> [options]\n"
            "   or: %s --image <image.png> --output <out.glb> [options]\n",
            argv0, argv0);
    }
    fprintf(stderr,
        "\n"
        "  -i, --image PATH        input image                  (image->3D)\n"
        "  -o, --output PATH       output .glb                  (default model.glb)\n"
        "      --copyright TEXT    glTF asset.copyright metadata\n"
        "  -m, --models DIR        GGUF model directory\n"
        "      --model FAMILY      trellis (default) | pixal3d — which flow weights the model\n"
        "                          directory holds. pixal3d swaps the DINOv3 cross-attention for\n"
        "                          view-aligned projection conditioning; the samplers, decoders\n"
        "                          and every postprocessing stage are shared.\n"
        "      --fov DEG           pixal3d: horizontal field of view of the input image, which\n"
        "                          fixes the projection camera (default 49.13, Pixal3D's own).\n"
        "                          Upstream estimates this with MoGe-2; that model is not ported,\n"
        "                          so a wrong FOV shows up as geometry drifting off the silhouette.\n"
        "      --mesh-scale F      pixal3d: object scale inside the unit grid   (default 1.0)\n"
        "      --no-naf            pixal3d: skip NAF guided upsampling (needs no naf.gguf, but\n"
        "                          the shape/texture stages then lose their high-frequency\n"
        "                          projection branch)\n"
        "      --gpu N             GPU index, <0 = CPU          (default 0)\n"
        "  -s, --seed N            RNG seed                     (default 42)\n"
        "      --res 512|1024|1536 geometry resolution\n"
        "      --max-tokens N      HR token budget              (default 49152)\n"
        "      --bg-removal MODE   threshold | birefnet   (default: auto -- a pre-matted\n"
        "                          image keeps its alpha; otherwise BiRefNet when its model\n"
        "                          is present. The plain threshold matte cuts out specular\n"
        "                          highlights, which the flow then turns into holes.)\n"
        "      --birefnet          alias for --bg-removal birefnet\n"
        "      --no-texture        geometry only\n"
        "      --xatlas            xatlas UV unwrap (default)\n"
        "      --box-uv            voxel-native box projection (faster)\n"
        "      --band N            narrow-band DC remesh band width (default: auto —\n"
        "                          res/512, i.e. 1 @512 / 2 @1024, which suppresses the\n"
        "                          res-1024 outer-skin speckle; N forces that width)\n"
        "      --decim GRID        legacy cluster-grid decimation (default: quadric\n"
        "                          simplify to 300K faces @1024 / 150K @512; 0 = none)\n"
        "      --atlas PX          UV atlas size (default 2048 @1024 / 1024 @512)\n"
        "      --tex-res N         texture PBR resolution 512/1024 (default: auto — drops\n"
        "                          a dense res-1024 decode to a clean res-512 PBR volume)\n"
        "      --webp on|off       encode GLB textures as WebP (default: on when built with\n"
        "                          WebP support; off = PNG)\n"
        "      --dump-bg           also write the background-removal cutout as <out>_cutout.png\n"
        "      --bg-only           background removal only: write the cutout and skip the rest\n"
        "      --f32               f32 sparse-conv compute\n"
        "      --no-fa             disable FlashAttention\n"
        "      --require-gpu       refuse CPU fallback\n"
        "      --threads N         CPU backend threads      (default all cores)\n"
        "      --gss F  --gsh F    guidance strengths\n"
        "      --host H  --port P  trellis-server bind address\n"
        "      --voxply            also dump the voxel point cloud as .ply\n"
        "      --dump-slat         dump the structured latent to disk\n"
        "  -h, --help              show this help\n");
}

bool parse_args(int argc, char** argv, TrellisParams& p) {
    int positional = 0;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* name) -> const char* {
            if (i + 1 >= argc) { fprintf(stderr, "[trellis] %s needs a value\n", name); return nullptr; }
            return argv[++i];
        };
        auto need = [&](const char* name) -> const char* {
            const char* v = next(name);
            return v;
        };

        if      (a == "-h" || a == "--help")    { p.help = true; return false; }
        else if (a == "-i" || a == "--image")   { const char* v = need(a.c_str()); if (!v) return false; p.image = v; }
        else if (a == "-o" || a == "--output")  { const char* v = need(a.c_str()); if (!v) return false; p.output = v; }
        else if (a == "--copyright")            { const char* v = need(a.c_str()); if (!v) return false; p.copyright = v; }
        else if (a == "-m" || a == "--models")  { const char* v = need(a.c_str()); if (!v) return false; p.models = v; }
        else if (a == "--model")                { const char* v = need(a.c_str()); if (!v) return false;
                                                  if      (std::strcmp(v, "trellis") == 0) p.family = ModelFamily::Trellis;
                                                  else if (std::strcmp(v, "pixal3d") == 0) p.family = ModelFamily::Pixal3D;
                                                  else { fprintf(stderr, "[trellis] unknown model family: %s (trellis|pixal3d)\n", v); return false; } }
        else if (a == "--fov")                  { const char* v = need(a.c_str()); if (!v) return false; p.fov_deg = (float)atof(v); }
        else if (a == "--mesh-scale")           { const char* v = need(a.c_str()); if (!v) return false; p.mesh_scale = (float)atof(v); }
        else if (a == "--no-naf")               { p.naf = false; }
        else if (a == "--gpu")                  { const char* v = need(a.c_str()); if (!v) return false; p.gpu = atoi(v); }
        else if (a == "-s" || a == "--seed")    { const char* v = need(a.c_str()); if (!v) return false; p.seed = (uint32_t)atoi(v); }
        else if (a == "--res")                  { const char* v = need(a.c_str()); if (!v) return false; p.set_res(atoi(v)); }
        else if (a == "--max-tokens")           { const char* v = need(a.c_str()); if (!v) return false; p.max_tokens = atoi(v); }
        else if (a == "--bg-removal")           { const char* v = need(a.c_str()); if (!v) return false; p.birefnet = (std::strcmp(v, "birefnet") == 0) ? 1 : 0; }
        else if (a == "--birefnet")             { p.birefnet = 1; }
        else if (a == "--no-texture")           { p.texture = false; }
        else if (a == "--xatlas")               { p.xatlas = true; }
        else if (a == "--box-uv")               { p.xatlas = false; }
        else if (a == "--band")                 { const char* v = need(a.c_str()); if (!v) return false; p.band = atoi(v); }
        else if (a == "--decim")                { const char* v = need(a.c_str()); if (!v) return false; p.decim = atoi(v); }
        else if (a == "--atlas" || a == "--tex"){ const char* v = need(a.c_str()); if (!v) return false; p.tex = atoi(v); }
        else if (a == "--tex-res")              { const char* v = need(a.c_str()); if (!v) return false; p.tex_res = atoi(v); }
        else if (a == "--webp")                 { const char* v = need(a.c_str()); if (!v) return false;
                                                  p.webp = (std::strcmp(v,"off")==0 || std::strcmp(v,"0")==0 || std::strcmp(v,"false")==0) ? 0
                                                         : (std::strcmp(v,"on")==0 || std::strcmp(v,"1")==0 || std::strcmp(v,"true")==0) ? 1 : -1; }
        else if (a == "--dump-bg")              { p.dump_bg = true; }
        else if (a == "--bg-only")              { p.bg_only = true; p.dump_bg = true; }
        else if (a == "--f32")                  { p.f32 = true; }
        else if (a == "--no-fa")                { p.no_fa = true; }
        else if (a == "--require-gpu")          { p.require_gpu = true; }
        else if (a == "--threads")              { const char* v = need(a.c_str()); if (!v) return false; p.threads = atoi(v); }
        else if (a == "--gss")                  { const char* v = need(a.c_str()); if (!v) return false; p.gss = (float)atof(v); }
        else if (a == "--gsh")                  { const char* v = need(a.c_str()); if (!v) return false; p.gsh = (float)atof(v); }
        else if (a == "--host")                 { const char* v = need(a.c_str()); if (!v) return false; p.host = v; }
        else if (a == "--port")                 { const char* v = need(a.c_str()); if (!v) return false; p.port = atoi(v); }
        else if (a == "--voxply")               { p.voxply = true; }
        else if (a == "--dump-slat")            { p.dump_slat = true; }
        else if (!a.empty() && a[0] == '-')     { fprintf(stderr, "[trellis] unknown option: %s\n", a.c_str()); return false; }
        else if (positional == 0)               { p.image  = a; positional = 1; }
        else if (positional == 1)               { p.output = a; positional = 2; }
        else                                    { fprintf(stderr, "[trellis] unexpected argument: %s\n", a.c_str()); return false; }
    }
    return true;
}

}  // namespace trellis

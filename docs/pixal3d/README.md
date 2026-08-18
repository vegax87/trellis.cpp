# Pixal3D backend for trellis.cpp

`--model pixal3d` runs [TencentARC/Pixal3D](https://github.com/TencentARC/Pixal3D)
(*Pixal3D: Pixel-Aligned 3D Generation from Images*, SIGGRAPH 2026) on the existing
trellis.cpp engine. `--model trellis` (the default) is unchanged TRELLIS.2.

Pixal3D is a fine-tune of TRELLIS.2, not a new architecture. Same 1.3B DiT, same 30
blocks, same flow-Euler sampler with guidance intervals, same sparse-structure /
shape / texture cascade, and **literally the same decoder checkpoints**. One thing
changes: how the image reaches the denoiser. That is why the integration is a new
conditioning module and two lines inside the DiT block rather than a second engine.

---

## What actually differs

TRELLIS.2 cross-attends over the full DINOv3 token stream — 1 cls + 4 register +
`(S/16)²` patch tokens — and lets attention figure out which part of the image a
voxel corresponds to.

Pixal3D splits that into two paths:

1. **Global.** Cross-attention over the 5 global tokens only (cls + registers). The
   patch tokens never enter the cross-attention.
2. **Pixel-aligned.** Every DiT token *is* a cell of a 3-D grid. That cell's centre is
   projected into the image with a fixed frontal camera, the DINOv3 patch feature map
   is bilinearly sampled at the projected pixel, and a per-block `proj_linear` maps
   the sampled vector into model space. The result is **added to the cross-attention
   output** and takes its place as the residual branch.

In checkpoint terms (`ProjectAttention` / `SparseProjectAttention`):

```
TRELLIS.2                          Pixal3D
blocks.N.cross_attn.to_q           blocks.N.cross_attn.cross_attn_block.to_q
blocks.N.cross_attn.to_kv          blocks.N.cross_attn.cross_attn_block.to_kv
blocks.N.cross_attn.to_out         blocks.N.cross_attn.cross_attn_block.to_out
                                   blocks.N.cross_attn.proj_linear.{weight,bias}
```

Everything else in the block — `self_attn`, `norm1..3`, `modulation`, `mlp`,
`input_layer`, `out_layer`, the RoPE tables — is bit-for-bit the same layout, which is
why `build_dit_dense` builds both with one `proj_mode` branch.

The shape and texture stages additionally sample a **NAF-upsampled** copy of the same
feature map and concatenate it, which is why their `proj_in_channels` is 2048 instead
of 1024. `trellis.cpp` never guesses that number: it reads it off
`blocks.0.cross_attn.proj_linear.weight`, which also serves as the check that the
GGUF directory matches `--model`.

### Per-stage projection parameters

Taken from Pixal3D's stage configs (`configs/gen/*_proj_finetune*.json`):

| stage | GGUF | grid | DINOv3 image | NAF target | proj ch |
|---|---|---|---|---|---|
| sparse structure | `pixal3d_ss_flow.gguf` | 16³ (dense, = the DiT's own token grid) | 512 | — | 1024 |
| shape LR | `pixal3d_shape_flow_512.gguf` | 32³ (active voxels) | 512 | 512 | 2048 |
| shape HR | `pixal3d_shape_flow_1024.gguf` | `hr_res/16` | 1024 | 512 | 2048 |
| texture | `pixal3d_tex_flow_1024.gguf` | `hr_res/16` | 1024 | 1024 | 2048 |

The HR grid follows the cascade's token-budget backoff (`--max-tokens`): when the
`1536` target steps down, the projection grid steps with it, exactly as the reference
overrides its cond model's `grid_resolution` per call.

### Camera

The projection needs the camera the photo was taken with. The reference estimates the
horizontal FOV with **MoGe-2** and then derives the camera distance from it in closed
form (`distance_from_fov`), by requiring the grid corner `x = -1` to land on the image
border.

MoGe-2 is **not ported**. The closed-form distance is, so the FOV is the only free
parameter and it is a flag:

```
--fov DEG           horizontal field of view (default 49.13°, Pixal3D's own default)
--mesh-scale F      object scale inside the unit grid (default 1.0)
--extend-pixel N    how far the subject continues past the image border (default 0)
```

`--extend-pixel` matters more than it sounds. The solve registers the grid corner onto
the image border, which silently assumes the subject is *fully inside the frame*. Feed it
an image cropped at the edges — a generator asked for a close-up, say — and the object is
squeezed into the grid: the visible part comes out compressed and the unseen part is
invented at the wrong scale. Raising it moves the virtual border outward, and the camera
steps back to match (49.13° at 512: 0 px → distance 1.094, 64 px → 0.875, 128 px → 0.729).
Upstream has the same parameter but never exposes it, since MoGe-2 sees the whole frame.

The projection is resolution-independent once normalized, so one camera solved at 512
serves the 1024 stages too. `trellis-test-pixal3d` pins both functions to golden values
taken from the reference implementation:

```
$ trellis-test-pixal3d
distance 1.093750000 (expected 1.093750014, delta 1.40e-08)
R= 16 res= 512 tok=      0  x=   80.313721/   80.313726  ...  ok
...
all checks passed
```

A wrong FOV does not crash anything — it shows up as generated geometry drifting off
the input silhouette, thicker or flatter than the object. If that is what you see, try
`--fov` a few degrees either side before blaming the weights.

---

## NAF

The shape and texture stages upsample the DINOv3 feature map before sampling it, using
[valeoai/NAF](https://github.com/valeoai/NAF) (*Neighborhood Attention Filtering*).
NAF is ported here in full (`src/naf.cpp`), which needs one extra GGUF.

NAF is small and unusually easy to port because **the upsampling itself has no
parameters**. The only learned part is a two-branch convolutional encoder over the RGB
guide (a 1×1 branch and a 3×3 branch, each `Conv2d → 2× EncBlock(GroupNorm/SiLU)`,
reflect-padded, concatenated to 256 channels). Everything after that is a
parameter-free neighborhood cross-attention:

- **queries** — the guide features at the target resolution, with axial RoPE applied,
- **keys** — those same features average-pooled back onto the DINOv3 grid,
- **values** — the DINOv3 features themselves, split into 4 heads of 256 channels.

The attention is dilated by exactly the upsampling factor. That detail is what makes a
CPU implementation practical: with dilation `d`, tap `t` of query `i` reads index
`(i%d) + (s+t)·d`, whose low-resolution cell is just `s+t`. NATTEN's dilated 2-D kernel
therefore collapses to a plain **9×9 window of low-resolution cells centred on the
query's own cell**, clamped at the border. `src/naf.cpp` computes exactly that, on
CPU, threaded, and only for the pixels the projection actually reads (the four bilinear
taps of each grid point, deduplicated) rather than for the whole upsampled map.

Two consequences worth knowing:

- **The texture stage is the memory peak.** It upsamples to the full 1024, so the guide
  encoder runs at 1024² and the query map alone is ~1 GB host, plus up to ~800 MB of
  attended pixels at the 49152-token budget, plus the ~2.4 GB im2col below — all while
  the 1.3B flow model is resident. If a 16 GB device OOMs there, lower `--max-tokens`
  or fall back to `--no-naf`.
- The guide encoder runs at the full image size in ggml. `ggml_conv_2d` goes through
  im2col, which materializes ~2.4 GB for the 1024-guide blocks. Set
  `TRELLIS_NAF_CONV_DIRECT=1` to use `ggml_conv_2d_direct` instead where the backend
  implements it (CUDA), trading portability for that buffer.
- `--no-naf` runs the shape/texture stages without `naf.gguf` by repeating the
  low-resolution samples in place of the upsampled branch. The stage runs and stays
  roughly on-distribution, but every high-frequency cue the second branch was trained
  to carry is gone. It is an escape hatch, not an equivalent.

The port was checked end to end against an independent transcription of the reference
(`naf.py` + `layers/{convolutions,rope,attentions}.py`) driven by random weights, on
both a pooled and an unpooled configuration, with sample points outside the frame to
exercise the border clamp: **max absolute error 2.3e-6**. The collapsed window and
NATTEN's own `get_window_start` on the upsampled grid agreed exactly, so the reduction
above is an identity, not an approximation, for every stage here (it needs
`length % dilation == 0`, which all four satisfy).

---

## Models

**Pre-built GGUFs:** [`vegax87/Pixal3D`](https://huggingface.co/vegax87/Pixal3D) — the five
Pixal3D-specific files, already named as the loader expects. Drop them into the model
directory you already use for TRELLIS.2:

```bash
for f in ss_flow shape_flow_512 shape_flow_1024 tex_flow_1024 naf; do
  curl -fL -o "$MODELS/pixal3d_$f.gguf" \
    "https://huggingface.co/vegax87/Pixal3D/resolve/main/pixal3d_$f.gguf"
done
```

**One model directory serves both families.** Pixal3D's checkpoints carry the same
upstream filenames as TRELLIS.2's, so the family-specific ones are written with a
`pixal3d_` prefix. Everything that is byte-identical between the two keeps its plain
name and is shared — adding Pixal3D to a working TRELLIS.2 set is **5 new files**, not
a second copy of everything.

| file | source | shared? |
|------|--------|---------|
| `pixal3d_ss_flow.gguf` | `TencentARC/Pixal3D` `ss_flow_img_dit_1_3B_64` | no — proj-conditioned |
| `pixal3d_shape_flow_512.gguf` | `…/slat_flow_img2shape_dit_1_3B_512` | no — `proj_in_channels` 2048 |
| `pixal3d_shape_flow_1024.gguf` | `…/slat_flow_img2shape_dit_1_3B_1024` | no |
| `pixal3d_tex_flow_1024.gguf` | `…/slat_flow_imgshape2tex_dit_1_3B_1024` | no; **no res-512 variant exists** |
| `pixal3d_naf.gguf` | `valeoai/NAF` `naf_release.pth` | no — Pixal3D only, a few MB |
| `shape_dec.gguf`, `tex_dec.gguf`, `ss_dec.gguf` | `…/{shape,tex}_dec_next_dc_f16c32`, `ss_dec_conv3d_16l8` | **yes** — unchanged from TRELLIS.2 |
| `dinov3.gguf` | DINOv3 ViT-L/16 | **yes** |
| `birefnet.gguf` | BiRefNet | **yes** |

To convert from the source checkpoints instead, the same tool handles both families and
applies the prefix automatically. Paths come from the environment, so nothing in the
script needs editing:

```bash
export TRELLIS_FAMILY=pixal3d
export PIXAL3D_MODELS=/path/to/TencentARC-Pixal3D    # the ckpts/ tree
export TRELLIS_GGUF_OUT=/path/to/models
export NAF_CKPT=/path/to/naf_release.pth             # optional; defaults to $PIXAL3D_MODELS/naf/

python tools/convert.py             # everything
python tools/convert.py naf         # just the upsampler
```

NAF is the one component read through torch rather than safetensors, because it ships as
a `.pth` GitHub release. It converts to ~1.3 MB: only `image_encoder.*` carries weights.

Tensor names are preserved verbatim, so the extra `proj_linear` / `cross_attn_block`
tensors need no remapping.

Because Pixal3D publishes only the 1024 texture flow:

- `--res 512` has no texture model to run and falls back to geometry only,
- the cascade's mixed-resolution shortcut (res-1024 geometry + res-512 texture, which
  suppresses the dense-decode speckle) is disabled; Pixal3D always textures at the
  cascade resolution.

Third-party GGUF conversions of Pixal3D exist on the Hub (search `Pixal3D gguf`). Most
were produced for the PyTorch pipeline with ComfyUI-style tooling, which stores tensors
in a quantization-friendly 2-D layout — a 1536-element RMS-norm gamma becomes `[256, 6]`
rather than `[128, 12]` — and records the real shape in a `*.orig_shape.<tensor>`
metadata key. The loader restores those shapes on load, so such files work; it logs
`restored N reshaped tensor(s)` when it does. Element counts must match exactly, so a
padded (rather than merely reshaped) tensor is left alone and will fail loudly.

What still has to hold is the naming: the tensors must carry the verbatim torch
`state_dict` names. `tools/gguf_probe.py` answers that from ~4 MB of HTTP Range, without
downloading the weights:

```bash
tools/gguf_probe.py https://huggingface.co/USER/REPO/resolve/main/some_flow.gguf
```

It reports the tensor names, the per-block structure (a SLat flow is 30 × 23 + 10 =
700 tensors), `proj_in_channels`, any foreign-runtime metadata, and the dtypes
trellis.cpp is picky about — the 1-D parameters must be F32, because `dit.cpp` adds
`modulation` to an f32 timestep embedding and multiplies `norm2.weight` into an f32
activation. BF16 matmul weights are fine, but note `--f32` only casts F16 and so will
not affect them.

---

## Usage

```bash
# Pixal3D, default cascade — same model directory as TRELLIS.2
trellis-cli in.png out.glb --model pixal3d --models /models

# with a measured FOV, and BiRefNet matting
trellis-cli in.png out.glb --model pixal3d --fov 38 --bg-removal birefnet

# no naf.gguf available
trellis-cli in.png out.glb --model pixal3d --no-naf
```

`trellis-server` takes the same flags at launch.

Every other flag — `--res`, `--max-tokens`, `--gss`/`--gsh`, `--band`, `--decim`,
`--atlas`, `--box-uv`, `--tex-res`, `--seed` — behaves identically, because everything
downstream of the conditioning is shared code.

Mismatched weights fail fast rather than producing garbage:

```
--model pixal3d but the checkpoint has no cross_attn.proj_linear (these are TRELLIS.2 weights)
--model trellis but the checkpoint has cross_attn.proj_linear (these are Pixal3D weights; pass --model pixal3d)
```

---

## Code map

| file | role |
|---|---|
| `include/pixal3d.h`, `src/pixal3d.cpp` | camera solve, grid projection, feature sampling, per-stage `ProjCond` assembly |
| `include/naf.h`, `src/naf.cpp` | NAF guide encoder (ggml) + neighborhood attention (CPU, threaded) |
| `src/dit.cpp` | `proj_mode` branch in the block: deeper cross-attn prefix + `proj_linear` add |
| `include/flow_runner.h`, `src/flow_runner.cpp` | `FlowCond` carries `(cond, proj)` through CFG; `DitRunner` gains the proj input |
| `src/trellis_cli.cpp` | per-stage projection parameters, family detection, guide preparation |
| `src/test_pixal3d.cpp` | golden-value test for the camera and projection |
| `tools/convert.py` | `TRELLIS_FAMILY=pixal3d` manifest + NAF `.pth` converter |

## Known gaps

- **MoGe-2 FOV estimation is not ported.** Use `--fov`.
- **No end-to-end numerical parity run on real weights.** The projection is pinned to
  golden values from the reference and NAF was diffed against a transcription of it
  under random weights, but the full pipeline has not been run against the PyTorch
  implementation on the released checkpoints — that needs the weights and a CUDA box.
- **NAF's pre-downsample branch is not implemented.** It only triggers when the guide
  exceeds 4× the NAF target, which no Pixal3D stage does; the code refuses rather than
  diverging if a stage ever would.
- **Multi-view conditioning is out of scope.** Pixal3D trains with `num_views: 2` but
  the released inference pipeline is single-view, and so is this.

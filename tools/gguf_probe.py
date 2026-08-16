#!/usr/bin/env python3
"""Print a GGUF's tensor table without downloading the weights.

GGUF puts its metadata and full tensor index at the head of the file, so a few MB of
HTTP Range is enough to answer the only two questions that decide whether a third-party
conversion will load in trellis.cpp:

  1. are the tensor names the verbatim torch state_dict names, and
  2. what dtype are the 1-D parameters (the per-block `modulation` vector, the norm
     weights) stored as — trellis.cpp adds `modulation` to an f32 timestep embedding and
     multiplies `norm2.weight` into an f32 activation, so f16 there is a silent
     mismatch, not a slow path.

Usage:
    tools/gguf_probe.py <file.gguf | https://.../file.gguf> [name-substring ...]

    tools/gguf_probe.py https://huggingface.co/USER/REPO/resolve/main/model.gguf

With no substrings it prints a summary plus the tensors trellis.cpp is picky about.
Only the standard library is used, so it runs anywhere python does.
"""
import struct
import sys
import urllib.request

GGUF_MAGIC = 0x46554747

# ggml_type -> name, for the types a checkpoint conversion realistically emits.
TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0",
    9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K",
    15: "Q8_K", 30: "BF16",
}

# What trellis.cpp requires beyond "the tensor exists". See src/dit.cpp.
PROBES = [
    ("blocks.0.cross_attn.proj_linear.weight", None,  "Pixal3D marker; ne[0] is proj_in_channels"),
    ("blocks.0.cross_attn.cross_attn_block.to_q.weight", None, "proj-mode cross-attn nesting"),
    ("blocks.0.modulation",                    "F32",  "added to the f32 timestep embedding"),
    ("blocks.0.norm2.weight",                  "F32",  "multiplied into an f32 activation"),
    ("blocks.0.norm2.bias",                    "F32",  "added to an f32 activation"),
    ("blocks.0.self_attn.q_rms_norm.gamma",    None,   "f16 tolerated (cast at graph build)"),
    ("input_layer.weight",                     None,   "matmul weight, f16/quant fine"),
]


class Head:
    """Reads a local file or an HTTP URL, fetching in chunks and caching what it read."""

    def __init__(self, src, chunk=1 << 22):
        self.src, self.chunk, self.buf = src, chunk, b""
        self.remote = src.startswith("http://") or src.startswith("https://")
        if not self.remote:
            self.fh = open(src, "rb")

    def _grow(self, need):
        while len(self.buf) < need:
            if self.remote:
                lo, hi = len(self.buf), len(self.buf) + max(self.chunk, need - len(self.buf)) - 1
                req = urllib.request.Request(self.src, headers={"Range": f"bytes={lo}-{hi}"})
                with urllib.request.urlopen(req) as r:
                    if r.status != 206:
                        raise SystemExit("server ignored the Range request; download the file instead")
                    part = r.read()
            else:
                self.fh.seek(len(self.buf))
                part = self.fh.read(max(self.chunk, need - len(self.buf)))
            if not part:
                raise SystemExit("unexpected end of file while reading the GGUF header")
            self.buf += part

    def at(self, off, n):
        self._grow(off + n)
        return self.buf[off:off + n]


def main(argv):
    if len(argv) < 2:
        raise SystemExit(__doc__)
    src, wanted = argv[1], argv[2:]
    h = Head(src)
    pos = 0

    def take(n):
        nonlocal pos
        b = h.at(pos, n)
        pos += n
        return b

    def u32():
        return struct.unpack("<I", take(4))[0]

    def u64():
        return struct.unpack("<Q", take(8))[0]

    def string():
        return take(u64()).decode("utf-8", "replace")

    if u32() != GGUF_MAGIC:
        raise SystemExit(f"{src}: not a GGUF file")
    version = u32()
    n_tensors = u64()
    n_kv = u64()

    # Value readers by gguf_metadata_value_type. Only the scalar payloads need real
    # decoding; everything else just has to be skipped by the right number of bytes.
    fixed = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

    def value(t):
        if t in fixed:
            return take(fixed[t])
        if t == 8:
            return string()
        if t == 9:
            et, n = u32(), u64()
            return [value(et) for _ in range(n)]
        raise SystemExit(f"unhandled metadata value type {t}")

    kv = {}
    for _ in range(n_kv):
        k = string()
        v = value(u32())
        kv[k] = v

    tensors = []
    for _ in range(n_tensors):
        name = string()
        ne = [u64() for _ in range(u32())]
        ttype = u32()
        u64()  # offset
        tensors.append((name, ne, ttype))

    def ty(t):
        return TYPES.get(t, f"type{t}")

    print(f"{src}")
    print(f"  gguf v{version}  {n_tensors} tensors  {n_kv} kv  "
          f"(header read: {len(h.buf) / 1e6:.1f} MB)")
    arch = kv.get("general.architecture")
    if arch:
        print(f"  general.architecture = {arch}")
    counts = {}
    for _, _, t in tensors:
        counts[ty(t)] = counts.get(ty(t), 0) + 1
    print("  dtypes: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    by_name = {n: (ne, t) for n, ne, t in tensors}

    if wanted:
        for w in wanted:
            hits = [(n, ne, t) for n, ne, t in tensors if w in n]
            print(f"\n  '{w}': {len(hits)} match(es)")
            for n, ne, t in hits[:20]:
                print(f"    {n:<62} {ty(t):>5}  ne={ne}")
        return 0

    print("\n  trellis.cpp compatibility probes:")
    bad = 0
    for name, want, why in PROBES:
        got = by_name.get(name)
        if got is None:
            print(f"    MISSING  {name:<52} ({why})")
            bad += 1
            continue
        ne, t = got
        note = ""
        if want and ty(t) != want:
            note = f"  <-- expected {want}: {why}"
            bad += 1
        print(f"    {ty(t):>5}  {name:<52} ne={ne}{note}")

    # Structural completeness. A DiT is 30 identical blocks plus a fixed preamble, so every
    # block must carry the same set of suffixes. A converter that renamed, merged or dropped
    # tensors shows up here as a ragged block rather than as a crash at inference time.
    blocks = {}
    top = []
    for n, ne, t in tensors:
        if n.startswith("blocks."):
            i, _, suffix = n[len("blocks."):].partition(".")
            if i.isdigit():
                blocks.setdefault(int(i), set()).add(suffix)
                continue
        top.append(n)
    if blocks:
        nb = max(blocks) + 1
        ref = blocks.get(0, set())
        ragged = [i for i in range(nb) if blocks.get(i) != ref]
        print(f"\n  structure: {nb} blocks x {len(ref)} tensors + {len(top)} top-level "
              f"= {nb * len(ref) + len(top)} (file has {n_tensors})")
        if len(blocks) != nb or ragged:
            print(f"    RAGGED: block indices missing or inconsistent: {ragged[:8]}")
            bad += 1
        if nb * len(ref) + len(top) != n_tensors:
            print("    COUNT MISMATCH: some blocks share suffixes unevenly")
            bad += 1

    # BF16 matmul weights load and run (ggml has a dedicated cuBLAS BF16 path and CPU type
    # traits), but --f32 only casts F16, so that escape hatch does not apply to them.
    mm = [t for n, ne, t in tensors if len(ne) >= 2 and "rms_norm" not in n]
    hist = {}
    for t in mm:
        hist[ty(t)] = hist.get(ty(t), 0) + 1
    if hist:
        print(f"  matmul weights: " + ", ".join(f"{k}={v}" for k, v in sorted(hist.items())))
    if "BF16" in hist:
        print("    note: BF16 runs, but --f32 only casts F16 — it will not affect these")
    quant = [k for k in hist if k.startswith("Q")]
    if quant:
        print(f"    note: k-quants/{'/'.join(quant)} are untested in trellis.cpp's graphs")

    proj = by_name.get("blocks.0.cross_attn.proj_linear.weight")
    print()
    if proj:
        print(f"  => Pixal3D weights, proj_in_channels = {proj[0][0]}  (--model pixal3d)")
    else:
        print("  => no proj_linear: TRELLIS.2 weights  (--model trellis)")
    print(f"  => {bad} compatibility problem(s)" if bad else "  => no compatibility problems found")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

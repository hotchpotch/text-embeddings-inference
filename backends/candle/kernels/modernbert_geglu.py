"""Optional ModernBERT F16 GeGLU compiler; inference only needs the exported PTX.

Requires triton==3.5.1. Example:
  python modernbert_geglu.py --arch 89 --model-size xsmall --output-dir modernbert/xsmall
"""
import argparse
import json
import re
from pathlib import Path
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget

@triton.jit(do_not_specialize=['M'])
def dual_gemm(X, W, Y, M, N: tl.constexpr, K: tl.constexpr,
              BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    ks = tl.arange(0, BK)
    acc_a = tl.full((BM, BN), 0, tl.float32)
    acc_g = tl.full((BM, BN), 0, tl.float32)
    for start in range(tl.cdiv(K, BK)):
        kk = start * BK + ks
        x = tl.load(X + rows[:, None] * K + kk[None, :], (rows[:, None] < M) & (kk[None, :] < K), 0)
        wa = tl.load(W + cols[None, :] * K + kk[:, None], (cols[None, :] < N) & (kk[:, None] < K), 0)
        wg = tl.load(W + (N + cols[None, :]) * K + kk[:, None], (cols[None, :] < N) & (kk[:, None] < K), 0)
        acc_a = tl.dot(x, wa, acc_a)
        acc_g = tl.dot(x, wg, acc_g)
    # Match the combined FP16 GEMM's rounding boundary, then activation/product.
    a = acc_a.to(tl.float16).to(tl.float32)
    g = acc_g.to(tl.float16).to(tl.float32)
    a = (0.5 * a * (1 + libdevice.tanh(0.7978845608028654 * (a + 0.044715 * a * a * a)))).to(tl.float16).to(tl.float32)
    tl.store(Y + rows[:, None] * N + cols[None, :], a * g, (rows[:, None] < M) & (cols[None, :] < N))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--arch', type=int, default=89)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--model-size', choices=['xsmall', 'base'], required=True)
    a = p.parse_args()
    hidden, intermediate = (256, 1024) if a.model_size == 'xsmall' else (512, 2048)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    for name, (bm, bn, bk, warps) in {
        'small': (16, 32, 32, 4),
        'medium': (32, 64, 64, 8),
        'large': (128, 64, 32, 4),
    }.items():
        constants = dict(N=intermediate, K=hidden, BM=bm, BN=bn, BK=bk)
        signature = dict(X='*fp16', W='*fp16', Y='*fp16', M='i32')
        signature.update({key: 'constexpr' for key in constants})
        src = ASTSource(dual_gemm, signature, constants,
            attrs={(i,): [('tt.divisibility', 16)] for i in range(3)})
        kernel = triton.compile(src, target=GPUTarget('cuda', a.arch, 32), options=dict(num_warps=warps, num_stages=3))
        m = kernel.metadata
        if m.global_scratch_size or m.profile_scratch_size or m.num_ctas != 1:
            raise ValueError('unsupported scratch or cluster ABI')
        # Strip debug-only source paths and DWARF sections for reproducible distribution.
        ptx = re.split(r'\n\s*\.section\s+\.debug_', kernel.asm['ptx'], maxsplit=1)[0]
        ptx = '\n'.join(line for line in ptx.splitlines()
                        if not line.lstrip().startswith(('.file', '.loc'))).rstrip() + '\n'
        (a.output_dir / (name + '.ptx')).write_text(ptx)
        metadata = dict(abi=1, name=m.name, block_m=bm, block_n=bn, num_warps=warps,
            shared=m.shared, arch=a.arch, triton_version=triton.__version__, hidden=hidden, intermediate=intermediate)
        (a.output_dir / (name + '.json')).write_text(json.dumps(metadata, indent=2) + '\n')
        print(name, metadata)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Count morph files per split and compare against the 50:50 reference directory.
Writes a generation_log_alpha_<ratio>.txt file with per-suffix counts and PASS/FAIL.

Usage:
  python -m morphing.validation.verify_blend_generation \
      --ratio_dir /path/to/blended_morphs/alpha_40 \
      --ref_dir   /path/to/sim_morphs_3way \
      --log_out   /path/to/blended_morphs/alpha_40/generation_log_alpha_40.txt
"""

import argparse
from pathlib import Path

SPLITS = ("train", "val", "test")
SUFFIXES = ("_full.png", "_inA.png", "_inB.png")


def count_by_suffix(directory: Path) -> dict:
    counts = {s: 0 for s in SUFFIXES}
    if not directory.is_dir():
        return counts
    for f in directory.iterdir():
        for s in SUFFIXES:
            if f.name.endswith(s):
                counts[s] += 1
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratio_dir", required=True, help="alpha_XX/ directory being verified")
    ap.add_argument("--ref_dir",   required=True, help="50:50 reference directory (sim_morphs_3way/)")
    ap.add_argument("--log_out",   required=True, help="Path for the output log file")
    args = ap.parse_args()

    ratio_dir = Path(args.ratio_dir)
    ref_dir   = Path(args.ref_dir)

    lines = [
        f"=== Generation verification: {ratio_dir.name} vs {ref_dir.name} ===",
    ]
    all_ok = True

    for split in SPLITS:
        new_counts = count_by_suffix(ratio_dir / split)
        ref_counts = count_by_suffix(ref_dir   / split)

        lines.append(f"\n[{split}]")
        split_ok = True
        for suffix in SUFFIXES:
            new_n = new_counts[suffix]
            ref_n = ref_counts[suffix]
            ok = new_n == ref_n
            status = "OK" if ok else "MISMATCH"
            if not ok:
                all_ok = False
                split_ok = False
            lines.append(f"  {suffix:15s}  new={new_n:>8,}  ref={ref_n:>8,}  {status}")

        lines.append(f"  split result: {'PASSED' if split_ok else 'FAILED'}")

    lines.append(f"\nOverall: {'PASSED' if all_ok else 'FAILED — see mismatches above'}")

    log_text = "\n".join(lines) + "\n"
    print(log_text)

    Path(args.log_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.log_out).write_text(log_text)
    print(f"Log written to: {args.log_out}")

    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
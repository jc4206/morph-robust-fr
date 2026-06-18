# Changes — multi-ratio morph loading fix

## Problem

When passing multiple morph folders (one per blending ratio, e.g. `--morph_root /alpha30 /alpha50 /alpha70`),
two bugs caused silent data loss:

1. **Deduplication collision** — `index_parser.py` used `(x, y, variant)` as the dedupe key.
   Both `/alpha30/03423__09982_inA.png` and `/alpha50/03423__09982_inA.png` share the same
   `(x, y, variant)`, so only whichever folder was scanned first survived.

2. **Variant invisible downstream** — morph tuples were stored as `(path, a, b)`, dropping
   the variant string. `dataset.py` propagated the same 3-tuple, so nothing downstream could
   filter, log, or weight by variant or ratio.

## Files changed

### `index_parser.py`

| Location | Before | After |
|---|---|---|
| Line 54 (dedupe key) | `key = (x, y, variant)` | `key = str(p)` |
| Lines 59–60 (append) | `(p, x, y)` | `(p, x, y, variant)` |

The dedupe key is now the full absolute path. This means:
- Files with identical names from different ratio folders are kept as distinct entries.
- If the same folder is passed twice, the same absolute path deduplicates correctly.

### `dataset.py`

| Location | Before | After |
|---|---|---|
| Line 74 (morph_list unpack) | `for morph_path, a, b in morph_list` | `for morph_path, a, b, variant in morph_list` |
| Line 79 (morph_samples append) | `(str(morph_path), int(a), int(b))` | `(str(morph_path), int(a), int(b), variant)` |
| Line 92 (hard-weight loop) | `for morph_path, _, _ in self.morph_samples` | `for morph_path, _, _, _ in self.morph_samples` |
| Lines 136, 142 (_sample_morph unpack) | `morph_path, a, b = ...` | `morph_path, a, b, variant = ...` |
| Batch dict | no variant field | `"variant": variant` added |

## What still works unchanged

- Multi-root support in `build_morph_index` (list of paths) — untouched.
- Variant regex `(full|inA|inB)` — untouched.
- Hard-sampling weights, pairlist metadata lookup, contributor-B / probe-B loading — untouched.
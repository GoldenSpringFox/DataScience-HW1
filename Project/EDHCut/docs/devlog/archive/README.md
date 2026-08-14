# Archived devlogs

Superseded work, kept for the final report's "what we tried and abandoned" material. Nothing here
describes the active pipeline — do not follow these documents as instructions.

| file | superseded by | why |
|---|---|---|
| `6.3b-communities-next-iteration.md` | `docs/devlog/6.3c-louvain-communities.md` | Soft clustering (NMF → symmetric NMF over the colour-conditioned t-score, recursive hierarchy, `card_tags` theme labelling). Abandoned 2026-08-13 at the user's direction: the communities were too large, the colour-bias problem it set out to solve was never solved, and the four bespoke modules it introduced were not understandable enough to reason about. Replaced by plain Louvain on the same graph. |

The corresponding plan documents live in `docs/plans/archive/`.

## Code left in place, no longer used

These modules were built for the archived approach and are **not** imported by the active pipeline
or by `notebooks/louvain_communities.ipynb`. They are still present and still pass their tests;
they have not been deleted pending an explicit call on whether the final report needs them.

- `edhcut/analysis/nmf_packages.py` (+ `tests/test_nmf_packages.py`)
- `edhcut/analysis/symnmf_packages.py` (+ `tests/test_symnmf_packages.py`)
- `edhcut/analysis/symnmf_hierarchy.py` (+ `tests/test_symnmf_hierarchy.py`)
- `edhcut/analysis/theme_labels.py` (+ `tests/test_theme_labels.py`)

`edhcut/analysis/communities.py` is **not** in this list — it is still live. The Louvain notebook
reuses its `sparsify_top_k_union` and `basic_land_mask`, and its `build_and_save` remains the only
place in the codebase that applies deck weighting correctly.

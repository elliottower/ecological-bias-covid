# `mexico_contextual_cells_20260926204400_SYNTHETIC.csv` was not snapshot output

Written 2026-09-26 20:44 by a run of `s12_composition_ladder.contextual_model` on **synthetic
records**, while testing whether `lme4::glmer` could fit the H6 design at all. It carried
34,295 cells, against the 161,655 the same construction gives on the snapshot.

The file itself is not kept: 4.8 MB of simulated cells is not a result, and no number in the
paper comes from it. This note is the record that it existed and what it was, so the name
cannot later be mistaken for output of a registered analysis.

H6 writes no cell table of its own. The model is fitted with `lme4::glmer`, and `glmm.py`
reproduces it independently; the comparison is in `results/glmm_validation.json`.

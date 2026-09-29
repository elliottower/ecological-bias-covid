# S11, first run, 2026-09-24 — superseded and not recoverable as a file

This run was overwritten by the corrected rerun before the archiving rule existed. Its JSON is
gone; these values are transcribed from the run's console output and are marked as reconstructed.

**Why it was superseded.** The paired municipality contrast keyed on treating state *and*
municipality together, producing 4,498 cells rather than 386 municipalities, and compared them
with the treating state rather than the state of residence in which municipalities nest.

| Quantity | First run (wrong grouping) | Reported run |
|---|---|---|
| Municipality discrepancy | +0.0418 | +0.4748 |
| State discrepancy on the same records | +1.0392 (treating state) | +1.0069 (residence state) |
| Paired difference | −0.9974 | −0.5321 |
| Paired 95% CI | −1.6915 to −0.6067 | −1.2123 to −0.1652 |

Every other S11 output was identical between the two runs: the six site definitions, both
partitions, and the H1 verdict. Only the paired contrast changed.

From 2026-09-26 every result file is written through `paths.write_result`, which moves any
existing file into this directory before writing, so no later run can lose its predecessor.

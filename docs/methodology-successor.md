# OLS conditioning and the scope of comparison evidence

This 2026-10-15 successor repairs a numerical defect and clarifies inference labels. The original
market experiment, registered configs, decision rule and archived result JSON are preserved.
It adds no alpha, profitability, equal-compute or future-market claim.

## The reproduced OLS defect

Regressing `y=(0,1,1,3)` on an intercept and `x=(0,1,2,3)` gives exact residuals
`(0.1,0.2,-0.7,0.4)`. Changing x to x+1,000,000 preserves the design's column space, so it must
preserve that projection. The old normal-equation solver instead changed a residual by
`5.056171e-5`. Smaller changes of units could also be mistaken for singularity.

Let M=I−11ᵀ/n. With an intercept, the residual is

$$r=(I-Q_1Q_1^T)My,$$

where Q₁ spans the centered regressor columns MC. Translation adds a multiple of the intercept;
nonzero column scaling preserves the span. Forming XᵀX squares the spectral condition number,
since its singular values are the squared singular values of X. Large origins then make a raw
intercept/regressor system a poor coordinate frame for a small residual.

The [LAPACK least-squares guide](https://www.netlib.org/lapack/lug/node27.html) uses QR for
full-rank least squares; its [QR description](https://www.netlib.org/lapack/lug/node69.html)
uses Householder reflectors. This repository independently implements that mathematical pattern
with the standard library. No LAPACK code or new dependency is included.

## Implementation and acceptance scope

Subtract an anchor before normalizing, so a large origin does not erase local differences.
If subtracting opposite finite extremes overflows, normalize those inputs before subtraction.
Center normalized differences before any large sum or norm. Normalize each centered regressor
to unit length; pivot on the largest remaining component, and apply Householder reflections
to both the columns and centered response. Zero the fitted coordinates, reverse the reflections
and restore response units. Direct projection avoids reconstructing and subtracting a large
fitted intercept.

The existing refusal contract remains: a constant, underdetermined or numerically unresolved
design returns `None`; no pseudoinverse or clipping is introduced. The numerical rank policy
uses a squared remaining-column norm below `1e-12` after unit normalization. This is a declared
conservative rank floor, not a universal condition-number estimator. Inputs are finite aligned
columns; finite precision cannot recover differences already lost in supplied floats, or
represent a residual outside floating-point range.

The original statistics assertions are unchanged. Eight additive cases check exact projections
under origins, units, column bases, row order, intercept/response translation, singularity,
nonbinary means and extreme finite units. Independent null-space vectors give the multi-column
oracles. The old solver and a version without anchor centering fail the causal tests. A reviewer
also found overflow in the first QR attempt; its regression fails before normalization is moved
ahead of the vulnerable arithmetic. The final fixture tolerance remains `1e-12`.

## Fixed diagnostic replay

The already pinned 34-coin data passed `verify-data`. The unchanged diagnostic config was replayed
with the successor into a fresh output, including its original bootstrap seed/settings and its
existing portfolio-loop reproduction check. The replay completed in 7.557 seconds. Every parsed
JSON value matched `results/binance_diagnostics.json` exactly: zero changed fields. The archived
file and all configs/results stayed byte-identical. This compares the existing spent window;
it is a numerical audit, not a new market experiment or new evidence of an edge.

[Separate replay and comparison receipt](evidence/methodology-2026-10-15/diagnostic-comparison.json)
identifies source/output hashes. [Separate successor output](evidence/methodology-2026-10-15/diagnostic-successor.json)
is retained apart from the archive. Historical source-bound bundles still require their recorded
source/runtime for exact replay; this successor does not make them silently adopt a new solver.

## What the Welch comparison establishes

The original Welch standard error compares selected GP/random search-seed outcomes **conditional
on one fixed panel, universe, splits and validation rule**. It describes optimizer variability
there. More seeds refine that conditional comparison, not uncertainty across markets, independent
windows or future regimes. The original difference, standard error and two-SE decision rule are
unchanged; `comparison_scope` in the canonical result and CLI, and the readable report, label
this scope explicitly. A rule result is not a calibrated probability of future outperformance.

Admitted slots include copied elites and cached scores. Completing the slot cap does not establish
equal unique evaluations, validation effort, runtime or compute. The `equal_compute` flag in
comparison scope states what this protocol establishes, rather than a measured runtime ratio.
Existing survivorship, reused-window, regime and cost-model caveats remain in [limits.md](limits.md).

Suggested engineering wording for root/CV assessment: “Audited a controlled GP-versus-random
research workflow, added exact OLS projection regressions and stable residualisation, and
clarified conditional search-seed uncertainty while preserving negative findings.” Final root
acceptance and human merge review remain separate from these checks; no surface is changed here.

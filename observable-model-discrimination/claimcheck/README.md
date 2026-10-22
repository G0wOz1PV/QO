# Input and guarantees

claimcheck is a solver-independent engine for a fixed list of candidate predictions.
Start with [the tutorial](TUTORIAL.md). Use `python -m claimcheck --version` to identify
the software and `--report report.md` to export a readable report with the input hash.

The software uses user-supplied prediction intervals. It does not infer a Hamiltonian
or validate physical error bounds. A retained candidate is compatible with the
specified test; it is not established as the unique physical explanation.

## Input contract

```json
{
  "alpha": 0.05,
  "family_size": 1,
  "uncertainty": {
    "status": "assumed",
    "description": "Illustrative bounds; no independent calibration"
  },
  "settings": [
    {"id": "observable_1", "kind": "binomial", "support": [-1, 1],
     "shots": 100, "successes": 90}
  ],
  "candidates": [
    {"id": "A", "set_semantics": "outer_box", "intervals": {"observable_1": [0.6, 0.9]}},
    {"id": "B", "set_semantics": "outer_box", "intervals": {"observable_1": [-0.8, -0.5]}}
  ]
}
```

- `successes` counts the **upper** outcome. A probability uses support `[0,1]`;
  a Pauli measurement uses `[-1,1]`. Counts must be independent and identically
  distributed within each setting. Aggregated counts cannot establish that premise.
- For general bounded observables use `kind: "bounded_mean"`, `shots`, `mean`
  and a known two-endpoint `support`. The engine uses Hoeffding intervals, requiring
  independent bounded observations with a common target mean (or their fixed average).
  Do not pass a continuous sample mean as a binomial count.
- Every candidate supplies one closed prediction interval for every setting,
  already including solver uncertainty, state preparation, measurement and dynamical
  discrepancies that the caller wishes to admit. Inputs must use the same units
  as the recorded observable. No gate-level model is inferred automatically.
- `family_size` is the fixed reporting family, at least the number of recorded
  settings. Unobserved planned settings must not silently reduce this number.
  All candidates share this single confidence family; their number does not
  introduce another Bonferroni factor. Adaptive selection using the tested counts
  is not supported. Prospective design uses reference predictions before sampling.
- `set_semantics: "outer_box"` is the default for conservative coordinate bounds.
  Use `"attainable_product"` only if **every** coordinate combination is allowed.
  Shared nuisance parameters usually make this assertion false.
- `uncertainty.status: "external_bound"` additionally requires
  `failure_probability` and a `description` of a separately justified joint bound.
  The output reports observation alpha plus that failure probability, conditional
  on the supplied external statement being valid. The software does not certify
  calibration files or permit fitting bounds to the same data without a valid
  simultaneous analysis. `assumed` produces only conditional conclusions.

## Outputs and interpretation

`confidence_intervals` use exact Clopper-Pearson inversion for binary counts,
with per-setting error alpha/family_size. `decisions` lists each excluded candidate,
the settings witnessing exclusion and a per-setting interval distance.
`common_additive_expansion_to_compatibility` is the maximum of those distances:
expand all supplied intervals by beta and clip to support. Box exclusion holds
for beta strictly below this value; equality retains the box. For outer bounds
this is a conservative box-compatibility boundary, not the exact frontier of a
correlated nuisance set. A common beta has physical meaning only if coordinates
have been placed in a specified common scale.

The `design` report compares every candidate pair. A greedy cover chooses settings
separating the largest number of remaining pairs, then the largest minimum
normalized gap, with input order breaking ties. It is not an optimal set-cover
algorithm and does not claim minimum experimental cost.

For support width R, gap g and m chosen settings, the reported sufficient count is
`floor(2 R^2 log(2m/alpha)/g^2)+1`. This guarantees a **Hoeffding interval** decision
on its simultaneous coverage event. It is not a minimum-shot bound and is not
advertised as a guarantee for the separately evaluated Clopper-Pearson rule.
Basis-rotation gates and different acquisition costs must be accounted for by the
experimentalist; the engine compares shot counts for the supplied measurements.

### Three distinct diagnostics

1. **Separated supplied intervals:** a positive gap provides a sufficient design
   certificate, conditional on validity of the prediction bounds.
2. **Overlapping outer boxes:** no box-separation certificate. Correlations inside
   either joint prediction set might still distinguish the models. This is not an
   impossibility proof.
3. **Common attainable binary mean vector:** under the product-binomial observation
   model, the candidates can induce exactly the same data law at all shot numbers.
   No rule can uniformly identify both with success probability exceeding 1/2.
   This is not a statement about all experimental settings, arbitrary correlations,
   or general multi-outcome distributions sharing a mean.

## Reporting and limitations

Report the candidate list, preparation, measured event, treatment of discarded
outcomes, prediction intervals and their provenance, sampling assumptions,
confidence-family size, confidence level, excluded/retained labels and witnesses.
See `REPORT_TEMPLATE.md`. Output files retain the scope of the inference.

There is no arbitrary-noise guarantee, shot-correlation estimator, adaptive
confidence sequence, tomography or global optimization in this implementation.
The elementary interval mathematics is established; the reusable contribution is
the explicit interface between physical predictions, observational restrictions,
finite-count exclusion and the next measurement/calibration decision.

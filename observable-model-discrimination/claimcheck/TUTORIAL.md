# A short tutorial

The examples below use artificial counts and predictions chosen to demonstrate the interface. They are not device measurements or results of a physical simulation.

## 1. Compare predictions with counts

From the package directory, run:

```sh
python -m claimcheck claimcheck/examples/synthetic_models.json --output result.json --report report.md
```

The example records 90 upper outcomes in 100 trials of a two-outcome observable with values -1 and +1. Its sample mean is 0.8. Candidate A allows means from 0.6 to 0.9; candidate B allows -0.8 to -0.5. The 95% count interval overlaps A and is disjoint from B, so A is retained and B is excluded under the example's assumptions.

`result.json` contains the intervals, decisions and additive expansion thresholds. `report.md` records the question, candidate list, sampling assumptions, error bounds and result. No network service is contacted.

## 2. Supply your own predictions

Replace the counts and candidate intervals in the JSON file. Every candidate must have an interval for every setting. Include numerical and implementation uncertainties in those intervals before testing the data.

For a probability, use support `[0, 1]`. For a Pauli observable, use `[-1, 1]`; `successes` counts the +1 outcome. For other bounded observables, use `kind: "bounded_mean"`, the sample `mean`, the `shots` and the known support.

`family_size` describes the fixed collection of reported settings, including planned settings omitted from the input. Bounds and measurement choices must not be fitted to the same tested counts without a valid joint analysis.

## 3. Plan measurements

```sh
python -m claimcheck claimcheck/examples/synthetic_design.json --design-only --output design.json
```

Three artificial candidates have the same allowed range at `probe_1` and separated ranges at `probe_2`. The output selects `probe_2` and computes a conservative sufficient shot count for a Hoeffding-interval decision. This is not the minimum required count or a guarantee for a different decision rule.

## 4. Interpret the result

- Exclusion is conditional on valid error bounds and the specified sampling model.
- Overlapping outer intervals give no separation certificate. Shared constraints can still distinguish the underlying models.
- The additive threshold expands all candidate intervals in the supplied common units; equality restores box compatibility.
- `uncertainty.status: "external_bound"` needs a separately justified bound, description and failure probability. The software records that claim but does not verify it.

See [Input and guarantees](README.md) for the precise assumptions and [the report template](REPORT_TEMPLATE.md) for what to document.

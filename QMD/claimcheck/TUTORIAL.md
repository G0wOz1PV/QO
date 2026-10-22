# Using your own predictions

New here? Start with the [two-command example](../README.md).

## Prepare the input

Save this artificial example as `my_models.json`:

```json
{
  "alpha": 0.05,
  "data_kind": "synthetic",
  "data_origin": "Artificial tutorial counts",
  "uncertainty": {"status": "assumed", "description": "Illustrative error bounds"},
  "settings": [
    {"id": "probe", "kind": "binomial", "support": [-1, 1], "shots": 100, "successes": 90}
  ],
  "candidates": [
    {"id": "A", "intervals": {"probe": [0.6, 0.9]}},
    {"id": "B", "intervals": {"probe": [-0.8, -0.5]}}
  ]
}
```

Replace the counts and prediction ranges, and update the data description. `successes` counts the upper outcome: +1 for support `[-1, 1]`, or the event of interest for `[0, 1]`. A prediction range is a range of **expected means after allowing errors**, not the range of individual outcomes. You supply it from your own physical model and error analysis.

```sh
python -m claimcheck my_models.json
```

For multiple measurements, add entries with distinct IDs to `settings` and a matching interval for every candidate. Confidence intervals then cover all settings together. If you report only part of a fixed measurement family, set `family_size` to the full family size. Do not fit bounds or choose measurements using the counts being tested.

## Read and save the result

| Result | Meaning |
|---|---|
| `EXCLUDED` | At least one measured confidence interval is separate from the candidate's allowed range |
| `NOT EXCLUDED` | This test has not ruled the candidate out; it is not proof that the candidate is correct |

```sh
python -m claimcheck my_models.json --output result.json --report report.md
```

The Markdown report starts with the decisions. The JSON also includes `common_additive_expansion_to_compatibility`: how far all of a candidate's ranges must expand to overlap the count intervals. Equality restores overlap. Compare this number only across measurements expressed in a justified common scale.

All decisions assume valid prediction bounds and the stated sampling model. `assumed` records this dependence; `external_bound` records a separately justified calibration bound, but does not verify it. See [the reference](README.md) for exact guarantees and bounded-mean inputs.

## Choose measurements before taking shots

```sh
python -m claimcheck --demo --design-only
```

This second artificial example recommends `probe_2` with 185 sufficient shots. To use your own predictions, copy [the design input](examples/synthetic_design.json), replace its candidate ranges, then run `python -m claimcheck my_design.json --design-only`.

Shot counts use a conservative Hoeffding-interval rule under valid bounds and independent sampling. They are not minimum counts or guarantees for the separate Clopper-Pearson count test. Overlapping outer ranges do not by themselves prove physical indistinguishability.

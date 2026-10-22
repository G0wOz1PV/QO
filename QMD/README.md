# QMD

Quantum Model Discrimination with `claimcheck`.

**Which models can your quantum measurements rule out?** Supply measurement counts and each model's allowed prediction range, including implementation errors. claimcheck compares them with a confidence interval for the measured mean.

## Try it

Requires Python 3.10 or later. Run these two commands; no Git installation or device account is needed.

```sh
python -m pip install "https://github.com/G0wOz1PV/QO/archive/refs/tags/claimcheck-v0.2.2.zip#subdirectory=QMD"
python -m claimcheck --demo
```

The artificial example prints:

```text
Artificial example (no device data)
95% count confidence
  probe: mean 0.800 | CI [0.648, 0.902]

A: NOT EXCLUDED | allowed prediction [0.600, 0.900]
B: EXCLUDED | allowed prediction [-0.800, -0.500]
```

**B is excluded:** its range does not overlap the measured confidence interval. **A is not excluded:** it still overlaps; this does not prove A correct.

Exclusion assumes valid prediction ranges and independent shots within each setting. Fix the ranges and measurement choices before testing the counts. The 95% refers to count uncertainty, not the probability that A is correct.

## Use your own measurements

Copy [this example](claimcheck/examples/synthetic_models.json) into `my_models.json`, then change:

| Field | What to enter |
|---|---|
| `shots`, `successes` | Total shots and number of +1 outcomes |
| `support` | `[-1, 1]` for a Pauli measurement; `[0, 1]` for an event probability |
| Each candidate's `intervals` | Allowed mean range, including numerical and experimental errors |
| `data_kind`, `data_origin`, `question`, `uncertainty` | Describe your data, physical question and how the bounds were obtained |

```sh
python -m claimcheck my_models.json
```

Results appear in the terminal. Nothing is uploaded. To save a result, add `--output result.json --report report.md`.

If several candidates remain, this test cannot separate them with these inputs. If none remain, check the candidates, error bounds and sampling assumptions.

[More than one measurement, error thresholds and design](claimcheck/TUTORIAL.md) · [Full input reference and guarantees](claimcheck/README.md)

Version 0.2.2 · [Citation](CITATION.cff) · [Noncommercial license](LICENSE.md)

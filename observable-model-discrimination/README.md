# claimcheck

Test competing physical predictions against finite measurement counts. Supply your own model predictions and error bounds; no Hamiltonian solver or device account is required.

The tool reports which candidate intervals are excluded, how much additional error would restore compatibility, and which proposed measurements separate candidate pairs. Conclusions are conditional on the supplied error bounds and sampling assumptions.

## Install and try

Requires Python 3.10 or later, NumPy and SciPy. From this directory:

```sh
python -m pip install .
python -m claimcheck claimcheck/examples/synthetic_models.json --output result.json --report report.md
python -m claimcheck claimcheck/examples/synthetic_design.json --design-only --output design.json
```

Both examples are artificial and contain no experimental records. The first retains candidate A and excludes B. The second compares two possible measurements before collecting counts.

## Documentation

- [Tutorial](claimcheck/TUTORIAL.md): counts, predictions and interpretation
- [Input and guarantees](claimcheck/README.md): uncertainty, confidence levels and scope
- [Input schema](claimcheck/input.schema.json) and [report template](claimcheck/REPORT_TEMPLATE.md)

Run the checks from this directory:

```sh
python -m unittest claimcheck.test_core claimcheck.test_reporting -v
```

Version 0.2.0. [Citation](CITATION.cff) · [License](LICENSE.md)

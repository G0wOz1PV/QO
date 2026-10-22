# Electron-gated proton transfer (EGPT)

Numerical models of electrostatic gating, proton dynamics and coupling to an environment.

## Local calculations

Run from this directory:

```sh
python -m pip install -r requirements.txt
python simulation.py
```

`simulation.py` writes the baseline tables to `csv/` and plots to `png/`. Existing outputs are included for inspection.

| File | Purpose |
| --- | --- |
| `pcet_gating_simulation.py` | Gate correspondence, coupled dynamics and reaction-rate calculations |
| `TLS phase.py` | Two-level-system phase analysis |
| `non_markovian_analysis.py` | Environment-dependent dynamics |
| `resonant_analysis.py` | Resonant-dynamics calculations |
| `SI/generate_si_figs_tables.py` | Supplementary numerical tables and figures |
| `validation/cco/` | Cytochrome-c-oxidase comparison data and calculations |

These are separate model studies; their parameters and approximations are specified in the scripts.

## Optional hardware execution

`geometric_witness.py` prepares and submits a two-qubit IBM Runtime study. It requires an independently configured IBM account, optional dependencies in `requirements-hardware.txt`, an explicitly selected backend and `--execute-hardware`. It does not submit jobs when imported or invoked with `--help`.

```sh
python geometric_witness.py --help
```

Hardware execution uses the selected account's computing allocation. Credentials belong in the provider's local account configuration, never in this repository.

## License

The original [license notice](LICENSE.txt) is retained.

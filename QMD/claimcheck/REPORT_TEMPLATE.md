# Model Claim Checklist

A proposed reporting checklist for model-discrimination claims in quantum experiments. Use it with claimcheck or another appropriate analysis method.

| Check | What to report |
|---|---|
| Physical claim | The interaction or dynamics being tested; all candidate models and relevant alternatives not tested. |
| Measurements | Initial preparation, observables, outcome coding, counts, shot totals and any discarded outcomes. Identify hardware, emulator or synthetic data. |
| Prediction bounds | Numerical and implementation errors, units, how bounds were obtained, and which were independently calibrated or only assumed. State shared error parameters and constraints lost by using separate intervals. |
| Sampling | Independence or correlation assumptions, the fixed measurement family, and any use of the tested data to select measurements or fit bounds. |
| Statistical guarantee | Confidence level, correction across settings and any separate calibration failure probability. State exactly which event is guaranteed. |
| Decision | Every retained and excluded candidate, the settings supporting exclusion and the additional error threshold that changes the decision. A surviving model is not thereby proven correct. |
| Next measurement | Whether more shots, tighter calibration, additional candidates or a different observable could resolve the remaining ambiguity. Interval overlap alone need not prove physical indistinguishability. |

## Short report template

> We tested [claim] against [candidates] using [observables and counts]. Under [prediction bounds, calibration status and sampling assumptions], [candidates] are excluded at [confidence level and simultaneous scope]. The exclusion ceases when [additional error threshold, with units] is allowed. [Remaining candidates] are not excluded. [Unexamined alternatives or assumptions] limit the conclusion; [next measurement or calibration] would address the remaining ambiguity.

With claimcheck, save a report using:

```sh
python -m claimcheck my_models.json --output result.json --report report.md
```

Check the generated report against the items above. The software records supplied assumptions; it does not independently establish calibration, candidate completeness or shot independence.

This is a proposed reporting aid, not an adopted community standard. Please cite the [software version](../CITATION.cff) when using the tool or this checklist in research.

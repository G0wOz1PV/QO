"""Compare model predictions with counts. Try: python -m claimcheck --demo"""
import argparse
import json
from importlib.resources import files
from pathlib import Path
from claimcheck.core import analyze, design
from claimcheck.reporting import evidence_report, markdown_report, console_summary
from claimcheck.version import __version__


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', action='version', version='claimcheck '+__version__)
    parser.add_argument('input', nargs='?', type=Path, help='JSON file containing counts and candidate prediction ranges')
    parser.add_argument('--demo', action='store_true', help='Run the bundled artificial example; no input file needed')
    parser.add_argument('--output', type=Path, help='Save the full result as JSON (optional)')
    parser.add_argument('--design-only', action='store_true', help='Use predictions and supports before taking shots; no counts required.')
    parser.add_argument('--report', type=Path, help='Also write a human-readable evidence report; count analysis only.')
    args = parser.parse_args()
    if args.demo and args.input:
        parser.error('choose either --demo or an input file')
    if not args.demo and args.input is None:
        parser.error('provide an input file, or try --demo')
    try:
        if args.input and args.output and args.output.resolve() == args.input.resolve():
            raise ValueError('output must not overwrite the input file')
        if args.report and args.report.resolve() in [p.resolve() for p in (args.input,args.output) if p]:
            raise ValueError('report must have a different path from input and output')
        if args.design_only and args.report:
            raise ValueError('--report requires observed counts; omit it with --design-only')
        if args.demo:
            name = 'synthetic_design.json' if args.design_only else 'synthetic_models.json'
            source = files('claimcheck').joinpath('examples', name).read_text(encoding='utf-8')
        else:
            source = args.input.read_text(encoding='utf-8-sig')
        problem = json.loads(source)
        if not isinstance(problem,dict): raise ValueError('input must be a JSON object')
        for field in ('settings','candidates'):
            if not isinstance(problem.get(field),list) or not problem[field]:
                raise ValueError(field+' must be a nonempty JSON array')
        report = design(problem['settings'], problem['candidates'], problem.get('alpha', .05)) if args.design_only else analyze(problem)
        if not args.design_only:
            report['evidence_report'] = evidence_report(problem,report)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        if args.report:
            args.report.parent.mkdir(parents=True,exist_ok=True)
            args.report.write_text(markdown_report(report['evidence_report']),encoding='utf-8')
    except KeyError as exc:
        parser.error('missing required field '+str(exc)+'; see claimcheck/input.schema.json and TUTORIAL.md')
    except (ValueError,TypeError,OSError) as exc:
        parser.error(str(exc))
    print(console_summary(problem, report, design_only=args.design_only))
    if args.output:
        print('JSON saved: ' + str(args.output))
    if args.report:
        print('Report saved: ' + str(args.report))


if __name__ == '__main__':
    main()

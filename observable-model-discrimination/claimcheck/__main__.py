"""python -m claimcheck problem.json --output report.json"""
import argparse
import json
from pathlib import Path
from claimcheck.core import analyze, design
from claimcheck.reporting import evidence_report, markdown_report
from claimcheck.version import __version__


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', action='version', version='claimcheck '+__version__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--design-only', action='store_true', help='Use predictions and supports before taking shots; no counts required.')
    parser.add_argument('--report', type=Path, help='Also write a human-readable evidence report; count analysis only.')
    args = parser.parse_args()
    try:
        if args.output.resolve() == args.input.resolve():
            raise ValueError('output must not overwrite the input file')
        if args.report and args.report.resolve() in (args.input.resolve(),args.output.resolve()):
            raise ValueError('report must have a different path from input and output')
        if args.design_only and args.report:
            raise ValueError('--report requires observed counts; omit it with --design-only')
        problem = json.loads(args.input.read_text(encoding='utf-8-sig'))
        if not isinstance(problem,dict): raise ValueError('input must be a JSON object')
        for field in ('settings','candidates'):
            if not isinstance(problem.get(field),list) or not problem[field]:
                raise ValueError(field+' must be a nonempty JSON array')
        report = design(problem['settings'], problem['candidates'], problem.get('alpha', .05)) if args.design_only else analyze(problem)
        if not args.design_only:
            report['evidence_report'] = evidence_report(problem,report)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        if args.report:
            args.report.parent.mkdir(parents=True,exist_ok=True)
            args.report.write_text(markdown_report(report['evidence_report']),encoding='utf-8')
    except KeyError as exc:
        parser.error('missing required field '+str(exc)+'; see claimcheck/input.schema.json and TUTORIAL.md')
    except (ValueError,TypeError,OSError) as exc:
        parser.error(str(exc))
    print('Wrote ' + str(args.output))


if __name__ == '__main__':
    main()

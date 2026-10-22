"""Check the first-run interface from an unrelated working directory."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from claimcheck import analyze
from claimcheck.reporting import console_summary
from claimcheck.test_reporting import example


class CliTests(unittest.TestCase):
    def run_cli(self, directory, *args):
        env = os.environ.copy()
        env['PYTHONPATH'] = str(Path(__file__).resolve().parent.parent) + os.pathsep + env.get('PYTHONPATH', '')
        return subprocess.run([sys.executable, '-m', 'claimcheck', *args], cwd=directory,
                              env=env, capture_output=True, text=True)

    def test_demo_from_empty_directory_without_writing_files(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_cli(directory, '--demo')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('Artificial example', result.stdout)
            self.assertIn('mean 0.800 | CI [0.648, 0.902]', result.stdout)
            self.assertIn('A: NOT EXCLUDED', result.stdout)
            self.assertIn('B: EXCLUDED', result.stdout)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_count_input_needs_no_output_path(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'input.json'
            path.write_text(json.dumps(example()), encoding='utf-8')
            result = self.run_cli(directory, str(path))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('A: NOT EXCLUDED', result.stdout)
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_design_demo_displays_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_cli(directory, '--demo', '--design-only')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('probe_2: 185 sufficient shots', result.stdout)
            self.assertIn('Pairs without a separation guarantee: 0 / 3', result.stdout)

    def test_invalid_argument_combinations(self):
        with tempfile.TemporaryDirectory() as directory:
            for args, message in [((), 'provide an input file'),
                                  (('--demo', 'input.json'), 'choose either'),
                                  (('--demo', '--design-only', '--report', 'report.md'), '--report requires observed counts')]:
                with self.subTest(args=args):
                    result = self.run_cli(directory, *args)
                    self.assertEqual(result.returncode, 2)
                    self.assertIn(message, result.stderr)
                    self.assertNotIn('Traceback', result.stderr)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_report_without_json_and_input_protection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'input.json'
            path.write_text(json.dumps(example()), encoding='utf-8')
            report = Path(directory) / 'report.md'
            result = self.run_cli(directory, str(path), '--report', str(report))
            self.assertEqual(result.returncode, 0, result.stderr)
            text = report.read_text(encoding='utf-8')
            self.assertLess(text.index('B | EXCLUDED'), text.index('## Inputs and assumptions'))
            result = self.run_cli(directory, str(path), '--report', str(path))
            self.assertEqual(result.returncode, 2)
            self.assertEqual(json.loads(path.read_text(encoding='utf-8')), example())

    def test_summary_preserves_family_and_external_bound(self):
        problem = example()
        second = copy.deepcopy(problem['settings'][0])
        second['id'] = 'other'
        problem['settings'].append(second)
        problem['family_size'] = 3
        for candidate in problem['candidates']:
            candidate['intervals']['other'] = candidate['intervals']['probe']
        problem['uncertainty'] = {'status': 'external_bound', 'description': 'Test bound', 'failure_probability': .01}
        text = console_summary(problem, analyze(problem))
        self.assertIn('95% count confidence (simultaneous over 3 settings)', text)
        self.assertIn('total failure <= 0.06', text)
        self.assertIn('if that bound is valid', text)

    def test_summary_uses_support_scale_and_bounded_mean(self):
        problem = example()
        problem['settings'][0]['support'] = [0, 1]
        problem['candidates'] = [{'id': 'A', 'intervals': {'probe': [.5, .7]}}]
        self.assertIn('mean 0.600', console_summary(problem, analyze(problem)))
        setting = problem['settings'][0]
        setting['kind'] = 'bounded_mean'
        setting['mean'] = .65
        setting.pop('successes')
        self.assertIn('mean 0.650', console_summary(problem, analyze(problem)))


if __name__ == '__main__':
    unittest.main()

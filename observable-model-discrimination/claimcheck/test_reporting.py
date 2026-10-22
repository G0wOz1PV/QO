import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from claimcheck import analyze
from claimcheck.reporting import evidence_report, markdown_report


def example():
    return {'alpha':.05,'data_kind':'synthetic','data_origin':'Test fixture',
            'settings':[{'id':'probe','kind':'binomial','support':[-1,1],'shots':1000,'successes':600}],
            'candidates':[{'id':'A','intervals':{'probe':[.1,.3]}},{'id':'B','intervals':{'probe':[-.3,-.1]}}],
            'uncertainty':{'status':'assumed','description':'Test intervals'}}


class ReportingTests(unittest.TestCase):
    def test_report_preserves_scope_and_candidate_decision(self):
        p=example(); r=evidence_report(p,analyze(p))
        self.assertEqual(r['retained_candidates'],['A'])
        self.assertEqual(r['evidence']['kind'],'synthetic')
        self.assertEqual(r['calibration']['status'],'assumed')
        self.assertIsNone(r['confidence']['external_combined_failure_bound'])
        self.assertIn('not a standard certification',markdown_report(r))

    def test_hash_is_order_independent_and_data_sensitive(self):
        p=example(); q=dict(reversed(list(p.items())))
        self.assertEqual(evidence_report(p,analyze(p))['input_sha256'],evidence_report(q,analyze(q))['input_sha256'])
        q=copy.deepcopy(p);q['settings'][0]['successes']+=1
        self.assertNotEqual(evidence_report(p,analyze(p))['input_sha256'],evidence_report(q,analyze(q))['input_sha256'])

    def test_unprovided_provenance_is_not_inferred(self):
        p=example();p.pop('data_kind');p.pop('data_origin')
        self.assertEqual(evidence_report(p,analyze(p))['evidence']['kind'],'unspecified')

    def test_cli_missing_field_has_no_traceback(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'invalid.json';path.write_text('{}')
            r=subprocess.run([sys.executable,'-m','claimcheck',str(path),'--output',str(path.with_suffix('.out'))],capture_output=True,text=True)
            self.assertEqual(r.returncode,2)
            self.assertIn('settings must be a nonempty JSON array',r.stderr)
            self.assertNotIn('Traceback',r.stderr)

    def test_cli_report_roundtrip_and_input_protection(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);p=root/'input.json';p.write_text(json.dumps(example()))
            args=[sys.executable,'-m','claimcheck',str(p)]
            r=subprocess.run(args+['--output',str(root/'out.json'),'--report',str(root/'out.md')],capture_output=True,text=True)
            self.assertEqual(r.returncode,0,r.stderr)
            self.assertEqual(json.loads((root/'out.json').read_text())['evidence_report']['retained_candidates'],['A'])
            r=subprocess.run(args+['--output',str(p)],capture_output=True,text=True)
            self.assertEqual(r.returncode,2)
            self.assertEqual(json.loads(p.read_text()),example())


if __name__=='__main__':unittest.main()

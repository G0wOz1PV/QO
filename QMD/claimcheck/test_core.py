"""Independent probability, counterexample and scope tests for the public engine."""
import copy
import unittest
import numpy as np
from scipy.stats import binom
from claimcheck.core import analyze, confidence_interval, design, phase_envelope


def problem():
    return {'alpha': .05,
            'settings': [{'id': 'x', 'kind': 'binomial', 'support': [-1, 1], 'shots': 100, 'successes': 90}],
            'candidates': [{'id': 'A', 'intervals': {'x': [.6, .9]}}, {'id': 'B', 'intervals': {'x': [-.8, -.5]}}]}


class TestCore(unittest.TestCase):
    def test_exact_coverage_enumeration(self):
        # Direct binomial masses rather than the implementation's beta inverse.
        for n in (1, 2, 8, 32):
            intervals = np.array([confidence_interval({'kind': 'binomial', 'support': [0, 1], 'shots': n, 'successes': k}, .05)
                                  for k in range(n+1)])
            for p in np.linspace(0, 1, 101):
                covered = (intervals[:, 0] <= p) & (p <= intervals[:, 1])
                self.assertGreaterEqual(binom.pmf(np.arange(n+1), n, p)[covered].sum(), .95-1e-13)

    def test_tail_equations(self):
        for k in range(1, 20):
            lo, hi = confidence_interval({'kind': 'binomial', 'support': [0, 1], 'shots': 20, 'successes': k}, .01)
            self.assertAlmostEqual(binom.sf(k-1, 20, lo), .005, places=12)
            self.assertAlmostEqual(binom.cdf(k, 20, hi), .005, places=12)

    def test_frontier_boundary(self):
        p = problem()
        out = analyze(p)
        f = out['decisions'][1]['common_additive_expansion_to_compatibility']
        for factor, excluded in ((.999, True), (1.001, False)):
            q = copy.deepcopy(p)
            q['candidates'][1]['intervals']['x'] = [max(-1, -.8-factor*f), min(1, -.5+factor*f)]
            self.assertEqual(analyze(q)['decisions'][1]['excluded'], excluded)

    def test_no_candidate_multiplicity_penalty(self):
        p = problem(); first = analyze(p)['confidence_intervals']
        p['candidates'] += [{'id': str(i), 'intervals': {'x': [-1, 1]}} for i in range(30)]
        self.assertEqual(first, analyze(p)['confidence_intervals'])

    def test_family_size_not_observed_subset(self):
        p = problem(); p['family_size'] = 16
        self.assertLess(analyze(p)['confidence_intervals']['x'][0], analyze(problem())['confidence_intervals']['x'][0])

    def test_outer_box_not_impossibility(self):
        settings = [{'id': 'x', 'kind': 'binomial', 'support': [0, 1]}]
        cs = [{'id': c, 'intervals': {'x': [.2, .8]}} for c in ('A', 'B')]
        self.assertEqual(design(settings, cs)['pairs'][0]['diagnosis'], 'no_box_separation_certificate')
        for c in cs: c['set_semantics'] = 'attainable_product'
        p = design(settings, cs)['pairs'][0]
        self.assertEqual(p['diagnosis'], 'no_uniform_binary_discrimination')
        w = p['common_mean_witness']['x']
        self.assertTrue(.2 <= w <= .8)

    def test_correlated_nuisance_counterexample(self):
        # A={(0,0),(1,1)} and B={(0,1),(1,0)} are disjoint joint sets,
        # but have identical coordinate boxes. No impossibility claim.
        ss = [{'id': i, 'kind': 'binomial', 'support': [0, 1]} for i in ('x', 'y')]
        cs = [{'id': 'A', 'intervals': {'x': [0, 1], 'y': [0, 1]}},
              {'id': 'B', 'intervals': {'x': [0, 1], 'y': [0, 1]}}]
        self.assertEqual(design(ss, cs)['pairs'][0]['diagnosis'], 'no_box_separation_certificate')

    def test_bounded_mean_overlap_not_binary_certificate(self):
        ss = [{'id': 'x', 'kind': 'bounded_mean', 'support': [-2, 3]}]
        cs = [{'id': c, 'intervals': {'x': [0, 1]}, 'set_semantics': 'attainable_product'} for c in ('a', 'b')]
        self.assertEqual(design(ss, cs)['pairs'][0]['diagnosis'], 'no_box_separation_certificate')

    def test_bounded_interval(self):
        r = {'kind': 'bounded_mean', 'support': [-2, 3], 'shots': 100, 'mean': .5}
        lo, hi = confidence_interval(r, .05)
        self.assertAlmostEqual(hi-lo, 10*np.sqrt(np.log(40)/200))

    def test_exact_phase_extrema_against_dense_grid(self):
        rng = np.random.default_rng(2307)
        for _ in range(30):
            theta, phase, amin = rng.uniform(-np.pi, np.pi), rng.uniform(0, np.pi), rng.uniform(0, 1)
            x, y = .9*np.cos(theta), .9*np.sin(theta)
            lo, hi = phase_envelope(x, y, phase, amin, .02)
            phi = np.linspace(-phase, phase, 20001)
            vals = np.r_[x*np.cos(phi)+y*np.sin(phi), amin*(x*np.cos(phi)+y*np.sin(phi))]
            self.assertLessEqual(lo, vals.min()-.02+1e-12)
            self.assertGreaterEqual(hi, vals.max()+.02-1e-12)
            self.assertLess(abs(lo-(vals.min()-.02)), 1e-7)
            self.assertLess(abs(hi-(vals.max()+.02)), 1e-7)

    def test_external_failure_union(self):
        p = problem(); self.assertIsNone(analyze(p)['unconditional_failure_bound_if_external_statement_valid'])
        p['uncertainty'] = {'status': 'external_bound', 'failure_probability': .01, 'description': 'Separate supplied calibration bound'}
        self.assertAlmostEqual(analyze(p)['unconditional_failure_bound_if_external_statement_valid'], .06)

    def test_three_candidate_pair_cover(self):
        ss = [{'id': i, 'kind': 'binomial', 'support': [-1, 1]} for i in ('x', 'y')]
        cs = [{'id': 'A', 'intervals': {'x': [-.8, -.6], 'y': [-.2, .2]}},
              {'id': 'B', 'intervals': {'x': [.6, .8], 'y': [-.2, .2]}},
              {'id': 'C', 'intervals': {'x': [-.2, .2], 'y': [-.2, .2]}}]
        d = design(ss, cs)
        self.assertEqual(len(d['greedy_cover']), 1)
        self.assertEqual(len(d['greedy_cover'][0]['pairs']), 3)
        n = d['greedy_cover'][0]['hoeffding_sufficient_shots']
        self.assertLess(2*2*np.sqrt(np.log(40)/(2*n)), .4)

    def test_invalid_inputs(self):
        for field, value in [('shots', 0), ('successes', 101), ('shots', 2.5), ('successes', True)]:
            p = problem(); p['settings'][0][field] = value
            with self.assertRaises(ValueError): analyze(p)
        p = problem(); p['candidates'][0]['intervals']['x'] = [0, float('nan')]
        with self.assertRaises(ValueError): analyze(p)
        p = problem(); p['family_size'] = 0
        with self.assertRaises(ValueError): analyze(p)


if __name__ == '__main__':
    unittest.main()

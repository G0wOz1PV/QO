"""Conservative confidence-box inversion for a fixed list of candidate models.

Predictions are closed intervals, not learned from the tested counts. The engine
records calibration evidence supplied by a caller; it cannot verify that evidence.
Only attainable Cartesian products of binary means support the implemented
indistinguishability certificate. Outer boxes support exclusion but not that claim.
"""
from itertools import combinations
import math
from numbers import Integral, Real
from scipy.stats import beta


def number(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(name + ' must be a finite number')
    return float(value)


def integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(name + ' must be an integer >= ' + str(minimum))
    return int(value)


def interval(value, name):
    if len(value) != 2:
        raise ValueError(name + ' must contain two endpoints')
    lo, hi = (number(x, name) for x in value)
    if lo > hi:
        raise ValueError(name + ' has reversed endpoints')
    return lo, hi


def level(alpha):
    alpha = number(alpha, 'alpha')
    if not 0 < alpha < 1:
        raise ValueError('alpha must be between 0 and 1')
    return alpha


def confidence_interval(record, alpha):
    """CP for two-outcome counts; Hoeffding for independent bounded observations."""
    alpha = level(alpha)
    lo, hi = interval(record['support'], 'support')
    if lo == hi:
        raise ValueError('support must have positive width')
    n = integer(record['shots'], 'shots', 1)
    if record['kind'] == 'binomial':
        k = integer(record['successes'], 'successes')
        if k > n:
            raise ValueError('successes exceeds shots')
        p_lo = 0.0 if k == 0 else float(beta.ppf(alpha/2, k, n-k+1))
        p_hi = 1.0 if k == n else float(beta.ppf(1-alpha/2, k+1, n-k))
        return [lo+(hi-lo)*p_lo, lo+(hi-lo)*p_hi]
    if record['kind'] == 'bounded_mean':
        mean = number(record['mean'], 'mean')
        if not lo <= mean <= hi:
            raise ValueError('mean outside support')
        radius = (hi-lo)*math.sqrt(math.log(2/alpha)/(2*n))
        return [max(lo, mean-radius), min(hi, mean+radius)]
    raise ValueError('kind must be binomial or bounded_mean')


def phase_envelope(x, y, phase=0.0, contrast_min=0.0, bias=0.0):
    """Adapter for a*(x*cos(phi)+y*sin(phi))+b, a in [amin,1]."""
    x, y = number(x, 'x'), number(y, 'y')
    phase, amin, bias = number(phase, 'phase'), number(contrast_min, 'contrast_min'), number(bias, 'bias')
    if not 0 <= phase <= math.pi or not 0 <= amin <= 1 or bias < 0:
        raise ValueError('invalid phase, contrast or bias bound')
    if math.hypot(x, y) > 1+1e-12:
        raise ValueError('transverse reference exceeds unit disk')
    values = [x*math.cos(phase)+y*math.sin(phase), x*math.cos(phase)-y*math.sin(phase)]
    theta = math.atan2(y, x)
    for stationary in (theta, theta-math.pi, theta+math.pi):
        if -phase <= stationary <= phase:
            values.append(x*math.cos(stationary)+y*math.sin(stationary))
    lo, hi = min(values), max(values)
    return [max(-1.0, min(lo, amin*lo)-bias), min(1.0, max(hi, amin*hi)+bias)]


def distance(a, b):
    return max(0.0, a[0]-b[1], b[0]-a[1])


def validate_predictions(settings, candidates):
    ids = [s['id'] for s in settings]
    if not ids or len(ids) != len(set(ids)) or any(not isinstance(i, str) or not i for i in ids):
        raise ValueError('nonempty unique setting ids required')
    names = [c['id'] for c in candidates]
    if not names or len(names) != len(set(names)) or any(not isinstance(i, str) or not i for i in names):
        raise ValueError('nonempty unique candidate ids required')
    for setting in settings:
        lo, hi = interval(setting['support'], 'support')
        if lo == hi or setting['kind'] not in ('binomial', 'bounded_mean'):
            raise ValueError('invalid observation support or kind')
    for candidate in candidates:
        if set(candidate['intervals']) != set(ids):
            raise ValueError('candidate intervals must match every setting exactly')
        if candidate.get('set_semantics', 'outer_box') not in ('outer_box', 'attainable_product'):
            raise ValueError('invalid set_semantics')
        for setting in settings:
            lo, hi = interval(candidate['intervals'][setting['id']], 'prediction')
            a, b = setting['support']
            if lo < a or hi > b:
                raise ValueError('prediction outside observation support')


def design(settings, candidates, alpha=0.05):
    """Pairwise observable gaps and deterministic greedy measurement cover.

The shot certificate uses a Hoeffding confidence-interval decision, not CP.
An overlap of outer boxes means the supplied boxes give no uniform certificate.
It does not prove overlap of the underlying correlated nuisance sets.
"""
    alpha = level(alpha)
    validate_predictions(settings, candidates)
    pairs = []
    for left, right in combinations(candidates, 2):
        rows = []
        witness = {}
        for s in settings:
            a, b = left['intervals'][s['id']], right['intervals'][s['id']]
            gap = distance(a, b)
            width = s['support'][1]-s['support'][0]
            rows.append({'setting': s['id'], 'gap': gap, 'normalized_gap': gap/width})
            if not gap:
                witness[s['id']] = (max(a[0], b[0])+min(a[1], b[1]))/2
        best = max(rows, key=lambda r: r['normalized_gap'])
        overlaps = best['gap'] == 0
        exact_binary = all(s['kind'] == 'binomial' for s in settings) and all(
            c.get('set_semantics', 'outer_box') == 'attainable_product' for c in (left, right))
        pairs.append({'candidates': [left['id'], right['id']], 'settings': rows,
                      'best_setting': best['setting'],
                      'diagnosis': ('no_uniform_binary_discrimination' if exact_binary else 'no_box_separation_certificate')
                      if overlaps else 'separated_by_supplied_intervals',
                      'common_mean_witness': witness if overlaps else None})
    # Each selected setting separates at least one as-yet-uncovered pair.
    uncovered = {i for i, p in enumerate(pairs) if p['diagnosis'] == 'separated_by_supplied_intervals'}
    selected = []
    while uncovered:
        def score(s):
            covered = [i for i in uncovered if distance(candidates_by_id(candidates, pairs[i]['candidates'][0])['intervals'][s['id']],
                                                      candidates_by_id(candidates, pairs[i]['candidates'][1])['intervals'][s['id']]) > 0]
            margins = [next(r['normalized_gap'] for r in pairs[i]['settings'] if r['setting'] == s['id']) for i in covered]
            return len(covered), min(margins) if margins else 0.0
        s = max(settings, key=score)
        covered = [i for i in uncovered if next(r['gap'] for r in pairs[i]['settings'] if r['setting'] == s['id']) > 0]
        selected.append({'setting': s['id'], 'pairs': [pairs[i]['candidates'] for i in sorted(covered)]})
        uncovered.difference_update(covered)
    m = len(selected)
    for choice in selected:
        s = next(s for s in settings if s['id'] == choice['setting'])
        gaps = [next(r['gap'] for r in p['settings'] if r['setting'] == s['id']) for p in pairs if p['candidates'] in choice['pairs']]
        g = min(gaps)
        width = s['support'][1]-s['support'][0]
        choice['minimum_assigned_gap'] = g
        choice['hoeffding_sufficient_shots'] = math.floor(2*width**2*math.log(2*m/alpha)/g**2)+1
    return {'pairs': pairs, 'greedy_cover': selected, 'cover_is_globally_optimal': False,
            'shot_bound_rule': 'Hoeffding intervals; fixed design; per-setting independent bounded shots; no cross-setting independence required',
            'impossibility_scope': 'Common attainable binary mean vectors admit identical joint product-binomial laws; one cannot uniformly identify both candidates above probability 1/2. Outer boxes and bounded means alone do not establish this.'}


def candidates_by_id(candidates, name):
    return next(c for c in candidates if c['id'] == name)


def analyze(problem):
    """Analyze a fixed reporting family. No adaptive reuse or nuisance fitting."""
    alpha = level(problem.get('alpha', 0.05))
    settings, candidates = problem['settings'], problem['candidates']
    validate_predictions(settings, candidates)
    m = integer(problem.get('family_size', len(settings)), 'family_size', 1)
    if m < len(settings):
        raise ValueError('family_size cannot be smaller than observed family')
    uncertainty = problem.get('uncertainty', {'status': 'assumed', 'description': 'Unspecified externally supplied intervals'})
    if uncertainty['status'] not in ('assumed', 'external_bound'):
        raise ValueError('uncertainty status must be assumed or external_bound')
    if not isinstance(uncertainty.get('description'), str) or not uncertainty['description'].strip():
        raise ValueError('uncertainty description is required')
    cal_alpha = None
    if uncertainty['status'] == 'external_bound':
        cal_alpha = number(uncertainty['failure_probability'], 'failure_probability')
        if not 0 <= cal_alpha < 1-alpha:
            raise ValueError('total failure budget must be < 1')
    cis = {s['id']: confidence_interval(s, alpha/m) for s in settings}
    decisions = []
    for candidate in candidates:
        distances = {i: distance(cis[i], candidate['intervals'][i]) for i in cis}
        frontier = max(distances.values())
        decisions.append({'candidate': candidate['id'], 'excluded': frontier > 0,
                          'witness_settings': [i for i, d in distances.items() if d > 0],
                          'per_setting_additive_frontier': distances,
                          'common_additive_expansion_to_compatibility': frontier,
                          'frontier_convention': 'Expand all intervals by beta in supplied units and clip to support. Excluded iff beta < frontier; equality retains.'})
    return {'schema_version': 1, 'alpha_observation': alpha, 'family_size': m,
            'confidence_intervals': cis, 'decisions': decisions,
            'retained_candidates': [d['candidate'] for d in decisions if not d['excluded']],
            'uncertainty': uncertainty,
            'unconditional_failure_bound_if_external_statement_valid': None if cal_alpha is None else alpha+cal_alpha,
            'interpretation': 'Retained means not excluded; even a unique survivor is only relative to the supplied candidates and bounds. Assumed bounds give conditional statements only. No independence across settings or extra correction for candidate count is required.',
            'design': design(settings, candidates, alpha)}

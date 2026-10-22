"""A structured evidence record with an inspectable link to the input."""
import hashlib
import json
from .version import __version__


def console_summary(problem, result, design_only=False):
    """Show decisions first; keep detailed assumptions in the full report."""
    lines = []
    if problem.get('data_kind') == 'synthetic':
        lines.append('Artificial example (no device data)')
    if design_only:
        lines.append('Suggested measurements:')
        for choice in result['greedy_cover']:
            lines.append(f"  {choice['setting']}: {choice['hoeffding_sufficient_shots']} sufficient shots (Hoeffding rule)")
        unresolved = sum(p['diagnosis'] != 'separated_by_supplied_intervals' for p in result['pairs'])
        lines.append(f'Pairs without a separation guarantee: {unresolved} / {len(result["pairs"])}')
        lines.append('Shot bounds assume valid prediction ranges and independent shots within each setting.')
        lines.append('This is a sufficient plan, not a minimum shot budget. Use --output for pairwise details.')
        return '\n'.join(lines)

    confidence = 100 * (1 - result['alpha_observation'])
    family = result['family_size']
    label = f'{confidence:g}% count confidence'
    if family > 1:
        label += f' (simultaneous over {family} settings)'
    lines.append(label)
    for setting in problem['settings']:
        lo, hi = setting['support']
        mean = lo + (hi-lo)*setting['successes']/setting['shots'] if setting['kind'] == 'binomial' else setting['mean']
        a, b = result['confidence_intervals'][setting['id']]
        lines.append(f"  {setting['id']}: mean {mean:.3f} | CI [{a:.3f}, {b:.3f}]")
    lines.append('')
    for decision, candidate in zip(result['decisions'], problem['candidates']):
        status = 'EXCLUDED' if decision['excluded'] else 'NOT EXCLUDED'
        line = f"{decision['candidate']}: {status}"
        if len(problem['settings']) == 1:
            a, b = candidate['intervals'][problem['settings'][0]['id']]
            line += f' | allowed prediction [{a:.3f}, {b:.3f}]'
        elif decision['excluded']:
            line += ' | separated at ' + ', '.join(decision['witness_settings'])
        lines.append(line)
    lines += ['', 'Exclusion assumes valid prediction ranges and independent shots within each setting.',
              'Ranges and measurements must be fixed before testing these counts.',
              'NOT EXCLUDED does not mean proven correct.']
    combined = result['unconditional_failure_bound_if_external_statement_valid']
    if combined is not None:
        lines.append(f'Including the supplied calibration failure bound: total failure <= {combined:.4g} (if that bound is valid).')
    if not result['retained_candidates']:
        lines.append('No candidate remains: check the candidate set, error bounds and sampling assumptions.')
    return '\n'.join(lines)


def evidence_report(problem, result):
    """Report supplied provenance without inferring calibration or physical truth."""
    origin = problem.get('data_origin', 'Not supplied')
    kind = problem.get('data_kind', 'unspecified')
    if kind not in ('hardware','emulator','synthetic','reference','unspecified'):
        raise ValueError('data_kind must be hardware, emulator, synthetic, reference or unspecified')
    canonical = json.dumps(problem,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
    return {
        'report_schema_version':1, 'software':{'name':'claimcheck','version':__version__},
        'input_sha256':hashlib.sha256(canonical.encode('utf-8')).hexdigest(),
        'question':problem.get('question','Exclude supplied candidate predictions at the recorded settings'),
        'evidence':{'kind':kind,'provenance':origin,'settings':len(problem['settings']),
                    'shots':sum(s['shots'] for s in problem['settings'])},
        'candidate_list':[c['id'] for c in problem['candidates']],
        'confidence':{'observation_alpha':result['alpha_observation'],'family_size':result['family_size'],
                      'external_combined_failure_bound':result['unconditional_failure_bound_if_external_statement_valid'],
                      'sampling':'Independent shots within each setting; binary counts require a common success probability. No cross-setting independence is required for coverage.',
                      'scope':'Fixed reporting family; bounds and design not fitted to the tested outcomes.'},
        'calibration':result['uncertainty'],
        'decision':result['decisions'], 'retained_candidates':result['retained_candidates'],
        'interpretation':result['interpretation'],
        'next_action':'Independently justify prediction bounds before interpreting exclusion physically. If boxes overlap, examine shared constraints or change observations; increase shots only when a useful gap survives.',
        'status':'Candidate-relative evidence report, not a standard certification or full-state verification.'
    }


def markdown_report(report):
    def clean(value): return str(value).replace('|','\\|').replace('\n',' ')
    e,c = report['evidence'],report['confidence']
    lines=['# Model-discrimination report','',
           '| Candidate | Result |','|---|---|']
    for d in report['decision']:
        lines.append('| '+clean(d['candidate'])+' | '+('EXCLUDED' if d['excluded'] else 'NOT EXCLUDED')+' |')
    lines += ['', 'Exclusion assumes valid prediction ranges and the stated sampling conditions. NOT EXCLUDED does not mean proven correct.',
           '', '## Inputs and assumptions', '',
           '| Field | Value |','|---|---|',
           '| Question | '+clean(report['question'])+' |',
           '| Evidence | '+clean(f"{e['kind']}; {e['settings']} settings; {e['shots']} shots. {e['provenance']}")+' |',
           '| Candidates | '+clean(', '.join(report['candidate_list']))+' |',
           '| Count uncertainty | '+clean(f"Observation alpha={c['observation_alpha']}; fixed family={c['family_size']}")+' |',
           '| Calibration | '+clean(report['calibration']['status']+': '+report['calibration']['description'])+' |',
           '| Combined failure bound | '+clean(c['external_combined_failure_bound'] if c['external_combined_failure_bound'] is not None else 'Not supplied; physical bounds remain assumptions')+' |',
           '| Retained | '+clean(', '.join(report['retained_candidates']) or 'None')+' |','',
           '## Candidate decisions','',
           '| Candidate | Excluded | Witness settings | Common additive boundary |',
           '|---|---|---|---|']
    for d in report['decision']:
        lines.append('| '+clean(d['candidate'])+' | '+str(d['excluded'])+' | '+clean(', '.join(d['witness_settings']) or 'None')+' | '+format(d['common_additive_expansion_to_compatibility'],'.10g')+' |')
    lines += ['', 'The boundary expands all supplied intervals in their stated common scale. Equality retains the candidate box.',
              '', '## Scope and next action','',c['sampling'],c['scope'],report['interpretation'],report['next_action'],
              '',report['status'],'',f"Software: claimcheck {report['software']['version']}; report schema {report['report_schema_version']}.",
              'Canonical input SHA-256: `'+report['input_sha256']+'`.','']
    return '\n'.join(lines)

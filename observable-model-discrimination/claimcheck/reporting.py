"""A structured evidence record with an inspectable link to the input."""
import hashlib
import json
from .version import __version__


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

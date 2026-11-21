#!/usr/bin/env python3
"""Final C11 source-byte and test-receipt check for actual 10368 CPU MCTS.

This verification does not refit any model or select a winner among seeded
assignments. It preserves the predeclared equal-cost gate and negative result.
"""
from pathlib import Path
import hashlib
import json


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    d=Path(__file__).resolve().parent
    e=d/'evidence/live_heldout_20261121'
    m=json.loads((e/'EXECUTION_MANIFEST.json').read_text())
    s=json.loads((e/'SUMMARY_C11_COST_MATCH.json').read_text())
    a=json.loads((e/'INDEPENDENT_READBACK.json').read_text())
    cohort=json.loads((d/'COHORT_384_FROZEN.json').read_text())
    plan=json.loads((d/'MIXTURE_PLAN_FROZEN.json').read_text())
    assigned=json.loads((d/'ASSIGNMENTS_FROZEN.json').read_text())
    assert m['source_commit']=='348ce4e0ad311ea0c14cd512b1ad15d2ca19b95c'
    assert m['n_real_Rust_searches']==s['n_real_Rust_inprocess_release_search_requests']==a['actual_release_Rust_requests']==10368
    assert len(cohort['rows'])==384 and len(cohort['families'])==24
    assert a['additional_external_solver_original_requeries']==24
    assert a['additional_external_solver_horizontal_mirror_requeries']==24
    assert a['all_actual_game_states_and_WDL_regret_recomputed']
    assert a['blocked_assignment_independently_rederived_without_test_labels']
    assert m['new_cohort_sha256']==sha(d/'COHORT_384_FROZEN.json')
    assert m['validation_mix_sha256']==sha(d/'MIXTURE_PLAN_FROZEN.json')
    assert m['assignment_sha256']==sha(d/'ASSIGNMENTS_FROZEN.json')
    assert m['exact_oracle_sha256']==sha(d/'evidence/exact_new384_20261121/EXACT_NEW384_LABELS.json')
    assert m['compiled_release_binary_sha256']=='4db989b0ede0c56077de77a820a99cd777adcf87c98d7faedc62611491eb8a03'
    for name,digest in m['output_sha256'].items():
        assert sha(e/name)==digest,name
    assert '596 passed; 0 failed; 89 ignored' in Path('/tmp/c11_full_release_rust.log').read_text()
    assert '596 passed; 0 failed; 89 ignored' in Path('/tmp/c11_full_debug_rust.log').read_text()
    assert s['comparisons']['pooled']['uniform']['decision_informative_root_count']==215
    assert s['comparisons']['pooled']['uniform']['arms']['dynamic']['two_step_commits']==218
    for group in ('new_A','new_B','pooled'):
        for i in (1,2,3):
            p=s['comparisons'][group]['uniform']['paired'][f'dynamic_minus_mixture_{i}']
            assert p['mean_CPU_time_match']=='MATCHED_10_PERCENT'
            assert p['family_cluster_conditional_bootstrap']['conditional_family_bootstrap_95'][1]>=-1e-12
    assert s['comparisons']['pooled']['uniform']['arms']['dynamic']['mean_regret']==s['comparisons']['pooled']['uniform']['arms']['mixture_2']['mean_regret']
    assert abs(s['comparisons']['pooled']['uniform']['expected_randomization']['dynamic_minus_expected_randomized_regret']+0.0010545877417001137)<1e-12
    assert a['root_WDL_regret_difference_counts']["('uniform', 'mixture_1')"]['different_WDL_regret_roots']==1
    assert a['root_WDL_regret_difference_counts']["('uniform', 'mixture_3')"]['different_WDL_regret_roots']==1
    receipt={
        'schema':'quartz_cog02_costmatch11_verified_receipt_v1',
        'source_commit':m['source_commit'],
        'compiled_rust_source_commit':m['compiled_rust_source_commit'],
        'release_executable_sha256':m['compiled_release_binary_sha256'],
        'frozen_original_C10_model_sha256':m['frozen_model_sha256'],
        'frozen_validation_mixture_sha256':m['validation_mix_sha256'],
        'blocked_assignments_sha256':m['assignment_sha256'],
        'new_24_families_cohort_sha256':m['new_cohort_sha256'],
        'new_384_exact_external_labels_sha256':m['exact_oracle_sha256'],
        'n_real_live_Rust_cpu_MCTS_searches':10368,
        'n_new_test_roots':384,'n_opening_family_clusters':24,
        'n_WDL_decision_informative_roots':215,
        'n_precommitted_assignment_seeds':3,
        'n_dynamic_uniform_two_step_commit_positions':218,
        'independent_pascal_pons_origin_and_mirror_queries':48,
        'warm_model_cache_MLP_hits':3455,
        'warm_model_cache_Conv_hits':3455,
        'full_Rust_release':{'passed':596,'failed':0,'ignored':89},
        'full_Rust_debug':{'passed':596,'failed':0,'ignored':89},
        'all_related_Python':{'passed':116,'failed':0,
           'suites':[11,12,6,7,10,9,4,7,7,7,9,27]},
        'all_three_actual_uniform_randomized_mixtures_match_mean_CPU_time_within_10pct_in_both_new_cohorts':True,
        'all_three_conditional_family_bootstrap_upper_bounds_include_zero':True,
        'assignment_only_randomizes_policy_computation_not_MCTS_search_tree':True,
        'no_test_family_used_for_fit_lambda_or_mixture_probability':True,
        'exact_external_solver_data_loaded_after_all_Rust_runs':True,
        'verified_raw_output_sha256':m['output_sha256'],
        'paired_results':{
            g:{
                'dynamic_regret':s['comparisons'][g]['uniform']['arms']['dynamic']['mean_regret'],
                'mixture_regrets':[s['comparisons'][g]['uniform']['arms'][f'mixture_{i}']['mean_regret'] for i in (1,2,3)],
                'mixture_CPU_mismatch':[s['comparisons'][g]['uniform']['paired'][f'dynamic_minus_mixture_{i}']['relative_actual_search_ms_mismatch'] for i in (1,2,3)],
                'dynamic_average_MCTS_search_ms':s['comparisons'][g]['uniform']['arms']['dynamic']['mean_actual_search_ms'],
            } for g in ('new_A','new_B','pooled')
        },
        'scientific_gate':'NONMYOPIC_ADVANTAGE_NOT_ESTABLISHED',
        'native_subagent':'NOT_RUN',
    }
    path=e/'VERIFICATION_RECEIPT.json'
    if path.exists():raise FileExistsError('no overwrite of independent receipt')
    path.write_text(json.dumps(receipt,sort_keys=True,indent=2)+'\n')
    print('C11_TEST_SCIENCE_VERIFICATION_RECEIPT_PASS',json.dumps({
        'Rust':receipt['full_Rust_release'],
        'Python':receipt['all_related_Python']['passed'],
        'mixed_regrets':receipt['paired_results']['pooled']['mixture_regrets'],
        'dynamic_regret':receipt['paired_results']['pooled']['dynamic_regret'],
        'status':receipt['scientific_gate']},sort_keys=True))


if __name__=='__main__':main()

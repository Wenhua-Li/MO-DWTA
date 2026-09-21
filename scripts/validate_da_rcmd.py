"""validate_da_rcmd.py - 快速验证 RCMD-EA 相比原算法的改善

对比三个版本：
1. RCMD-EA（双档案 + 动态预算）
2. RCMD-EA原版（单档案，qa_ratio=1.0模拟原行为）
3. NSGA-II+TOPSIS（基线）

实例：I1-I5（小中实例），各5次
"""
import sys
sys.path.insert(0, '..')
import numpy as np
import json
from pathlib import Path
from alg_code.problems.instance import load_instance
from alg_code.rcmd_ea import RCMDParams, run_rcmd_ea
from alg_code.baselines.nsga2 import run_nsga2, NSGA2Params
from alg_code.post_selection import topsis_select
from alg_code.metrics import hypervolume, compute_reference_point, hamming_diversity

def run_comparison(inst_name, seed):
    """单次运行三算法对比"""
    inst_path = f'../../paper2/source_code/instances/{inst_name}.mat'
    inst = load_instance(inst_path)

    results = {}

    # 1. RCMD-EA（新版：双档案 + 动态预算）
    params_da = RCMDParams(seed=seed, G_max=100, qa_ratio=0.5, G_max_dynamic=150)
    res_da = run_rcmd_ea(inst, params_da, verbose=False)
    # 提取最后阶段的 QA Pareto 集
    last_pr = res_da.phase_results[-1]
    qa_objs = last_pr.pareto_objs  # QA档案的 Pareto 集
    da_codes = last_pr.da_codes if last_pr.da_codes is not None else last_pr.pareto_codes

    # 计算 HV（需要参考点）
    ref_pt = compute_reference_point([qa_objs], factor=1.1)
    hv_da = hypervolume(qa_objs, ref_pt) if len(qa_objs) > 1 else 0.0

    results['da_rcmd'] = {
        'total_f1_res': float(res_da.total_f1_res),
        'qa_size': int(len(qa_objs)),
        'da_size': int(len(da_codes)),
        'hd': float(last_pr.hd),
        'hv': float(hv_da),
    }

    # 2. 原RCMD-EA（单档案，qa_ratio=1.0 模拟原版选择）
    params_orig = RCMDParams(seed=seed, G_max=100, qa_ratio=1.0, G_max_dynamic=None)
    res_orig = run_rcmd_ea(inst, params_orig, verbose=False)
    last_pr_orig = res_orig.phase_results[-1]
    qa_objs_orig = last_pr_orig.pareto_objs
    ref_pt_orig = compute_reference_point([qa_objs_orig], factor=1.1)
    hv_orig = hypervolume(qa_objs_orig, ref_pt_orig) if len(qa_objs_orig) > 1 else 0.0

    results['rcmd_orig'] = {
        'total_f1_res': float(res_orig.total_f1_res),
        'qa_size': int(len(qa_objs_orig)),
        'hd': float(last_pr_orig.hd),
        'hv': float(hv_orig),
    }

    # 3. NSGA-II（基线）
    nsga_params = NSGA2Params(seed=seed, G_max=100)
    res_nsga = run_nsga2(inst, nsga_params, verbose=False)
    last_pr_nsga = res_nsga.phase_results[-1]
    pareto_objs_nsga = last_pr_nsga.pareto_objs
    ref_pt_nsga = compute_reference_point([pareto_objs_nsga], factor=1.1)
    hv_nsga = hypervolume(pareto_objs_nsga, ref_pt_nsga) if len(pareto_objs_nsga) > 1 else 0.0
    hd_nsga = hamming_diversity(last_pr_nsga.pareto_codes) if len(last_pr_nsga.pareto_codes) > 1 else 0.0

    results['nsga2'] = {
        'pareto_size': int(len(pareto_objs_nsga)),
        'hd': float(hd_nsga),
        'hv': float(hv_nsga),
    }

    return results

if __name__ == '__main__':
    instances = ['I1', 'I2', 'I3', 'I4', 'I5']
    n_runs = 5
    seeds = [42, 43, 44, 45, 46]

    all_results = {}

    for inst_name in instances:
        print(f'\n=== {inst_name} ===')
        all_results[inst_name] = []

        for run_id, seed in enumerate(seeds):
            print(f'  Run {run_id+1}/{n_runs}...', end=' ', flush=True)
            try:
                res = run_comparison(inst_name, seed)
                all_results[inst_name].append(res)
                print(f"DA_HV={res['da_rcmd']['hv']:.2f}, Orig_HV={res['rcmd_orig']['hv']:.2f}, NSGA_HV={res['nsga2']['hv']:.2f}")
            except Exception as e:
                print(f'ERROR: {e}')
                import traceback
                traceback.print_exc()
                continue

    # 保存结果
    output_dir = Path('../../reports/da_rcmd_validation_2026-06-12')
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    # 计算平均值
    print('\n\n=== 汇总（最后阶段平均HV） ===')
    print(f"{'实例':<8} {'DA-RCMD':<12} {'原RCMD':<12} {'NSGA-II':<12} {'DA vs 原':<12} {'DA vs NSGA':<12}")
    for inst_name in instances:
        runs = all_results[inst_name]
        if not runs:
            continue
        hv_da = np.mean([r['da_rcmd']['hv'] for r in runs])
        hv_orig = np.mean([r['rcmd_orig']['hv'] for r in runs])
        hv_nsga = np.mean([r['nsga2']['hv'] for r in runs])
        diff_orig = ((hv_da - hv_orig) / hv_orig * 100) if hv_orig > 0 else 0
        diff_nsga = ((hv_da - hv_nsga) / hv_nsga * 100) if hv_nsga > 0 else 0
        print(f"{inst_name:<8} {hv_da:<12.2f} {hv_orig:<12.2f} {hv_nsga:<12.2f} {diff_orig:>+10.1f}% {diff_nsga:>+10.1f}%")

    print(f'\n结果已保存到 {output_dir / "results.json"}')

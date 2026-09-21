"""scripts/gen_assignment_results.py — 重新生成 fig7/figS1 用的分配方案数据

用新RMI公式（固定参考点）重新生成 papre_prepare/figures/assignment_results.json
中 RCMD-EA 和 NSGA-II+TOPSIS 在指定实例、每个阶段的 best_c/f1/f2/rmi。

用法：
  python scripts/gen_assignment_results.py --instances I2 I3 --seed 0
"""
import sys, json, argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
INST_DIR = ROOT / 'instances'

from alg_code.problems.instance import load_instance
from alg_code.problems.encoding import build_combo_set
from alg_code.problems.robustness import compute_rmi, adaptive_sample_size
from alg_code.rcmd_ea import run_rcmd_ea, RCMDParams
from alg_code.baselines.nsga2 import run_nsga2, NSGA2Params
from alg_code.post_selection import topsis_select
import numpy as np


def compute_rmi_adaptive(code, f1_val, combo_sets, phase, inst, seed):
    rng = np.random.default_rng(seed)
    S = adaptive_sample_size(
        n_t=len(phase.targets), combo_sets=combo_sets, r=2,
        delta=0.1, eps_rel=0.15, delta_prob=0.10, S_floor=10, S_cap=500,
    )
    return compute_rmi(code, float(f1_val), combo_sets, phase, inst, r=2, S=S, rng=rng)


def gen_rcmd_ea(inst, seed):
    r = run_rcmd_ea(inst, RCMDParams(G_max=100, seed=seed, explore_ratio=0.4,
                                      qa_ratio=0.5, G_max_dynamic=None))
    out = []
    for pr in r.phase_results:
        out.append({
            't': pr.t,
            'best_c': pr.best_c.tolist() if hasattr(pr.best_c, 'tolist') else list(pr.best_c),
            'f1': float(pr.best_f1_res),
            'f2': float(pr.best_f2),
            'rmi': float(pr.best_rmi),
        })
    return out


def gen_nsga2_topsis(inst, seed):
    r = run_nsga2(inst, NSGA2Params(G_max=100, seed=seed))
    out = []
    for t, pr in enumerate(r.phase_results):
        f1 = pr.pareto_objs[:, 0]
        f2 = pr.pareto_objs[:, 1]
        best_c, bf1, bf2 = topsis_select(pr.pareto_codes, f1, f2)
        combo_sets = build_combo_set(inst.phases[t], inst)
        rmi = compute_rmi_adaptive(best_c, bf1, combo_sets, inst.phases[t], inst, seed)
        out.append({
            't': t,
            'best_c': best_c.tolist() if hasattr(best_c, 'tolist') else list(best_c),
            'f1': float(bf1),
            'f2': float(bf2),
            'rmi': float(rmi),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--instances', nargs='+', default=['I2', 'I3'])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=str(ROOT.parent / 'papre_prepare' / 'figures' / 'assignment_results.json'))
    args = ap.parse_args()

    results = {}
    for inst_name in args.instances:
        inst = load_instance(str(INST_DIR / f'{inst_name}.mat'))
        print(f'[{inst_name}] running RCMD-EA (seed={args.seed})...')
        results[f'{inst_name}_rcmd_ea'] = gen_rcmd_ea(inst, args.seed)
        print(f'[{inst_name}] running NSGA-II+TOPSIS (seed={args.seed})...')
        results[f'{inst_name}_nsga2_topsis'] = gen_nsga2_topsis(inst, args.seed)

    out_path = Path(args.out)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()

"""scripts/run_baselines.py — 6个对比基线算法全量实验驱动

支持 NSGA-II / MOEA-D / SPEA2 / CoMMEA / HDMMODE / MMOEADC
结果保存到 reports/<algo>/<inst>_run<NNN>.json，格式与runner.py一致。

用法：
  python scripts/run_baselines.py --instances I1-I10 --runs 4 --workers 6
"""
import sys, argparse, json, time, traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

_LOCAL_INST = ROOT.parent / 'paper2' / 'source_code' / 'instances'
_SERVER_INST = ROOT / 'instances'
INST_DIR = _LOCAL_INST if _LOCAL_INST.exists() else _SERVER_INST
RESULTS_DIR = ROOT / 'reports'
RESULTS_DIR.mkdir(exist_ok=True)

from alg_code.problems.instance import load_instance


# ── 后处理：从Pareto集选最优（TOPSIS权重0.5/0.5）─────────────────────

def _topsis_best(pareto_codes, f1, f2):
    """简单TOPSIS，等权，选最靠近理想点的方案。"""
    import numpy as np
    n = len(f1)
    if n == 0:
        return pareto_codes[0], f1[0], f2[0]
    mn1, mx1 = f1.min(), f1.max()
    mn2, mx2 = f2.min(), f2.max()
    r1 = f1 / (mx1 - mn1 + 1e-12)
    r2 = f2 / (mx2 - mn2 + 1e-12)
    ideal = np.array([r1.min(), r2.min()])
    nadir = np.array([r1.max(), r2.max()])
    d_pos = np.sqrt((r1 - ideal[0])**2 + (r2 - ideal[1])**2)
    d_neg = np.sqrt((r1 - nadir[0])**2 + (r2 - nadir[1])**2)
    cc = d_neg / (d_pos + d_neg + 1e-12)
    best = int(cc.argmax())
    return pareto_codes[best], float(f1[best]), float(f2[best])


def _phases_from_result(r, inst, seed):
    """将 run_* 返回的 Result 对象转为 phases 列表（与runner.py格式兼容）。"""
    from alg_code.problems.encoding import build_combo_set
    from alg_code.problems.robustness import compute_rmi
    import numpy as np
    rng = __import__('numpy').random.default_rng(seed)
    phases_out = []
    for t, pr in enumerate(r.phase_results):
        f1 = pr.pareto_objs[:, 0]; f2 = pr.pareto_objs[:, 1]
        best_c, bf1, bf2 = _topsis_best(pr.pareto_codes, f1, f2)
        combo_sets = build_combo_set(inst.phases[t], inst)
        rmi = compute_rmi(best_c, float(bf1), combo_sets,
                          inst.phases[t], inst, r=2, S=50, rng=rng)
        phases_out.append({
            't': t,
            'best_f1_res': float(bf1),
            'best_f2':     float(bf2),
            'best_rmi':    float(rmi),
            'hd':          float(pr.hd),
            'pareto_size': len(pr.pareto_objs),
            'pareto_objs': pr.pareto_objs.tolist(),
        })
    return phases_out


# ── 各算法入口 ────────────────────────────────────────────────────────

def _run_nsga2(inst, seed):
    from alg_code.baselines.nsga2 import run_nsga2, NSGA2Params
    r = run_nsga2(inst, NSGA2Params(G_max=100, seed=seed))
    phases = _phases_from_result(r, inst, seed)
    return {'total_f1_res': sum(p['best_f1_res'] for p in phases), 'phases': phases}


def _run_moead(inst, seed):
    from alg_code.baselines.moead import run_moead, MOEADParams
    r = run_moead(inst, MOEADParams(G_max=100, seed=seed))
    phases = _phases_from_result(r, inst, seed)
    return {'total_f1_res': sum(p['best_f1_res'] for p in phases), 'phases': phases}


def _run_spea2(inst, seed):
    from alg_code.baselines.spea2 import run_spea2, SPEA2Params
    r = run_spea2(inst, SPEA2Params(G_max=100, seed=seed))
    phases = _phases_from_result(r, inst, seed)
    return {'total_f1_res': sum(p['best_f1_res'] for p in phases), 'phases': phases}


def _run_commea(inst, seed):
    from alg_code.baselines.commea import run_commea, CoMMEAParams
    r = run_commea(inst, CoMMEAParams(G_max=100, seed=seed))
    phases = _phases_from_result(r, inst, seed)
    return {'total_f1_res': sum(p['best_f1_res'] for p in phases), 'phases': phases}


def _run_hdmmode(inst, seed):
    from alg_code.baselines.hdmmode import run_hdmmode, HDMMODEParams
    r = run_hdmmode(inst, HDMMODEParams(G_max=100, seed=seed))
    phases = _phases_from_result(r, inst, seed)
    return {'total_f1_res': sum(p['best_f1_res'] for p in phases), 'phases': phases}


def _run_mmoeadc(inst, seed):
    from alg_code.baselines.mmoeadc import run_mmoeadc, MMOEADCParams
    r = run_mmoeadc(inst, MMOEADCParams(G_max=100, seed=seed))
    phases = _phases_from_result(r, inst, seed)
    return {'total_f1_res': sum(p['best_f1_res'] for p in phases), 'phases': phases}


ALGOS = {
    'nsga2':   _run_nsga2,
    'moead':   _run_moead,
    'spea2':   _run_spea2,
    'commea':  _run_commea,
    'hdmmode': _run_hdmmode,
    'mmoeadc': _run_mmoeadc,
}

INSTANCE_NAMES = [f'I{i}' for i in range(1, 11)]


def _result_path(algo, inst_name, run):
    return RESULTS_DIR / algo / f'{inst_name}_run{run:03d}.json'


def _task(args):
    algo, inst_name, run, skip_existing = args
    out_path = _result_path(algo, inst_name, run)
    if skip_existing and out_path.exists():
        return algo, inst_name, run, 'skipped'
    try:
        inst = load_instance(str(INST_DIR / f'{inst_name}.mat'))
        t0 = time.time()
        result = ALGOS[algo](inst, seed=run)
        elapsed = time.time() - t0
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump({'algo': algo, 'instance': inst_name, 'run': run,
                       'elapsed_s': elapsed, **result}, f)
        return algo, inst_name, run, f'ok ({elapsed:.1f}s)'
    except Exception as e:
        return algo, inst_name, run, f'ERROR: {e}\n{traceback.format_exc()[:300]}'


def expand_instances(spec):
    if not spec:
        return INSTANCE_NAMES
    result = []
    for s in spec:
        if '-' in s and s.startswith('I'):
            a, b = s.split('-')
            result.extend(f'I{i}' for i in range(int(a[1:]), int(b[1:]) + 1))
        else:
            result.append(s)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--algo', nargs='+')
    p.add_argument('--all-algos', action='store_true')
    p.add_argument('--instances', nargs='+', default=['I1', 'I2', 'I3'])
    p.add_argument('--runs', type=int, default=4)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--skip-existing', action='store_true')
    args = p.parse_args()

    algos = list(ALGOS.keys()) if args.all_algos else (args.algo or list(ALGOS.keys()))
    instances = expand_instances(args.instances)
    tasks = [(a, i, r, args.skip_existing)
             for a in algos for i in instances for r in range(args.runs)]

    total = len(tasks)
    print(f'任务总数: {total}  算法: {algos}  实例: {instances}  runs: {args.runs}  workers: {args.workers}')

    done = skipped = errors = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_task, t): t for t in tasks}
        for fut in as_completed(futures):
            algo, inst, run, status = fut.result()
            done += 1
            if 'skipped' in status:
                skipped += 1
            elif 'ERROR' in status:
                errors += 1
                print(f'  ✗ {algo}/{inst}/run{run:03d}: {status[:120]}')
            if done % max(1, total // 20) == 0 or done == total:
                elapsed = time.time() - t0
                eta = elapsed / done * (total - done) if done > 0 else 0
                print(f'  {done}/{total}  skip={skipped} err={errors}'
                      f'  elapsed={elapsed:.0f}s  ETA={eta:.0f}s')

    print(f'\n完成 {done} 任务 (跳过{skipped}, 错误{errors})')


if __name__ == '__main__':
    main()

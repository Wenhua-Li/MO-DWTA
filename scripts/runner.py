"""scripts/runner.py — 统一实验驱动脚本

支持所有算法 × 全部实例 × 多次运行的并行执行，结果保存为JSON。
与 docs/server_run.md 中的部署方式配套。

用法：
  # 本地烟测
  python scripts/runner.py --algo rcmd_ea --instances I1 I2 I3 --runs 3 --workers 4

  # 服务器：主实验（RCMD-EA + 6基线）
  python3 scripts/runner.py --algo rcmd_ea nsga2_topsis nsga2_vikor moead_topsis spea2_topsis commea_topsis hdmmode_topsis mmoeadc_topsis --instances I1-I10 --runs 30 --workers 96 --skip-existing

  # 服务器：消融实验
  python3 scripts/runner.py --algo abl_no_explore abl_no_dual_archive abl_fixed_pref --instances I1-I10 --runs 30 --workers 96 --skip-existing

  # 服务器：全部实验一键运行
  python3 scripts/runner.py --all-algos --instances I1-I10 --runs 30 --workers 96 --skip-existing
"""
import sys, argparse, json, time, hashlib, traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
# 实例目录：本地用 paper2/source_code/instances，服务器用 mowta/instances
_LOCAL_INST = ROOT.parent / 'paper2' / 'source_code' / 'instances'
_SERVER_INST = ROOT / 'instances'
INST_DIR = _LOCAL_INST if _LOCAL_INST.exists() else _SERVER_INST
RESULTS_DIR = ROOT / 'reports'
RESULTS_DIR.mkdir(exist_ok=True)

from alg_code.problems.instance import load_instance

# ── 公共工具：计算RMI（自适应采样）────────────────────────────────────

def _compute_rmi_adaptive(code, f1_val, combo_sets, phase, inst, seed):
    """使用自适应采样数S*计算RMI，与RCMD-EA内部一致（ε_rel=0.15，置信度90%）。"""
    from alg_code.problems.robustness import compute_rmi, adaptive_sample_size
    import numpy as np
    rng = np.random.default_rng(seed)
    S = adaptive_sample_size(
        n_t=len(phase.targets),
        combo_sets=combo_sets,
        r=2,
        delta=0.1,
        eps_rel=0.15,
        delta_prob=0.10,
        S_floor=10,
        S_cap=500,
    )
    return compute_rmi(code, float(f1_val), combo_sets, phase, inst, r=2, S=S, rng=rng)

# ── RCMD-EA 主算法（4组件：C1 RCMD支配 / C2 两阶段 / C3 态势感知后选择 / C4 双档案）──

def _run_rcmd_ea(inst, seed):
    """RCMD-EA 完整版：双档案（QA+DA）C1+C2+C3+C4"""
    from alg_code.rcmd_ea import run_rcmd_ea, RCMDParams
    from alg_code.metrics import hypervolume, compute_reference_point
    import numpy as np
    r = run_rcmd_ea(inst, RCMDParams(G_max=100, seed=seed, explore_ratio=0.4,
                                     qa_ratio=0.5, G_max_dynamic=None))
    return _postprocess_rcmd(r)


def _run_rcmd_v2(inst, seed):
    """RCMD-EA v2（修改稿）：budget_aware=True，修复 Phase II 预算饥饿缺陷"""
    from alg_code.rcmd_ea import run_rcmd_ea, RCMDParams
    from alg_code.metrics import hypervolume, compute_reference_point
    import numpy as np
    r = run_rcmd_ea(inst, RCMDParams(G_max=100, seed=seed, explore_ratio=0.4,
                                     qa_ratio=0.5, G_max_dynamic=None,
                                     budget_aware=True, budget_g_target=10))
    return _postprocess_rcmd(r)


def _postprocess_rcmd(r):
    from alg_code.metrics import hypervolume, compute_reference_point
    import numpy as np
    all_objs = [pr.pareto_objs for pr in r.phase_results if len(pr.pareto_objs) >= 2]
    ref_pt = compute_reference_point(all_objs, factor=1.1) if all_objs else np.array([1e9, 1e9])

    def safe_hv(objs):
        if len(objs) < 2: return 0.0
        mask = np.all(objs < ref_pt, axis=1)
        return float(hypervolume(objs[mask], ref_pt)) if mask.sum() >= 2 else 0.0

    return {
        'total_f1_res': r.total_f1_res,
        'phases': [
            {
                't': pr.t,
                'best_f1_res': pr.best_f1_res,
                'best_f2': pr.best_f2,
                'best_rmi': pr.best_rmi,
                'hd': pr.hd,
                'pareto_size': len(pr.pareto_objs),
                'da_size': len(pr.da_codes) if pr.da_codes is not None else 0,
                'pareto_objs': pr.pareto_objs.tolist(),
                'hv': safe_hv(pr.pareto_objs),
            }
            for pr in r.phase_results
        ]
    }

# ── 基线算法：传统多目标（3个） ────────────────────────────────────────

def _run_nsga2_rmisel(inst, seed):
    """R3-1：NSGA-II + RMI最大化后选择（分离支配关系 vs 选择规则的贡献）。
    搜索与 nsga2_topsis 完全一致（同seed同轨迹），仅后选择改为
    对Pareto集逐候选计算RMI（同一自适应S*评估），取argmax（并列取F1最小）。"""
    from alg_code.baselines.nsga2 import run_nsga2, NSGA2Params
    from alg_code.problems.encoding import build_combo_set
    from alg_code.problems.robustness import compute_rmi, adaptive_sample_size
    import numpy as np
    r = run_nsga2(inst, NSGA2Params(G_max=100, seed=seed))
    phases_out = []
    for t, pr in enumerate(r.phase_results):
        f1 = pr.pareto_objs[:,0]; f2 = pr.pareto_objs[:,1]
        combo_sets = build_combo_set(inst.phases[t], inst)
        rng = np.random.default_rng(seed)
        S = adaptive_sample_size(
            n_t=len(inst.phases[t].targets), combo_sets=combo_sets, r=2,
            delta=0.1, eps_rel=0.15, delta_prob=0.10, S_floor=10, S_cap=500,
        )
        rmis = np.array([
            compute_rmi(pr.pareto_codes[k], float(f1[k]), combo_sets,
                        inst.phases[t], inst, r=2, S=S, rng=rng)
            for k in range(len(pr.pareto_codes))
        ])
        best = int(np.argmax(rmis))
        ties = np.where(rmis >= rmis[best] - 1e-12)[0]
        best = int(ties[np.argmin(f1[ties])])
        phases_out.append({
            't': t, 'best_f1_res': float(f1[best]), 'best_f2': float(f2[best]),
            'best_rmi': float(rmis[best]), 'hd': pr.hd,
            'pareto_size': len(pr.pareto_objs), 'pareto_objs': pr.pareto_objs.tolist(),
        })
    return {'total_f1_res': sum(p['best_f1_res'] for p in phases_out), 'phases': phases_out}


def _run_nsga2_topsis(inst, seed):
    from alg_code.baselines.nsga2 import run_nsga2, NSGA2Params
    from alg_code.post_selection import topsis_select
    from alg_code.problems.encoding import build_combo_set
    import numpy as np
    r = run_nsga2(inst, NSGA2Params(G_max=100, seed=seed))
    phases_out = []
    for t, pr in enumerate(r.phase_results):
        f1 = pr.pareto_objs[:,0]; f2 = pr.pareto_objs[:,1]
        best_c, bf1, bf2 = topsis_select(pr.pareto_codes, f1, f2)
        combo_sets = build_combo_set(inst.phases[t], inst)
        rmi = _compute_rmi_adaptive(best_c, bf1, combo_sets, inst.phases[t], inst, seed)
        phases_out.append({
            't': t, 'best_f1_res': float(bf1), 'best_f2': float(bf2),
            'best_rmi': float(rmi), 'hd': pr.hd,
            'pareto_size': len(pr.pareto_objs), 'pareto_objs': pr.pareto_objs.tolist(),
        })
    return {'total_f1_res': sum(p['best_f1_res'] for p in phases_out), 'phases': phases_out}


def _run_nsga2_vikor(inst, seed):
    from alg_code.baselines.nsga2 import run_nsga2, NSGA2Params
    from alg_code.post_selection import vikor_select
    from alg_code.problems.encoding import build_combo_set
    import numpy as np
    r = run_nsga2(inst, NSGA2Params(G_max=100, seed=seed))
    phases_out = []
    for t, pr in enumerate(r.phase_results):
        f1 = pr.pareto_objs[:,0]; f2 = pr.pareto_objs[:,1]
        best_c, bf1, bf2 = vikor_select(pr.pareto_codes, f1, f2)
        combo_sets = build_combo_set(inst.phases[t], inst)
        rmi = _compute_rmi_adaptive(best_c, bf1, combo_sets, inst.phases[t], inst, seed)
        phases_out.append({
            't': t, 'best_f1_res': float(bf1), 'best_f2': float(bf2),
            'best_rmi': float(rmi), 'hd': pr.hd,
            'pareto_size': len(pr.pareto_objs), 'pareto_objs': pr.pareto_objs.tolist(),
        })
    return {'total_f1_res': sum(p['best_f1_res'] for p in phases_out), 'phases': phases_out}


def _run_moead_topsis(inst, seed):
    from alg_code.baselines.moead import run_moead, MOEADParams
    from alg_code.post_selection import topsis_select
    from alg_code.problems.encoding import build_combo_set
    r = run_moead(inst, MOEADParams(G_max=100, seed=seed))
    phases_out = []
    for t, pr in enumerate(r.phase_results):
        f1 = pr.pareto_objs[:,0]; f2 = pr.pareto_objs[:,1]
        best_c, bf1, bf2 = topsis_select(pr.pareto_codes, f1, f2)
        combo_sets = build_combo_set(inst.phases[t], inst)
        rmi = _compute_rmi_adaptive(best_c, bf1, combo_sets, inst.phases[t], inst, seed)
        phases_out.append({
            't': t, 'best_f1_res': float(bf1), 'best_f2': float(bf2),
            'best_rmi': float(rmi), 'hd': pr.hd,
            'pareto_size': len(pr.pareto_objs), 'pareto_objs': pr.pareto_objs.tolist(),
        })
    return {'total_f1_res': sum(p['best_f1_res'] for p in phases_out), 'phases': phases_out}


def _run_spea2_topsis(inst, seed):
    from alg_code.baselines.spea2 import run_spea2, SPEA2Params
    from alg_code.post_selection import topsis_select
    from alg_code.problems.encoding import build_combo_set
    r = run_spea2(inst, SPEA2Params(G_max=100, seed=seed))
    phases_out = []
    for t, pr in enumerate(r.phase_results):
        f1 = pr.pareto_objs[:,0]; f2 = pr.pareto_objs[:,1]
        best_c, bf1, bf2 = topsis_select(pr.pareto_codes, f1, f2)
        combo_sets = build_combo_set(inst.phases[t], inst)
        rmi = _compute_rmi_adaptive(best_c, bf1, combo_sets, inst.phases[t], inst, seed)
        phases_out.append({
            't': t, 'best_f1_res': float(bf1), 'best_f2': float(bf2),
            'best_rmi': float(rmi), 'hd': pr.hd,
            'pareto_size': len(pr.pareto_objs), 'pareto_objs': pr.pareto_objs.tolist(),
        })
    return {'total_f1_res': sum(p['best_f1_res'] for p in phases_out), 'phases': phases_out}

# ── 基线算法：多模态多目标（3个） ──────────────────────────────────────

def _run_commea_topsis(inst, seed):
    from alg_code.baselines.commea import run_commea, CoMMEAParams
    from alg_code.post_selection import topsis_select
    from alg_code.problems.encoding import build_combo_set
    r = run_commea(inst, CoMMEAParams(G_max=100, seed=seed))
    phases_out = []
    for t, pr in enumerate(r.phase_results):
        f1 = pr.pareto_objs[:,0]; f2 = pr.pareto_objs[:,1]
        best_c, bf1, bf2 = topsis_select(pr.pareto_codes, f1, f2)
        combo_sets = build_combo_set(inst.phases[t], inst)
        rmi = _compute_rmi_adaptive(best_c, bf1, combo_sets, inst.phases[t], inst, seed)
        phases_out.append({
            't': t, 'best_f1_res': float(bf1), 'best_f2': float(bf2),
            'best_rmi': float(rmi), 'hd': pr.hd,
            'pareto_size': len(pr.pareto_objs), 'pareto_objs': pr.pareto_objs.tolist(),
        })
    return {'total_f1_res': sum(p['best_f1_res'] for p in phases_out), 'phases': phases_out}


def _run_hdmmode_topsis(inst, seed):
    from alg_code.baselines.hdmmode import run_hdmmode, HDMMODEParams
    from alg_code.post_selection import topsis_select
    from alg_code.problems.encoding import build_combo_set
    r = run_hdmmode(inst, HDMMODEParams(G_max=100, seed=seed))
    phases_out = []
    for t, pr in enumerate(r.phase_results):
        f1 = pr.pareto_objs[:,0]; f2 = pr.pareto_objs[:,1]
        best_c, bf1, bf2 = topsis_select(pr.pareto_codes, f1, f2)
        combo_sets = build_combo_set(inst.phases[t], inst)
        rmi = _compute_rmi_adaptive(best_c, bf1, combo_sets, inst.phases[t], inst, seed)
        phases_out.append({
            't': t, 'best_f1_res': float(bf1), 'best_f2': float(bf2),
            'best_rmi': float(rmi), 'hd': pr.hd,
            'pareto_size': len(pr.pareto_objs), 'pareto_objs': pr.pareto_objs.tolist(),
        })
    return {'total_f1_res': sum(p['best_f1_res'] for p in phases_out), 'phases': phases_out}


def _run_mmoeadc_topsis(inst, seed):
    from alg_code.baselines.mmoeadc import run_mmoeadc, MMOEADCParams
    from alg_code.post_selection import topsis_select
    from alg_code.problems.encoding import build_combo_set
    r = run_mmoeadc(inst, MMOEADCParams(G_max=100, seed=seed))
    phases_out = []
    for t, pr in enumerate(r.phase_results):
        f1 = pr.pareto_objs[:,0]; f2 = pr.pareto_objs[:,1]
        best_c, bf1, bf2 = topsis_select(pr.pareto_codes, f1, f2)
        combo_sets = build_combo_set(inst.phases[t], inst)
        rmi = _compute_rmi_adaptive(best_c, bf1, combo_sets, inst.phases[t], inst, seed)
        phases_out.append({
            't': t, 'best_f1_res': float(bf1), 'best_f2': float(bf2),
            'best_rmi': float(rmi), 'hd': pr.hd,
            'pareto_size': len(pr.pareto_objs), 'pareto_objs': pr.pareto_objs.tolist(),
        })
    return {'total_f1_res': sum(p['best_f1_res'] for p in phases_out), 'phases': phases_out}

# ── 消融实验（3个变体：C2/C3/C4）──────────────────────────────────────

def _run_ablation(inst, seed, variant, base_v2=False):
    """消融实验变体（通过参数控制关闭单个组件）。
    base_v2=True：以修改稿v2（budget_aware）为基座——修改稿消融表口径"""
    from alg_code.rcmd_ea import run_rcmd_ea, RCMDParams
    from alg_code.metrics import hypervolume, compute_reference_point
    import copy, numpy as np

    # 基础参数：RCMD-EA 完整版（与主算法一致，C5已移除）
    base = RCMDParams(G_max=100, seed=seed, explore_ratio=0.4,
                      qa_ratio=0.5, G_max_dynamic=None,
                      budget_aware=base_v2, budget_g_target=10)

    if variant == 'no_explore':
        # -C2：关闭两阶段探索（全程RCMD，不经历NSGA-II预热阶段）
        params = copy.copy(base); params.explore_ratio = 0.0
    elif variant == 'no_dual_archive':
        # -C4：关闭双档案（qa_ratio=1 = 纯RCMD选择，无QA/DA分离）
        params = copy.copy(base); params.qa_ratio = 1.0
    elif variant == 'fixed_pref':
        # -C3：固定后选择权重α=0.5，不使用φ_t态势信号
        params = copy.copy(base); params.fixed_alpha = 0.5
    elif variant == 'no_rcmd':
        # -C1（R2-5）：Phase II 选择与终选用纯 Pareto 替代 RCMD，
        # DA 不做鲁棒可行过滤；保留两阶段/双档案/φ_t后选择。
        # 等NFE预算下（省下的MORI预算自动转化为更多深化代数）。
        params = copy.copy(base); params.disable_rcmd = True
    elif variant == 'mori_p40':
        # R3-6 阈值敏感性：MORI 可行阈值 = 种群40百分位
        params = copy.copy(base); params.mori_threshold_mode = 'p40'
    elif variant == 'mori_p60':
        # R3-6 阈值敏感性：MORI 可行阈值 = 种群60百分位
        params = copy.copy(base); params.mori_threshold_mode = 'p60'
    elif variant == 'mori_abs0':
        # R3-6 阈值敏感性：MORI 可行阈值 = 绝对阈值 0
        params = copy.copy(base); params.mori_threshold_mode = 'abs0'
    else:
        params = base

    r = run_rcmd_ea(inst, params, verbose=False)

    all_objs = [pr.pareto_objs for pr in r.phase_results if len(pr.pareto_objs) >= 2]
    ref_pt = compute_reference_point(all_objs, factor=1.1) if all_objs else np.array([1e9, 1e9])

    def safe_hv(objs):
        if len(objs) < 2: return 0.0
        mask = np.all(objs < ref_pt, axis=1)
        return float(hypervolume(objs[mask], ref_pt)) if mask.sum() >= 2 else 0.0

    return {
        'total_f1_res': r.total_f1_res,
        'variant': variant,
        'phases': [
            {
                't': pr.t, 'best_f1_res': pr.best_f1_res, 'best_f2': pr.best_f2,
                'best_rmi': pr.best_rmi, 'hd': pr.hd,
                'pareto_size': len(pr.pareto_objs),
                'pareto_objs': pr.pareto_objs.tolist(),
                'hv': safe_hv(pr.pareto_objs),
            }
            for pr in r.phase_results
        ]
    }


# ── 算法注册表 ────────────────────────────────────────────────────────
# 主实验：1个主算法 + 6个基线
# 消融实验：3个变体（-C2/-C3/-C4）

ALGOS = {
    # ── 主算法 ──
    'rcmd_ea':          _run_rcmd_ea,       # RCMD-EA（C1+C2+C3+C4，双档案，自适应后选择）
    'rcmd_v2':          _run_rcmd_v2,       # RCMD-EA v2（修改稿：budget_aware 预算修复）

    # ── 传统多目标基线（3个）──
    'nsga2_topsis':     _run_nsga2_topsis,  # NSGA-II + TOPSIS（等权）
    'nsga2_vikor':      _run_nsga2_vikor,   # NSGA-II + VIKOR（等权）
    'moead_topsis':     _run_moead_topsis,  # MOEA/D + TOPSIS（等权）
    'spea2_topsis':     _run_spea2_topsis,  # SPEA2 + TOPSIS（等权）

    # ── 修改稿新增基线（R3-1）──
    'nsga2_rmisel':     _run_nsga2_rmisel,  # NSGA-II + RMI最大化后选择

    # ── 多模态多目标基线（3个）──
    'commea_topsis':    _run_commea_topsis,   # CoMMEA + TOPSIS（等权）
    'hdmmode_topsis':   _run_hdmmode_topsis,  # HDMMODE + TOPSIS（等权）
    'mmoeadc_topsis':   _run_mmoeadc_topsis,  # MMOEADC + TOPSIS（等权）

    # ── 消融变体（3个：-C2/-C3/-C4）──
    'abl_no_explore':       lambda inst, seed: _run_ablation(inst, seed, 'no_explore'),
    'abl_fixed_pref':       lambda inst, seed: _run_ablation(inst, seed, 'fixed_pref'),
    'abl_no_dual_archive':  lambda inst, seed: _run_ablation(inst, seed, 'no_dual_archive'),

    # ── 修改稿新增消融（R2-5 / R3-6）──
    'abl_no_rcmd':      lambda inst, seed: _run_ablation(inst, seed, 'no_rcmd'),
    'abl_mori_p40':     lambda inst, seed: _run_ablation(inst, seed, 'mori_p40'),
    'abl_mori_p60':     lambda inst, seed: _run_ablation(inst, seed, 'mori_p60'),
    'abl_mori_abs0':    lambda inst, seed: _run_ablation(inst, seed, 'mori_abs0'),

    # ── 修改稿v2基座消融（budget_aware=True，正式修改稿口径）──
    'abv2_no_rcmd':          lambda inst, seed: _run_ablation(inst, seed, 'no_rcmd', base_v2=True),
    'abv2_mori_p40':         lambda inst, seed: _run_ablation(inst, seed, 'mori_p40', base_v2=True),
    'abv2_mori_p60':         lambda inst, seed: _run_ablation(inst, seed, 'mori_p60', base_v2=True),
    'abv2_mori_abs0':        lambda inst, seed: _run_ablation(inst, seed, 'mori_abs0', base_v2=True),
    'abv2_no_explore':       lambda inst, seed: _run_ablation(inst, seed, 'no_explore', base_v2=True),
    'abv2_fixed_pref':       lambda inst, seed: _run_ablation(inst, seed, 'fixed_pref', base_v2=True),
    'abv2_no_dual_archive':  lambda inst, seed: _run_ablation(inst, seed, 'no_dual_archive', base_v2=True),
}

INSTANCE_NAMES = [f'I{i}' for i in range(1, 11)]

# ── 任务单元 ──────────────────────────────────────────────────────────

def _result_path(algo: str, inst_name: str, run: int) -> Path:
    return RESULTS_DIR / algo / f'{inst_name}_run{run:03d}.json'

def _task(args):
    algo, inst_name, run, skip_existing = args
    out_path = _result_path(algo, inst_name, run)
    if skip_existing and out_path.exists():
        return algo, inst_name, run, 'skipped'
    try:
        inst = load_instance(str(INST_DIR / f'{inst_name}.mat'))
        fn = ALGOS[algo]
        t0 = time.time()
        result = fn(inst, seed=run)
        elapsed = time.time() - t0
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump({'algo': algo, 'instance': inst_name, 'run': run,
                       'elapsed_s': elapsed, **result}, f)
        return algo, inst_name, run, f'ok ({elapsed:.1f}s)'
    except Exception as e:
        return algo, inst_name, run, f'ERROR: {e}\n{traceback.format_exc()[:200]}'

# ── CLI ───────────────────────────────────────────────────────────────

def expand_instances(spec):
    """'I1-I10' → ['I1',...,'I10']；'I1 I3' → ['I1','I3']"""
    if not spec:
        return INSTANCE_NAMES
    result = []
    for s in spec:
        if '-' in s and s.startswith('I'):
            a, b = s.split('-')
            lo, hi = int(a[1:]), int(b[1:])
            result.extend(f'I{i}' for i in range(lo, hi+1))
        else:
            result.append(s)
    return result

def main():
    p = argparse.ArgumentParser(description='RCMD-EA实验驱动')
    p.add_argument('--algo', nargs='+', help='算法名（多个空格分隔）')
    p.add_argument('--all-algos', action='store_true')
    p.add_argument('--instances', nargs='+', default=['I1','I2','I3'])
    p.add_argument('--runs', type=int, default=5)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--skip-existing', action='store_true')
    p.add_argument('--list', action='store_true', help='列出算法')
    args = p.parse_args()

    if args.list:
        print('可用算法:', ', '.join(ALGOS.keys()))
        return

    algos = list(ALGOS.keys()) if args.all_algos else (args.algo or ['rcmd_ea'])
    instances = expand_instances(args.instances)

    tasks = [
        (algo, inst, run, args.skip_existing)
        for algo in algos
        for inst in instances
        for run in range(args.runs)
    ]

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
                print(f'  ✗ {algo}/{inst}/run{run}: {status[:80]}')
            if done % max(1, total//20) == 0 or done == total:
                elapsed = time.time() - t0
                eta = elapsed / done * (total - done) if done > 0 else 0
                print(f'  {done}/{total}  skipped={skipped} err={errors}  '
                      f'elapsed={elapsed:.0f}s  ETA={eta:.0f}s')

    print(f'\n完成 {done} 任务 (跳过{skipped}, 错误{errors})，结果在 {RESULTS_DIR}')


if __name__ == '__main__':
    main()

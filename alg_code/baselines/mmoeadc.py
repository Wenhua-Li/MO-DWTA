"""baselines/mmoeadc.py — MMOEADC GA + 决策空间NCM聚类 + 目标空间层次聚类修剪，适配WTA离散问题

Reference: MMOEADC (see competitor_alg_ref/MMOEADC).
决策空间NCM聚类提取局部Pareto代表，候选集过大时用目标空间Ward层次聚类+汉明拥挤度修剪。
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from alg_code.problems.instance import Instance, PhaseData
from alg_code.problems.encoding import build_combo_set, decode, repair, random_individual
from alg_code.problems.evaluator import evaluate_f1_raw_batch, evaluate_f2_raw_batch
from alg_code.metrics import pareto_dominance_filter, hamming_diversity
from alg_code.baselines._mmea_utils import nd_sort, box_ncm, ga_offspring, hamming_crowd


@dataclass
class MMOEADCParams:
    N_factor: int = 5
    G_max: int = 100
    delta: int = 5             # NCM最小簇大小
    p_crossover: float = 0.8
    p_mut: float = 0.05
    seed: Optional[int] = None


@dataclass
class MMOEADCPhaseResult:
    t: int
    pareto_objs: np.ndarray
    pareto_codes: np.ndarray
    hd: float


@dataclass
class MMOEADCResult:
    instance_name: str
    phase_results: List[MMOEADCPhaseResult] = field(default_factory=list)


def _hier_trim(pop, f1, f2, N):
    """目标空间Ward层次聚类 + 汉明拥挤度修剪，从len(pop)个候选中选N个。"""
    try:
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import pdist
        objs = np.column_stack([f1, f2])
        z_min = objs.min(axis=0); z_rng = objs.max(axis=0) - z_min
        z_rng[z_rng < 1e-12] = 1.0
        objs_n = (objs - z_min) / z_rng
        if len(objs_n) <= N:
            return np.arange(len(pop))
        Z = linkage(pdist(objs_n, 'euclidean'), 'ward')
        labels = fcluster(Z, N, criterion='maxclust')
    except Exception:
        labels = np.arange(len(pop)) % N + 1   # 降级：按索引分组

    n_clusters = labels.max()
    selected = []
    for c in range(1, n_clusters + 1):
        members = np.where(labels == c)[0]
        if len(members) == 0:
            continue
        # 同簇内保留汉明最孤立的代表
        cd = hamming_crowd(pop[members], K=3)
        selected.append(members[np.argmax(cd)])

    # 若选出数量不足，按汉明拥挤度补充
    if len(selected) < N:
        selected_set = set(selected)
        rest = [i for i in range(len(pop)) if i not in selected_set]
        cd_rest = hamming_crowd(pop[rest], K=3)
        rest_ord = np.array(rest)[np.argsort(-cd_rest)]
        selected.extend(rest_ord[:N - len(selected)].tolist())

    return np.array(selected[:N])


def _env_sel_mmoeadc(pop, f1, f2, N, delta):
    """MMOEADC环境选择：NCM局部代表 + 全局非支配排序 + 超额时层次聚类修剪。"""
    clusters = box_ncm(pop, r_frac=0.1)
    local_idx = []
    for cl in clusters:
        if len(cl) <= delta:
            continue
        cl_arr = np.array(cl)
        ranks = nd_sort(f1[cl_arr], f2[cl_arr])
        local_idx.extend(cl_arr[ranks == 1].tolist())
    local_set = set(local_idx)

    ranks_all = nd_sort(f1, f2)
    P = list(local_idx)
    rest = set(range(len(f1))) - local_set
    k = 1
    while len(P) < N:
        add = [i for i in rest if ranks_all[i] == k]
        P.extend(add); rest -= set(add)
        k += 1
        if k > len(f1):
            break

    P = np.array(P)
    if len(P) > N:
        keep = _hier_trim(pop[P], f1[P], f2[P], N)
        P = P[keep]

    return pop[P], f1[P], f2[P]


def solve_mmoeadc_phase(
    phase_data: PhaseData, inst: Instance,
    params: MMOEADCParams, rng: np.random.Generator,
) -> MMOEADCPhaseResult:
    n_t = phase_data.n_t
    N = params.N_factor * n_t
    combo_sets = build_combo_set(phase_data, inst)

    pop = np.array([random_individual(combo_sets, rng) for _ in range(N)])
    pop = np.array([repair(pop[i], combo_sets, phase_data, inst) for i in range(N)])
    xb = np.stack([decode(pop[i], combo_sets, inst) for i in range(N)])
    f1 = evaluate_f1_raw_batch(xb, phase_data, inst)
    f2 = evaluate_f2_raw_batch(xb)

    for _ in range(params.G_max):
        Q = ga_offspring(pop, combo_sets, rng, params.p_crossover, params.p_mut)
        Q = np.array([repair(Q[i], combo_sets, phase_data, inst) for i in range(len(Q))])
        xq = np.stack([decode(Q[i], combo_sets, inst) for i in range(len(Q))])
        f1q = evaluate_f1_raw_batch(xq, phase_data, inst)
        f2q = evaluate_f2_raw_batch(xq)

        combined = np.vstack([pop, Q])
        f1c = np.concatenate([f1, f1q]); f2c = np.concatenate([f2, f2q])
        pop, f1, f2 = _env_sel_mmoeadc(combined, f1c, f2c, N, params.delta)

    pf = pareto_dominance_filter(np.column_stack([f1, f2]))
    pf_idx = np.where(pf)[0]
    return MMOEADCPhaseResult(
        t=0,
        pareto_objs=np.column_stack([f1[pf_idx], f2[pf_idx]]),
        pareto_codes=pop[pf_idx],
        hd=hamming_diversity(pop),
    )


def run_mmoeadc(
    inst: Instance,
    params: Optional[MMOEADCParams] = None,
    verbose: bool = False,
) -> MMOEADCResult:
    if params is None:
        params = MMOEADCParams()
    rng = np.random.default_rng(params.seed)
    result = MMOEADCResult(instance_name=inst.name)
    for t in range(inst.T_total):
        phase = inst.phases[t]
        if verbose:
            print(f"  Phase {t+1}/{inst.T_total}")
        pr = solve_mmoeadc_phase(phase, inst, params, rng)
        pr.t = t
        result.phase_results.append(pr)
    return result

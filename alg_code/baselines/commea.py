"""baselines/commea.py — CoMMEA 双种群协同进化多模态多目标算法，适配WTA离散问题

Reference: CoMMEA dual-population coevolutionary MMEA (see competitor_alg_ref/CoMMEA).
Pop1: 收敛导向（标准SPEA2环境选择）
Pop2: 多样性导向（epsilon-支配环境选择 + 双空间拥挤度修剪）
输出：Pop2的Pareto前沿
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
from alg_code.baselines._mmea_utils import (
    spea2_fitness, ga_offspring, nd_sort, hamming_crowd, obj_crowd,
)


@dataclass
class CoMMEAParams:
    N_factor: int = 5
    G_max: int = 100
    eps_floor: float = 0.2
    p_crossover: float = 0.8
    p_mut: float = 0.05
    seed: Optional[int] = None


@dataclass
class CoMMEAPhaseResult:
    t: int
    pareto_objs: np.ndarray
    pareto_codes: np.ndarray
    hd: float


@dataclass
class CoMMEAResult:
    instance_name: str
    phase_results: List[CoMMEAPhaseResult] = field(default_factory=list)


def _env_sel_s2(pop, f1, f2, N):
    """SPEA2环境选择（收敛种群用）。"""
    fit = spea2_fitness(pop, f1, f2)
    mask = fit < 1.0
    n_ok = int(mask.sum())
    if n_ok < N:
        sel = np.argsort(fit)[:N]
    elif n_ok > N:
        idx = np.where(mask)[0]
        objs = np.column_stack([f1[idx], f2[idx]])
        od = np.sqrt(np.sum((objs[:, None, :] - objs[None, :, :]) ** 2, axis=2))
        np.fill_diagonal(od, np.inf)
        sorted_od = np.sort(od, axis=1)
        lex = np.lexsort(sorted_od[:, ::-1].T)
        sel = idx[lex[n_ok - N:]]
    else:
        sel = np.where(mask)[0]
    return pop[sel], f1[sel], f2[sel]


def _env_sel_eps(pop, f1, f2, N, eps):
    """Epsilon-支配环境选择（多样性种群用）+ 双空间拥挤度修剪。"""
    n = len(f1)
    # epsilon-支配：A eps-dominates B 当 (1+eps)*f_A <= f_B 各分量成立且至少一个严格小
    F = np.column_stack([f1, f2])
    sc = (1.0 + eps) * F                                         # (n, 2) 缩放后的值
    diff = F[None, :, :] - sc[:, None, :]                        # (n, n, 2): F_j - sc_i
    eps_dom = np.all(diff >= 0, axis=2) & np.any(diff > 0, axis=2)
    np.fill_diagonal(eps_dom, False)
    is_dominated = eps_dom.any(axis=0)                           # 被某人eps-支配

    cands = np.where(~is_dominated)[0]
    if len(cands) < N:
        ranks = nd_sort(f1, f2)
        rest = np.where(is_dominated)[0]
        rest_ord = rest[np.argsort(ranks[rest])]
        cands = np.concatenate([cands, rest_ord[:N - len(cands)]])

    if len(cands) > N:
        sp = pop[cands]; sf1 = f1[cands]; sf2 = f2[cands]
        ch = hamming_crowd(sp, K=3)
        co = obj_crowd(sf1, sf2, K=3)
        mh, mo = ch.max() + 1e-12, co.max() + 1e-12
        cd = ch / mh + co / mo                                   # 越大越孤立
        order = np.argsort(-cd)                                  # 降序：最孤立在前
        cands = cands[order[:N]]

    return pop[cands], f1[cands], f2[cands]


def _local_fit(pop: np.ndarray, K: int = 3) -> np.ndarray:
    """局部决策空间拥挤度作为配对适应度（越孤立越被优选）。"""
    cd = hamming_crowd(pop, K)
    avg = cd.mean()
    if avg < 1e-12:
        return np.ones(len(pop))
    return 1.0 / (1.0 + cd / avg)


def solve_commea_phase(
    phase_data: PhaseData, inst: Instance,
    params: CoMMEAParams, rng: np.random.Generator,
) -> CoMMEAPhaseResult:
    n_t = phase_data.n_t
    N = params.N_factor * n_t
    combo_sets = build_combo_set(phase_data, inst)

    def _init():
        p = np.array([random_individual(combo_sets, rng) for _ in range(N)])
        p = np.array([repair(p[i], combo_sets, phase_data, inst) for i in range(N)])
        xb = np.stack([decode(p[i], combo_sets, inst) for i in range(N)])
        return p, evaluate_f1_raw_batch(xb, phase_data, inst), evaluate_f2_raw_batch(xb)

    pop1, f1_1, f2_1 = _init()
    pop2, f1_2, f2_2 = _init()

    for g in range(params.G_max):
        evoState = (g + 1) / params.G_max
        curEps = max(-np.log2(max(1.5 * evoState, 1e-9)), params.eps_floor)

        Q1 = ga_offspring(pop1, combo_sets, rng, params.p_crossover, params.p_mut)
        Q2 = ga_offspring(pop2, combo_sets, rng, params.p_crossover, params.p_mut)
        Qall = np.vstack([Q1, Q2])
        Qall = np.array([repair(Qall[i], combo_sets, phase_data, inst) for i in range(len(Qall))])
        xq = np.stack([decode(Qall[i], combo_sets, inst) for i in range(len(Qall))])
        f1q = evaluate_f1_raw_batch(xq, phase_data, inst)
        f2q = evaluate_f2_raw_batch(xq)

        # 两个种群各自与全部子代合并后再选择
        pop1, f1_1, f2_1 = _env_sel_s2(
            np.vstack([pop1, Qall]),
            np.concatenate([f1_1, f1q]),
            np.concatenate([f2_1, f2q]), N)

        pop2, f1_2, f2_2 = _env_sel_eps(
            np.vstack([pop2, Qall]),
            np.concatenate([f1_2, f1q]),
            np.concatenate([f2_2, f2q]), N, curEps)

    pf = pareto_dominance_filter(np.column_stack([f1_2, f2_2]))
    pf_idx = np.where(pf)[0]
    return CoMMEAPhaseResult(
        t=0,
        pareto_objs=np.column_stack([f1_2[pf_idx], f2_2[pf_idx]]),
        pareto_codes=pop2[pf_idx],
        hd=hamming_diversity(pop2),
    )


def run_commea(
    inst: Instance,
    params: Optional[CoMMEAParams] = None,
    verbose: bool = False,
) -> CoMMEAResult:
    if params is None:
        params = CoMMEAParams()
    rng = np.random.default_rng(params.seed)
    result = CoMMEAResult(instance_name=inst.name)
    for t in range(inst.T_total):
        phase = inst.phases[t]
        if verbose:
            print(f"  Phase {t+1}/{inst.T_total}")
        pr = solve_commea_phase(phase, inst, params, rng)
        pr.t = t
        result.phase_results.append(pr)
    return result

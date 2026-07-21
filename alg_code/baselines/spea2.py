"""baselines/spea2.py — SPEA2 适配WTA离散问题

Reference: E. Zitzler, M. Laumanns, L. Thiele, SPEA2: Improving the strength
Pareto evolutionary algorithm, EUROGEN 2001.
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
from alg_code.baselines._mmea_utils import spea2_fitness, ga_offspring


@dataclass
class SPEA2Params:
    N_factor: int = 5
    G_max: int = 100
    p_crossover: float = 0.8
    p_mut: float = 0.05
    seed: Optional[int] = None


@dataclass
class SPEA2PhaseResult:
    t: int
    pareto_objs: np.ndarray    # (K, 2)
    pareto_codes: np.ndarray   # (K, n_t)
    hd: float


@dataclass
class SPEA2Result:
    instance_name: str
    phase_results: List[SPEA2PhaseResult] = field(default_factory=list)


def _tournament(fitness: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """二元锦标赛选择（适应度越低越优先）。"""
    N = len(fitness)
    a = rng.integers(0, N, n); b = rng.integers(0, N, n)
    return np.where(fitness[a] <= fitness[b], a, b)


def _env_sel(pop: np.ndarray, f1: np.ndarray, f2: np.ndarray, N: int):
    """SPEA2环境选择：优先保留适应度<1的个体，不足填补，过多按目标空间kNN修剪。"""
    fit = spea2_fitness(pop, f1, f2)
    mask = fit < 1.0
    n_ok = int(mask.sum())

    if n_ok < N:
        order = np.argsort(fit)
        sel = order[:N]
    elif n_ok > N:
        idx = np.where(mask)[0]
        objs = np.column_stack([f1[idx], f2[idx]])
        od = np.sqrt(np.sum((objs[:, None, :] - objs[None, :, :]) ** 2, axis=2))
        np.fill_diagonal(od, np.inf)
        sorted_od = np.sort(od, axis=1)
        # 按目标空间邻距字典序排列，逐步删去最拥挤个体
        lex = np.lexsort(sorted_od[:, ::-1].T)  # 升序：lex[0]最拥挤
        keep = lex[n_ok - N:]                    # 保留最稀疏的N个
        sel = idx[keep]
    else:
        sel = np.where(mask)[0]

    return pop[sel], f1[sel], f2[sel], fit[sel]


def solve_spea2_phase(
    phase_data: PhaseData, inst: Instance,
    params: SPEA2Params, rng: np.random.Generator,
) -> SPEA2PhaseResult:
    n_t = phase_data.n_t
    N = params.N_factor * n_t
    combo_sets = build_combo_set(phase_data, inst)

    pop = np.array([random_individual(combo_sets, rng) for _ in range(N)])
    pop = np.array([repair(pop[i], combo_sets, phase_data, inst) for i in range(N)])
    xb = np.stack([decode(pop[i], combo_sets, inst) for i in range(N)])
    f1 = evaluate_f1_raw_batch(xb, phase_data, inst)
    f2 = evaluate_f2_raw_batch(xb)
    pop, f1, f2, fit = _env_sel(pop, f1, f2, N)

    for _ in range(params.G_max):
        mating = _tournament(fit, N, rng)
        Q = ga_offspring(pop[mating], combo_sets, rng, params.p_crossover, params.p_mut)
        Q = np.array([repair(Q[i], combo_sets, phase_data, inst) for i in range(len(Q))])
        xq = np.stack([decode(Q[i], combo_sets, inst) for i in range(len(Q))])
        f1q = evaluate_f1_raw_batch(xq, phase_data, inst)
        f2q = evaluate_f2_raw_batch(xq)

        combined = np.vstack([pop, Q])
        f1c = np.concatenate([f1, f1q]); f2c = np.concatenate([f2, f2q])
        pop, f1, f2, fit = _env_sel(combined, f1c, f2c, N)

    pf = pareto_dominance_filter(np.column_stack([f1, f2]))
    pf_idx = np.where(pf)[0]
    return SPEA2PhaseResult(
        t=0,
        pareto_objs=np.column_stack([f1[pf_idx], f2[pf_idx]]),
        pareto_codes=pop[pf_idx],
        hd=hamming_diversity(pop),
    )


def run_spea2(
    inst: Instance,
    params: Optional[SPEA2Params] = None,
    verbose: bool = False,
) -> SPEA2Result:
    if params is None:
        params = SPEA2Params()
    rng = np.random.default_rng(params.seed)
    result = SPEA2Result(instance_name=inst.name)
    for t in range(inst.T_total):
        phase = inst.phases[t]
        if verbose:
            print(f"  Phase {t+1}/{inst.T_total}")
        pr = solve_spea2_phase(phase, inst, params, rng)
        pr.t = t
        result.phase_results.append(pr)
    return result

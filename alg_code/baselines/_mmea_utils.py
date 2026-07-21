"""baselines/_mmea_utils.py — 多模态/分解基线算法共享工具"""
import numpy as np
from typing import List
from alg_code.problems.encoding import combo_count
from alg_code.metrics import pareto_dominance_filter


def nd_sort(f1: np.ndarray, f2: np.ndarray) -> np.ndarray:
    """非支配排序，返回1-indexed前沿等级。"""
    objs = np.column_stack([f1, f2])
    N = len(f1)
    ranks = np.zeros(N, dtype=int)
    remaining = np.ones(N, dtype=bool)
    rank = 1
    while remaining.any():
        pf = pareto_dominance_filter(objs[remaining])
        idx = np.where(remaining)[0][pf]
        ranks[idx] = rank
        remaining[idx] = False
        rank += 1
    return ranks


def box_ncm(pop: np.ndarray, r_frac: float = 0.1) -> List[List[int]]:
    """Box邻域聚类：所有维度差值 <= r_j 时两个体相邻，返回连通分量列表。"""
    N, n_t = pop.shape
    lo = pop.min(axis=0); hi = pop.max(axis=0)
    r = np.maximum(1, np.floor((hi - lo) * r_frac)).astype(int)
    adj = np.all(np.abs(pop[:, None, :] - pop[None, :, :]) <= r, axis=2)
    np.fill_diagonal(adj, False)
    visited = np.zeros(N, dtype=bool)
    components: List[List[int]] = []
    for start in range(N):
        if visited[start]:
            continue
        comp, queue = [start], [start]
        visited[start] = True
        while queue:
            node = queue.pop(0)
            for nb in np.where(adj[node])[0]:
                if not visited[nb]:
                    visited[nb] = True
                    comp.append(nb); queue.append(nb)
        components.append(comp)
    return components


def de_int(pop: np.ndarray, combo_sets: list, rng: np.random.Generator, F: float = 0.5) -> np.ndarray:
    """整数DE/rand/1变异，按各目标combo数取模回绕。"""
    N, _ = pop.shape
    offs = pop.copy()
    idx_all = np.arange(N)
    for i in range(N):
        others = np.delete(idx_all, i)
        r1, r2, r3 = rng.choice(others, 3, replace=False)
        for j, tc in enumerate(combo_sets):
            H = combo_count(tc)
            if H == 0:
                continue
            v = (int(pop[r1, j]) + round(F * (int(pop[r2, j]) - int(pop[r3, j])))) % (H + 1)
            offs[i, j] = v
    return offs.astype(np.int32)


def ga_offspring(pop: np.ndarray, combo_sets: list, rng: np.random.Generator,
                 p_cx: float = 0.8, p_mut: float = 0.05) -> np.ndarray:
    """均匀交叉+按位变异，返回N//2个子代。"""
    N, n_t = pop.shape
    perm = rng.permutation(N)
    children = []
    for k in range(N // 2):
        p1, p2 = pop[perm[2*k]], pop[perm[2*k+1]]
        child = (np.where(rng.random(n_t) < 0.5, p1, p2) if rng.random() < p_cx
                 else (p1 if rng.random() < 0.5 else p2).copy()).astype(np.int32)
        for j, tc in enumerate(combo_sets):
            H = combo_count(tc)
            if H > 0 and rng.random() < p_mut:
                child[j] = int(rng.integers(0, H + 1))
        children.append(child)
    return np.array(children, dtype=np.int32)


def spea2_fitness(pop: np.ndarray, f1: np.ndarray, f2: np.ndarray) -> np.ndarray:
    """SPEA2适应度 = R(i) + D(i)，密度用目标空间第k近邻距离计算。"""
    N = len(f1)
    dom = ((f1[:, None] <= f1[None, :]) & (f2[:, None] <= f2[None, :]) &
           ((f1[:, None] < f1[None, :]) | (f2[:, None] < f2[None, :])))
    np.fill_diagonal(dom, False)
    S = dom.sum(axis=1)
    R = np.array([float(S[dom[:, i]].sum()) for i in range(N)])
    k = max(1, int(np.ceil(np.sqrt(N))) - 1)
    objs = np.column_stack([f1, f2])
    od = np.sqrt(np.sum((objs[:, None, :] - objs[None, :, :]) ** 2, axis=2))
    np.fill_diagonal(od, np.inf)
    sigma = np.sort(od, axis=1)[:, min(k, N - 2)]
    return R + 1.0 / (sigma + 2.0)


def hamming_crowd(pop: np.ndarray, K: int = 3) -> np.ndarray:
    """前K个汉明近邻距离之和（值越大越孤立）。"""
    N = len(pop)
    if N <= 1:
        return np.zeros(N)
    hd = np.sum(pop[:, None, :] != pop[None, :, :], axis=2).astype(float)
    np.fill_diagonal(hd, np.inf)
    return np.sort(hd, axis=1)[:, :min(K, N - 1)].sum(axis=1)


def obj_crowd(f1: np.ndarray, f2: np.ndarray, K: int = 3) -> np.ndarray:
    """目标空间前K欧氏近邻距离之和。"""
    N = len(f1)
    if N <= 1:
        return np.zeros(N)
    objs = np.column_stack([f1, f2])
    d = np.sqrt(np.sum((objs[:, None, :] - objs[None, :, :]) ** 2, axis=2))
    np.fill_diagonal(d, np.inf)
    return np.sort(d, axis=1)[:, :min(K, N - 1)].sum(axis=1)

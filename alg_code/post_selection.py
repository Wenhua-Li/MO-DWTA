"""post_selection.py — 后选择方法：TOPSIS、VIKOR 适配WTA双目标

用于给NSGA-II等基线算法提供后选择，以便与RCMD-EA的偏好感知后选择做PSR/PSS对比。
"""
import numpy as np
from typing import List, Optional
from .problems.encoding import hamming_distance


def topsis_select(
    pareto_codes: np.ndarray,   # (K, n_t) 编码
    f1_res: np.ndarray,          # (K,) 残余威胁（越小越好）
    f2: np.ndarray,              # (K,) 弹药消耗（越小越好）
    w1: float = 0.5, w2: float = 0.5,
    prev_c: Optional[np.ndarray] = None,
    epsilon_ps: float = 0.01,
) -> tuple:
    """
    TOPSIS后选择（固定权重，经典用法）
    返回 (best_c, best_f1_res, best_f2)
    若只有1个解，直接返回
    """
    K = len(pareto_codes)
    if K == 0:
        raise ValueError("空Pareto集")
    if K == 1:
        return pareto_codes[0], float(f1_res[0]), float(f2[0])

    # 归一化决策矩阵（列归一化：每列除以L2范数）
    objs = np.column_stack([f1_res, f2]).astype(float)
    norms = np.linalg.norm(objs, axis=0)
    norms = np.where(norms < 1e-12, 1.0, norms)
    v = objs / norms * np.array([w1, w2])

    # 理想点（越小越好，故ideal=列最小值，nadir=列最大值）
    ideal = v.min(axis=0)
    nadir = v.max(axis=0)

    d_pos = np.sqrt(((v - ideal) ** 2).sum(axis=1))
    d_neg = np.sqrt(((v - nadir) ** 2).sum(axis=1))

    score = d_neg / (d_pos + d_neg + 1e-12)  # 越大越优
    best_idx = int(np.argmax(score))
    return pareto_codes[best_idx], float(f1_res[best_idx]), float(f2[best_idx])


def vikor_select(
    pareto_codes: np.ndarray,
    f1_res: np.ndarray,
    f2: np.ndarray,
    w1: float = 0.5, w2: float = 0.5,
    v: float = 0.5,   # 最大群体效益权重
) -> tuple:
    """
    VIKOR后选择（固定权重，经典用法）
    返回 (best_c, best_f1_res, best_f2)
    """
    K = len(pareto_codes)
    if K == 1:
        return pareto_codes[0], float(f1_res[0]), float(f2[0])

    objs = np.column_stack([f1_res, f2]).astype(float)
    f_star = objs.min(axis=0)  # 理想值
    f_minus = objs.max(axis=0)  # 最差值
    rng = np.where(np.abs(f_minus - f_star) < 1e-12, 1.0, f_minus - f_star)

    weights = np.array([w1, w2])
    # S: 加权曼哈顿距离（群体效益）
    S = ((weights * (objs - f_star) / rng)).sum(axis=1)
    # R: 最大加权距离（个体遗憾）
    R = (weights * (objs - f_star) / rng).max(axis=1)

    S_star, S_minus = S.min(), S.max()
    R_star, R_minus = R.min(), R.max()
    sR = S_minus - S_star
    rR = R_minus - R_star

    Q = (v * (S - S_star) / (sR + 1e-12) +
         (1 - v) * (R - R_star) / (rR + 1e-12))

    best_idx = int(np.argmin(Q))
    return pareto_codes[best_idx], float(f1_res[best_idx]), float(f2[best_idx])


def random_select(
    pareto_codes: np.ndarray,
    f1_res: np.ndarray,
    f2: np.ndarray,
    rng_seed: int = 0,
) -> tuple:
    """随机后选择（下界基准）"""
    rng = np.random.default_rng(rng_seed)
    idx = int(rng.integers(0, len(pareto_codes)))
    return pareto_codes[idx], float(f1_res[idx]), float(f2[idx])


def best_f1_select(
    pareto_codes: np.ndarray,
    f1_res: np.ndarray,
    f2: np.ndarray,
) -> tuple:
    """纯效能优先选择（α=1基准）"""
    idx = int(np.argmin(f1_res))
    return pareto_codes[idx], float(f1_res[idx]), float(f2[idx])

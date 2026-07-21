"""robustness.py — RMI/MORI鲁棒性指标（精确+蒙特卡洛）"""
import numpy as np
from math import comb, ceil, log, sqrt
from typing import List, Optional
from .encoding import TargetCombos, combo_count, hamming_distance
from .evaluator import evaluate_f1_raw
from .instance import Instance, PhaseData

# 精确枚举邻域上限：|N_r| ≤ N_ENUM 时枚举所有邻居（零误差）
_N_ENUM = 300


def adaptive_sample_size(
    n_t: int,
    combo_sets: List[TargetCombos],
    r: int,
    delta: float,
    eps_rel: float = 0.15,
    delta_prob: float = 0.10,
    alpha: float = 1.0,
    S_floor: int = 10,
    S_cap: int = 500,
) -> int:
    """
    自适应采样数 S*，基于三成分公式：

      S_acc  = ⌈ln(2/δ_prob) / (2·ε_rel²)⌉           — Hoeffding精度下界
      S_cov  = ⌈α·√(C(n_t,r)·H̄^r)⌉                  — 邻域覆盖下界
      S_cross = S_acc·max(1, r/δ)                     — 跨峰修正（r≥δ时生效）

      S* = max(S_floor, min(S_cap, max(S_acc, S_cov, S_cross)))

    返回 -1 表示邻域足够小（|N_r| ≤ _N_ENUM），应走精确枚举路径。

    参数
    ----
    eps_rel   : 相对估计误差（占 fitness range 比例），默认 0.15
    delta_prob: 允许失败概率，默认 0.05
    alpha     : 覆盖项系数，默认 1.0
    delta     : niche 半径（峰间最小 Hamming 距离），用于跨峰修正
    """
    H_bar = float(np.mean([combo_count(cs) for cs in combo_sets]))
    H_bar = max(H_bar, 1.0)

    N_r = comb(n_t, r) * (H_bar ** r)

    if N_r <= _N_ENUM:
        return -1  # 精确枚举

    # 第一成分：Hoeffding 精度下界
    S_acc = ceil(log(2.0 / delta_prob) / (2.0 * eps_rel ** 2))

    # 第二成分：邻域覆盖下界
    S_cov = ceil(alpha * sqrt(N_r))

    # 第三成分：跨峰修正
    gamma = max(1.0, r / max(delta, 1.0))
    S_cross = ceil(S_acc * gamma)

    S = max(S_floor, min(S_cap, max(S_acc, S_cov, S_cross)))
    return S


def _generate_neighbors_exact(
    c: np.ndarray,
    combo_sets: List[TargetCombos],
    r: int,
) -> np.ndarray:
    """精确枚举所有 r 阶汉明邻居（仅在 |N_r| ≤ _N_ENUM 时调用）。"""
    from itertools import combinations
    n_t = len(combo_sets)
    neighbors = []
    for positions in combinations(range(n_t), r):
        ranges = []
        for pos in positions:
            H = combo_count(combo_sets[pos])
            alt = [v for v in range(0, H + 1) if v != c[pos]]
            ranges.append(alt)
        # 所有位置的笛卡尔积
        from itertools import product
        for vals in product(*ranges):
            nb = c.copy()
            for pos, val in zip(positions, vals):
                nb[pos] = val
            neighbors.append(nb)
    if not neighbors:
        return c.reshape(1, -1)
    return np.array(neighbors, dtype=c.dtype)


# ── RMI（单目标，继承Paper2逻辑）────────────────────────────────────

def _generate_neighbors_mc(
    c: np.ndarray, combo_sets: List[TargetCombos],
    r: int, S: int, rng: np.random.Generator
) -> np.ndarray:
    """
    MC生成 S 个 r 阶汉明邻居，返回 (S, n_t) 矩阵。
    对所有 n_t 均使用此路径，确保 S 参数严格有效（Bug1修复）。
    """
    n_t = len(combo_sets)
    nbs = np.tile(c, (S, 1))                          # (S, n_t)
    positions = rng.integers(0, n_t, size=(S, r))     # 每行随机选r个位置
    for s in range(S):
        for pos in positions[s]:
            H = combo_count(combo_sets[int(pos)])
            if H > 0:
                nbs[s, int(pos)] = rng.integers(0, H + 1)
    return nbs


def compute_rmi(
    c: np.ndarray,
    f_c: float,
    combo_sets: List[TargetCombos],
    phase_data: PhaseData,
    inst: Instance,
    r: int = 2,
    S: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    _nbs: Optional[np.ndarray] = None,   # 可外部传入已生成的邻居矩阵（共享用）
    delta: float = 1.0,
    eps_rel: float = 0.15,
    delta_prob: float = 0.10,
) -> float:
    """RMI(c) — 批量向量化版本。

    RMI(c) = F1_ref - E_{c' in N_r(c)}[F1(c')]，F1_ref为该阶段零毁伤上界
    （固定参考点，见 evaluator.f1_ref）。数值越大表示邻域期望残余威胁越低，
    鲁棒性越强；与当前解自身取值无关（不再是 c 与邻居的差值）。
    f_c 参数保留以兼容既有调用签名，新公式不再使用该值。

    S=None 时自动调用 adaptive_sample_size()；若邻域小（≤_N_ENUM）则精确枚举。
    S 为正整数时作为 override，直接使用该采样数（向后兼容）。
    """
    from .evaluator import evaluate_f1_raw_batch, f1_ref
    if rng is None:
        rng = np.random.default_rng()
    if _nbs is None:
        if S is None:
            S_eff = adaptive_sample_size(
                len(combo_sets), combo_sets, r, delta, eps_rel, delta_prob
            )
        else:
            S_eff = S
        if S_eff == -1:
            nbs = _generate_neighbors_exact(c, combo_sets, r)
        else:
            nbs = _generate_neighbors_mc(c, combo_sets, r, S_eff, rng)
    else:
        nbs = _nbs
    x_batch = _decode_batch_fast(nbs, combo_sets, inst)
    fs = evaluate_f1_raw_batch(x_batch, phase_data, inst)
    return f1_ref(phase_data) - float(np.mean(fs))


def compute_rmi_q(
    c: np.ndarray,
    f_q_c: float,
    combo_sets: List[TargetCombos],
    phase_data_q,
    inst: Instance,
    q: int,
    r: int = 2,
    S: Optional[int] = None,
    rng=None,
    _nbs: Optional[np.ndarray] = None,
    delta: float = 1.0,
    eps_rel: float = 0.15,
    delta_prob: float = 0.10,
) -> float:
    """第q个目标的RMI — 批量向量化版本。S=None 时自动自适应。

    RMI_q(c) = F_ref_q - E[F_q(c')]，F_ref_q为该目标的理论最坏参考值
    （q=0: 零毁伤上界 f1_ref；q=1: 满额分配上界 f2_ref），与 c 自身取值无关。
    f_q_c 参数保留以兼容既有调用签名，新公式不再使用该值。
    """
    from .evaluator import evaluate_f1_raw_batch, evaluate_f2_raw_batch, f1_ref, f2_ref
    if rng is None:
        rng = np.random.default_rng()
    if _nbs is None:
        if S is None:
            S_eff = adaptive_sample_size(
                len(combo_sets), combo_sets, r, delta, eps_rel, delta_prob
            )
        else:
            S_eff = S
        if S_eff == -1:
            nbs = _generate_neighbors_exact(c, combo_sets, r)
        else:
            nbs = _generate_neighbors_mc(c, combo_sets, r, S_eff, rng)
    else:
        nbs = _nbs
    x_batch = _decode_batch_fast(nbs, combo_sets, inst)
    if q == 0:
        fs = evaluate_f1_raw_batch(x_batch, phase_data_q, inst)
        ref = f1_ref(phase_data_q)
    else:
        fs = evaluate_f2_raw_batch(x_batch)
        ref = f2_ref(phase_data_q)
    return ref - float(np.mean(fs))




# ── MORI（多目标扩展）────────────────────────────────────────────────

def compute_mori(
    c: np.ndarray,
    f1_c: float, f2_c: float,
    combo_sets: List[TargetCombos],
    phase_data: PhaseData,
    inst: Instance,
    f1_range: float, f2_range: float,
    r: int = 2,
    S: Optional[int] = None,
    rng=None,
    delta: float = 1.0,
    eps_rel: float = 0.15,
    delta_prob: float = 0.10,
) -> float:
    """
    MORI(c) = 0.5 * [norm_RMI_1(c) + norm_RMI_2(c)]

    RMI_q采用固定参考点公式 F_ref_q - E[F_q(c')]（见 compute_rmi_q），
    与c自身取值无关，因此MORI衡量的是"邻域相对理论最坏情况的期望改善量"，
    在种群内做min-max归一化（eq:mori）后仍保持[0,1]、方向不变。

    Bug2修复：生成一次邻居矩阵，同时用于 q=0 和 q=1 的 RMI 计算，
    而非各自独立采样（节省 1× 评价次数，每次 MORI 调用只消耗 S 次评价而非 2S）。

    S=None 时自动调用 adaptive_sample_size()；邻域小时精确枚举。
    """
    if rng is None:
        rng = np.random.default_rng()
    # 确定采样数 / 精确枚举
    if S is None:
        S_eff = adaptive_sample_size(
            len(combo_sets), combo_sets, r, delta, eps_rel, delta_prob
        )
    else:
        S_eff = S
    if S_eff == -1:
        nbs = _generate_neighbors_exact(c, combo_sets, r)
    else:
        nbs = _generate_neighbors_mc(c, combo_sets, r, S_eff, rng)
    # 一次生成，两目标共享
    rmi1 = compute_rmi_q(c, f1_c, combo_sets, phase_data, inst, q=0, r=r, S=S_eff, rng=rng, _nbs=nbs,
                         delta=delta, eps_rel=eps_rel, delta_prob=delta_prob)
    rmi2 = compute_rmi_q(c, f2_c, combo_sets, phase_data, inst, q=1, r=r, S=S_eff, rng=rng, _nbs=nbs,
                         delta=delta, eps_rel=eps_rel, delta_prob=delta_prob)
    n1 = rmi1 / f1_range if f1_range > 1e-9 else rmi1
    n2 = rmi2 / f2_range if f2_range > 1e-9 else rmi2
    return 0.5 * (n1 + n2)


# ── 内部辅助：快速解码（避免循环导入）─────────────────────────────

def _decode_fast(c: np.ndarray, combo_sets: List[TargetCombos], inst) -> np.ndarray:
    """轻量解码单个个体"""
    n_t = len(combo_sets)
    m, L = inst.m, inst.L
    x = np.zeros((m, n_t, L), dtype=np.int32)
    for j in range(n_t):
        h = int(c[j])
        if h == 0:
            continue
        combos = combo_sets[j].combos
        if h - 1 >= len(combos):
            continue
        combo = combos[h - 1]
        if combo.weapon < 0:
            continue
        x[combo.weapon, j, combo.radar] = combo.shots
    return x


def _decode_batch_fast(
    nbs: np.ndarray,      # (K, n_t) int32
    combo_sets: List[TargetCombos],
    inst,
) -> np.ndarray:
    """批量解码邻居矩阵 → (K, m, n_t, L)"""
    K, n_t = nbs.shape
    m, L = inst.m, inst.L
    x = np.zeros((K, m, n_t, L), dtype=np.int32)
    for j in range(n_t):
        combos = combo_sets[j].combos
        hs = nbs[:, j]  # (K,)
        for k_idx in range(K):
            h = int(hs[k_idx])
            if h == 0 or h - 1 >= len(combos):
                continue
            combo = combos[h - 1]
            if combo.weapon >= 0:
                x[k_idx, combo.weapon, j, combo.radar] = combo.shots
    return x



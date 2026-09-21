"""rcmd_ea.py — RCMD-EA主算法（Dynamic-Adaptive Robust-Constrained Multimodal Dominance EA）

核心创新：
  C1. RCMD三层支配关系（鲁棒约束多模态支配）
  C2. 双信号零参数自适应（w_g内部搜索信号 + φ_t外部态势信号）
  C3. 态势感知鲁棒后选择 + ε-并列时序一致性择优
  C4. 双档案机制（QA质量档案 + DA多样性档案）— 解决HV问题
  C5. 威胁变化感知的动态预算分配（Δ_t信号）— 动态特性利用

所有参数均继承自Paper2或由信号派生，新增用户参数=0。
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .problems.instance import Instance, PhaseData, load_instance
from .problems.encoding import (
    build_combo_set, decode, repair, hamming_distance, hamming_matrix,
    TargetCombos, combo_count,
)
from .problems.evaluator import evaluate_f1_raw, evaluate_f1_raw_batch, evaluate_f2_raw, evaluate_f2_raw_batch
from .problems.robustness import compute_rmi, compute_mori, _decode_fast, adaptive_sample_size
from .knowledge import KnowledgeBase, init_population, crossover, mutation


@dataclass
class RCMDParams:
    """所有参数继承Paper2或由信号派生，新增用户参数=0。
    explore_ratio: 两阶段比例，前explore_ratio用纯NSGA-II探索，后用RCMD深化。
    max_nfe: 总函数评价预算（含MORI采样），None表示用 N_factor*n_t*G_max（与NSGA-II等预算）。
             每次MORI调用消耗 S_fast 次评价（两目标共享邻居集，Bug2修复后）。

    采样精度参数（自适应S模式）：
      eps_rel   : Hoeffding相对精度（占fitness range），默认0.15
      delta_prob: 允许失败概率，默认0.05
    当 S > 0 时作为固定采样数 override（向后兼容）；S=-1 表示自动自适应。

    C4. 双档案参数（由比例派生，无用户参数）：
      qa_ratio: QA占种群比例，DA = 1 - qa_ratio，默认0.5
    C5. 动态预算参数（由信号派生）：
      G_min: 小变化阶段最少代数（默认60）
      G_max_dynamic: 大变化阶段最多代数（默认150，None表示固定G_max不启用动态预算）
    """
    N_factor: int = 5
    G_max: int = 100
    explore_ratio: float = 0.4
    delta_factor: float = 0.2
    epsilon_ps: float = 0.01
    r: int = 2
    S: int = -1              # -1 表示自动自适应；正整数为 override
    eps_rel: float = 0.15    # Hoeffding 相对精度
    delta_prob: float = 0.10 # 允许失败概率
    p_crossover: float = 0.8
    p_mut: float = 0.05
    p_guided: float = 0.7
    theta_c: float = 5.0
    seed: Optional[int] = None
    max_nfe: Optional[int] = None   # None → N*G_max，与NSGA-II等预算
    # C4: 双档案
    qa_ratio: float = 0.5           # QA容量占比，DA = N*(1-qa_ratio)
    # C5: 动态预算（None = 禁用，保持固定G_max）
    G_min: int = 60
    G_max_dynamic: Optional[int] = 150
    warm_start: bool = True         # 是否启用上阶段DA解热启动（C5组件之一）
    # C3消融：固定后选择权重（None = 使用φ_t态势感知；0.0~1.0 = 固定α值）
    fixed_alpha: Optional[float] = None
    # ── 修改稿新增消融开关（2026-09，审稿回复用）──
    # disable_rcmd=True 即 -C1 变体：Phase II 选择与终选均用纯 Pareto 支配替代 RCMD，
    # DA 档案不再做鲁棒可行过滤（保留两阶段/双档案结构/φ_t后选择，MORI 仅用于后选择打分）
    disable_rcmd: bool = False
    # MORI 可行性阈值的定义方式（R3-6 阈值敏感性实验）：
    #   'mean' = 种群均值（默认，与主实验一致）
    #   'p40' / 'p60' = 种群 40/60 百分位
    #   'abs0' = 绝对阈值 MORI >= 0
    mori_threshold_mode: str = 'mean'
    # budget_aware=True（修改稿v2）：Phase II 的 MORI 采样数按剩余预算自适应缩放，
    # 保证初始 MORI 与至少 G_target 代深化循环都在等额预算内执行。
    # 修复原实现的预算饥饿缺陷（S_fast 过大导致深化循环 0 代 + 初始MORI超支）
    budget_aware: bool = False
    budget_g_target: int = 10   # budget_aware 模式下希望保证的最少深化代数


@dataclass
class RCMDPhaseResult:
    t: int
    best_c: np.ndarray          # 后选择方案编码
    best_f1_res: float          # 残余威胁（内部最小化形式）
    best_f2: float              # 弹药消耗
    best_rmi: float             # 后选择方案RMI（F1维度）
    pareto_objs: np.ndarray     # 非支配解集目标向量 (K, 2)：来自QA档案
    pareto_codes: np.ndarray    # 非支配解编码 (K, n_t)：来自QA档案（用于HV计算）
    da_codes: Optional[np.ndarray] = None  # DA档案解编码（用于后选择和HD计算）
    da_objs: Optional[np.ndarray] = None   # DA档案解目标向量
    hd: float = 0.0             # 种群汉明多样性（基于DA档案）
    fitness_history: List[float] = field(default_factory=list)
    w_g_history: List[float] = field(default_factory=list)
    delta_g_history: List[float] = field(default_factory=list)


@dataclass
class RCMDResult:
    instance_name: str
    phase_results: List[RCMDPhaseResult] = field(default_factory=list)
    total_f1_res: float = 0.0


# ── 双信号计算 ────────────────────────────────────────────────────────

def compute_wg(pop: np.ndarray, objs: np.ndarray, eps: float = 1e-12) -> float:
    """
    内部搜索信号 w_g = c_X / (c_X + c_F)
    c_X = CV(汉明距离分布), c_F = CV(目标空间距离分布)
    全部用 numpy 向量化计算，消除 O(N²) Python 循环。
    """
    N = len(pop)
    if N < 2:
        return 0.5

    # 上三角汉明距离（向量化）
    diff = (pop[:, None, :] != pop[None, :, :]).sum(axis=-1)   # (N, N)
    idx = np.triu_indices(N, k=1)
    D_ham = diff[idx].astype(float)

    # 上三角目标空间欧氏距离（向量化）
    d_obj = objs[:, None, :] - objs[None, :, :]                # (N, N, 2)
    D_obj = np.sqrt((d_obj ** 2).sum(axis=-1))[idx]

    mu_X, sig_X = D_ham.mean() + eps, D_ham.std()
    mu_F, sig_F = D_obj.mean() + eps, D_obj.std()
    c_X = sig_X / mu_X
    c_F = sig_F / mu_F
    w_g = c_X / (c_X + c_F + eps)
    return float(np.clip(w_g, 0.1, 0.9))


def compute_phi(phase_data: PhaseData) -> float:
    """
    外部态势信号φ_t：归一化Herfindahl威胁集中度（零参数）
      φ = (n * Σ th̃² - 1) / (n - 1)，th̃_j = Th_j / ΣTh
    φ=0：威胁完全均匀；φ=1：完全集中于单一目标
    """
    n_t = phase_data.n_t
    if n_t <= 1:
        return 1.0
    ths = np.array([phase_data.targets[j].threat for j in range(n_t)])
    total = ths.sum()
    if total < 1e-12:
        return 0.0
    th_norm = ths / total
    hhi = float(np.sum(th_norm ** 2))
    phi = (n_t * hhi - 1.0) / (n_t - 1.0)
    return float(np.clip(phi, 0.0, 1.0))


def compute_delta_g(D_ham_upper: np.ndarray, w_g: float, n_t: int) -> float:
    """
    模态半径 δ_g：基于PRD设计的自适应版本
    δ_g = ceil(0.2 * n_t) * 0.5^(1+w_g)
    w_g大（决策空间结构信息丰富）→ δ_g小（保留更多模态）
    下限 ≥ 1
    注：使用0.2*n_t而非median，因为WTA汉明距离分布高度右偏
       （大多数随机解彼此差异很大，median偏大不适合作模态基准）
    """
    base = max(1.0, 0.2 * n_t)
    delta = base * (0.5 ** (1.0 + w_g))
    return max(1.0, delta)


# ── RCMD 支配关系与非支配排序 ────────────────────────────────────────

def _mori_threshold(mori_vals: np.ndarray, mode: str = 'mean') -> float:
    """MORI 可行性阈值（R3-6 阈值敏感性：mean / p40 / p60 / abs0）"""
    if mode == 'p40':
        return float(np.percentile(mori_vals, 40))
    if mode == 'p60':
        return float(np.percentile(mori_vals, 60))
    if mode == 'abs0':
        return 0.0
    return float(mori_vals.mean())

def rcmd_dominates(
    i: int, j: int,
    f1_res: np.ndarray, f2: np.ndarray,
    mori: np.ndarray, mori_threshold: float,
    D_ham: np.ndarray, delta_g: float,
) -> bool:
    """
    c_i RCMD支配 c_j 当且仅当满足R1/R2/R3之一
    R1: i可行且j不可行（g(c)=avg_MORI-MORI(c)<=0 ↔ MORI(c)>=avg_MORI为可行）
    R2: 均可行，同模态(d_TC<=δ_g)，i Pareto支配j
    R3: 均不可行，MORI(i) > MORI(j)
    注意：MORI越大=鲁棒性越强，g(c)<=0 ↔ MORI(c)>=threshold
    """
    feasible_i = mori[i] >= mori_threshold
    feasible_j = mori[j] >= mori_threshold

    # R1
    if feasible_i and not feasible_j:
        return True
    # R3
    if not feasible_i and not feasible_j:
        return mori[i] > mori[j]
    # R2
    if feasible_i and feasible_j:
        if D_ham[i, j] <= delta_g:
            # 同模态：i Pareto支配j（均越小越好）
            return ((f1_res[i] <= f1_res[j] and f2[i] <= f2[j]) and
                    (f1_res[i] < f1_res[j] or f2[i] < f2[j]))
    # 跨模态：不建立支配（多模态保持）
    return False


def rcmd_nondominated_sort(
    f1_res: np.ndarray, f2: np.ndarray,
    mori: np.ndarray, mori_threshold: float,
    D_ham: np.ndarray, delta_g: float,
) -> List[List[int]]:
    """
    RCMD非支配排序（返回分层列表，L[0]为第一前沿）
    无环性由RCMD理论性质保证（见docs/算法设计_RCMD-EA.md §3.4）
    """
    N = len(f1_res)
    dom_count = np.zeros(N, dtype=int)     # 被支配次数
    dominated_by_me = [[] for _ in range(N)]  # 我支配的集合

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if rcmd_dominates(i, j, f1_res, f2, mori, mori_threshold, D_ham, delta_g):
                dominated_by_me[i].append(j)
                dom_count[j] += 1

    fronts: List[List[int]] = []
    current_front = [i for i in range(N) if dom_count[i] == 0]
    while current_front:
        fronts.append(current_front)
        next_front = []
        for i in current_front:
            for j in dominated_by_me[i]:
                dom_count[j] -= 1
                if dom_count[j] == 0:
                    next_front.append(j)
        current_front = next_front

    return fronts


# ── 双空间拥挤度 ────────────────────────────────────────────────────

def dual_space_crowding(
    indices: List[int],
    f1_res: np.ndarray, f2: np.ndarray,
    D_ham: np.ndarray, w_g: float,
) -> np.ndarray:
    """
    CD = (1-w_g)·ĈD_obj + w_g·ĈD_ham
    返回前沿内各个体的CD值（越大越好，优先保留多样性高的）
    """
    idx = np.array(indices)
    n = len(idx)
    if n <= 2:
        return np.full(n, np.inf)

    # 目标空间拥挤距离（标准NSGA-II方式）
    cd_obj = np.zeros(n)
    for m_dim, vals in enumerate([f1_res[idx], f2[idx]]):
        order = np.argsort(vals)
        cd_obj[order[0]] = np.inf
        cd_obj[order[-1]] = np.inf
        r = vals[order[-1]] - vals[order[0]]
        if r < 1e-12:
            continue
        for k in range(1, n - 1):
            cd_obj[order[k]] += (vals[order[k+1]] - vals[order[k-1]]) / r

    # 决策空间拥挤度（maximin：最近邻汉明距离）
    cd_ham = np.zeros(n)
    for k in range(n):
        i = idx[k]
        others = [idx[k2] for k2 in range(n) if k2 != k]
        if others:
            cd_ham[k] = min(D_ham[i, j] for j in others)

    # 归一化并加权融合
    cd_obj_n = cd_obj / (cd_obj[np.isfinite(cd_obj)].max() + 1e-12) if np.isfinite(cd_obj).any() else cd_obj
    cd_ham_n = cd_ham / (cd_ham.max() + 1e-12) if cd_ham.max() > 0 else cd_ham
    cd = (1.0 - w_g) * cd_obj_n + w_g * cd_ham_n
    # inf传播（边界点保持inf优先级）
    cd[np.isinf(cd_obj)] = np.inf
    return cd


# ── 后选择 ────────────────────────────────────────────────────────────

def preference_selection(
    pareto_codes: np.ndarray,
    f1_res: np.ndarray, f2: np.ndarray,
    mori: np.ndarray,
    phi_t: float,
    epsilon_ps: float,
    prev_c: Optional[np.ndarray],
) -> Tuple[np.ndarray, float, float, float]:
    """
    态势感知鲁棒后选择（零参数）
    α_t = w_{1,t} = φ_t（直接恒等映射，无sigmoid参数）

    步骤：
    1. 计算Score = φ_t[φ_t·f̂1 + (1-φ_t)·f̂2] + (1-φ_t)·(1-MORÎ)
    2. 候选集 = {c: Score <= (1+ε)·min_Score}
    3. 候选集内取 d_TC(c, prev_c) 最小者（时序一致性）
    4. 仍并列取 f1_res 更小者

    返回: (best_c, best_f1_res, best_f2, best_rmi)
    """
    N = len(pareto_codes)
    if N == 0:
        return pareto_codes[0], f1_res[0], f2[0], mori[0]

    # 归一化（min-max，统一越小越好方向）
    def _norm(v):
        lo, hi = v.min(), v.max()
        if hi - lo < 1e-12:
            return np.zeros_like(v, dtype=float)
        return (v - lo) / (hi - lo)

    f1_n = _norm(f1_res)   # f1_res越小越好，归一化后越小越好
    f2_n = _norm(f2)
    mori_n = _norm(-mori)  # MORI越大鲁棒性越强，取负后越小越好（但归一化后一致）

    alpha = phi_t
    score = alpha * (phi_t * f1_n + (1.0 - phi_t) * f2_n) + (1.0 - alpha) * mori_n

    s_min = score.min()
    cand_mask = score <= s_min * (1.0 + epsilon_ps) + 1e-9
    cand_idx = np.where(cand_mask)[0]

    if prev_c is not None and len(cand_idx) > 1 and len(prev_c) == len(pareto_codes[0]):
        # 时序一致性择优
        dists = np.array([hamming_distance(pareto_codes[k], prev_c) for k in cand_idx])
        min_dist = dists.min()
        cand_idx = cand_idx[dists == min_dist]

    # 仍并列：取f1_res最小者
    best_local = cand_idx[np.argmin(f1_res[cand_idx])]
    return (pareto_codes[best_local], float(f1_res[best_local]),
            float(f2[best_local]), float(mori[best_local]))


# ── C4. 双档案机制（QA + DA）─────────────────────────────────────────

def dual_archive_select(
    R_pop: np.ndarray,
    f1_R: np.ndarray,
    f2_R: np.ndarray,
    mori_R: np.ndarray,
    D_ham_R: np.ndarray,
    delta_g: float,
    w_g: float,
    N: int,
    qa_ratio: float = 0.5,
    mori_threshold_mode: str = 'mean',
    disable_rcmd: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    C4. 双档案环境选择（Phase 2 替代原单档案选择）

    QA（质量档案，N_qa = N*qa_ratio）：
      - 纯 Pareto 非支配排序 + 目标空间拥挤距离
      - 负责 HV 竞争力，向 NSGA-II 对齐

    DA（多样性档案，N_da = N - N_qa）：
      - 从 QA 第一前沿中取鲁棒可行解，按汉明 maximin 距离选取
      - 不足时从 R 中按 MORI 降序补充
      - 负责 HD/PSR 优势，保持多模态多样性
      - disable_rcmd=True（-C1 变体）时不做鲁棒可行过滤：
        候选 = QA 第一前沿全体，补充按索引序（完全移除 MORI 对搜索的影响）

    返回:
      qa_pop, qa_f1, qa_f2,   — QA档案（用于 HV 计算和论文对比）
      da_pop, da_f1, da_f2    — DA档案（用于后选择和 HD 计算）
    """
    from .baselines.nsga2 import fast_nondom_sort, crowding_distance

    N_R = len(R_pop)
    N_qa = max(2, int(round(N * qa_ratio)))
    N_da = max(2, N - N_qa)
    mori_threshold = _mori_threshold(mori_R, mori_threshold_mode)

    # ── QA：纯 Pareto 前沿 + 目标拥挤距离 ──
    fronts_ns = fast_nondom_sort(f1_R, f2_R)
    qa_idx: List[int] = []
    for front in fronts_ns:
        if len(qa_idx) + len(front) <= N_qa:
            qa_idx.extend(front)
        else:
            need = N_qa - len(qa_idx)
            cd = crowding_distance(front, f1_R, f2_R)
            order = np.argsort(-cd)
            qa_idx.extend([front[k] for k in order[:need]])
            break
    # 保护：至少保留 N_qa 个解（完全收敛时 fronts[0] 可能很小）
    if len(qa_idx) < N_qa:
        existing = set(qa_idx)
        for i in range(N_R):
            if i not in existing:
                qa_idx.append(i)
            if len(qa_idx) >= N_qa:
                break
    qa_idx = qa_idx[:N_qa]

    # ── DA：从 QA 第一前沿（的鲁棒可行子集）中按汉明 maximin 选取 ──
    qa_front0 = fronts_ns[0] if fronts_ns else list(range(N_R))
    if disable_rcmd:
        # -C1 变体：无鲁棒可行过滤，QA 第一前沿全体为候选；补充按索引序
        robust_cands = list(qa_front0)
        if len(robust_cands) < N_da:
            chosen = set(robust_cands)
            extra_pool = [i for i in range(N_R) if i not in chosen]
            robust_cands = robust_cands + extra_pool
    else:
        # 鲁棒可行候选：QA第一前沿中满足 MORI >= threshold 的解
        robust_cands = [i for i in qa_front0 if mori_R[i] >= mori_threshold]
        if len(robust_cands) < N_da:
            # 补充：从所有 R 中按 MORI 降序填充（不限于 QA 前沿）
            extra_pool = sorted(
                [i for i in range(N_R) if i not in robust_cands],
                key=lambda i: -mori_R[i],
            )
            robust_cands = robust_cands + extra_pool

    # Hamming maximin 贪心选取 N_da 个解（最大化最小汉明距离，向量化加速）
    if len(robust_cands) <= N_da:
        da_idx = list(robust_cands[:N_da])
    else:
        cands = np.array(robust_cands)
        # 提取汉明子矩阵
        sub_D = D_ham_R[np.ix_(cands, cands)]
        # 贪心：每次选与已选集合最小距离最大的候选
        selected = [0]   # 索引到 cands 的下标
        unsel = list(range(1, len(cands)))
        min_dist_to_sel = sub_D[np.array(unsel), 0].copy()  # 初始：到第0个的距离

        while len(selected) < N_da and unsel:
            best_pos = int(np.argmax(min_dist_to_sel))
            best_unsel_idx = unsel[best_pos]
            selected.append(best_unsel_idx)
            unsel.pop(best_pos)
            min_dist_to_sel = np.delete(min_dist_to_sel, best_pos)
            if unsel:
                # 更新 min_dist：与新加入解的距离 vs 原来的 min
                new_dists = sub_D[np.array(unsel), best_unsel_idx]
                min_dist_to_sel = np.minimum(min_dist_to_sel, new_dists)

        da_idx = [int(cands[k]) for k in selected]

    da_idx = da_idx[:N_da]

    qa_arr = np.array(qa_idx)
    da_arr = np.array(da_idx)
    return (
        R_pop[qa_arr], f1_R[qa_arr], f2_R[qa_arr],
        R_pop[da_arr], f1_R[da_arr], f2_R[da_arr],
    )


# ── C5. 威胁变化感知：Δ_t 计算 ──────────────────────────────────────

def compute_threat_change(
    prev_phase: PhaseData,
    curr_phase: PhaseData,
) -> float:
    """
    C5. 阶段间威胁变化量 Δ_t ∈ [0, 1]
      Δ_t = 1 - Jaccard(T_{t-1}, T_t) × max(0, corr(Th_{t-1}, Th_t))
    Δ_t → 0：威胁集几乎不变（热启动为主）
    Δ_t → 1：全新威胁集（重新探索）
    """
    prev_ids = {t.id for t in prev_phase.targets}
    curr_ids = {t.id for t in curr_phase.targets}
    union = prev_ids | curr_ids
    if not union:
        return 0.0
    intersection = prev_ids & curr_ids
    jaccard = len(intersection) / len(union)

    if len(intersection) >= 2:
        prev_th = {t.id: t.threat for t in prev_phase.targets}
        curr_th = {t.id: t.threat for t in curr_phase.targets}
        common = list(intersection)
        p_vals = np.array([prev_th[i] for i in common], dtype=float)
        c_vals = np.array([curr_th[i] for i in common], dtype=float)
        # Pearson 相关系数（标准差为0时退化为0）
        if p_vals.std() < 1e-12 or c_vals.std() < 1e-12:
            threat_corr = 1.0 if np.allclose(p_vals, c_vals) else 0.0
        else:
            threat_corr = float(np.corrcoef(p_vals, c_vals)[0, 1])
        threat_corr = max(0.0, threat_corr)
    else:
        threat_corr = 0.0

    delta_t = 1.0 - jaccard * threat_corr
    return float(np.clip(delta_t, 0.0, 1.0))


# ── 主求解器：单决策周期 ─────────────────────────────────────────────

def solve_single_phase_rcmd(
    phase_data: PhaseData,
    inst: Instance,
    kb: KnowledgeBase,
    params: RCMDParams,
    rng: np.random.Generator,
    prev_c: Optional[np.ndarray] = None,
    warm_start_codes: Optional[np.ndarray] = None,  # C5: 上阶段DA解（热启动）
    verbose: bool = False,
) -> RCMDPhaseResult:
    n_t = phase_data.n_t
    N = params.N_factor * n_t
    phi_t = compute_phi(phase_data)

    # 约束预处理
    combo_sets = build_combo_set(phase_data, inst)

    # 知识引导初始化（含C5热启动）
    pop = init_population(N, combo_sets, kb, rng)
    # C5: 热启动 —— 将上阶段 DA 解注入前 N_warm 个位置
    if warm_start_codes is not None and len(warm_start_codes) > 0:
        N_warm = min(len(warm_start_codes), N // 3)
        for k in range(N_warm):
            # 跨阶段维度不匹配，只能作为启发：随机初始化后部分继承
            # 简化：只在 n_t 匹配时注入，否则跳过
            if len(warm_start_codes[k]) == n_t:
                try:
                    repaired = repair(warm_start_codes[k], combo_sets, phase_data, inst)
                    pop[k] = repaired
                except:
                    # repair 失败（编码不兼容），跳过
                    pass
    pop = np.array([repair(pop[i], combo_sets, phase_data, inst) for i in range(N)])

    # 批量评估（F1_res, F2）
    x_batch = np.stack([decode(pop[i], combo_sets, inst) for i in range(N)])
    f1_arr = evaluate_f1_raw_batch(x_batch, phase_data, inst)
    f2_arr = evaluate_f2_raw_batch(x_batch)

    history_f1 = [float(f1_arr.min())]
    history_wg = []
    history_dg = []

    # 两阶段代数划分：前 explore_ratio 用纯NSGA-II探索，后用RCMD深化
    G_explore = max(1, int(round(params.G_max * params.explore_ratio)))
    G_refine  = params.G_max - G_explore

    # ── 阶段一：宽泛探索（纯NSGA-II，不计算MORI，目标铺满Pareto前沿）──────
    from .baselines.nsga2 import fast_nondom_sort, crowding_distance

    for g in range(G_explore):
        f1_range = max(f1_arr.max() - f1_arr.min(), 1e-9)
        f2_range = max(f2_arr.max() - f2_arr.min(), 1e-9)
        objs_n = np.column_stack([
            (f1_arr - f1_arr.min()) / f1_range,
            (f2_arr - f2_arr.min()) / f2_range,
        ])
        w_g = compute_wg(pop, objs_n)
        history_wg.append(w_g)
        D_ham = hamming_matrix(pop)
        D_ham_upper = D_ham[np.triu_indices(N, k=1)].astype(float)
        delta_g = compute_delta_g(D_ham_upper, w_g, n_t)
        history_dg.append(delta_g)

        Q = crossover(pop, rng, params.p_crossover)
        Q = mutation(Q, combo_sets, kb, rng, params.p_guided, params.p_mut)
        Q = np.array([repair(Q[i], combo_sets, phase_data, inst) for i in range(len(Q))])
        x_q = np.stack([decode(Q[i], combo_sets, inst) for i in range(len(Q))])
        f1_q = evaluate_f1_raw_batch(x_q, phase_data, inst)
        f2_q = evaluate_f2_raw_batch(x_q)

        R = np.vstack([pop, Q])
        f1_R = np.concatenate([f1_arr, f1_q])
        f2_R = np.concatenate([f2_arr, f2_q])

        # 纯NSGA-II选择（无MORI，无鲁棒约束）
        fronts_ns = fast_nondom_sort(f1_R, f2_R)
        new_idx: List[int] = []
        for front in fronts_ns:
            if len(new_idx) + len(front) <= N:
                new_idx.extend(front)
            else:
                need = N - len(new_idx)
                cd = crowding_distance(front, f1_R, f2_R)
                order = np.argsort(-cd)
                new_idx.extend([front[k] for k in order[:need]])
                break

        sel = np.array(new_idx[:N])
        pop = R[sel]; f1_arr = f1_R[sel]; f2_arr = f2_R[sel]
        history_f1.append(float(f1_arr.min()))

        if verbose and (g + 1) % 20 == 0:
            print(f"    [探索 gen {g+1}/{G_explore}] f1_min={f1_arr.min():.4f} w_g={w_g:.3f}")

    # ── 阶段二：鲁棒深化（RCMD，与原设计一致）────────────────────────────
    delta_niche = max(1.0, params.delta_factor * n_t)  # niche半径，用于跨峰修正

    # 自适应采样数计算（一次，全阶段复用）
    if params.S > 0:
        S_full = params.S      # 用户 override
    else:
        S_full_or_exact = adaptive_sample_size(
            n_t, combo_sets, params.r, delta_niche,
            params.eps_rel, params.delta_prob,
        )
        S_full = S_full_or_exact if S_full_or_exact > 0 else 30  # exact路径兜底

    S_fast = max(10, S_full // 3)  # Phase 2 快速代理：精度的 √3 倍误差，速度优先
    # 每代 MORI 消耗：N（offspring评价）+ N×S_fast（offspring MORI，两目标共享邻居）
    # 此处 S_fast 每次 compute_mori 消耗 S_fast 次评价（Bug2修复后共享邻居）
    if params.budget_aware:
        # v2：按剩余预算自适应缩放 S_fast，保证初始 MORI + ≥G_target 代深化在预算内
        budget_total_pre = params.max_nfe if params.max_nfe is not None else N * params.G_max
        B_refine = budget_total_pre - N * G_explore
        S_aff = int(B_refine // (N * (1 + params.budget_g_target))) - 1 if N > 0 else 10
        S_fast = int(max(10, min(S_fast, S_aff)))
    nfe_per_refine_gen = N + (0 if params.disable_rcmd else N * S_fast)
    # -C1 变体（disable_rcmd）：搜索不使用 MORI，省下的预算用于更多 Pareto 深化代数
    #（总 NFE 预算与完整版严格一致，等预算公平比较）
    budget_used = N * G_explore           # Phase1消耗
    budget_total = params.max_nfe if params.max_nfe is not None else N * params.G_max

    # 初始化 pop 的 MORI 缓存（第一次进 Phase 2 时计算；-C1 变体跳过——搜索不用）
    if params.disable_rcmd:
        mori_pop = np.zeros(N)
    else:
        f1_range0 = max(f1_arr.max() - f1_arr.min(), 1e-9)
        f2_range0 = max(f2_arr.max() - f2_arr.min(), 1e-9)
        mori_pop = np.array([
            compute_mori(
                pop[k], float(f1_arr[k]), float(f2_arr[k]),
                combo_sets, phase_data, inst,
                f1_range0, f2_range0,
                r=params.r, S=S_fast, rng=rng,
                delta=delta_niche, eps_rel=params.eps_rel, delta_prob=params.delta_prob,
            )
            for k in range(N)
        ])
        budget_used += N * S_fast   # 初始 pop MORI 计入预算

    for g in range(params.G_max):   # 循环上限用 G_max，内部按预算退出
        if budget_used + nfe_per_refine_gen > budget_total:
            break

        f1_range = max(f1_arr.max() - f1_arr.min(), 1e-9)
        f2_range = max(f2_arr.max() - f2_arr.min(), 1e-9)
        objs_n = np.column_stack([
            (f1_arr - f1_arr.min()) / f1_range,
            (f2_arr - f2_arr.min()) / f2_range,
        ])
        N_cur = len(pop)   # 双档案去重后实际种群大小
        w_g = compute_wg(pop, objs_n)
        history_wg.append(w_g)
        D_ham = hamming_matrix(pop)
        D_ham_upper = D_ham[np.triu_indices(N_cur, k=1)].astype(float)
        delta_g = compute_delta_g(D_ham_upper, w_g, n_t)
        history_dg.append(delta_g)

        Q = crossover(pop, rng, params.p_crossover)
        if len(Q) == 0:
            # pop 过小时 crossover 可能产生空集（奇数种群 N_cur=1），直接用 pop 副本
            Q = pop.copy()
        Q = mutation(Q, combo_sets, kb, rng, params.p_guided, params.p_mut)
        if len(Q) == 0:
            Q = pop.copy()
        Q = np.array([repair(Q[i], combo_sets, phase_data, inst) for i in range(len(Q))])
        if len(Q) == 0:
            # 保护：跳过本代
            continue
        x_q = np.stack([decode(Q[i], combo_sets, inst) for i in range(len(Q))])
        f1_q = evaluate_f1_raw_batch(x_q, phase_data, inst)
        f2_q = evaluate_f2_raw_batch(x_q)
        budget_used += len(Q)   # offspring 实际数量

        # Bug3修复：只对 offspring Q 计算 MORI（pop 的 MORI 从上一代缓存）
        # -C1 变体：跳过（搜索不使用 MORI）
        f1_R_range = max(max(f1_arr.max(), f1_q.max()) - min(f1_arr.min(), f1_q.min()), 1e-9)
        f2_R_range = max(max(f2_arr.max(), f2_q.max()) - min(f2_arr.min(), f2_q.min()), 1e-9)
        if params.disable_rcmd:
            mori_q = np.zeros(len(Q))
        else:
            mori_q = np.array([
                compute_mori(
                    Q[k], float(f1_q[k]), float(f2_q[k]),
                    combo_sets, phase_data, inst,
                    f1_R_range, f2_R_range,
                    r=params.r, S=S_fast, rng=rng,
                    delta=delta_niche, eps_rel=params.eps_rel, delta_prob=params.delta_prob,
                )
                for k in range(len(Q))
            ])
            budget_used += len(Q) * S_fast   # offspring MORI 计入预算

        R = np.vstack([pop, Q])
        f1_R = np.concatenate([f1_arr, f1_q])
        f2_R = np.concatenate([f2_arr, f2_q])
        mori_R = np.concatenate([mori_pop, mori_q])

        D_ham_R = hamming_matrix(R)

        # C4: 双档案选择（替代原 RCMD 单档案）
        qa_pop, qa_f1, qa_f2, da_pop, da_f1, da_f2 = dual_archive_select(
            R, f1_R, f2_R, mori_R, D_ham_R, delta_g, w_g, N, params.qa_ratio,
            mori_threshold_mode=params.mori_threshold_mode,
            disable_rcmd=params.disable_rcmd,
        )

        # 合并去重（优先保留 DA，QA 补充）
        combined_codes = np.vstack([da_pop, qa_pop])
        combined_f1 = np.concatenate([da_f1, qa_f1])
        combined_f2 = np.concatenate([da_f2, qa_f2])
        # 去重：保留编码唯一的前 N 个
        unique_mask = np.ones(len(combined_codes), dtype=bool)
        seen = set()
        for idx in range(len(combined_codes)):
            key = tuple(combined_codes[idx])
            if key in seen:
                unique_mask[idx] = False
            else:
                seen.add(key)
        unique_idx = np.where(unique_mask)[0][:N]
        pop = combined_codes[unique_idx]
        f1_arr = combined_f1[unique_idx]
        f2_arr = combined_f2[unique_idx]

        # MORI 缓存更新：用字典映射加速（O(N) 而非 O(N²)）
        mori_dict = {tuple(R[j]): mori_R[j] for j in range(len(R))}
        mori_pop = np.array([mori_dict.get(tuple(pop[i]), 0.0) for i in range(len(pop))])

        history_f1.append(float(f1_arr.min()))

        if verbose and (g + 1) % 20 == 0:
            print(f"    [深化 gen {g+1}] nfe={budget_used}/{budget_total} f1_min={f1_arr.min():.4f} w_g={w_g:.3f} delta_g={delta_g:.1f}")

    # ── 后选择 ──────────────────────────────────────────────────────
    # C4: 最终非支配排序使用 QA 档案（HV计算基准）和 DA 档案（后选择基准）
    # 从最终种群重建 QA/DA 以确保一致性
    D_ham_final = hamming_matrix(pop)
    final_mori_threshold = _mori_threshold(mori_pop, params.mori_threshold_mode)
    w_g_final = history_wg[-1] if history_wg else 0.5
    delta_final = compute_delta_g(
        D_ham_final[np.triu_indices(len(pop), k=1)].astype(float), w_g_final, n_t
    )

    # QA：纯 Pareto 非支配排序，用于 HV
    from .baselines.nsga2 import fast_nondom_sort
    fronts_qa = fast_nondom_sort(f1_arr, f2_arr)
    qa_pareto_idx = np.array(fronts_qa[0]) if fronts_qa else np.arange(len(pop))

    # DA：鲁棒可行解中的 RCMD 非支配解，用于后选择和 HD
    # -C1 变体：用纯 Pareto 非支配排序替代 RCMD 排序
    if params.disable_rcmd:
        fronts_rcmd = fast_nondom_sort(f1_arr, f2_arr)
    else:
        fronts_rcmd = rcmd_nondominated_sort(
            f1_arr, f2_arr, mori_pop, final_mori_threshold, D_ham_final, delta_final
        )
    da_pareto_idx = np.array(fronts_rcmd[0]) if fronts_rcmd else np.arange(len(pop))

    # QA 档案 Pareto 集（HV 计算用）
    qa_pareto_codes = pop[qa_pareto_idx]
    qa_pareto_f1    = f1_arr[qa_pareto_idx]
    qa_pareto_f2    = f2_arr[qa_pareto_idx]

    # DA 档案 Pareto 集（后选择用）
    da_pareto_codes = pop[da_pareto_idx]
    da_pareto_f1    = f1_arr[da_pareto_idx]
    da_pareto_f2    = f2_arr[da_pareto_idx]

    # 后选择 RMI：对 DA Pareto 集用 S_full 精算
    pareto_rmi = np.array([
        compute_rmi(
            da_pareto_codes[k], float(da_pareto_f1[k]),
            combo_sets, phase_data, inst,
            r=params.r, S=S_full, rng=rng,
            delta=delta_niche, eps_rel=params.eps_rel, delta_prob=params.delta_prob,
        )
        for k in range(len(da_pareto_idx))
    ])

    best_c, best_f1, best_f2, best_rmi_val = preference_selection(
        da_pareto_codes, da_pareto_f1, da_pareto_f2, pareto_rmi,
        phi_t if params.fixed_alpha is None else params.fixed_alpha,
        params.epsilon_ps, prev_c,
    )

    # 汉明多样性（基于DA档案Pareto集，体现决策空间多样性）
    from .metrics import hamming_diversity
    hd = hamming_diversity(da_pareto_codes) if len(da_pareto_codes) > 1 else 0.0

    return RCMDPhaseResult(
        t=0,
        best_c=best_c,
        best_f1_res=best_f1,
        best_f2=best_f2,
        best_rmi=best_rmi_val,
        pareto_objs=np.column_stack([qa_pareto_f1, qa_pareto_f2]),
        pareto_codes=qa_pareto_codes,
        da_codes=da_pareto_codes,
        da_objs=np.column_stack([da_pareto_f1, da_pareto_f2]),
        hd=hd,
        fitness_history=history_f1,
        w_g_history=history_wg,
        delta_g_history=history_dg,
    )


# ── 完整多阶段运行入口 ───────────────────────────────────────────────

def run_rcmd_ea(
    inst: Instance,
    params: Optional[RCMDParams] = None,
    verbose: bool = False,
) -> RCMDResult:
    if params is None:
        params = RCMDParams()
    rng = np.random.default_rng(params.seed)
    kb = KnowledgeBase(theta_c=params.theta_c)
    result = RCMDResult(instance_name=inst.name)
    prev_c = None
    total_f1 = 0.0
    prev_da_codes: Optional[np.ndarray] = None   # C5: 上阶段 DA 档案（热启动）
    prev_phase_data: Optional[PhaseData] = None  # C5: 上阶段威胁数据

    for t in range(inst.T_total):
        phase_data = inst.phases[t]

        # C5: 计算威胁变化量，动态调整 G_max
        if prev_phase_data is not None and params.G_max_dynamic is not None:
            delta_t = compute_threat_change(prev_phase_data, phase_data)
            g_max_t = int(round(
                params.G_min + (params.G_max_dynamic - params.G_min) * delta_t
            ))
            # 临时覆盖 params.G_max（不修改原始 params）
            import copy
            params_t = copy.copy(params)
            params_t.G_max = g_max_t
        else:
            delta_t = 1.0  # 第一阶段视为全新
            params_t = params

        if verbose:
            print(f"  Phase {t+1}/{inst.T_total}, n_t={phase_data.n_t}"
                  + (f", Δ_t={delta_t:.3f}, G_max={params_t.G_max}" if params.G_max_dynamic else ""))

        pr = solve_single_phase_rcmd(
            phase_data, inst, kb, params_t, rng,
            prev_c=prev_c,
            warm_start_codes=prev_da_codes,  # C5: 热启动
            verbose=verbose,
        )
        pr.t = t

        # 知识库更新
        combo_sets = build_combo_set(phase_data, inst)
        kb.update(pr.best_c, combo_sets)
        prev_c = pr.best_c
        prev_da_codes = pr.da_codes   # C5: 保存 DA 档案供下阶段热启动（受 warm_start 参数控制）
        if not params.G_max_dynamic or not params.warm_start:
            prev_da_codes = None   # 禁用热启动：不传下阶段
        prev_phase_data = phase_data  # C5: 保存阶段数据供下阶段计算 Δ_t

        result.phase_results.append(pr)
        total_f1 += pr.best_f1_res

    result.total_f1_res = total_f1
    return result

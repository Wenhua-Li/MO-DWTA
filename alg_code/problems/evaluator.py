"""evaluator.py — F1/F2目标函数、MRFDQI、毁伤概率"""
import numpy as np
from .instance import Instance, PhaseData

# ── 物理模型 ───────────────────────────────────────────────────────────

def compute_kill_prob(d_ij: float, R_max_i: float, rcs: float) -> float:
    """
    单武器-目标对毁伤概率（精确复现MATLAB compute_kill_prob.m）
      P_ij = min(0.99, 0.7*(1 - d/R_max)*min(1, RCS/1.0) + 0.25)
      若 d > R_max，返回0
    """
    if d_ij > R_max_i or R_max_i == 0:
        return 0.0
    p = 0.7 * (1.0 - d_ij / R_max_i) * min(1.0, rcs / 1.0) + 0.25
    return min(0.99, p)


def compute_mrfdqi(
    target_x: float, target_y: float, rcs: float,
    radar_x: np.ndarray, radar_y: np.ndarray,
    lam: np.ndarray, R_radar: np.ndarray, p_th: float
) -> float:
    """
    多雷达融合探测质量指数（复现MATLAB compute_mrfdqi.m）
      eta = max(0.1, 1 - prod(1 - p_kj for k in B_jt))
      若B_jt为空，返回0
    """
    survival = 1.0
    any_detect = False
    sigma = max(rcs, 1e-6)
    for k in range(len(radar_x)):
        d_kj = np.hypot(radar_x[k] - target_x, radar_y[k] - target_y)
        if d_kj > R_radar[k]:
            continue
        p_kj = max(0.05, float(np.exp(-lam[k] * d_kj**4 / sigma)))
        if p_kj >= p_th:
            survival *= (1.0 - p_kj)
            any_detect = True
    if not any_detect:
        return 0.0
    return max(0.1, 1.0 - survival)


def precompute_phase(phase_data: PhaseData, inst: Instance) -> None:
    """
    如果 .mat 文件中的 P_ij / eta_j 未预计算（全零或形状不对），在此补算。
    正常情况下 .mat 已有预计算值，此函数相当于验证/fallback。
    """
    m, n_t = inst.m, phase_data.n_t
    targets = phase_data.targets

    # 若 P_ij 形状正确则直接用 mat 中的值
    if phase_data.P_ij.shape == (m, n_t):
        return

    P_ij = np.zeros((m, n_t))
    eta_j = np.zeros(n_t)
    d_ij = np.zeros((m, n_t))

    for j, tgt in enumerate(targets):
        eta_j[j] = compute_mrfdqi(
            tgt.x, tgt.y, tgt.rcs,
            inst.radar_x, inst.radar_y,
            inst.lam, inst.R_radar, inst.p_th
        )
        for i in range(m):
            d = np.hypot(inst.weapon_x[i] - tgt.x, inst.weapon_y[i] - tgt.y)
            d_ij[i, j] = d
            P_ij[i, j] = compute_kill_prob(d, inst.R_max[i], tgt.rcs)

    phase_data.P_ij = P_ij
    phase_data.eta_j = eta_j
    phase_data.d_ij = d_ij


# ── 目标函数评估 ───────────────────────────────────────────────────────

PENALTY_WEIGHT = 1000.0


def evaluate_f1_raw(x_ijkt: np.ndarray, phase_data: PhaseData, inst: Instance) -> float:
    """
    计算F1残余威胁（内部最小化形式，与Paper2完全一致）：
      F1_res = Σ_j eta_j * Th_j * Π_{i,k} (1 - P_ij)^x_ijkt[i,j,k]
    加C1/C9惩罚项（若存在违约）。

    x_ijkt: (m, n_t, L) int数组
    返回：float（残余威胁值+惩罚，越小越好）
    """
    m, n_t, L = x_ijkt.shape
    targets = phase_data.targets
    P_ij = phase_data.P_ij    # (m, n_t)
    eta_j = phase_data.eta_j  # (n_t,)
    avail = phase_data.available.copy()  # (m,)

    residual = 0.0
    weapon_used = np.zeros(m, dtype=int)
    radar_used = np.zeros(L, dtype=int)

    for j in range(n_t):
        tgt = targets[j]
        survival = 1.0
        for i in range(m):
            for k in range(L):
                s = int(x_ijkt[i, j, k])
                if s > 0:
                    p = float(P_ij[i, j])
                    if p > 0:
                        survival *= (1.0 - p) ** s
                    weapon_used[i] += s
                    radar_used[k] += s
        residual += eta_j[j] * tgt.threat * survival

    # C1惩罚
    penalty = 0.0
    for i in range(m):
        overshoot = weapon_used[i] - int(avail[i])
        if overshoot > 0:
            penalty += overshoot * PENALTY_WEIGHT
    # C9惩罚
    for k in range(L):
        overshoot = radar_used[k] - int(inst.C_track[k])
        if overshoot > 0:
            penalty += overshoot * PENALTY_WEIGHT

    result = residual + penalty
    if not np.isfinite(result):
        return 1e10
    return result


def f1_ref(phase_data: PhaseData) -> float:
    """
    F1的理论最坏参考值（零毁伤上界）：
      F1_ref = Σ_j eta_j * Th_j
    即所有目标均未分配任何武器时的残余威胁，用作RMI/MORI的固定参考点
    （与个体自身取值无关，跨代/跨解可比）。
    """
    return sum(
        phase_data.eta_j[j] * phase_data.targets[j].threat
        for j in range(phase_data.n_t)
    )


def f2_ref(phase_data: PhaseData) -> float:
    """
    F2的理论最坏参考值（满额分配上界）：
      F2_ref = Σ_j U_jt
    即每个目标都用满最大可分配弹药数时的弹药消耗，用作RMI/MORI的固定参考点。
    """
    return float(np.sum(phase_data.U_jt))


def f1_max_from_residual(f1_res: float, phase_data: PhaseData) -> float:
    """
    将残余威胁（内部最小化）转换为加权拦截成功概率（论文呈现形式）：
      F1_max = Σ_j eta_j * Th_j - F1_res
    """
    return f1_ref(phase_data) - f1_res


def evaluate_f2_raw(x_ijkt: np.ndarray) -> int:
    """
    F2弹药消耗（最小化）：
      F2 = Σ_{i,j,k} x_ijkt
    """
    return int(x_ijkt.sum())


def evaluate_f1_raw_batch(
    x_batch: np.ndarray,   # (B, m, n_t, L) int
    phase_data: PhaseData,
    inst: Instance,
) -> np.ndarray:
    """
    批量评估F1_res（向量化，比逐个调用快约5-10倍）
    x_batch: (B, m, n_t, L)
    返回: (B,) float
    """
    B, m, n_t, L = x_batch.shape
    targets = phase_data.targets
    P_ij = phase_data.P_ij    # (m, n_t)
    eta_j = phase_data.eta_j  # (n_t,)
    avail = phase_data.available.astype(int)
    C_track = inst.C_track.astype(int)

    # 计算每个解的每个目标的生存概率 (B, n_t)
    # survival[b, j] = Π_{i,k} (1-P_ij)^x[b,i,j,k]
    # 先取 (B, m, n_t)：对每个(b,i,j)求所有k的连乘
    x = x_batch  # (B, m, n_t, L)
    # (1-P_ij) shape (m, n_t) → 扩展到 (1, m, n_t, 1)
    one_minus_p = 1.0 - P_ij[np.newaxis, :, :, np.newaxis]  # (1,m,n_t,1)
    # 幂次: (B,m,n_t,L)
    log_surv = x * np.log(np.clip(one_minus_p, 1e-300, 1.0))  # 用log避免0^0
    # 对k维求和后exp → (B, m, n_t) 每个武器对目标的生存贡献
    log_surv_per_ij = log_surv.sum(axis=3)   # (B, m, n_t)
    # 对i求和（连乘=exp(sum log)）→ (B, n_t)
    log_surv_per_j = log_surv_per_ij.sum(axis=1)   # (B, n_t)
    survival = np.exp(log_surv_per_j)   # (B, n_t)

    # 残余威胁 (B, n_t) * eta_j * threat，求和得 (B,)
    weights = eta_j * np.array([t.threat for t in targets])  # (n_t,)
    residual = (survival * weights[np.newaxis, :]).sum(axis=1)  # (B,)

    # C1/C9惩罚
    weapon_used = x.sum(axis=(2, 3))   # (B, m) 对n_t和L维求和
    radar_used = x.sum(axis=(1, 2))    # (B, L) 对m和n_t维求和

    c1_penalty = np.maximum(0, weapon_used - avail[np.newaxis, :]).sum(axis=1) * PENALTY_WEIGHT
    c9_penalty = np.maximum(0, radar_used - C_track[np.newaxis, :]).sum(axis=1) * PENALTY_WEIGHT

    result = residual + c1_penalty + c9_penalty
    result = np.where(np.isfinite(result), result, 1e10)
    return result


def evaluate_f2_raw_batch(x_batch: np.ndarray) -> np.ndarray:
    """批量评估F2（弹药消耗）"""
    return x_batch.sum(axis=(1, 2, 3)).astype(float)


def evaluate_solution(
    x_ijkt: np.ndarray,
    phase_data: PhaseData,
    inst: Instance
) -> tuple:
    """
    返回 (f1_res, f2)：内部均为最小化形式
      f1_res = 加权残余威胁（越小越好）
      f2     = 弹药消耗（越小越好）
    """
    f1 = evaluate_f1_raw(x_ijkt, phase_data, inst)
    f2 = float(evaluate_f2_raw(x_ijkt))
    return f1, f2
    return f1, f2

"""instance.py — WTA实例加载与双目标扩展参数"""
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional
import scipy.io


@dataclass
class TargetData:
    id: int
    x: float
    y: float
    speed: float
    heading: float
    threat: float
    rcs: float
    W: float        # 打击窗口 s
    U: int          # 最大可分配弹药数
    P_min: float    # 最低毁伤概率要求
    category: str


@dataclass
class PhaseData:
    phase: int
    n_t: int
    targets: List[TargetData]
    P_ij: np.ndarray     # (m, n_t) 预计算毁伤概率
    eta_j: np.ndarray    # (n_t,) MRFDQI
    d_ij: np.ndarray     # (m, n_t) 武器-目标距离
    W_jt: np.ndarray     # (n_t,) 打击窗口
    U_jt: np.ndarray     # (n_t,) 最大分配弹药数（int）
    P_min_jt: np.ndarray # (n_t,) 最低毁伤概率
    available: np.ndarray # (m,) 本阶段开始时可用弹药


@dataclass
class Instance:
    name: str
    n: int           # 总目标数（跨阶段）
    m: int           # 武器平台数
    L: int           # 雷达数
    T_total: int     # 总阶段数
    phase_targets: np.ndarray  # (T_total,) 每阶段目标数
    # 武器
    weapon_x: np.ndarray   # (m,)
    weapon_y: np.ndarray   # (m,)
    R_max: np.ndarray      # (m,) 最大射程
    tau_react: np.ndarray  # (m,) 反应时间
    v_weapon: np.ndarray   # (m,) 飞行速度
    priority: np.ndarray   # (m,) 武器优先级
    ammo: np.ndarray       # (m,) 初始总载弹量
    alpha: np.ndarray      # (m, L) 协同能力矩阵
    # 雷达
    radar_x: np.ndarray    # (L,)
    radar_y: np.ndarray    # (L,)
    lam: np.ndarray        # (L,) 雷达性能参数（lambda）
    R_radar: np.ndarray    # (L,) 探测距离
    p_th: float            # 探测概率阈值
    C_track: np.ndarray    # (L,) 火控通道容量
    # 通用
    dt: float              # 决策间隔 s
    phases: List[PhaseData] = field(default_factory=list)


def _squeeze(x):
    """MATLAB squeeze_me兼容：将0维数组转标量，1维保留"""
    if isinstance(x, np.ndarray):
        if x.ndim == 0:
            return x.item()
        if x.ndim == 1 and x.size == 1:
            return x.item()
    return x


def _to_1d(x):
    return np.asarray(x).ravel()


def _parse_target(t):
    """从MATLAB struct解析TargetData"""
    get = lambda f: _squeeze(getattr(t, f))
    return TargetData(
        id=int(get('id')),
        x=float(get('position_x')),
        y=float(get('position_y')),
        speed=float(get('speed')),
        heading=float(get('heading')),
        threat=float(get('threat')),
        rcs=float(get('RCS')),
        W=float(get('W')),
        U=int(get('U')),
        P_min=float(get('P_min')),
        category=str(get('category')),
    )


def _parse_phase(ph_mat):
    """从MATLAB struct解析PhaseData"""
    get = lambda f: _squeeze(getattr(ph_mat, f))
    # targets数组
    tgt_raw = getattr(ph_mat, 'targets')
    if hasattr(tgt_raw, '__len__'):
        targets = [_parse_target(tgt_raw[i]) for i in range(len(tgt_raw))]
    else:
        targets = [_parse_target(tgt_raw)]

    n_t = int(get('n_t'))
    P_ij = np.array(getattr(ph_mat, 'P_ij'), dtype=float)   # (m, n_t)
    eta_j = _to_1d(np.array(getattr(ph_mat, 'eta_j'), dtype=float))  # (n_t,)
    d_ij = np.array(getattr(ph_mat, 'd_ij'), dtype=float)            # (m, n_t)
    W_jt = _to_1d(np.array(getattr(ph_mat, 'W_jt'), dtype=float))
    U_jt = _to_1d(np.array(getattr(ph_mat, 'U_jt'), dtype=int))
    P_min_jt = _to_1d(np.array(getattr(ph_mat, 'P_min_jt'), dtype=float))
    available = _to_1d(np.array(getattr(ph_mat, 'available'), dtype=int))

    return PhaseData(
        phase=int(get('phase')),
        n_t=n_t,
        targets=targets,
        P_ij=P_ij,
        eta_j=eta_j,
        d_ij=d_ij,
        W_jt=W_jt,
        U_jt=U_jt,
        P_min_jt=P_min_jt,
        available=available,
    )


def load_instance(mat_path: str) -> Instance:
    """加载.mat实例文件，返回Instance对象"""
    raw = scipy.io.loadmat(mat_path, struct_as_record=False, squeeze_me=True)
    inst_mat = raw['instance']
    get = lambda f: _squeeze(getattr(inst_mat, f))

    phases_raw = getattr(inst_mat, 'phases')
    if hasattr(phases_raw, '__len__'):
        phases = [_parse_phase(phases_raw[i]) for i in range(len(phases_raw))]
    else:
        phases = [_parse_phase(phases_raw)]

    return Instance(
        name=str(get('name')),
        n=int(get('n')),
        m=int(get('m')),
        L=int(get('L')),
        T_total=int(get('T_total')),
        phase_targets=_to_1d(np.array(getattr(inst_mat, 'phase_targets'), dtype=int)),
        weapon_x=_to_1d(np.array(getattr(inst_mat, 'weapon_x'), dtype=float)),
        weapon_y=_to_1d(np.array(getattr(inst_mat, 'weapon_y'), dtype=float)),
        R_max=_to_1d(np.array(getattr(inst_mat, 'R_max'), dtype=float)),
        tau_react=_to_1d(np.array(getattr(inst_mat, 'tau_react'), dtype=float)),
        v_weapon=_to_1d(np.array(getattr(inst_mat, 'v_weapon'), dtype=float)),
        priority=_to_1d(np.array(getattr(inst_mat, 'priority'), dtype=int)),
        ammo=_to_1d(np.array(getattr(inst_mat, 'ammo'), dtype=int)),
        alpha=np.array(getattr(inst_mat, 'alpha'), dtype=int),   # (m, L)
        radar_x=_to_1d(np.array(getattr(inst_mat, 'radar_x'), dtype=float)),
        radar_y=_to_1d(np.array(getattr(inst_mat, 'radar_y'), dtype=float)),
        lam=_to_1d(np.array(getattr(inst_mat, 'lambda'), dtype=float)),
        R_radar=_to_1d(np.array(getattr(inst_mat, 'R_radar'), dtype=float)),
        p_th=float(get('p_th')),
        C_track=_to_1d(np.array(getattr(inst_mat, 'C_track'), dtype=int)),
        dt=float(get('dt')),
        phases=phases,
    )


# ── 双目标扩展参数 ────────────────────────────────────────────────────
@dataclass
class BiObjParams:
    """双目标扩展的额外参数（阶段0单目标实验不需要）"""
    ammo_replenish_rate: float = 0.0   # 每阶段弹药补充率（0=不补充，0.1=10%）
    threat_dist: str = 'uniform'        # 威胁分布：'uniform' 或 'heavy_tail'

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import argparse
import csv
import math
import json

@dataclass
class BatteryParams:
    Q_nom_Ah: float = 4.8
    V_nom: float = 3.85
    R0_ohm: float = 0.055
    T_ref_C: float = 25.0
    cap_temp_drop_per_C_below: float = 0.004
    cap_temp_drop_per_C_above: float = 0.001
    R_temp_rise_per_C_below: float = 0.03
    R_temp_drop_per_C_above: float = 0.005
    health_capacity_factor: float = 0.92
    health_resistance_factor: float = 1.25
    ocv_min: float = 3.0
    ocv_max: float = 4.35
    eta_dc: float = 0.92

@dataclass
class ThermalParams:
    T_amb_C: float = 22.0
    C_th_J_per_C: float = 120.0
    h_W_per_C: float = 0.9
    sys_heat_fraction: float = 0.75
    T_throttle_start_C: float = 38.0
    T_throttle_full_C: float = 48.0
    throttle_min_factor: float = 0.55

@dataclass
class ScreenParams:
    is_oled: bool = True
    size_inch: float = 6.6
    resolution_scale: float = 1.0
    refresh_hz: float = 120.0
    base_power_W: float = 0.9
    brightness_gamma: float = 1.6
    oled_content_factor: float = 0.85

@dataclass
class SoCComputeParams:
    p_idle_W: float = 0.18
    p_cpu_max_W: float = 2.2
    p_gpu_max_W: float = 3.2
    p_isp_max_W: float = 1.6
    p_npu_max_W: float = 1.0
    p_dram_on_W: float = 0.25
    p_storage_active_W: float = 0.35

@dataclass
class SoCArchParams:
    process_node_nm: float = 4.0
    efficiency_exp: float = 0.7
    has_hw_video_decode: bool = True
    has_hw_photo_pipeline: bool = True
    has_npu: bool = True

@dataclass
class InteractionParams:
    k_screen_render_W: float = 0.55
    screen_render_brightness_exp: float = 1.2
    k_radio_bg_mult: float = 0.65
    bg_intensity_exp: float = 0.5
    k_thermal_leak_W_per_C: float = 0.015

@dataclass
class RadioParams:
    tau_tail_s: float = 10.0
    tau_active_hold_s: float = 2.0
    p_idle_W: float = 0.1
    p_tail_W: float = 0.55
    p_active_W: float = 1.25
    weak_signal_penalty_max: float = 1.9
    p_switch_J: float = 2.5

@dataclass
class SensorParams:
    p_gps_W: float = 0.35
    p_bt_scan_W: float = 0.12

@dataclass
class OSParams:
    wake_cost_J: float = 0.35
    wake_duration_s: float = 0.4
    sched_aggressiveness: float = 1.0

@dataclass
class ModelParams:
    battery: BatteryParams = field(default_factory=BatteryParams)
    thermal: ThermalParams = field(default_factory=ThermalParams)
    screen: ScreenParams = field(default_factory=ScreenParams)
    compute: SoCComputeParams = field(default_factory=SoCComputeParams)
    soc_arch: SoCArchParams = field(default_factory=SoCArchParams)
    radio: RadioParams = field(default_factory=RadioParams)
    sensors: SensorParams = field(default_factory=SensorParams)
    os: OSParams = field(default_factory=OSParams)
    interaction: InteractionParams = field(default_factory=InteractionParams)

@dataclass
class ScenarioStep:
    dt_s: float
    screen_on: bool
    brightness: float
    refresh_hz: float
    oled_content_factor: Optional[float] = None
    foreground: str = 'game'
    net_requests_per_s: float = 0.0
    signal_quality: float = 1.0
    network_type: str = 'wifi'
    gps_on: bool = False
    bt_scan: bool = False
    background_wake_per_s: float = 0.05
    ambient_temp_C: Optional[float] = None

def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x

def ocv_from_soc(b: BatteryParams, z: float, T_C: float) -> float:
    z = clamp(z, 0.0, 1.0)
    s = 0.12
    v = b.ocv_min + (b.ocv_max - b.ocv_min) * (z - s) / (1.0 - 2 * s)
    v = clamp(v, b.ocv_min, b.ocv_max)
    v += 0.0008 * (T_C - b.T_ref_C)
    return v

def q_eff_Ah(b: BatteryParams, T_C: float) -> float:
    dT = T_C - b.T_ref_C
    if dT < 0:
        temp_factor = 1.0 + dT * b.cap_temp_drop_per_C_below
    else:
        temp_factor = 1.0 - dT * b.cap_temp_drop_per_C_above
    temp_factor = clamp(temp_factor, 0.55, 1.05)
    return b.Q_nom_Ah * b.health_capacity_factor * temp_factor

def r_int_ohm(b: BatteryParams, T_C: float) -> float:
    dT = T_C - b.T_ref_C
    if dT < 0:
        temp_factor = 1.0 + -dT * b.R_temp_rise_per_C_below
    else:
        temp_factor = 1.0 - dT * b.R_temp_drop_per_C_above
    temp_factor = clamp(temp_factor, 0.6, 4.0)
    return b.R0_ohm * b.health_resistance_factor * temp_factor

def throttle_factor(th: ThermalParams, T_C: float) -> float:
    if T_C <= th.T_throttle_start_C:
        return 1.0
    if T_C >= th.T_throttle_full_C:
        return th.throttle_min_factor
    x = (T_C - th.T_throttle_start_C) / (th.T_throttle_full_C - th.T_throttle_start_C)
    return 1.0 - x * (1.0 - th.throttle_min_factor)

class RadioRRC:

    def __init__(self, params: RadioParams):
        self.p = params
        self.state = 'IDLE'
        self.timer_s = 0.0
        self.prev_network_type = 'wifi'

    def step(self, dt_s: float, net_requests_per_s: float, signal_quality: float, network_type: str) -> Tuple[float, float]:
        switching_J = 0.0
        if network_type != self.prev_network_type:
            switching_J = self.p.p_switch_J
            self.prev_network_type = network_type
        req = net_requests_per_s * dt_s
        has_req = req > 1e-09
        if has_req:
            self.state = 'ACTIVE'
            self.timer_s = self.p.tau_active_hold_s
        elif self.state == 'ACTIVE':
            self.timer_s -= dt_s
            if self.timer_s <= 0:
                self.state = 'TAIL'
                self.timer_s = self.p.tau_tail_s
        elif self.state == 'TAIL':
            self.timer_s -= dt_s
            if self.timer_s <= 0:
                self.state = 'IDLE'
                self.timer_s = 0.0
        else:
            self.state = 'IDLE'
        base = {'IDLE': self.p.p_idle_W, 'TAIL': self.p.p_tail_W, 'ACTIVE': self.p.p_active_W}[self.state]
        sq = clamp(signal_quality, 0.0, 1.0)
        penalty = 1.0 + (1.0 - sq) * (self.p.weak_signal_penalty_max - 1.0)
        return (base * penalty, switching_J)

def screen_power_W(p: ScreenParams, screen_on: bool, brightness: float, refresh_hz: float, oled_content_factor: Optional[float]) -> float:
    if not screen_on:
        return 0.0
    b = clamp(brightness, 0.0, 1.0)
    hz_factor = clamp(refresh_hz / 60.0, 0.8, 2.5)
    res_factor = clamp(p.resolution_scale, 0.7, 1.8)
    content = p.oled_content_factor if oled_content_factor is None else clamp(oled_content_factor, 0.5, 1.2)
    P = p.base_power_W * (0.15 + 0.85 * b ** p.brightness_gamma) * hz_factor * res_factor
    if p.is_oled:
        P *= content
    return P

def workload_utilization(foreground: str) -> Dict[str, float]:
    fg = foreground.lower().strip()
    if fg == 'video':
        return {'cpu': 0.35, 'gpu': 0.2, 'isp': 0.0, 'npu': 0.05, 'storage': 0.2}
    if fg == 'game':
        return {'cpu': 0.55, 'gpu': 0.85, 'isp': 0.0, 'npu': 0.05, 'storage': 0.1}
    if fg == 'camera':
        return {'cpu': 0.4, 'gpu': 0.15, 'isp': 0.9, 'npu': 0.25, 'storage': 0.65}
    if fg == 'navigation':
        return {'cpu': 0.35, 'gpu': 0.25, 'isp': 0.0, 'npu': 0.05, 'storage': 0.1}
    if fg == 'social':
        return {'cpu': 0.3, 'gpu': 0.15, 'isp': 0.0, 'npu': 0.02, 'storage': 0.1}
    return {'cpu': 0.06, 'gpu': 0.01, 'isp': 0.0, 'npu': 0.0, 'storage': 0.02}

def soc_efficiency_factor(arch: SoCArchParams) -> float:
    node = clamp(arch.process_node_nm, 2.0, 28.0)
    return (7.0 / node) ** clamp(arch.efficiency_exp, 0.3, 1.2)

def compute_power_W(c: SoCComputeParams, arch: SoCArchParams, os_p: OSParams, screen_on: bool, fg: str, perf_factor: float) -> Tuple[float, Dict[str, float]]:
    u = workload_utilization(fg)
    fg_l = fg.lower().strip()
    if fg_l == 'video' and arch.has_hw_video_decode:
        u = {**u, 'cpu': u['cpu'] * 0.75, 'gpu': u['gpu'] * 0.85}
    if fg_l == 'camera' and arch.has_hw_photo_pipeline:
        u = {**u, 'cpu': u['cpu'] * 0.85, 'isp': min(1.0, u['isp'] * 1.0)}
    if not arch.has_npu:
        u = {**u, 'cpu': clamp(u['cpu'] + 0.25 * u['npu'], 0.0, 1.0), 'npu': 0.0}
    pf = clamp(perf_factor, 0.2, 1.0)
    sched = clamp(os_p.sched_aggressiveness, 0.7, 1.4)
    u_cpu = clamp(u['cpu'] / pf, 0.0, 1.0)
    u_gpu = clamp(u['gpu'] / pf, 0.0, 1.0)
    u_isp = clamp(u['isp'] / pf, 0.0, 1.0)
    u_npu = clamp(u['npu'] / pf, 0.0, 1.0)
    eff = soc_efficiency_factor(arch)
    p_scale = 1.0 / clamp(eff, 0.55, 1.8)
    alpha = 1.35
    p_cpu = c.p_cpu_max_W * u_cpu ** alpha * sched * p_scale
    p_gpu = c.p_gpu_max_W * u_gpu ** alpha * sched * p_scale
    p_isp = c.p_isp_max_W * u_isp ** 1.15 * p_scale
    p_npu = c.p_npu_max_W * u_npu ** 1.2 * p_scale
    p_dram = c.p_dram_on_W if screen_on else 0.0
    p_storage = c.p_storage_active_W * clamp(u['storage'] / pf, 0.0, 1.0)
    total = p_cpu + p_gpu + p_isp + p_npu + p_dram + p_storage
    detail = {'p_cpu_W': p_cpu, 'p_gpu_W': p_gpu, 'p_isp_W': p_isp, 'p_npu_W': p_npu, 'p_dram_W': p_dram, 'p_storage_W': p_storage, 'u_cpu': u_cpu, 'u_gpu': u_gpu, 'u_isp': u_isp, 'u_npu': u_npu, 'p_scale_soc': p_scale}
    return (total, detail)

def background_wake_power_W(os_p: OSParams, wake_per_s: float) -> float:
    lam = max(0.0, wake_per_s)
    return lam * os_p.wake_cost_J

@dataclass
class SimState:
    t_s: float = 0.0
    z: float = 1.0
    T_C: float = 25.0

def solve_current_A(ocv_V: float, R_ohm: float, P_W: float) -> float:
    P = max(0.0, P_W)
    R = max(1e-06, R_ohm)
    O = max(0.001, ocv_V)
    disc = O * O - 4.0 * R * P
    if disc <= 0.0:
        return O / (2.0 * R)
    return (O - math.sqrt(disc)) / (2.0 * R)

def simulate(params: ModelParams, steps: List[ScenarioStep], z0: float=1.0, T0_C: Optional[float]=None, *, extend_with_idle: bool=False, idle_step: Optional[ScenarioStep]=None, max_time_s: Optional[float]=None) -> Dict[str, object]:
    st = SimState(t_s=0.0, z=clamp(z0, 0.0, 1.0), T_C=params.thermal.T_amb_C if T0_C is None else T0_C)
    radio = RadioRRC(params.radio)
    rows: List[Dict[str, float]] = []
    t_empty_s: Optional[float] = None
    terminated_reason: str = 'unknown'
    step_i = 0
    while True:
        if st.z <= 0.0:
            t_empty_s = st.t_s
            terminated_reason = 'empty'
            break
        if max_time_s is not None and st.t_s >= max_time_s:
            terminated_reason = 'max_time'
            break
        if step_i < len(steps):
            step = steps[step_i]
            step_i += 1
        else:
            if not extend_with_idle or idle_step is None:
                terminated_reason = 'steps_end'
                break
            step = idle_step
        dt = max(0.001, step.dt_s)
        T_amb_C = params.thermal.T_amb_C if step.ambient_temp_C is None else float(step.ambient_temp_C)
        perf = throttle_factor(params.thermal, st.T_C)
        p_scr = screen_power_W(params.screen, step.screen_on, step.brightness, step.refresh_hz, step.oled_content_factor)
        p_comp, comp_detail = compute_power_W(params.compute, params.soc_arch, params.os, screen_on=step.screen_on, fg=step.foreground, perf_factor=perf)
        p_radio, switch_J = radio.step(dt_s=dt, net_requests_per_s=step.net_requests_per_s, signal_quality=step.signal_quality, network_type=step.network_type)
        p_bg = background_wake_power_W(params.os, step.background_wake_per_s)
        p_gps = params.sensors.p_gps_W if step.gps_on else 0.0
        p_bt = params.sensors.p_bt_scan_W if step.bt_scan else 0.0
        p_idle = params.compute.p_idle_W + (0.06 if step.screen_on else 0.0)
        inter = params.interaction
        u_cpu = float(comp_detail.get('u_cpu', 0.0))
        u_gpu = float(comp_detail.get('u_gpu', 0.0))
        render_extra_W = 0.0
        if step.screen_on:
            render_extra_W = inter.k_screen_render_W * clamp(step.brightness, 0.0, 1.0) ** inter.screen_render_brightness_exp * clamp(step.refresh_hz / 60.0, 0.8, 2.5) * clamp(0.35 * u_cpu + 0.65 * u_gpu, 0.0, 1.0)
        sq = clamp(step.signal_quality, 0.0, 1.0)
        bg_intensity = clamp(step.background_wake_per_s, 0.0, 5.0) ** inter.bg_intensity_exp
        radio_bg_extra_W = p_radio * inter.k_radio_bg_mult * (1.0 - sq) * clamp(bg_intensity, 0.0, 3.0)
        thermal_leak_W = inter.k_thermal_leak_W_per_C * max(0.0, st.T_C - T_amb_C)
        p_sys = p_idle + p_scr + p_comp + p_radio + p_bg + p_gps + p_bt + render_extra_W + radio_bg_extra_W + thermal_leak_W
        p_switch = switch_J / dt
        p_sys += p_switch
        b = params.battery
        ocv = ocv_from_soc(b, st.z, st.T_C)
        R = r_int_ohm(b, st.T_C)
        Q = q_eff_Ah(b, st.T_C)
        p_batt = p_sys / max(1e-06, b.eta_dc)
        I = solve_current_A(ocv, R, p_batt)
        V_term = ocv - I * R
        V_term = max(2.5, V_term)
        dz = -I * dt / (3600.0 * max(1e-06, Q))
        z_next = clamp(st.z + dz, 0.0, 1.0)
        th = params.thermal
        p_heat = I * I * R + th.sys_heat_fraction * p_sys
        dT = (p_heat - th.h_W_per_C * (st.T_C - T_amb_C)) * dt / max(1e-06, th.C_th_J_per_C)
        T_next = st.T_C + dT
        rows.append({'t_s': st.t_s, 'soc': st.z, 'T_C': st.T_C, 'T_amb_C': T_amb_C, 'ocv_V': ocv, 'V_term_V': V_term, 'I_A': I, 'p_sys_W': p_sys, 'p_screen_W': p_scr, 'p_compute_W': p_comp, 'p_radio_W': p_radio, 'p_bg_W': p_bg, 'p_gps_W': p_gps, 'p_bt_W': p_bt, 'p_render_extra_W': render_extra_W, 'p_radio_bg_extra_W': radio_bg_extra_W, 'p_thermal_leak_W': thermal_leak_W, 'radio_state': 0.0 if radio.state == 'IDLE' else 1.0 if radio.state == 'TAIL' else 2.0, 'perf_factor': perf, **comp_detail})
        st.t_s += dt
        st.z = z_next
        st.T_C = T_next
    if t_empty_s is None and st.z <= 0.0:
        t_empty_s = st.t_s
        terminated_reason = 'empty'
    return {'params': params, 'rows': rows, 't_empty_s': t_empty_s, 'terminated_reason': terminated_reason}

def load_steps_from_csv(path: str, default_dt_s: float=1.0) -> List[ScenarioStep]:

    def parse_bool(x: str) -> bool:
        s = (x or '').strip().lower()
        return s in {'1', 'true', 't', 'yes', 'y'}
    steps: List[ScenarioStep] = []
    with open(path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            dt_s = float(r.get('dt_s') or default_dt_s)
            steps.append(ScenarioStep(dt_s=dt_s, screen_on=parse_bool(r.get('screen_on', '1')), brightness=float(r.get('brightness') or 0.5), refresh_hz=float(r.get('refresh_hz') or 120.0), oled_content_factor=None if r.get('oled_content_factor') in (None, '') else float(r.get('oled_content_factor')), foreground=str(r.get('foreground') or 'idle'), net_requests_per_s=float(r.get('net_requests_per_s') or 0.0), signal_quality=float(r.get('signal_quality') or 1.0), network_type=str(r.get('network_type') or 'wifi'), gps_on=parse_bool(r.get('gps_on', '0')), bt_scan=parse_bool(r.get('bt_scan', '0')), background_wake_per_s=float(r.get('background_wake_per_s') or 0.05), ambient_temp_C=None if r.get('ambient_temp_C') in (None, '') else float(r.get('ambient_temp_C'))))
    return steps

def built_in_demo_scenario(dt_s: float=1.0, hours: float=8.0, ambient_temp_C: Optional[float]=None) -> List[ScenarioStep]:
    total_s = int(hours * 3600)
    steps: List[ScenarioStep] = []

    def add_block(duration_s: int, **kw):
        n = max(1, int(duration_s / dt_s))
        for _ in range(n):
            steps.append(ScenarioStep(dt_s=dt_s, **kw))
    add_block(20 * 60, screen_on=True, brightness=0.55, refresh_hz=120.0, foreground='video', net_requests_per_s=0.7, signal_quality=0.95, network_type='wifi', gps_on=False, bt_scan=False, background_wake_per_s=0.12, oled_content_factor=0.8, ambient_temp_C=ambient_temp_C)
    add_block(15 * 60, screen_on=True, brightness=0.62, refresh_hz=120.0, foreground='social', net_requests_per_s=0.25, signal_quality=0.9, network_type='wifi', gps_on=False, bt_scan=False, background_wake_per_s=0.18, oled_content_factor=0.95, ambient_temp_C=ambient_temp_C)
    add_block(30 * 60, screen_on=True, brightness=0.7, refresh_hz=120.0, foreground='game', net_requests_per_s=0.18, signal_quality=0.9, network_type='wifi', gps_on=False, bt_scan=False, background_wake_per_s=0.1, oled_content_factor=0.92, ambient_temp_C=ambient_temp_C)
    add_block(15 * 60, screen_on=True, brightness=0.75, refresh_hz=60.0, foreground='camera', net_requests_per_s=0.08, signal_quality=0.85, network_type='wifi', gps_on=False, bt_scan=False, background_wake_per_s=0.08, oled_content_factor=0.9, ambient_temp_C=ambient_temp_C)
    add_block(50 * 60, screen_on=True, brightness=0.85, refresh_hz=60.0, foreground='navigation', net_requests_per_s=0.35, signal_quality=0.35, network_type='cell', gps_on=True, bt_scan=True, background_wake_per_s=0.12, oled_content_factor=0.75, ambient_temp_C=ambient_temp_C)
    remaining = total_s - len(steps)
    if remaining > 0:
        add_block(remaining, screen_on=False, brightness=0.0, refresh_hz=60.0, foreground='idle', net_requests_per_s=0.03, signal_quality=0.6, network_type='cell', gps_on=False, bt_scan=False, background_wake_per_s=0.25, oled_content_factor=None, ambient_temp_C=ambient_temp_C)
    return steps

def summarize(sim: Dict[str, object], t_check_s: float=50.0) -> Dict[str, float]:
    rows: List[Dict[str, float]] = sim['rows']
    if not rows:
        return {'soc_at_check': float('nan'), 't_empty_s': float('nan')}
    idx = min(range(len(rows)), key=lambda i: abs(rows[i]['t_s'] - t_check_s))
    soc_at = float(rows[idx]['soc'])
    t_empty = sim['t_empty_s']
    t_empty_val = float('nan') if t_empty is None else float(t_empty)
    return {'soc_at_check': soc_at, 't_empty_s': t_empty_val}

def save_csv(rows: List[Dict[str, float]], out_path: str) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def main():
    ap = argparse.ArgumentParser(description='智能手机电池混合连续时间模型')
    ap.add_argument('--csv', type=str, default='', help='输入场景 CSV（列名见 load_steps_from_csv 的说明）')
    ap.add_argument('--dt', type=float, default=1.0, help='内置场景的时间步长 dt（秒）')
    ap.add_argument('--hours', type=float, default=0.0, help='内置场景仿真总时长（小时）；<=0 表示默认一直仿真到耗尽')
    ap.add_argument('--max-hours', type=float, default=240.0, help='仿真最长时长上限（小时）')
    ap.add_argument('--ambient', type=float, default=None, help='外部环境温度（℃）；不填则使用 ThermalParams.T_amb_C')
    ap.add_argument('--t-check', type=float, default=50.0, help='输出 t=t_check 的 SoC（秒）')
    ap.add_argument('--out', type=str, default='sim_output.csv', help='输出 CSV 路径')
    ap.add_argument('--params-out', type=str, default='model_params_used.json', help='输出参数 JSON 路径')
    args = ap.parse_args()
    params = ModelParams()
    dt_s = max(0.001, float(args.dt))
    ambient = None if args.ambient is None else float(args.ambient)
    if args.csv:
        steps = load_steps_from_csv(str(args.csv), default_dt_s=dt_s)
        sim = simulate(params, steps, z0=1.0)
    elif float(args.hours) > 0.0:
        steps = built_in_demo_scenario(dt_s=dt_s, hours=max(0.01, float(args.hours)), ambient_temp_C=ambient)
        sim = simulate(params, steps, z0=1.0)
    else:
        steps = built_in_demo_scenario(dt_s=dt_s, hours=3.0, ambient_temp_C=ambient)
        idle_step = ScenarioStep(dt_s=dt_s, screen_on=False, brightness=0.0, refresh_hz=60.0, oled_content_factor=None, foreground='idle', net_requests_per_s=0.02, signal_quality=0.85, network_type='cell', gps_on=False, bt_scan=False, background_wake_per_s=0.06, ambient_temp_C=ambient)
        sim = simulate(params, steps, z0=1.0, extend_with_idle=True, idle_step=idle_step, max_time_s=max(0.01, float(args.max_hours)) * 3600.0)
    s = summarize(sim, t_check_s=float(args.t_check))
    soc50 = s['soc_at_check']
    t_empty_s = s['t_empty_s']
    print('=== 智能手机电池 SoC 混合连续时间模型：演示 ===')
    print(f't={float(args.t_check):.0f}s 时 SoC: {soc50 * 100:.2f}%')
    if math.isfinite(t_empty_s):
        print(f'预测耗尽时间（SoC→0）：{t_empty_s / 3600:.2f} 小时')
    else:
        reason = str(sim.get('terminated_reason', 'unknown'))
        if reason == 'max_time':
            print('预测耗尽时间：在当前仿真时域内未耗尽（已达到 --max-hours/时域上限）')
        elif reason == 'steps_end':
            print('预测耗尽时间：在当前仿真时域内未耗尽（输入场景结束且未启用自动延长）')
        else:
            print('预测耗尽时间：在当前仿真时域内未耗尽')
    out_csv = str(args.out)
    save_csv(sim['rows'], out_csv)
    print(f'已保存轨迹到: {out_csv}')
    out_json = str(args.params_out)
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump({'battery': params.battery.__dict__, 'thermal': params.thermal.__dict__, 'screen': params.screen.__dict__, 'compute': params.compute.__dict__, 'soc_arch': params.soc_arch.__dict__, 'radio': params.radio.__dict__, 'sensors': params.sensors.__dict__, 'os': params.os.__dict__, 'interaction': params.interaction.__dict__}, f, ensure_ascii=False, indent=2)
    print(f'已保存参数到: {out_json}')
if __name__ == '__main__':
    main()

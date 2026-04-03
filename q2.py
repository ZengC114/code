from __future__ import annotations
import argparse
import csv
import datetime
import math
import os
import json
import re
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple
try:
    import final_unified_model as m
except Exception:
    try:
        import q1 as m
    except Exception as e:
        raise SystemExit(f'无法导入第一问模型模块原始错误: {e}')

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def make_dated_outdir(base: str) -> str:
    today = datetime.date.today().strftime('%Y%m%d')
    candidate = f'{base}_{today}'
    if not os.path.exists(candidate):
        return candidate
    k = 2
    while True:
        cand2 = f'{candidate}_{k}'
        if not os.path.exists(cand2):
            return cand2
        k += 1

def sanitize_filename(name: str) -> str:
    name = re.sub('[<>:\\"/\\\\|?*]', '_', name)
    name = name.strip().strip('.')
    return name if name else 'scenario'

def write_rows_csv(path: str, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    keys_no_t = sorted([k for k in keys if k != 't_s'])
    fieldnames = ['t_s'] + keys_no_t if 't_s' in rows[0] else keys_no_t
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, 0.0) for k in fieldnames})

def configure_cn_font() -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
    except Exception:
        return
    candidates = ['Microsoft YaHei', 'Microsoft YaHei UI', 'SimHei', 'SimSun', 'DengXian', 'PingFang SC', 'Noto Sans CJK SC', 'Source Han Sans SC']
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next((name for name in candidates if name in available), None)
    if chosen is not None:
        plt.rcParams['font.sans-serif'] = [chosen] + list(plt.rcParams.get('font.sans-serif', []))
    plt.rcParams['axes.unicode_minus'] = False

def dt_series_from_rows(rows: List[Dict[str, float]]) -> List[float]:
    if len(rows) <= 1:
        return [0.0]
    dts: List[float] = []
    for i in range(len(rows) - 1):
        dts.append(max(0.0, float(rows[i + 1]['t_s']) - float(rows[i]['t_s'])))
    dts_sorted = sorted(dts)
    dt_last = dts_sorted[len(dts_sorted) // 2] if dts_sorted else 0.0
    dts.append(dt_last)
    return dts

@dataclass
class ScenarioResult:
    name: str
    t_empty_h: float
    terminated_reason: str
    soc_end: float
    t_end_h: float
    T_max_C: float
    perf_min: float
    avg_p_sys_W: float
    energy_Wh: float
    energy_breakdown_Wh: Dict[str, float]

def integrate_energy_Wh(rows: List[Dict[str, float]]) -> Tuple[float, Dict[str, float], float]:
    if not rows:
        return (0.0, {}, 0.0)
    dts = dt_series_from_rows(rows)
    keys = [('screen', 'p_screen_W'), ('compute', 'p_compute_W'), ('radio', 'p_radio_W'), ('background', 'p_bg_W'), ('gps', 'p_gps_W'), ('bt', 'p_bt_W'), ('coupling_render', 'p_render_extra_W'), ('coupling_radio_bg', 'p_radio_bg_extra_W'), ('coupling_thermal', 'p_thermal_leak_W')]
    e_sys_J = 0.0
    t_total_s = 0.0
    e_parts_J: Dict[str, float] = {k: 0.0 for k, _ in keys}
    for row, dt in zip(rows, dts):
        t_total_s += dt
        p_sys = float(row.get('p_sys_W', 0.0))
        e_sys_J += p_sys * dt
        for part_name, col in keys:
            e_parts_J[part_name] += float(row.get(col, 0.0)) * dt
    e_sys_Wh = e_sys_J / 3600.0
    e_parts_Wh = {k: v / 3600.0 for k, v in e_parts_J.items()}
    avg_p = e_sys_J / t_total_s if t_total_s > 1e-09 else 0.0
    return (e_sys_Wh, e_parts_Wh, avg_p)

def top_drivers(e_parts_Wh: Dict[str, float], top_k: int=3) -> List[Tuple[str, float]]:
    items = [(k, float(v)) for k, v in e_parts_Wh.items()]
    items.sort(key=lambda x: x[1], reverse=True)
    return items[:top_k]

def run_scenario(name: str, params: m.ModelParams, steps: List[m.ScenarioStep], *, extend_until_empty: bool, max_hours: float, idle_step: Optional[m.ScenarioStep]=None) -> ScenarioResult:
    sim = m.simulate(params, steps, z0=1.0, extend_with_idle=extend_until_empty, idle_step=idle_step, max_time_s=max_hours * 3600.0)
    rows: List[Dict[str, float]] = sim['rows']
    t_empty_s = sim.get('t_empty_s')
    terminated_reason = str(sim.get('terminated_reason', 'unknown'))
    t_end_s = float(rows[-1]['t_s']) if rows else 0.0
    soc_end = float(rows[-1]['soc']) if rows else float('nan')
    T_max = max((float(r.get('T_C', 0.0)) for r in rows), default=float('nan'))
    perf_min = min((float(r.get('perf_factor', 1.0)) for r in rows), default=float('nan'))
    e_sys_Wh, e_parts_Wh, avg_p = integrate_energy_Wh(rows)
    if t_empty_s is None:
        t_empty_h = float('nan')
    else:
        t_empty_h = float(t_empty_s) / 3600.0
    return ScenarioResult(name=name, t_empty_h=t_empty_h, terminated_reason=terminated_reason, soc_end=soc_end, t_end_h=t_end_s / 3600.0, T_max_C=T_max, perf_min=perf_min, avg_p_sys_W=float(avg_p), energy_Wh=float(e_sys_Wh), energy_breakdown_Wh=e_parts_Wh)

def run_scenario_with_rows(name: str, params: m.ModelParams, steps: List[m.ScenarioStep], *, extend_until_empty: bool, max_hours: float, idle_step: Optional[m.ScenarioStep]=None) -> Tuple[ScenarioResult, List[Dict[str, float]], Dict[str, object]]:
    sim = m.simulate(params, steps, z0=1.0, extend_with_idle=extend_until_empty, idle_step=idle_step, max_time_s=max_hours * 3600.0)
    rows: List[Dict[str, float]] = sim['rows']
    r = run_scenario(name, params, steps, extend_until_empty=extend_until_empty, max_hours=max_hours, idle_step=idle_step)
    return (r, rows, sim)

def export_per_scenario_outputs(*, outdir: str, scenario_name: str, params: m.ModelParams, step: m.ScenarioStep, rows: List[Dict[str, float]], result: ScenarioResult, dpi: int, xlim_h: Optional[float]) -> None:
    scenario_dir = os.path.join(outdir, 'scenarios', sanitize_filename(scenario_name))
    ensure_dir(scenario_dir)
    csv_path = os.path.join(scenario_dir, 'sim_output.csv')
    write_rows_csv(csv_path, rows)
    meta = {'scenario': scenario_name, 'step': {'dt_s': float(step.dt_s), 'foreground': str(step.foreground), 'screen_on': bool(step.screen_on), 'brightness': float(step.brightness), 'refresh_hz': float(step.refresh_hz), 'oled_content_factor': None if step.oled_content_factor is None else float(step.oled_content_factor), 'net_requests_per_s': float(step.net_requests_per_s), 'signal_quality': float(step.signal_quality), 'network_type': str(step.network_type), 'gps_on': bool(step.gps_on), 'bt_scan': bool(step.bt_scan), 'background_wake_per_s': float(step.background_wake_per_s), 'ambient_temp_C': None if step.ambient_temp_C is None else float(step.ambient_temp_C)}, 'result': {'t_empty_h': float(result.t_empty_h), 'terminated_reason': str(result.terminated_reason), 'T_max_C': float(result.T_max_C), 'perf_min': float(result.perf_min), 'avg_p_sys_W': float(result.avg_p_sys_W), 'energy_Wh': float(result.energy_Wh)}, 'params_used': {'battery': {'Q_nom_Ah': float(params.battery.Q_nom_Ah), 'health_capacity_factor': float(params.battery.health_capacity_factor), 'health_resistance_factor': float(params.battery.health_resistance_factor), 'eta_dc': float(params.battery.eta_dc)}, 'thermal': {'T_amb_C': float(params.thermal.T_amb_C), 'h_W_per_C': float(params.thermal.h_W_per_C), 'T_throttle_start_C': float(params.thermal.T_throttle_start_C), 'T_throttle_full_C': float(params.thermal.T_throttle_full_C), 'throttle_min_factor': float(params.thermal.throttle_min_factor)}, 'radio': {'tau_tail_s': float(params.radio.tau_tail_s), 'tau_active_hold_s': float(params.radio.tau_active_hold_s)}, 'interaction': {'k_screen_render_W': float(params.interaction.k_screen_render_W), 'k_radio_bg_mult': float(params.interaction.k_radio_bg_mult), 'k_thermal_leak_W_per_C': float(params.interaction.k_thermal_leak_W_per_C)}}}
    with open(os.path.join(scenario_dir, 'scenario_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    try:
        import plot_sim_output as pso
        fig_dir = os.path.join(scenario_dir, 'figures')
        pso.plot_from_csv(csv_path, fig_dir, dpi=int(dpi), title_prefix=str(scenario_name), xlim_h=None if xlim_h is None else float(xlim_h))
    except Exception as e:
        print(f'[逐场景绘图] 跳过 {scenario_name}：{e}')

def make_constant_step(*, dt_s: float, foreground: str, screen_on: bool, brightness: float, refresh_hz: float, net_requests_per_s: float, signal_quality: float, network_type: str, gps_on: bool, bt_scan: bool, background_wake_per_s: float, ambient_temp_C: Optional[float], oled_content_factor: Optional[float]=None) -> m.ScenarioStep:
    return m.ScenarioStep(dt_s=dt_s, screen_on=screen_on, brightness=brightness, refresh_hz=refresh_hz, oled_content_factor=oled_content_factor, foreground=foreground, net_requests_per_s=net_requests_per_s, signal_quality=signal_quality, network_type=network_type, gps_on=gps_on, bt_scan=bt_scan, background_wake_per_s=background_wake_per_s, ambient_temp_C=ambient_temp_C)

def repeat_steps(step: m.ScenarioStep, hours: float) -> List[m.ScenarioStep]:
    n = max(1, int(hours * 3600.0 / max(1e-06, step.dt_s)))
    return [step for _ in range(n)]

def scenario_suite(dt_s: float, suite: str='core') -> List[Tuple[str, m.ScenarioStep, bool]]:
    core: List[Tuple[str, m.ScenarioStep, bool]] = [('重度游戏_室温', make_constant_step(dt_s=dt_s, foreground='game', screen_on=True, brightness=0.85, refresh_hz=120.0, net_requests_per_s=0.1, signal_quality=0.85, network_type='wifi', gps_on=False, bt_scan=False, background_wake_per_s=0.1, ambient_temp_C=22.0, oled_content_factor=0.9), True), ('短视频_室温', make_constant_step(dt_s=dt_s, foreground='video', screen_on=True, brightness=0.65, refresh_hz=120.0, net_requests_per_s=0.7, signal_quality=0.9, network_type='wifi', gps_on=False, bt_scan=False, background_wake_per_s=0.12, ambient_temp_C=22.0, oled_content_factor=0.8), True), ('导航_弱信号_夏季', make_constant_step(dt_s=dt_s, foreground='navigation', screen_on=True, brightness=0.85, refresh_hz=90.0, net_requests_per_s=0.25, signal_quality=0.3, network_type='cell', gps_on=True, bt_scan=True, background_wake_per_s=0.18, ambient_temp_C=35.0, oled_content_factor=0.9), True), ('待机推送_冬季', make_constant_step(dt_s=dt_s, foreground='idle', screen_on=False, brightness=0.0, refresh_hz=60.0, net_requests_per_s=0.02, signal_quality=0.6, network_type='cell', gps_on=False, bt_scan=False, background_wake_per_s=0.1, ambient_temp_C=-5.0, oled_content_factor=None), True), ('室内浏览_高亮度', make_constant_step(dt_s=dt_s, foreground='social', screen_on=True, brightness=0.95, refresh_hz=120.0, net_requests_per_s=0.3, signal_quality=0.9, network_type='wifi', gps_on=False, bt_scan=False, background_wake_per_s=0.2, ambient_temp_C=22.0, oled_content_factor=1.0), True)]
    if suite == 'core':
        return core
    if suite != 'extended':
        raise ValueError(f'未知 suite: {suite}')
    extended: List[Tuple[str, m.ScenarioStep, bool]] = list(core)
    extended += [('长视频_暗色主题_WIFI', make_constant_step(dt_s=dt_s, foreground='video', screen_on=True, brightness=0.35, refresh_hz=60.0, net_requests_per_s=0.08, signal_quality=0.95, network_type='wifi', gps_on=False, bt_scan=False, background_wake_per_s=0.08, ambient_temp_C=22.0, oled_content_factor=0.65), True), ('直播弹幕_高刷新_WIFI', make_constant_step(dt_s=dt_s, foreground='video', screen_on=True, brightness=0.7, refresh_hz=120.0, net_requests_per_s=0.35, signal_quality=0.9, network_type='wifi', gps_on=False, bt_scan=False, background_wake_per_s=0.18, ambient_temp_C=22.0, oled_content_factor=1.0), True), ('轻度游戏_60Hz', make_constant_step(dt_s=dt_s, foreground='game', screen_on=True, brightness=0.55, refresh_hz=60.0, net_requests_per_s=0.05, signal_quality=0.9, network_type='wifi', gps_on=False, bt_scan=False, background_wake_per_s=0.08, ambient_temp_C=22.0, oled_content_factor=0.9), True), ('联机游戏_蜂窝_中信号', make_constant_step(dt_s=dt_s, foreground='game', screen_on=True, brightness=0.75, refresh_hz=120.0, net_requests_per_s=0.18, signal_quality=0.55, network_type='cell', gps_on=False, bt_scan=True, background_wake_per_s=0.1, ambient_temp_C=25.0, oled_content_factor=0.95), True), ('拍照_连续取景', make_constant_step(dt_s=dt_s, foreground='camera', screen_on=True, brightness=0.7, refresh_hz=60.0, net_requests_per_s=0.02, signal_quality=0.9, network_type='wifi', gps_on=False, bt_scan=False, background_wake_per_s=0.1, ambient_temp_C=22.0, oled_content_factor=0.95), True), ('录像_1080p_户外夏季', make_constant_step(dt_s=dt_s, foreground='camera', screen_on=True, brightness=0.85, refresh_hz=60.0, net_requests_per_s=0.05, signal_quality=0.85, network_type='wifi', gps_on=True, bt_scan=False, background_wake_per_s=0.12, ambient_temp_C=33.0, oled_content_factor=1.05), True), ('社交聊天_高频消息', make_constant_step(dt_s=dt_s, foreground='social', screen_on=True, brightness=0.55, refresh_hz=60.0, net_requests_per_s=0.22, signal_quality=0.9, network_type='wifi', gps_on=False, bt_scan=False, background_wake_per_s=0.35, ambient_temp_C=22.0, oled_content_factor=0.95), True), ('刷网页_图文混排', make_constant_step(dt_s=dt_s, foreground='social', screen_on=True, brightness=0.65, refresh_hz=60.0, net_requests_per_s=0.1, signal_quality=0.88, network_type='wifi', gps_on=False, bt_scan=False, background_wake_per_s=0.18, ambient_temp_C=22.0, oled_content_factor=1.1), True), ('电子书阅读_低亮度', make_constant_step(dt_s=dt_s, foreground='social', screen_on=True, brightness=0.2, refresh_hz=60.0, net_requests_per_s=0.01, signal_quality=0.95, network_type='wifi', gps_on=False, bt_scan=False, background_wake_per_s=0.05, ambient_temp_C=22.0, oled_content_factor=0.7), True), ('音乐播放_锁屏', make_constant_step(dt_s=dt_s, foreground='idle', screen_on=False, brightness=0.0, refresh_hz=60.0, net_requests_per_s=0.03, signal_quality=0.9, network_type='wifi', gps_on=False, bt_scan=True, background_wake_per_s=0.1, ambient_temp_C=22.0, oled_content_factor=None), True), ('后台下载_锁屏_WIFI', make_constant_step(dt_s=dt_s, foreground='idle', screen_on=False, brightness=0.0, refresh_hz=60.0, net_requests_per_s=0.25, signal_quality=0.9, network_type='wifi', gps_on=False, bt_scan=False, background_wake_per_s=0.2, ambient_temp_C=22.0, oled_content_factor=None), True), ('视频会议_蜂窝_中信号', make_constant_step(dt_s=dt_s, foreground='social', screen_on=True, brightness=0.6, refresh_hz=60.0, net_requests_per_s=0.4, signal_quality=0.55, network_type='cell', gps_on=False, bt_scan=True, background_wake_per_s=0.25, ambient_temp_C=24.0, oled_content_factor=0.95), True)]
    return extended

def write_scenario_summary_csv(out_path: str, results: List[ScenarioResult]) -> None:
    ensure_dir(os.path.dirname(out_path) or '.')
    fieldnames = ['scenario', 't_empty_h', 'terminated_reason', 't_end_h', 'soc_end', 'T_max_C', 'perf_min', 'avg_p_sys_W', 'energy_Wh', 'top1_driver', 'top1_Wh', 'top2_driver', 'top2_Wh', 'top3_driver', 'top3_Wh', 'E_screen_Wh', 'E_compute_Wh', 'E_radio_Wh', 'E_background_Wh', 'E_gps_Wh', 'E_bt_Wh', 'E_coupling_render_Wh', 'E_coupling_radio_bg_Wh', 'E_coupling_thermal_Wh']
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            top = top_drivers(r.energy_breakdown_Wh, top_k=3)
            row = {'scenario': r.name, 't_empty_h': r.t_empty_h, 'terminated_reason': r.terminated_reason, 't_end_h': r.t_end_h, 'soc_end': r.soc_end, 'T_max_C': r.T_max_C, 'perf_min': r.perf_min, 'avg_p_sys_W': r.avg_p_sys_W, 'energy_Wh': r.energy_Wh, 'top1_driver': top[0][0] if len(top) > 0 else '', 'top1_Wh': top[0][1] if len(top) > 0 else 0.0, 'top2_driver': top[1][0] if len(top) > 1 else '', 'top2_Wh': top[1][1] if len(top) > 1 else 0.0, 'top3_driver': top[2][0] if len(top) > 2 else '', 'top3_Wh': top[2][1] if len(top) > 2 else 0.0}
            for k in ['screen', 'compute', 'radio', 'background', 'gps', 'bt', 'coupling_render', 'coupling_radio_bg', 'coupling_thermal']:
                row[f'E_{k}_Wh'] = float(r.energy_breakdown_Wh.get(k, 0.0))
            w.writerow(row)

def plot_time_to_empty(out_png: str, results: List[ScenarioResult]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    configure_cn_font()
    names = [r.name for r in results]
    vals = [r.t_empty_h for r in results]
    fig = plt.figure(figsize=(10, 4.8))
    ax = fig.add_subplot(111)
    ax.bar(range(len(names)), vals, color='#1f77b4', alpha=0.85)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha='right')
    ax.set_ylabel('预计耗尽时间 (小时)')
    ax.set_title('不同场景下的续航差异（同一模型解释）')
    ax.grid(True, axis='y', alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=170)
    plt.close(fig)

def plot_energy_share(out_png: str, results: List[ScenarioResult]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    configure_cn_font()
    parts = [('screen', '屏幕'), ('compute', '计算'), ('radio', '无线'), ('background', '后台'), ('gps', 'GPS'), ('bt', '蓝牙'), ('coupling_render', '耦合:渲染'), ('coupling_radio_bg', '耦合:弱信号×后台'), ('coupling_thermal', '耦合:温度漏损')]
    names = [r.name for r in results]
    totals = [max(1e-09, r.energy_Wh) for r in results]
    fig = plt.figure(figsize=(10, 6.2))
    ax = fig.add_subplot(111)
    bottoms = [0.0 for _ in results]
    xs = list(range(len(results)))
    for key, label in parts:
        vals = [r.energy_breakdown_Wh.get(key, 0.0) / tot for r, tot in zip(results, totals)]
        ax.bar(xs, vals, bottom=bottoms, label=label, alpha=0.88)
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    ax.set_xticks(xs)
    ax.set_xticklabels(names, rotation=20, ha='right')
    ax.set_ylabel('能量占比')
    ax.set_ylim(0.0, 1.0)
    ax.set_title('能量分解占比：识别不同场景的主要耗电驱动')
    ax.legend(ncol=3, fontsize=9, loc='upper right')
    ax.grid(True, axis='y', alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=170)
    plt.close(fig)

@dataclass
class SensitivityCase:
    group: str
    name: str
    apply: Callable[[m.ModelParams, m.ScenarioStep], None]
    delta_label: str

def compute_sensitivity(baseline_t: float, new_t: float, *, delta_frac: float) -> float:
    if not (math.isfinite(baseline_t) and math.isfinite(new_t)):
        return float('nan')
    if baseline_t <= 1e-09:
        return float('nan')
    dy_over_y = (new_t - baseline_t) / baseline_t
    if abs(delta_frac) <= 1e-12:
        return float('nan')
    return dy_over_y / delta_frac

def sensitivity_suite(delta_frac: float) -> Tuple[List[SensitivityCase], List[SensitivityCase]]:

    def mul_batt_capacity(f: float) -> Callable[[m.ModelParams, m.ScenarioStep], None]:

        def _apply(p: m.ModelParams, s: m.ScenarioStep) -> None:
            p.battery.health_capacity_factor = m.clamp(p.battery.health_capacity_factor * f, 0.5, 1.05)
        return _apply

    def mul_batt_resistance(f: float) -> Callable[[m.ModelParams, m.ScenarioStep], None]:

        def _apply(p: m.ModelParams, s: m.ScenarioStep) -> None:
            p.battery.health_resistance_factor = m.clamp(p.battery.health_resistance_factor * f, 0.8, 3.0)
        return _apply

    def mul_cooling(f: float) -> Callable[[m.ModelParams, m.ScenarioStep], None]:

        def _apply(p: m.ModelParams, s: m.ScenarioStep) -> None:
            p.thermal.h_W_per_C = m.clamp(p.thermal.h_W_per_C * f, 0.2, 3.0)
        return _apply

    def mul_sched(f: float) -> Callable[[m.ModelParams, m.ScenarioStep], None]:

        def _apply(p: m.ModelParams, s: m.ScenarioStep) -> None:
            p.os.sched_aggressiveness = m.clamp(p.os.sched_aggressiveness * f, 0.7, 1.4)
        return _apply

    def set_process(node_nm: float) -> Callable[[m.ModelParams, m.ScenarioStep], None]:

        def _apply(p: m.ModelParams, s: m.ScenarioStep) -> None:
            p.soc_arch.process_node_nm = float(node_nm)
        return _apply

    def set_ambient(temp_C: float) -> Callable[[m.ModelParams, m.ScenarioStep], None]:

        def _apply(p: m.ModelParams, s: m.ScenarioStep) -> None:
            s.ambient_temp_C = float(temp_C)
        return _apply

    def mul_brightness(f: float) -> Callable[[m.ModelParams, m.ScenarioStep], None]:

        def _apply(p: m.ModelParams, s: m.ScenarioStep) -> None:
            s.brightness = m.clamp(s.brightness * f, 0.0, 1.0)
            if s.brightness <= 1e-09:
                s.screen_on = False
        return _apply

    def mul_requests(f: float) -> Callable[[m.ModelParams, m.ScenarioStep], None]:

        def _apply(p: m.ModelParams, s: m.ScenarioStep) -> None:
            s.net_requests_per_s = max(0.0, s.net_requests_per_s * f)
        return _apply

    def set_signal(q: float) -> Callable[[m.ModelParams, m.ScenarioStep], None]:

        def _apply(p: m.ModelParams, s: m.ScenarioStep) -> None:
            s.signal_quality = m.clamp(float(q), 0.0, 1.0)
        return _apply

    def toggle_gps(on: bool) -> Callable[[m.ModelParams, m.ScenarioStep], None]:

        def _apply(p: m.ModelParams, s: m.ScenarioStep) -> None:
            s.gps_on = bool(on)
        return _apply
    internal = [SensitivityCase('internal', f'容量健康度 ×(1+{delta_frac:.0%})', mul_batt_capacity(1.0 + delta_frac), f'+{delta_frac:.0%}'), SensitivityCase('internal', f'容量健康度 ×(1-{delta_frac:.0%})', mul_batt_capacity(1.0 - delta_frac), f'-{delta_frac:.0%}'), SensitivityCase('internal', f'内阻健康度 ×(1+{delta_frac:.0%})', mul_batt_resistance(1.0 + delta_frac), f'+{delta_frac:.0%}'), SensitivityCase('internal', f'内阻健康度 ×(1-{delta_frac:.0%})', mul_batt_resistance(1.0 - delta_frac), f'-{delta_frac:.0%}'), SensitivityCase('internal', f'散热系数h ×(1+{delta_frac:.0%})', mul_cooling(1.0 + delta_frac), f'+{delta_frac:.0%}'), SensitivityCase('internal', f'散热系数h ×(1-{delta_frac:.0%})', mul_cooling(1.0 - delta_frac), f'-{delta_frac:.0%}'), SensitivityCase('internal', f'调度激进度 ×(1+{delta_frac:.0%})', mul_sched(1.0 + delta_frac), f'+{delta_frac:.0%}'), SensitivityCase('internal', f'调度激进度 ×(1-{delta_frac:.0%})', mul_sched(1.0 - delta_frac), f'-{delta_frac:.0%}'), SensitivityCase('internal', '制程节点: 4nm→7nm', set_process(7.0), 'set'), SensitivityCase('internal', '制程节点: 4nm→14nm', set_process(14.0), 'set')]
    external = [SensitivityCase('external', '外温: 22℃→35℃', set_ambient(35.0), 'set'), SensitivityCase('external', '外温: 22℃→-5℃', set_ambient(-5.0), 'set'), SensitivityCase('external', f'屏幕亮度 ×(1+{delta_frac:.0%})', mul_brightness(1.0 + delta_frac), f'+{delta_frac:.0%}'), SensitivityCase('external', f'屏幕亮度 ×(1-{delta_frac:.0%})', mul_brightness(1.0 - delta_frac), f'-{delta_frac:.0%}'), SensitivityCase('external', f'网络请求率 ×(1+{delta_frac:.0%})', mul_requests(1.0 + delta_frac), f'+{delta_frac:.0%}'), SensitivityCase('external', f'网络请求率 ×(1-{delta_frac:.0%})', mul_requests(1.0 - delta_frac), f'-{delta_frac:.0%}'), SensitivityCase('external', '信号质量: 0.9→0.3', set_signal(0.3), 'set'), SensitivityCase('external', '信号质量: 0.9→1.0', set_signal(1.0), 'set'), SensitivityCase('external', 'GPS: OFF→ON', toggle_gps(True), 'set'), SensitivityCase('external', 'GPS: ON→OFF', toggle_gps(False), 'set')]
    return (internal, external)

def sensitivity_suite_quick(delta_frac: float) -> Tuple[List[SensitivityCase], List[SensitivityCase]]:
    internal, external = sensitivity_suite(delta_frac)
    internal_keep = {'容量健康度 ×(1-10%)', '内阻健康度 ×(1+10%)', '散热系数h ×(1-10%)', '调度激进度 ×(1+10%)', '制程节点: 4nm→7nm'}
    external_keep = {'外温: 22℃→35℃', '外温: 22℃→-5℃', '屏幕亮度 ×(1+10%)', '网络请求率 ×(1+10%)', '信号质量: 0.9→0.3', 'GPS: OFF→ON'}
    internal2 = [c for c in internal if c.name in internal_keep]
    external2 = [c for c in external if c.name in external_keep]
    return (internal2, external2)

def write_sensitivity_csv(path: str, rows: List[Dict[str, object]]) -> None:
    ensure_dir(os.path.dirname(path) or '.')
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})

def plot_sensitivity_bar(out_png: str, title: str, rows: List[Dict[str, object]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    configure_cn_font()
    rows_sorted = sorted(rows, key=lambda r: abs(float(r.get('S', 0.0) or 0.0)), reverse=True)
    rows_sorted = rows_sorted[:14]
    labels = [str(r['case']) for r in rows_sorted]
    vals = [float(r.get('S', float('nan'))) for r in rows_sorted]
    fig = plt.figure(figsize=(10, 5.8))
    ax = fig.add_subplot(111)
    colors = ['#d62728' if v < 0 else '#2ca02c' for v in vals]
    ax.barh(range(len(labels)), vals, color=colors, alpha=0.85)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.axvline(0.0, color='black', linewidth=1.0)
    ax.set_xlabel('一阶灵敏度 S = (Δ续航/续航)/(Δ参数/参数)')
    ax.set_title(title)
    ax.grid(True, axis='x', alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=170)
    plt.close(fig)

def main() -> None:
    ap = argparse.ArgumentParser(description='第一问模型：场景对比 + 驱动因素归因 + 灵敏度分析')
    ap.add_argument('--mode', type=str, default='all', choices=['compare', 'sensitivity', 'all'], help='运行模式')
    ap.add_argument('--outdir', type=str, default='analysis_results', help='输出目录（默认会自动追加日期避免覆盖；想固定目录可显式传入）')
    ap.add_argument('--dt', type=float, default=2.0, help='仿真步长 dt（秒）；灵敏度分析建议 >=2 以加速')
    ap.add_argument('--max-hours', type=float, default=120.0, help='最长仿真时长上限（小时），避免无限跑')
    ap.add_argument('--delta', type=float, default=0.1, help='OAT 扰动比例（内部/外部连续参数）')
    ap.add_argument('--baseline', type=str, default='短视频_室温', help='灵敏度分析的基准场景名称')
    ap.add_argument('--suite', type=str, default='extended', choices=['core', 'extended'], help='场景库规模：core(少量代表性)/extended(更丰富的用户行为场景)')
    ap.add_argument('--no-plots', action='store_true', help='只输出 CSV，不生成 PNG 图（更快）')
    ap.add_argument('--quick', action='store_true', help='快速模式：减少灵敏度用例数量')
    ap.add_argument('--no-per-scenario-figures', action='store_true', help='关闭逐场景导出（默认开启：每个对比场景都会输出 sim_output.csv + figures/）')
    ap.add_argument('--per-scenario-dpi', type=int, default=170, help='逐场景 figures 的 DPI')
    args = ap.parse_args()
    if str(args.outdir) == 'analysis_results':
        args.outdir = make_dated_outdir('analysis_results')
    args.per_scenario_figures = not bool(getattr(args, 'no_per_scenario_figures', False))
    ensure_dir(args.outdir)
    dt_s = max(0.2, float(args.dt))
    max_hours = max(0.1, float(args.max_hours))
    suite = scenario_suite(dt_s, suite=str(args.suite))
    if args.mode in {'compare', 'all'}:
        results: List[ScenarioResult] = []
        per_payload: List[Tuple[str, m.ModelParams, m.ScenarioStep, List[Dict[str, float]], ScenarioResult]] = []
        for name, step, until_empty in suite:
            print(f'[场景] 开始: {name}')
            params = m.ModelParams()
            steps = repeat_steps(step, hours=1.0)
            r, rows, _sim = run_scenario_with_rows(name, params, steps, extend_until_empty=until_empty, idle_step=step, max_hours=max_hours)
            results.append(r)
            print(f'[场景] 完成: {name} | t_empty_h={r.t_empty_h:.2f} | reason={r.terminated_reason}')
            if args.per_scenario_figures and (not args.no_plots):
                per_payload.append((str(name), params, step, rows, r))
        if args.per_scenario_figures and (not args.no_plots) and per_payload:
            finite_ts = [x[4].t_empty_h for x in per_payload if math.isfinite(x[4].t_empty_h)]
            xlim_h = max(finite_ts) if finite_ts else None
            for scenario_name, params, step, rows, r in per_payload:
                export_per_scenario_outputs(outdir=str(args.outdir), scenario_name=str(scenario_name), params=params, step=step, rows=rows, result=r, dpi=int(args.per_scenario_dpi), xlim_h=xlim_h)
        out_csv = os.path.join(args.outdir, 'scenario_summary.csv')
        write_scenario_summary_csv(out_csv, results)
        if not args.no_plots:
            plot_time_to_empty(os.path.join(args.outdir, '01_time_to_empty.png'), results)
            plot_energy_share(os.path.join(args.outdir, '02_energy_share.png'), results)
        print(f'[场景对比] 已输出: {out_csv}')
    if args.mode in {'sensitivity', 'all'}:
        baseline_tuple = next((x for x in suite if x[0] == str(args.baseline)), None)
        if baseline_tuple is None:
            raise SystemExit(f'未找到 baseline 场景: {args.baseline}。可选: {[x[0] for x in suite]}')
        baseline_name, baseline_step, _ = baseline_tuple
        base_params = m.ModelParams()
        base_steps = repeat_steps(baseline_step, hours=1.0)
        base_res = run_scenario(f'baseline:{baseline_name}', base_params, base_steps, extend_until_empty=True, idle_step=baseline_step, max_hours=max_hours)
        base_t = base_res.t_empty_h
        if not math.isfinite(base_t):
            print('[警告] 基准场景在 max-hours 内未耗尽，灵敏度结果可能缺失；可尝试调大 --max-hours')
        if args.quick:
            internal_cases, external_cases = sensitivity_suite_quick(delta_frac=max(0.01, float(args.delta)))
        else:
            internal_cases, external_cases = sensitivity_suite(delta_frac=max(0.01, float(args.delta)))

        def eval_case(case: SensitivityCase) -> Dict[str, object]:
            params = m.ModelParams()
            step = make_constant_step(dt_s=baseline_step.dt_s, foreground=baseline_step.foreground, screen_on=baseline_step.screen_on, brightness=baseline_step.brightness, refresh_hz=baseline_step.refresh_hz, net_requests_per_s=baseline_step.net_requests_per_s, signal_quality=baseline_step.signal_quality, network_type=baseline_step.network_type, gps_on=baseline_step.gps_on, bt_scan=baseline_step.bt_scan, background_wake_per_s=baseline_step.background_wake_per_s, ambient_temp_C=baseline_step.ambient_temp_C, oled_content_factor=baseline_step.oled_content_factor)
            case.apply(params, step)
            print(f'[灵敏度] 开始: {case.group} | {case.name}')
            res = run_scenario(case.name, params, repeat_steps(step, hours=1.0), extend_until_empty=True, idle_step=step, max_hours=max_hours)
            print(f'[灵敏度] 完成: {case.group} | {case.name} | t_empty_h={res.t_empty_h:.2f} | reason={res.terminated_reason}')
            if case.delta_label.startswith('+') or case.delta_label.startswith('-'):
                delta = float(args.delta) if case.delta_label.startswith('+') else -float(args.delta)
            else:
                delta = 1.0
            S = compute_sensitivity(base_t, res.t_empty_h, delta_frac=delta)
            return {'group': case.group, 'case': case.name, 'delta': case.delta_label, 't_empty_h': res.t_empty_h, 'S': S, 'terminated_reason': res.terminated_reason}
        internal_rows = [eval_case(c) for c in internal_cases]
        external_rows = [eval_case(c) for c in external_cases]
        out_internal = os.path.join(args.outdir, 'sensitivity_internal.csv')
        out_external = os.path.join(args.outdir, 'sensitivity_external.csv')
        write_sensitivity_csv(out_internal, internal_rows)
        write_sensitivity_csv(out_external, external_rows)
        if not args.no_plots:
            plot_sensitivity_bar(os.path.join(args.outdir, '03_sensitivity_internal.png'), f'内部条件灵敏度（基准：{baseline_name}）', internal_rows)
            plot_sensitivity_bar(os.path.join(args.outdir, '04_sensitivity_external.png'), f'外部条件/行为灵敏度（基准：{baseline_name}）', external_rows)
        print(f'[灵敏度分析] 已输出: {out_internal}')
        print(f'[灵敏度分析] 已输出: {out_external}')
if __name__ == '__main__':
    main()

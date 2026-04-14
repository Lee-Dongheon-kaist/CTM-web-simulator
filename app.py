import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.lines import Line2D

BASE_DIR = Path(__file__).resolve().parent
CTM_PATH = BASE_DIR / "code.py"

spec = importlib.util.spec_from_file_location("ctm_core", CTM_PATH)
ctm_core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ctm_core)

run_ctm = ctm_core.run_ctm
DT = ctm_core.DT
VF = ctm_core.VF
ALPHA = ctm_core.ALPHA
QMAX_PER_LANE = ctm_core.QMAX_PER_LANE
KJ_PER_LANE = ctm_core.KJ_PER_LANE
W = ALPHA * VF
QK_K_MAX = 4.5 * KJ_PER_LANE
QK_Q_MAX = 4.5 * QMAX_PER_LANE
QK_LAYOUT_RECT = [0.0, 0.03, 1.0, 0.96]
N_MAIN_DEFAULT = ctm_core.N_MAIN
DIVERGE_IDX_DEFAULT = ctm_core.DIVERGE_IDX
MERGE_TOP_IDX_DEFAULT = ctm_core.MERGE_TOP_IDX
MERGE_BOTTOM_IDX_DEFAULT = ctm_core.MERGE_BOTTOM_IDX
MAIN_INFLOW_MAX = int(round(ctm_core.QMAX_MAIN))
ONRAMP_INFLOW_MAX = int(round(ctm_core.QMAX_RAMP))
N_ON_SIMPLIFIED = 1
OFF_PEAK_DEFAULT = dict(ctm_core.OFF_PEAK)
PEAK_DEFAULT = dict(ctm_core.PEAK_INCIDENT)


st.set_page_config(page_title="CTM Web Simulator", layout="wide")
st.title("Cell Transmission Model Web Simulator")

# ---------------------------------------------------------
# Input controls panel
# ---------------------------------------------------------
with st.expander("Input Parameters", expanded=True):
    st.header("Input Parameters")
    st.caption(
        f"Demand max auto-set by lane count x q_max: "
        f"main={MAIN_INFLOW_MAX} veh/hr, on-ramp={ONRAMP_INFLOW_MAX} veh/hr"
    )

    T_hours = st.slider("Simulation time (hr)", min_value=1.0, max_value=24.0, value=6.0, step=0.5)
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.subheader("Off-peak Demand")
        offpeak_main_inflow = st.slider(
            "Off-peak main inflow (veh/hr)",
            min_value=0,
            max_value=MAIN_INFLOW_MAX,
            value=min(int(OFF_PEAK_DEFAULT["main_inflow"]), MAIN_INFLOW_MAX),
            step=100,
        )
        offpeak_onramp_top_inflow = st.slider(
            "Off-peak upper on-ramp inflow (veh/hr)",
            min_value=0,
            max_value=ONRAMP_INFLOW_MAX,
            value=min(int(OFF_PEAK_DEFAULT.get("onramp_top_inflow", OFF_PEAK_DEFAULT.get("onramp_inflow", 500))), ONRAMP_INFLOW_MAX),
            step=50,
        )
        offpeak_onramp_bottom_inflow = st.slider(
            "Off-peak lower on-ramp inflow (veh/hr)",
            min_value=0,
            max_value=ONRAMP_INFLOW_MAX,
            value=min(int(OFF_PEAK_DEFAULT.get("onramp_bottom_inflow", OFF_PEAK_DEFAULT.get("onramp_inflow", 800))), ONRAMP_INFLOW_MAX),
            step=50,
        )

        st.subheader("Peak Demand")
        peak_main_inflow = st.slider(
            "Peak main inflow (veh/hr)",
            min_value=0,
            max_value=MAIN_INFLOW_MAX,
            value=min(int(PEAK_DEFAULT["main_inflow"]), MAIN_INFLOW_MAX),
            step=100,
        )
        peak_onramp_top_inflow = st.slider(
            "Peak upper on-ramp inflow (veh/hr)",
            min_value=0,
            max_value=ONRAMP_INFLOW_MAX,
            value=min(int(PEAK_DEFAULT.get("onramp_top_inflow", PEAK_DEFAULT.get("onramp_inflow", 800))), ONRAMP_INFLOW_MAX),
            step=50,
        )
        peak_onramp_bottom_inflow = st.slider(
            "Peak lower on-ramp inflow (veh/hr)",
            min_value=0,
            max_value=ONRAMP_INFLOW_MAX,
            value=min(int(PEAK_DEFAULT.get("onramp_bottom_inflow", PEAK_DEFAULT.get("onramp_inflow", 1200))), ONRAMP_INFLOW_MAX),
            step=50,
        )

    with col2:
        st.subheader("Merge / Diverge Ratio")
        beta_off = st.slider(
            "Off-ramp split ratio β",
            min_value=0.00,
            max_value=0.80,
            value=float(OFF_PEAK_DEFAULT["beta_off"]),
            step=0.01,
        )

        p_main = st.slider("Mainline priority", min_value=0.0, max_value=1.0, value=float(OFF_PEAK_DEFAULT["p_main"]), step=0.05)
        p_on = round(1.0 - p_main, 2)
        st.write(f"On-ramp priority: **{p_on:.2f}**")

    with col3:
        st.subheader("Network Structure & Initial State")

        n_main_cells = st.slider("Mainline cells", min_value=8, max_value=30, value=int(N_MAIN_DEFAULT), step=1)

        # scale default indices with number of cells
        default_diverge = min(n_main_cells - 2, max(0, round(DIVERGE_IDX_DEFAULT / (N_MAIN_DEFAULT - 1) * (n_main_cells - 1))))
        default_merge_top = min(n_main_cells - 2, max(1, round(MERGE_TOP_IDX_DEFAULT / (N_MAIN_DEFAULT - 1) * (n_main_cells - 1))))
        default_merge_bottom = min(n_main_cells - 1, max(2, round(MERGE_BOTTOM_IDX_DEFAULT / (N_MAIN_DEFAULT - 1) * (n_main_cells - 1))))
        if default_merge_top - 1 == default_diverge:
            default_merge_top = min(n_main_cells - 1, default_merge_top + 1)
        if default_merge_bottom - 1 == default_diverge:
            default_merge_bottom = min(n_main_cells - 1, default_merge_bottom + 1)
        if default_merge_bottom == default_merge_top:
            default_merge_bottom = max(1, min(n_main_cells - 1, default_merge_bottom - 1))

        diverge_idx = st.slider("Off-ramp diverge index", min_value=0, max_value=n_main_cells - 2, value=int(default_diverge), step=1)
        top_merge_default_safe = max(1, min(n_main_cells - 1, default_merge_top))
        merge_top_idx = st.slider(
            "Upper on-ramp merge index",
            min_value=1,
            max_value=n_main_cells - 1,
            value=int(top_merge_default_safe),
            step=1,
        )
        bottom_merge_default_safe = max(1, min(n_main_cells - 1, default_merge_bottom))
        merge_bottom_idx = st.slider(
            "Lower on-ramp merge index",
            min_value=1,
            max_value=n_main_cells - 1,
            value=int(bottom_merge_default_safe),
            step=1,
        )

        initial_main_density = st.slider("Initial mainline density (veh/km/lane)", min_value=0.0, max_value=100.0, value=10.0, step=5.0)

    with col4:
        st.subheader("Incident")
        incident_cell = st.slider("Incident cell", min_value=0, max_value=n_main_cells - 1, value=min(6, n_main_cells - 1), step=1)
        incident_start_hour = st.slider("Incident start time (hr)", min_value=0.0, max_value=T_hours, value=min(1.0, T_hours), step=DT)
        max_duration = max(DT, T_hours - incident_start_hour)
        incident_duration_hour = st.slider("Incident duration (hr)", min_value=DT, max_value=max_duration, value=min(1.0, max_duration), step=DT)
        blocked_lanes = st.slider("Blocked lanes", min_value=1, max_value=3, value=1, step=1)

run_clicked = st.button("Run 4-case comparison", type="primary", width="stretch")

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def make_line_plot(time, series_dict, title, ylabel):
    fig, ax = plt.subplots(figsize=(10, 4))
    for label, values in series_dict.items():
        ax.plot(time, values, label=label)
    ax.set_xlabel("Time (hr)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def make_detector_flow_by_scenario_grid(results_by_case, middle_detector_cell_idx=None):
    detector_specs = [
        ("Main upstream", "inflow_main", "#9ecae1"),
        ("Off-ramp", "outflow_off", "#fdd0a2"),
        ("Upper on-ramp", "inflow_on_top", "#a1d99b"),
        ("Lower on-ramp", "inflow_on_bottom", "#fcbba1"),
        ("Main downstream", "outflow_main", "#cbc9e2"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)
    axes_flat = axes.ravel()
    time = next(iter(results_by_case.values()))["time"]
    n_cells = next(iter(results_by_case.values()))["flow_main_cell"].shape[1]
    if middle_detector_cell_idx is None:
        middle_detector_cell_idx = n_cells // 2
    middle_detector_cell_idx = int(max(0, min(n_cells - 1, middle_detector_cell_idx)))

    for ax, (scenario_label, res) in zip(axes_flat, results_by_case.items()):
        for detector_label, detector_key, color in detector_specs:
            ax.plot(time, res[detector_key], label=detector_label, color=color, linewidth=1.6, alpha=0.55)

        # Configurable middle detector from mainline cell flow.
        ax.plot(
            time,
            res["flow_main_cell"][:, middle_detector_cell_idx],
            label=f"Main middle (cell {middle_detector_cell_idx})",
            color="#d60000",
            linewidth=2.6,
            alpha=0.95,
        )

        ax.set_title(scenario_label)
        ax.set_xlabel("Time (hr)")
        ax.set_ylabel("Flow (veh/hr)")
        ax.set_ylim(0.0, 8000.0)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    fig.suptitle("Detector flow comparison by scenario")
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    return fig


def make_three_lane_cumulative_curve_grid(results_by_case, background_flow_by_case=None):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)
    axes_flat = axes.ravel()

    for ax, (scenario_label, res) in zip(axes_flat, results_by_case.items()):
        time = res["time"]
        lanes = res["main_lanes_by_cell"]
        f_link = res["f_main_link"]
        n_cells = len(lanes)

        lane3_cells = np.where(np.isclose(lanes, 3.0))[0]
        if lane3_cells.size == 0:
            ax.text(0.5, 0.5, "No 3-lane cells in this scenario", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(scenario_label)
            ax.set_xlabel("Time (hr)")
            ax.set_ylabel("Cumulative count (veh)")
            ax.grid(True, alpha=0.3)
            continue

        upstream_boundary_links = [
            i - 1 for i in lane3_cells
            if i > 0 and (not np.isclose(lanes[i - 1], 3.0))
        ]
        downstream_boundary_links = [
            i for i in lane3_cells
            if i < n_cells - 1 and (not np.isclose(lanes[i + 1], 3.0))
        ]

        q_up = np.zeros_like(time, dtype=float)
        q_down = np.zeros_like(time, dtype=float)
        if len(upstream_boundary_links) > 0:
            q_up = np.sum(f_link[:, upstream_boundary_links], axis=1)
        if len(downstream_boundary_links) > 0:
            q_down = np.sum(f_link[:, downstream_boundary_links], axis=1)

        # Use edge-aligned cumulative counts so curves start at exactly 0 at t=t0.
        t_edge = np.concatenate(([float(time[0])], time + DT))
        n_up = np.concatenate(([0.0], np.cumsum(q_up * DT)))
        n_down = np.concatenate(([0.0], np.cumsum(q_down * DT)))

        # Detector-specific time alignment:
        # subtract free-flow travel time from mainline origin to each detector.
        up_det_link = int(min(upstream_boundary_links)) if len(upstream_boundary_links) > 0 else 0
        down_det_link = int(max(downstream_boundary_links)) if len(downstream_boundary_links) > 0 else max(0, n_cells - 2)
        t_up_shift_ff = ((up_det_link + 1) * ctm_core.DX) / max(VF, 1e-9)
        t_down_shift_ff = ((down_det_link + 1) * ctm_core.DX) / max(VF, 1e-9)

        t_up_edge = t_edge - t_up_shift_ff
        t_down_edge = t_edge - t_down_shift_ff

        # Determine background flow q0.
        if background_flow_by_case is not None and scenario_label in background_flow_by_case:
            q0 = float(background_flow_by_case[scenario_label])
        else:
            n_ref_steps = max(3, int(0.15 * len(time)))
            q0 = float(np.mean(q_up[:n_ref_steps])) if len(time) > 0 else 0.0

        # Fixed detector-based t0 logic (user-requested):
        # upstream t0 step = upstream detector link index + 2
        # downstream t0 step = downstream detector link index + 2
        # Example with diverge_idx=1 and merge_idx=9:
        # up_link=1 -> t0 step=3, down_link=8 -> t0 step=10.
        i0_up = int(up_det_link + 2)
        i0_down = int(down_det_link + 2)
        i0_up = max(0, min(i0_up, len(t_up_edge) - 1))
        i0_down = max(0, min(i0_down, len(t_down_edge) - 1))

        t0_up = float(t_up_edge[i0_up])
        t0_down = float(t_down_edge[i0_down])

        tau_up_plot = t_up_edge[i0_up:] - t0_up
        tau_down_plot = t_down_edge[i0_down:] - t0_down

        n_up_reacc = n_up[i0_up:] - float(n_up[i0_up])
        n_down_reacc = n_down[i0_down:] - float(n_down[i0_down])

        n_up_plot = n_up_reacc - q0 * tau_up_plot
        n_down_plot = n_down_reacc - q0 * tau_down_plot

        # Keep plotting from shifted origin only.
        keep_up = tau_up_plot >= 0.0
        keep_down = tau_down_plot >= 0.0
        tau_up_plot = tau_up_plot[keep_up]
        tau_down_plot = tau_down_plot[keep_down]
        n_up_plot = n_up_plot[keep_up]
        n_down_plot = n_down_plot[keep_down]

        if tau_up_plot.size == 0 or tau_up_plot[0] > 1e-12:
            tau_up_plot = np.insert(tau_up_plot, 0, 0.0)
            n_up_plot = np.insert(n_up_plot, 0, 0.0)
        if tau_down_plot.size == 0 or tau_down_plot[0] > 1e-12:
            tau_down_plot = np.insert(tau_down_plot, 0, 0.0)
            n_down_plot = np.insert(n_down_plot, 0, 0.0)

        ax.plot(tau_up_plot, n_up_plot, color="#1f77b4", linewidth=2.0, label="Upstream Newell curve")
        ax.plot(tau_down_plot, n_down_plot, color="#d62728", linewidth=2.0, label="Downstream Newell curve")
        ax.axhline(0.0, color="#808080", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.axvline(0.0, color="#808080", linewidth=0.8, linestyle="--", alpha=0.6)

        ax.set_title(scenario_label)
        ax.set_xlabel("Shifted time from origin (hr)")
        ax.set_ylabel("N(t)-q0(t-t0) (veh)")
        ax.set_xlim(left=0.0)
        ax.grid(True, alpha=0.3)
        ax.legend(
            title=(
                f"q0={q0:.0f} veh/hr\n"
                f"t_up shift={t_up_shift_ff:.2f} hr\n"
                f"t_down shift={t_down_shift_ff:.2f} hr"
            ),
            loc="lower left",
            fontsize=8,
            framealpha=0.9,
        )
    fig.suptitle("3-lane cells: Newell N-curve from detector-based t0 timestep")
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    return fig


def _inverse_cumulative_time(cumulative_count, time_axis_end, targets):
    cum = np.maximum.accumulate(np.asarray(cumulative_count, dtype=float))
    t = np.asarray(time_axis_end, dtype=float)
    if cum.size == 0:
        return np.zeros_like(targets, dtype=float)

    keep = np.r_[True, np.diff(cum) > 1e-9]
    x = cum[keep]
    y = t[keep]
    if x.size == 1:
        return np.full_like(targets, y[0], dtype=float)

    return np.interp(targets, x, y, left=y[0], right=y[-1])


def compute_three_lane_segment_total_delay_veh_hr(res):
    time = res["time"]
    lanes = res["main_lanes_by_cell"]
    f_link = res["f_main_link"]
    n_cells = len(lanes)

    lane3_cells = np.where(np.isclose(lanes, 3.0))[0]
    if lane3_cells.size == 0 or len(time) == 0:
        return 0.0, 0, 0, 0

    # Split possibly disconnected 3-lane cells into contiguous segments.
    segments = []
    seg_start = int(lane3_cells[0])
    seg_prev = int(lane3_cells[0])
    for c in lane3_cells[1:]:
        c = int(c)
        if c == seg_prev + 1:
            seg_prev = c
            continue
        segments.append((seg_start, seg_prev))
        seg_start = c
        seg_prev = c
    segments.append((seg_start, seg_prev))

    total_delay_veh_hr = 0.0
    n_up_total = 0
    n_down_total = 0
    n_unmatched = 0

    # Approximate delay from vertical gap with the same alignment logic used in plots:
    # 1) detector-specific free-flow travel-time shift,
    # 2) detector-based fixed t0 timestep (link index + 2),
    # 3) sum_t max(0, N_up - N_down) * DT.
    for seg_start, seg_end in segments:
        up_link = seg_start - 1
        down_link = seg_end

        # Use only main->main boundary links.
        if up_link < 0 or down_link >= (n_cells - 1):
            continue

        q_up = f_link[:, up_link]
        q_down = f_link[:, down_link]

        n_ref_steps = max(3, int(0.15 * len(time)))
        q0 = float(np.mean(q_up[:n_ref_steps])) if len(time) > 0 else 0.0

        n_up = np.concatenate(([0.0], np.cumsum(q_up * DT)))
        n_down = np.concatenate(([0.0], np.cumsum(q_down * DT)))
        t_edge = np.concatenate(([float(time[0])], time + DT))
        seg_up_total = int(np.floor(n_up[-1] + 1e-9))
        seg_down_total = int(np.floor(n_down[-1] + 1e-9))

        n_up_total += seg_up_total
        n_down_total += seg_down_total

        if seg_up_total <= 0:
            continue

        t_up_shift_ff = ((up_link + 1) * ctm_core.DX) / max(VF, 1e-9)
        t_down_shift_ff = ((down_link + 1) * ctm_core.DX) / max(VF, 1e-9)
        t_up_edge = t_edge - t_up_shift_ff
        t_down_edge = t_edge - t_down_shift_ff

        i0_up = int(up_link + 2)
        i0_down = int(down_link + 2)
        i0_up = max(0, min(i0_up, len(t_up_edge) - 1))
        i0_down = max(0, min(i0_down, len(t_down_edge) - 1))

        t0_up = float(t_up_edge[i0_up])
        t0_down = float(t_down_edge[i0_down])

        tau_up = t_up_edge[i0_up:] - t0_up
        tau_down = t_down_edge[i0_down:] - t0_down

        n_up_reacc = n_up[i0_up:] - float(n_up[i0_up])
        n_down_reacc = n_down[i0_down:] - float(n_down[i0_down])

        n_up_shift = n_up_reacc - q0 * tau_up
        n_down_shift = n_down_reacc - q0 * tau_down

        tau_end = float(min(np.max(tau_up), np.max(tau_down)))
        if tau_end <= 0.0:
            continue

        tau_grid = np.arange(0.0, tau_end + 1e-12, DT)
        n_up_grid = np.interp(tau_grid, tau_up, n_up_shift, left=n_up_shift[0], right=n_up_shift[-1])
        n_down_grid = np.interp(tau_grid, tau_down, n_down_shift, left=n_down_shift[0], right=n_down_shift[-1])

        vertical_gap = np.maximum(0.0, n_up_grid - n_down_grid)
        total_delay_veh_hr += float(np.sum(vertical_gap) * DT)
        n_unmatched += int(max(0, seg_up_total - seg_down_total))

    return total_delay_veh_hr, n_up_total, n_down_total, n_unmatched


def format_hr_m(hours):
    if hours is None:
        return "-"
    total_minutes = int(round(float(hours) * 60.0))
    sign = "-" if total_minutes < 0 else ""
    total_minutes = abs(total_minutes)
    hh, mm = divmod(total_minutes, 60)
    return f"{sign}{hh} hr {mm} min"


def compute_network_weighted_series(res):
    n_main = res["n_main"][:-1]
    speed_main = res["speed_main"]
    time = res["time"]
    total_veh = np.sum(n_main, axis=1)

    weighted_speed = np.divide(
        np.sum(speed_main * n_main, axis=1),
        total_veh,
        out=np.full_like(total_veh, VF, dtype=float),
        where=total_veh > 1e-9,
    )

    network_length_km = speed_main.shape[1] * ctm_core.DX
    network_density = total_veh / max(1e-9, network_length_km)
    network_flow = network_density * weighted_speed

    return {
        "time": time,
        "speed": weighted_speed,
        "density": network_density,
        "flow": network_flow,
    }


def make_network_qk_grid(network_series_by_case):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)
    axes_flat = axes.ravel()

    for ax, (label, series) in zip(axes_flat, network_series_by_case.items()):
        k = series["density"]
        q = series["flow"]
        t = series["time"]

        ax.plot(k, q, color="#2d3748", linewidth=0.9, alpha=0.55)
        sc = ax.scatter(k, q, c=t, cmap="viridis", s=20, alpha=0.9)

        ax.set_title(label)
        ax.set_xlabel("Network density k (veh/km)")
        ax.set_ylabel("Network flow q (veh/hr)")
        ax.set_xlim(0.0, QK_K_MAX)
        ax.set_ylim(0.0, QK_Q_MAX)
        ax.grid(True, alpha=0.25)

        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label("Time (hr)")

    fig.suptitle("Network-level q-k diagram by timestep (4 scenarios)")
    fig.tight_layout(rect=QK_LAYOUT_RECT)
    return fig


def make_cell_qk_last_grid(results_by_case, analysis_step_idx=None, analysis_time_hr=None):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)
    axes_flat = axes.ravel()

    k_lane = np.linspace(0.0, KJ_PER_LANE, 400)
    q_lane = np.minimum(np.minimum(VF * k_lane, QMAX_PER_LANE), W * (KJ_PER_LANE - k_lane))
    k_fd_3 = 3.0 * k_lane
    q_fd_3 = 3.0 * q_lane
    k_fd_4 = 4.0 * k_lane
    q_fd_4 = 4.0 * q_lane

    for idx, (ax, (label, res)) in enumerate(zip(axes_flat, results_by_case.items())):
        step_idx = -1 if analysis_step_idx is None else int(np.clip(analysis_step_idx, 0, res["density_main"].shape[0] - 1))
        lanes = res["main_lanes_by_cell"]
        k_cell_last = res["density_main"][step_idx] * lanes
        q_cell_last = res["flow_main_cell"][step_idx]
        k_mean = float(np.mean(k_cell_last))
        q_mean = float(np.mean(q_cell_last))

        ax.plot(k_fd_3, q_fd_3, linestyle="--", color="#4a90e2", linewidth=1.4, label="FD x3")
        ax.plot(k_fd_4, q_fd_4, linestyle="--", color="#0057b8", linewidth=1.6, label="FD x4")
        ax.scatter(k_cell_last, q_cell_last, s=30, color="#2d3748", alpha=0.9)
        ax.annotate(
            "",
            xy=(k_mean, q_mean),
            xytext=(0.0, 0.0),
            arrowprops=dict(arrowstyle="->", color="red", lw=2.0),
        )
        ax.scatter(k_mean, q_mean, s=70, color="red", marker="o", zorder=5, label="Mean vector")
        ax.text(k_mean, q_mean, " mean", color="red", fontsize=8, fontweight="bold")

        ax.set_title(label)
        ax.set_xlabel("Cell density k (veh/km)")
        ax.set_ylabel("Cell flow q (veh/hr)")
        ax.set_xlim(0.0, QK_K_MAX)
        ax.set_ylim(0.0, QK_Q_MAX)
        ax.grid(True, alpha=0.25)

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc="upper right", fontsize=8, framealpha=0.9)

    if analysis_time_hr is None:
        fig.suptitle("Cell-level q-k at selected timestep (with FD x3/x4 background)")
    else:
        fig.suptitle(f"Cell-level q-k at selected timestep (t={format_hr_m(analysis_time_hr)}, with FD x3/x4 background)")
    fig.tight_layout(rect=QK_LAYOUT_RECT)
    return fig


def make_qk_overlay_comparison_grid(network_series_by_case, results_by_case, analysis_step_idx=None, analysis_time_hr=None):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)
    axes_flat = axes.ravel()

    for ax, (label, res) in zip(axes_flat, results_by_case.items()):
        series = network_series_by_case[label]
        k_net = series["density"]
        q_net = series["flow"]
        t_net = series["time"]

        step_idx = len(k_net) - 1 if analysis_step_idx is None else int(np.clip(analysis_step_idx, 0, len(k_net) - 1))
        lanes = res["main_lanes_by_cell"]
        k_cell_last = res["density_main"][step_idx] * lanes
        q_cell_last = res["flow_main_cell"][step_idx]
        k_mean = float(np.mean(k_cell_last))
        q_mean = float(np.mean(q_cell_last))

        sc = ax.scatter(k_net, q_net, c=t_net, cmap="viridis", s=18, alpha=0.85, label="Network q-k points")
        ax.scatter(
            [k_net[step_idx]],
            [q_net[step_idx]],
            s=110,
            facecolors="none",
            edgecolors="#f59e0b",
            linewidths=2.2,
            zorder=6,
            label="Selected timestep",
        )
        ax.annotate(
            "",
            xy=(k_mean, q_mean),
            xytext=(0.0, 0.0),
            arrowprops=dict(arrowstyle="->", color="red", lw=2.0),
        )
        ax.set_title(label)
        ax.set_xlabel("k (veh/km)")
        ax.set_ylabel("q (veh/hr)")
        ax.set_xlim(0.0, QK_K_MAX)
        ax.set_ylim(0.0, QK_Q_MAX)
        ax.grid(True, alpha=0.25)

        handles, labels = ax.get_legend_handles_labels()
        uniq = dict(zip(labels, handles))
        uniq["Red arrow: origin -> mean cell q-k"] = Line2D([0], [0], color="red", lw=2.0)
        ax.legend(
            uniq.values(),
            uniq.keys(),
            loc="upper right",
            fontsize=8,
            framealpha=0.9,
        )

        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label("Time (hr)")

    if analysis_time_hr is None:
        fig.suptitle("Overlay: network q-k points and cell q-k mean vector")
    else:
        fig.suptitle(f"Overlay: network q-k points and cell q-k mean vector (t={format_hr_m(analysis_time_hr)})")
    fig.tight_layout(rect=QK_LAYOUT_RECT)
    return fig


def make_case_heatmap_grid(
    results_by_case,
    key,
    title,
    cmap,
    vmin=None,
    vmax=None,
    cbar_label="",
    incident_case_flags=None,
    incident_start_hour=None,
    incident_duration_hour=None,
    incident_cell=None,
):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)
    axes_flat = axes.ravel()
    im = None

    for ax, (label, res) in zip(axes_flat, results_by_case.items()):
        data = res[key]
        im = ax.imshow(
            data.T,
            origin="lower",
            aspect="auto",
            extent=[0, data.shape[0] * DT, 0, data.shape[1]],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(label)
        ax.set_xlabel("Time (hr)")
        ax.set_ylabel("Mainline cell")

        if (
            incident_case_flags is not None
            and incident_case_flags.get(label, False)
            and incident_start_hour is not None
            and incident_duration_hour is not None
            and incident_cell is not None
        ):
            rect = plt.Rectangle(
                (incident_start_hour, incident_cell),
                incident_duration_hour,
                1.0,
                fill=False,
                edgecolor="yellow",
                linewidth=2.2,
            )
            ax.add_patch(rect)
            ax.text(
                incident_start_hour,
                incident_cell + 1.1,
                "incident",
                color="yellow",
                fontsize=8,
                fontweight="bold",
            )

    if im is not None:
        # Reserve space on the right and place a dedicated colorbar axis outside plots.
        fig.subplots_adjust(right=0.88)
        cax = fig.add_axes([0.90, 0.15, 0.02, 0.70])
        cbar = fig.colorbar(im, cax=cax)
        if cbar_label:
            cbar.set_label(cbar_label)

    fig.suptitle(title)
    fig.subplots_adjust(left=0.07, bottom=0.08, top=0.90, right=0.88, wspace=0.18, hspace=0.28)
    return fig


def make_fundamental_diagram_plot(qmax_per_lane, vf, w, kj_per_lane):
    k = np.linspace(0.0, kj_per_lane, 400)
    q_ff = vf * k
    q_cg = w * (kj_per_lane - k)
    q = np.minimum(np.minimum(q_ff, qmax_per_lane), q_cg)

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(k, q_ff, color="#2b6cb0", linestyle="--", linewidth=1.5, label="Free-flow branch")
    ax.plot(k, q_cg, color="#c53030", linestyle="--", linewidth=1.5, label="Congested branch")
    ax.plot(k, np.full_like(k, qmax_per_lane), color="#b7791f", linewidth=1.2, linestyle=":", label="q_max")
    ax.plot(k, q, color="black", linewidth=2.0, label="FD: min(free, q_max, congested)")

    ax.set_xlabel("Density k (veh/km/lane)")
    ax.set_ylabel("Flow q (veh/hr/lane)")
    ax.set_title("Triangular Fundamental Diagram")
    ax.set_xlim(left=0.0)
    ax.set_ylim(0.0, 2500.0)
    ax.spines["left"].set_position(("data", 0.0))
    ax.spines["bottom"].set_position(("data", 0.0))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def make_network_image_with_incident(image_path, n_main_cells, incident_cell):
    img = plt.imread(image_path)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.imshow(img)
    ax.axis("off")

    # Approximate mainline cell band location in Network_image.png (axes fraction).
    left = 0.095
    right = 0.905
    y0 = 0.31
    h = 0.36
    w = (right - left) / max(1, n_main_cells)
    x0 = left + incident_cell * w

    # Place a star marker at the center of the incident cell.
    x_center = x0 + 0.5 * w
    y_center = y0 + 0.5 * h
    ax.scatter(
        [x_center],
        [y_center+0.05],
        transform=ax.transAxes,
        marker="*",
        s=600,
        color="yellow",
        edgecolors="red",
        linewidths=1.2,
        zorder=5,
    )
    ax.text(
        x0,
        y_center - 0.05,
        f"Incident cell",
        color="yellow",
        fontsize=10,
        fontweight="bold",
        transform=ax.transAxes,
    )

    fig.tight_layout()
    return fig

def compute_vehicle_trajectory(res, start_time_hr, dt_sub_hr=None, extra_follow_hr=6.0):
    speed_main = res["speed_main"]
    n_cells = speed_main.shape[1]
    road_length_km = n_cells * ctm_core.DX
    sim_end_hr = float(res["time"][-1] + DT)

    if dt_sub_hr is None:
        dt_sub_hr = DT / 6.0

    max_end_hr = max(sim_end_hr + extra_follow_hr, start_time_hr + road_length_km / max(1e-6, VF) + extra_follow_hr)

    t = float(start_time_hr)
    x = 0.0
    times = [t]
    distances = [x]

    while x < road_length_km and t < max_end_hr:
        t_idx = min(int(t / DT), speed_main.shape[0] - 1)
        cell_idx = min(int(x / ctm_core.DX), n_cells - 1)
        v = max(0.0, float(speed_main[t_idx, cell_idx]))
        x = min(road_length_km, x + v * dt_sub_hr)
        t = t + dt_sub_hr
        times.append(t)
        distances.append(x)

    reached = x >= road_length_km - 1e-9
    free_flow_tt = road_length_km / max(1e-6, VF)

    if reached:
        arrival_time = t
        actual_tt = arrival_time - start_time_hr
        delay_tt = max(0.0, actual_tt - free_flow_tt)
    else:
        arrival_time = None
        actual_tt = None
        delay_tt = None

    return {
        "time_hr": np.array(times),
        "distance_km": np.array(distances),
        "road_length_km": road_length_km,
        "arrival_time_hr": arrival_time,
        "actual_travel_time_hr": actual_tt,
        "free_flow_travel_time_hr": free_flow_tt,
        "delay_time_hr": delay_tt,
        "reached": reached,
    }


def make_trajectory_speed_background_grid(results_by_case, trajectories_by_case, vehicle_start_hour):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)
    axes_flat = axes.ravel()
    im = None

    for idx, (ax, (label, res)) in enumerate(zip(axes_flat, results_by_case.items())):
        speed = res["speed_main"]
        road_length_km = speed.shape[1] * ctm_core.DX
        time_end = speed.shape[0] * DT

        im = ax.imshow(
            speed.T,
            origin="lower",
            aspect="auto",
            extent=[0, time_end, 0, road_length_km],
            cmap="coolwarm",
            vmin=0.0,
            vmax=VF,
        )

        traj = trajectories_by_case[label]
        traj_time = np.asarray(traj["time_hr"], dtype=float)
        traj_dist = np.asarray(traj["distance_km"], dtype=float)

        # Clip trajectory at simulation end time.
        keep_mask = traj_time <= time_end
        plot_time = traj_time[keep_mask]
        plot_dist = traj_dist[keep_mask]
        if np.any(traj_time > time_end) and len(plot_time) > 0 and plot_time[-1] < time_end:
            next_idx = int(np.argmax(traj_time > time_end))
            t0, t1 = traj_time[next_idx - 1], traj_time[next_idx]
            d0, d1 = traj_dist[next_idx - 1], traj_dist[next_idx]
            if t1 > t0:
                ratio = (time_end - t0) / (t1 - t0)
                d_end = d0 + ratio * (d1 - d0)
                plot_time = np.append(plot_time, time_end)
                plot_dist = np.append(plot_dist, d_end)

        ax.plot(plot_time, plot_dist, color="black", linewidth=1.8, label="Vehicle trajectory")

        ff_arrival = vehicle_start_hour + traj["free_flow_travel_time_hr"]
        ff_end_t = min(ff_arrival, time_end)
        if ff_arrival > vehicle_start_hour:
            ff_end_x = road_length_km * ((ff_end_t - vehicle_start_hour) / (ff_arrival - vehicle_start_hour))
        else:
            ff_end_x = 0.0
        ax.plot(
            [vehicle_start_hour, ff_end_t],
            [0.0, ff_end_x],
            color="#00c853",
            linestyle="-.",
            linewidth=1.8,
            label="Free-flow",
        )

        ax.set_title(label)
        ax.set_xlabel("Time (hr)")
        ax.set_ylabel("Distance (km)")

        if idx == 0:
            ax.legend(loc="upper left", fontsize=8, framealpha=0.85)

    if im is not None:
        fig.subplots_adjust(right=0.88)
        cax = fig.add_axes([0.90, 0.15, 0.02, 0.70])
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label("Speed (km/hr)")

    fig.suptitle("Trajectory on speed background (4 scenarios)")
    fig.subplots_adjust(left=0.07, bottom=0.08, top=0.92, right=0.88, wspace=0.18, hspace=0.25)
    return fig


layout_left, layout_right = st.columns([2.4, 1.0])

with layout_left:
    st.subheader("Initial Network")
    st.pyplot(make_network_image_with_incident(str(BASE_DIR / "Network_image.png"), n_main_cells, incident_cell))
    st.caption(
        f"Incident marker: cell {incident_cell}, from {format_hr_m(incident_start_hour)} "
        f"to {format_hr_m(incident_start_hour + incident_duration_hour)}"
    )

with layout_right:
    st.subheader("Fundamental Diagram")
    st.pyplot(make_fundamental_diagram_plot(QMAX_PER_LANE, VF, W, KJ_PER_LANE))
    st.subheader("FD Table")
    st.table([
        {"Parameter": "Maximum capacity (qmax)", "Value": f"{QMAX_PER_LANE:.0f} veh/hr/lane"},
        {"Parameter": "Free-flow speed (v)", "Value": f"{VF:.0f} km/hr"},
        {"Parameter": "Jam density (kj)", "Value": f"{KJ_PER_LANE:.0f} veh/km/lane"},
        {"Parameter": "Wave speed (w)", "Value": f"{W:.0f} km/hr"},
    ])

if merge_top_idx == merge_bottom_idx:
    st.warning("Upper and lower on-ramp merge indices must be different.")
if merge_top_idx - 1 == diverge_idx or merge_bottom_idx - 1 == diverge_idx:
    st.warning("A merge link overlaps with diverge link. Adjust merge indices.")

with st.expander("CTM Algorithm", expanded=False):
    st.subheader("CTM Algorithm")

    st.markdown("<h4 style='font-family: Georgia, serif; font-weight: 800;'>Intercell flow</h4>", unsafe_allow_html=True)
    st.markdown("For sending flow,")
    st.latex(r"S_i(t) = \min\left(q_{\max,i-1},\; v_f\,k_{i-1}(t)\right)")
    st.markdown("For receiving flow,")
    st.latex(r"R_i(t) = \min\left(q_{\max,i},\; w\left[k_j^i-k_i(t)\right]\right)")
    st.markdown("Inter-cell flow is the minimum of sending flow and receiving flow.")
    st.latex(r"y_i(t) \leq \min\left\{S_i(t),\; R_i(t)\right\}")

    st.markdown("<h4 style='font-family: Georgia, serif; font-weight: 800;'>Diverging flow</h4>", unsafe_allow_html=True)
    st.latex(
        r"""
\begin{aligned}
y_{i,\text{main}}(t) &= (1-\beta_{\text{off}})\,S_i(t) \\
y_{i,\text{off}}(t) &= \beta_{\text{off}}\,S_i(t)
\end{aligned}
"""
    )

    st.markdown("<h4 style='font-family: Georgia, serif; font-weight: 800;'>Merging flow</h4>", unsafe_allow_html=True)
    st.latex(
        r"""
\begin{aligned}
y_i(t)+y_{on}(t) &= \min\left(S_i(t)+S_{on}(t),\; R_i(t)\right) \\
y_i(t) &= \min\left(S_i(t),\; p_{\mathrm{main}}R_i(t)\right) \\
y_{on}(t) &= \min\left(S_{on}(t),\; (1-p_{\mathrm{main}})R_i(t)\right)
\end{aligned}
"""
    )

with st.expander("AI Policy", expanded=False):
    st.markdown("<h4 style='font-family: Georgia, serif; font-weight: 800;'>Used tool: GPT-5.3-Codex</h4>", unsafe_allow_html=True)
    st.markdown("<h4 style='font-family: Georgia, serif; font-weight: 500;'>AI was used for code writing and modification, as well as for web page creation and deployment, but was not used to propose or modify the logic of CTM and analyzation.</h4>", unsafe_allow_html=True)
# ---------------------------------------------------------
# Run and display
# ---------------------------------------------------------
if run_clicked:
    scenarios = [
        ("Off-peak / No incident", float(offpeak_main_inflow), float(offpeak_onramp_top_inflow), float(offpeak_onramp_bottom_inflow), False),
        ("Off-peak / Incident", float(offpeak_main_inflow), float(offpeak_onramp_top_inflow), float(offpeak_onramp_bottom_inflow), True),
        ("Peak / No incident", float(peak_main_inflow), float(peak_onramp_top_inflow), float(peak_onramp_bottom_inflow), False),
        ("Peak / Incident", float(peak_main_inflow), float(peak_onramp_top_inflow), float(peak_onramp_bottom_inflow), True),
    ]
    incident_case_flags = {label: is_incident for label, _, _, _, is_incident in scenarios}
    background_flow_by_case = {
        label: float(case_main_inflow) * (1.0 - float(beta_off))
        for label, case_main_inflow, _, _, _ in scenarios
    }
    results_by_case = {}
    try:
        for case_label, case_main_inflow, case_on_top, case_on_bottom, case_incident in scenarios:
            results_by_case[case_label] = run_ctm(
                T_hours=T_hours,
                main_inflow=case_main_inflow,
                onramp_top_inflow=case_on_top,
                onramp_bottom_inflow=case_on_bottom,
                beta_off=float(beta_off),
                p_main=float(p_main),
                p_on=float(p_on),
                n_main_cells=int(n_main_cells),
                n_on_cells=N_ON_SIMPLIFIED,
                diverge_idx=int(diverge_idx),
                merge_top_idx=int(merge_top_idx),
                merge_bottom_idx=int(merge_bottom_idx),
                with_incident=bool(case_incident),
                incident_cell=int(incident_cell),
                incident_start_hour=float(incident_start_hour),
                incident_duration_hour=float(incident_duration_hour),
                blocked_lanes=int(blocked_lanes),
                initial_main_density=float(initial_main_density),
            )
    except Exception as e:
        st.error(f"Simulation error: {e}")
    else:
        st.session_state["ctm_last_run"] = {
            "results_by_case": results_by_case,
            "incident_case_flags": incident_case_flags,
            "background_flow_by_case": background_flow_by_case,
            "setup": {
                "T_hours": float(T_hours),
                "n_main_cells": int(n_main_cells),
                "diverge_idx": int(diverge_idx),
                "merge_top_idx": int(merge_top_idx),
                "merge_bottom_idx": int(merge_bottom_idx),
                "incident_cell": int(incident_cell),
                "incident_start_hour": float(incident_start_hour),
                "incident_duration_hour": float(incident_duration_hour),
                "blocked_lanes": int(blocked_lanes),
            },
        }
        st.success("4-case simulation completed.")

sim_data = st.session_state.get("ctm_last_run")

if sim_data is not None:
    results_by_case = sim_data["results_by_case"]
    incident_case_flags = sim_data["incident_case_flags"]
    background_flow_by_case = sim_data.get("background_flow_by_case", {})
    setup = sim_data["setup"]

    st.subheader("Network setup")
    st.write(
        f"- Mainline cells: {setup['n_main_cells']}\n"
        f"- Ramp cells: simplified (no ramp cell state)\n"
        f"- Off-ramp diverge index: {setup['diverge_idx']}\n"
        f"- Upper on-ramp merge index: {setup['merge_top_idx']}\n"
        f"- Lower on-ramp merge index: {setup['merge_bottom_idx']}\n"
        f"- Incident cell: {setup['incident_cell']}\n"
        f"- Incident start/duration: {format_hr_m(setup['incident_start_hour'])} / {format_hr_m(setup['incident_duration_hour'])}\n"
        f"- Blocked lanes at incident cell: {setup['blocked_lanes']}\n"
        f"- Trajectory start time: set in Trajectory & Delay tab"
    )

    network_series_by_case = {
        label: compute_network_weighted_series(res)
        for label, res in results_by_case.items()
    }

    top_tab1, top_tab2, top_tab3, top_tab4 = st.tabs([
        "CTM Result",
        "Scenario Comparison",
        "Delay Analysis",
        "q-k analysis",
    ])

    with top_tab1:
        ctm_tab1, ctm_tab2 = st.tabs([
            "Density (2x2)",
            "Speed (2x2)",
        ])

        with ctm_tab1:
            density_vmax = max(float(np.max(res["density_main"])) for res in results_by_case.values())
            fig_density_grid = make_case_heatmap_grid(
                results_by_case,
                key="density_main",
                title="Mainline density heatmaps (4 cases, veh/km/lane)",
                cmap="hot",
                vmin=0.0,
                vmax=density_vmax,
                cbar_label="Density (veh/km/lane)",
                incident_case_flags=incident_case_flags,
                incident_start_hour=setup["incident_start_hour"],
                incident_duration_hour=setup["incident_duration_hour"],
                incident_cell=setup["incident_cell"],
            )
            st.pyplot(fig_density_grid)

        with ctm_tab2:
            fig_speed_grid = make_case_heatmap_grid(
                results_by_case,
                key="speed_main",
                title="Mainline speed heatmaps (4 cases, km/hr)",
                cmap="coolwarm",
                vmin=0.0,
                vmax=VF,
                cbar_label="Speed (km/hr)",
                incident_case_flags=incident_case_flags,
                incident_start_hour=setup["incident_start_hour"],
                incident_duration_hour=setup["incident_duration_hour"],
                incident_cell=setup["incident_cell"],
            )
            st.pyplot(fig_speed_grid)

    with top_tab2:
        cmp_tab1, cmp_tab2 = st.tabs([
            "Flow comparison",
            "Speed/Density comparison",
        ])

        with cmp_tab1:
            max_detector_cell = next(iter(results_by_case.values()))["flow_main_cell"].shape[1] - 1
            middle_detector_cell_idx = st.slider(
                "Middle detector cell index",
                min_value=0,
                max_value=max_detector_cell,
                value=max_detector_cell // 2,
                step=1,
                key="middle_detector_cell_idx",
            )
            st.pyplot(make_detector_flow_by_scenario_grid(results_by_case, middle_detector_cell_idx=middle_detector_cell_idx))

        with cmp_tab2:
            fig_speed_ts = make_line_plot(
                next(iter(results_by_case.values()))["time"],
                {label: series["speed"] for label, series in network_series_by_case.items()},
                "Average vehicle speed comparison",
                "Speed (km/hr)",
            )
            st.pyplot(fig_speed_ts)

            fig_density_ts = make_line_plot(
                next(iter(results_by_case.values()))["time"],
                {label: series["density"] for label, series in network_series_by_case.items()},
                "Network density comparison",
                "Density (veh/km)",
            )
            st.pyplot(fig_density_ts)

    with top_tab3:
        delay_tab1, delay_tab2 = st.tabs([
            "Trajectory & Delay",
            "3-lane segment delay",
        ])

        with delay_tab1:
            vehicle_start_hour = st.slider(
                "Vehicle start time from mainline origin",
                min_value=0.0,
                max_value=float(setup["T_hours"]),
                value=min(1.0, float(setup["T_hours"])),
                step=DT,
                key="trajectory_start_hour_tab",
            )
            trajectories_by_case = {
                case_label: compute_vehicle_trajectory(case_res, vehicle_start_hour)
                for case_label, case_res in results_by_case.items()
            }

            st.pyplot(
                make_trajectory_speed_background_grid(
                    results_by_case,
                    trajectories_by_case,
                    vehicle_start_hour,
                )
            )

            delay_rows = []
            for label, traj in trajectories_by_case.items():
                delay_rows.append(
                    {
                        "Scenario": label,
                        "Start time": format_hr_m(vehicle_start_hour),
                        "Free-flow TT": format_hr_m(traj["free_flow_travel_time_hr"]),
                        "Actual TT": format_hr_m(traj["actual_travel_time_hr"]),
                        "Delay": format_hr_m(traj["delay_time_hr"]),
                        "Arrival time": format_hr_m(traj["arrival_time_hr"]),
                    }
                )
            st.dataframe(delay_rows, width="stretch")

        with delay_tab2:
            st.image(str(BASE_DIR / "3-lane_image.png"), caption="3-lane segment detector layout", width="stretch")
            st.pyplot(make_three_lane_cumulative_curve_grid(results_by_case, background_flow_by_case=background_flow_by_case))

            seg_delay_rows = []
            for label, res in results_by_case.items():
                total_delay, n_up_total, n_down_total, n_unmatched = compute_three_lane_segment_total_delay_veh_hr(res)
                seg_delay_rows.append(
                    {
                        "Scenario": label,
                        "Total delay in 3-lane segment (veh-hr)": round(total_delay, 2),
                    }
                )
            st.dataframe(seg_delay_rows, width="stretch")

    with top_tab4:
        max_qk_step = next(iter(results_by_case.values()))["density_main"].shape[0] - 1
        if "qk_analysis_step" not in st.session_state:
            st.session_state["qk_analysis_step"] = max_qk_step
        st.session_state["qk_analysis_step"] = int(min(st.session_state["qk_analysis_step"], max_qk_step))

        qk_step_idx = st.slider(
            "q-k analysis timestep",
            min_value=0,
            max_value=max_qk_step,
            step=1,
            key="qk_analysis_step",
        )
        time_ref = next(iter(results_by_case.values()))["time"]
        sim_end_hr = float(len(time_ref) * DT)
        qk_time_hr = float(min(sim_end_hr, float(time_ref[qk_step_idx] + DT)))
        st.caption(f"Selected timestep: {qk_step_idx} (t = {format_hr_m(qk_time_hr)})")

        qk_tab1, qk_tab2, qk_tab3 = st.tabs([
            "Network q-k diagram",
            "Cell q-k at selected timestep",
            "Overlay q-k comparison",
        ])
        with qk_tab1:
            st.pyplot(make_network_qk_grid(network_series_by_case))
        with qk_tab2:
            st.pyplot(make_cell_qk_last_grid(results_by_case, analysis_step_idx=qk_step_idx, analysis_time_hr=qk_time_hr))
        with qk_tab3:
            st.pyplot(
                make_qk_overlay_comparison_grid(
                    network_series_by_case,
                    results_by_case,
                    analysis_step_idx=qk_step_idx,
                    analysis_time_hr=qk_time_hr,
                )
            )

    with st.expander("Simulation arrays preview (Peak / Incident)"):
        preview = results_by_case["Peak / Incident"]
        st.write("density_main shape:", preview["density_main"].shape)
        st.dataframe(preview["density_main"][: min(10, preview["density_main"].shape[0]), :])
else:
    st.info("Adjust parameters in the input panel and click Run 4-case comparison.")

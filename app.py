import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

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

    T_hours = st.slider("Simulation time (hr)", min_value=1.0, max_value=9.0, value=6.0, step=0.5)
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
        st.subheader("Network & Merge")
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

    with col3:
        st.subheader("Incident")
        incident_cell = st.slider("Incident cell", min_value=0, max_value=n_main_cells - 1, value=min(6, n_main_cells - 1), step=1)
        incident_start_hour = st.slider("Incident start time (hr)", min_value=0.0, max_value=T_hours, value=min(1.0, T_hours), step=DT)
        max_duration = max(DT, T_hours - incident_start_hour)
        incident_duration_hour = st.slider("Incident duration (hr)", min_value=DT, max_value=max_duration, value=min(1.0, max_duration), step=DT)
        blocked_lanes = st.slider("Blocked lanes", min_value=1, max_value=3, value=1, step=1)

    with col4:
        st.subheader("Trajectory")
        vehicle_start_hour = st.slider("Vehicle start time from mainline origin (hr)", min_value=0.0, max_value=T_hours, value=1.0, step=DT)

run_clicked = st.button("Run 4-case comparison", type="primary", use_container_width=True)

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
    fig.tight_layout(rect=[0.0, 0.0, 0.88, 0.95])
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
        f"Incident cell: {incident_cell}",
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
        ax.plot(traj["time_hr"], traj["distance_km"], color="black", linewidth=1.8, label="Vehicle trajectory")

        ff_arrival = vehicle_start_hour + traj["free_flow_travel_time_hr"]
        ax.plot(
            [vehicle_start_hour, ff_arrival],
            [0.0, road_length_km],
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
    fig.tight_layout(rect=[0.0, 0.0, 0.88, 0.96])
    return fig


layout_left, layout_right = st.columns([2.4, 1.0])

with layout_left:
    st.subheader("Network")
    st.pyplot(make_network_image_with_incident(str(BASE_DIR / "Network_image.png"), n_main_cells, incident_cell))
    st.caption(f"Incident marker: cell {incident_cell}, from {incident_start_hour:.2f} hr to {incident_start_hour + incident_duration_hour:.2f} hr")

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
        st.success("4-case simulation completed.")

        trajectories_by_case = {
            case_label: compute_vehicle_trajectory(case_res, vehicle_start_hour)
            for case_label, case_res in results_by_case.items()
        }

        st.subheader("Network setup")
        st.write(
            f"- Mainline cells: {n_main_cells}\n"
            f"- Ramp cells: simplified (no ramp cell state)\n"
            f"- Off-ramp diverge index: {diverge_idx}\n"
            f"- Upper on-ramp merge index: {merge_top_idx}\n"
            f"- Lower on-ramp merge index: {merge_bottom_idx}\n"
            f"- Incident cell: {incident_cell}\n"
            f"- Incident start/duration: {incident_start_hour:.2f} h / {incident_duration_hour:.2f} h\n"
            f"- Blocked lanes at incident cell: {blocked_lanes}\n"
            f"- Vehicle start time: {vehicle_start_hour:.2f} h"
        )

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Density (2x2)",
            "Speed (2x2)",
            "Flow comparison",
            "Speed/Density comparison",
            "Trajectory & Delay",
        ])

        with tab1:
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
                incident_start_hour=incident_start_hour,
                incident_duration_hour=incident_duration_hour,
                incident_cell=incident_cell,
            )
            st.pyplot(fig_density_grid)

        with tab2:
            fig_speed_grid = make_case_heatmap_grid(
                results_by_case,
                key="speed_main",
                title="Mainline speed heatmaps (4 cases, km/hr)",
                cmap="coolwarm",
                vmin=0.0,
                vmax=VF,
                cbar_label="Speed (km/hr)",
                incident_case_flags=incident_case_flags,
                incident_start_hour=incident_start_hour,
                incident_duration_hour=incident_duration_hour,
                incident_cell=incident_cell,
            )
            st.pyplot(fig_speed_grid)

        with tab3:
            fig_flow = make_line_plot(
                next(iter(results_by_case.values()))["time"],
                {
                    "Off-peak / No incident": results_by_case["Off-peak / No incident"]["outflow_main"],
                    "Off-peak / Incident": results_by_case["Off-peak / Incident"]["outflow_main"],
                    "Peak / No incident": results_by_case["Peak / No incident"]["outflow_main"],
                    "Peak / Incident": results_by_case["Peak / Incident"]["outflow_main"],
                },
                "Main downstream outflow comparison",
                "Outflow (veh/hr)",
            )
            st.pyplot(fig_flow)

        with tab4:
            fig_speed_ts = make_line_plot(
                next(iter(results_by_case.values()))["time"],
                {label: np.mean(res["speed_main"], axis=1) for label, res in results_by_case.items()},
                "Mean mainline speed comparison",
                "Speed (km/hr)",
            )
            st.pyplot(fig_speed_ts)

            fig_density_ts = make_line_plot(
                next(iter(results_by_case.values()))["time"],
                {label: np.mean(res["density_main"], axis=1) for label, res in results_by_case.items()},
                "Mean mainline density comparison",
                "Density (veh/km/lane)",
            )
            st.pyplot(fig_density_ts)

        with tab5:
            st.pyplot(make_trajectory_speed_background_grid(results_by_case, trajectories_by_case, vehicle_start_hour))

            delay_rows = []
            for label, traj in trajectories_by_case.items():
                delay_rows.append(
                    {
                        "Scenario": label,
                        "Start time (hr)": vehicle_start_hour,
                        "Free-flow TT (hr)": traj["free_flow_travel_time_hr"],
                        "Actual TT (hr)": traj["actual_travel_time_hr"],
                        "Delay (hr)": traj["delay_time_hr"],
                        "Arrival time (hr)": traj["arrival_time_hr"],
                    }
                )
            st.dataframe(delay_rows, use_container_width=True)

        with st.expander("Simulation arrays preview (Peak / Incident)"):
            preview = results_by_case["Peak / Incident"]
            st.write("density_main shape:", preview["density_main"].shape)
            st.dataframe(preview["density_main"][: min(10, preview["density_main"].shape[0]), :])

else:
    st.info("Adjust parameters in the input panel and click Run 4-case comparison.")

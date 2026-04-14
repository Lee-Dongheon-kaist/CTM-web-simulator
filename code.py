import numpy as np
import matplotlib.pyplot as plt
from ctm_inputs import (
    BETA_OFF,
    MAIN_INFLOW,
    OFF_PEAK,
    ONRAMP_BOTTOM_INFLOW,
    ONRAMP_INFLOW,
    ONRAMP_TOP_INFLOW,
    PEAK_INCIDENT,
    P_MAIN,
    P_ON,
)
from ctm_params import (
    ALPHA,
    DT,
    DX,
    DIVERGE_IDX,
    KJ_PER_LANE,
    MERGE_BOTTOM_IDX,
    MERGE_TOP_IDX,
    N_MAIN,
    N_ON,
    QMAX_MAIN,
    QMAX_PER_LANE,
    QMAX_RAMP,
    VF,
)

# =========================================================
# 3. CTM 기본 함수
# =========================================================
def sending(n, qmax):
    return min(qmax, n / DT)

def receiving(n, n_jam, qmax):
    return min(qmax, ALPHA * (n_jam - n) / DT)

def ordinary_flow(n_up, n_down, q_up, q_down, n_jam_down):
    s = sending(n_up, q_up)
    r = receiving(n_down, n_jam_down, q_down)
    return min(s, r)


def diverge_flow(n_up, n_main_down, q_up, q_main_down, n_jam_main_down, beta):
    """
    off-ramp는 별도 ramp state를 두지 않고,
    diverge cell에서 beta 비율만큼 바로 빠져나간다고 가정
    """
    s_up = sending(n_up, q_up)
    r_main = receiving(n_main_down, n_jam_main_down, q_main_down)

    # main downstream 수용능력 때문에 전체 유량 제한
    # f_main = (1-beta) * F <= r_main
    # 따라서 F <= r_main / (1-beta)
    F = min(s_up, r_main / max(1e-9, (1 - beta)))

    f_main = (1 - beta) * F
    f_off = beta * F
    return f_main, f_off


def merge_flow(n_main_up, on_demand, n_down, q_main_up, q_on_up, q_down, n_jam_down,
               p_main=0.7, p_on=0.3):
    """
    mainline + on-ramp -> mainline
    priority 기반 merge
    """
    s_main = sending(n_main_up, q_main_up)
    s_on = min(on_demand, q_on_up)
    r_down = receiving(n_down, n_jam_down, q_down)

    total = min(s_main + s_on, r_down)

    f_main = min(s_main, p_main * r_down)
    f_on = min(s_on, p_on * r_down)

    used = f_main + f_on
    rem = total - used

    if rem > 0:
        add_main = min(rem, s_main - f_main)
        f_main += max(0, add_main)
        rem = total - (f_main + f_on)

    if rem > 0:
        add_on = min(rem, s_on - f_on)
        f_on += max(0, add_on)

    return f_main, f_on


# =========================================================
# 4. Incident 함수
# =========================================================
def apply_lane_block_incident(qmax_main_time, njam_main_time, start_step, end_step,
                              incident_cell, total_lanes=3, blocked_lanes=1):
    """
    3차로 중 1차로 차단 -> capacity factor = 2/3
    """
    factor = (total_lanes - blocked_lanes) / total_lanes
    qmax_main_time[start_step:end_step, incident_cell] *= factor
    njam_main_time[start_step:end_step, incident_cell] *= factor
    return qmax_main_time, njam_main_time


def _scaled_index(base_idx, base_n, target_n):
    """Scale a reference cell index when the number of cells changes."""
    if target_n <= 1:
        return 0
    if base_n <= 1:
        return min(target_n - 1, max(0, base_idx))

    ratio = base_idx / (base_n - 1)
    return int(round(ratio * (target_n - 1)))


def _build_main_lanes_by_cell(n_main_cells, diverge_idx, merge_bottom_idx):
    """Set 3 lanes only between off-ramp and lower on-ramp; 4 lanes elsewhere."""
    lanes = np.full(n_main_cells, 4.0, dtype=float)
    for i in range(n_main_cells):
        if diverge_idx < i < merge_bottom_idx:
            lanes[i] = 3.0
    return lanes


# =========================================================
# 5. 시뮬레이션
# =========================================================
def run_ctm(
    T_hours=3.0,
    main_inflow=MAIN_INFLOW,     # veh/hr
    onramp_top_inflow=ONRAMP_TOP_INFLOW,
    onramp_bottom_inflow=ONRAMP_BOTTOM_INFLOW,
    onramp_inflow=ONRAMP_INFLOW, # backward-compatible alias to bottom on-ramp
    beta_off=BETA_OFF,
    p_main=P_MAIN,
    p_on=P_ON,
    n_main_cells=N_MAIN,
    n_on_cells=N_ON,
    diverge_idx=None,
    merge_top_idx=None,
    merge_bottom_idx=None,
    merge_idx=None,
    with_incident=False,
    incident_cell=None,
    incident_start_hour=1.0,
    incident_duration_hour=0.5,
    blocked_lanes=1,
    initial_main_density=0.0,    # veh/km/lane
):
    if diverge_idx is None:
        diverge_idx = _scaled_index(DIVERGE_IDX, N_MAIN, n_main_cells)
    if merge_top_idx is None:
        merge_top_idx = _scaled_index(MERGE_TOP_IDX, N_MAIN, n_main_cells)
    if merge_bottom_idx is None:
        merge_bottom_idx = _scaled_index(MERGE_BOTTOM_IDX, N_MAIN, n_main_cells)
    if merge_idx is not None:
        merge_bottom_idx = merge_idx
    if onramp_bottom_inflow is None:
        onramp_bottom_inflow = onramp_inflow
    if incident_cell is None:
        incident_cell = _scaled_index(6, 12, n_main_cells)

    if n_main_cells < 5:
        raise ValueError("n_main_cells must be at least 5.")
    if not (0 <= diverge_idx < n_main_cells - 1):
        raise ValueError("diverge_idx must satisfy 0 <= diverge_idx < n_main_cells - 1.")
    if not (1 <= merge_top_idx < n_main_cells):
        raise ValueError("merge_top_idx must satisfy 1 <= merge_top_idx < n_main_cells.")
    if not (1 <= merge_bottom_idx < n_main_cells):
        raise ValueError("merge_bottom_idx must satisfy 1 <= merge_bottom_idx < n_main_cells.")
    if merge_top_idx == merge_bottom_idx:
        raise ValueError("merge_top_idx and merge_bottom_idx must be different.")
    if (merge_top_idx - 1 == diverge_idx) or (merge_bottom_idx - 1 == diverge_idx):
        raise ValueError("diverge and merge links overlap. Adjust indices.")
    if not (0 <= incident_cell < n_main_cells):
        raise ValueError("incident_cell must satisfy 0 <= incident_cell < n_main_cells.")

    total_steps = int(T_hours / DT)
    time = np.arange(total_steps) * DT
    main_lanes_by_cell = _build_main_lanes_by_cell(n_main_cells, diverge_idx, merge_bottom_idx)

    # 상태변수
    n_main = np.zeros((total_steps + 1, n_main_cells))
    n_on = np.zeros((total_steps + 1, 0))

    # 초기조건
    n_main[0, :] = initial_main_density * DX * main_lanes_by_cell

    # 시간별 capacity
    qmax_main_time = np.tile(QMAX_PER_LANE * main_lanes_by_cell, (total_steps, 1))
    njam_main_time = np.tile(KJ_PER_LANE * DX * main_lanes_by_cell, (total_steps, 1))

    if with_incident:
        start_step = int(incident_start_hour / DT)
        end_step = int((incident_start_hour + incident_duration_hour) / DT)
        qmax_main_time, njam_main_time = apply_lane_block_incident(
            qmax_main_time=qmax_main_time,
            njam_main_time=njam_main_time,
            start_step=start_step,
            end_step=end_step,
            incident_cell=incident_cell,
            total_lanes=float(main_lanes_by_cell[incident_cell]),
            blocked_lanes=blocked_lanes
        )

    # 결과 저장
    f_main_link = np.zeros((total_steps, n_main_cells - 1))
    inflow_main = np.zeros(total_steps)
    inflow_on_top = np.zeros(total_steps)
    inflow_on_bottom = np.zeros(total_steps)
    inflow_on = np.zeros(total_steps)
    outflow_main = np.zeros(total_steps)
    outflow_off = np.zeros(total_steps)

    div_main = np.zeros(total_steps)
    div_off = np.zeros(total_steps)
    merge_top_main = np.zeros(total_steps)
    merge_top_on = np.zeros(total_steps)
    merge_bottom_main = np.zeros(total_steps)
    merge_bottom_on = np.zeros(total_steps)

    for t in range(total_steps):

        # -------------------------------------------------
        # 1) main upstream inflow
        # -------------------------------------------------
        inflow_main[t] = min(
            main_inflow,
            receiving(n_main[t, 0], njam_main_time[t, 0], qmax_main_time[t, 0])
        )

        inflow_on_top[t] = min(onramp_top_inflow, QMAX_RAMP)
        inflow_on_bottom[t] = min(onramp_bottom_inflow, QMAX_RAMP)
        inflow_on[t] = inflow_on_top[t] + inflow_on_bottom[t]

        # -------------------------------------------------
        # 4) mainline ordinary flow
        # diverge / merge 위치는 따로 계산
        # -------------------------------------------------
        for i in range(n_main_cells - 1):
            if i in (diverge_idx, merge_top_idx - 1, merge_bottom_idx - 1):
                continue

            f_main_link[t, i] = ordinary_flow(
                n_main[t, i],
                n_main[t, i + 1],
                qmax_main_time[t, i],
                qmax_main_time[t, i + 1],
                njam_main_time[t, i + 1]
            )

        # -------------------------------------------------
        # 5) diverge node (left off-ramp)
        # mainline cell DIVERGE_IDX -> mainline + off-ramp
        # -------------------------------------------------
        div_main[t], div_off[t] = diverge_flow(
            n_up=n_main[t, diverge_idx],
            n_main_down=n_main[t, diverge_idx + 1],
            q_up=qmax_main_time[t, diverge_idx],
            q_main_down=qmax_main_time[t, diverge_idx + 1],
            n_jam_main_down=njam_main_time[t, diverge_idx + 1],
            beta=beta_off
        )
        f_main_link[t, diverge_idx] = div_main[t]

        # -------------------------------------------------
        # 6) merge node (upper on-ramp)
        # -------------------------------------------------
        merge_top_main[t], merge_top_on[t] = merge_flow(
            n_main_up=n_main[t, merge_top_idx - 1],
            on_demand=inflow_on_top[t],
            n_down=n_main[t, merge_top_idx],
            q_main_up=qmax_main_time[t, merge_top_idx - 1],
            q_on_up=QMAX_RAMP,
            q_down=qmax_main_time[t, merge_top_idx],
            n_jam_down=njam_main_time[t, merge_top_idx],
            p_main=p_main,
            p_on=p_on
        )
        f_main_link[t, merge_top_idx - 1] = merge_top_main[t]

        # -------------------------------------------------
        # 7) merge node (lower/right on-ramp)
        # -------------------------------------------------
        merge_bottom_main[t], merge_bottom_on[t] = merge_flow(
            n_main_up=n_main[t, merge_bottom_idx - 1],
            on_demand=inflow_on_bottom[t],
            n_down=n_main[t, merge_bottom_idx],
            q_main_up=qmax_main_time[t, merge_bottom_idx - 1],
            q_on_up=QMAX_RAMP,
            q_down=qmax_main_time[t, merge_bottom_idx],
            n_jam_down=njam_main_time[t, merge_bottom_idx],
            p_main=p_main,
            p_on=p_on
        )
        f_main_link[t, merge_bottom_idx - 1] = merge_bottom_main[t]

        # -------------------------------------------------
        # 8) main downstream outflow
        # -------------------------------------------------
        outflow_main[t] = min(
            sending(n_main[t, -1], qmax_main_time[t, -1]),
            qmax_main_time[t, -1]
        )

        # off-ramp outflow는 diverge node에서 바로 빠진 양
        outflow_off[t] = div_off[t]

        # -------------------------------------------------
        # 9) 상태 업데이트 - mainline
        # -------------------------------------------------
        # cell 0
        n_main[t + 1, 0] = n_main[t, 0] + DT * (inflow_main[t] - f_main_link[t, 0])

        # 중간 cell
        for i in range(1, n_main_cells - 1):

            if i == diverge_idx:
                inflow_i = f_main_link[t, i - 1]
                outflow_i = div_main[t] + div_off[t]

            elif i == merge_top_idx:
                inflow_i = merge_top_main[t] + merge_top_on[t]
                outflow_i = f_main_link[t, i]

            elif i == merge_bottom_idx:
                inflow_i = merge_bottom_main[t] + merge_bottom_on[t]
                outflow_i = f_main_link[t, i]

            else:
                inflow_i = f_main_link[t, i - 1]
                outflow_i = f_main_link[t, i]

            n_main[t + 1, i] = n_main[t, i] + DT * (inflow_i - outflow_i)

        # 마지막 cell
        n_main[t + 1, -1] = n_main[t, -1] + DT * (f_main_link[t, -1] - outflow_main[t])

        # clip
        n_main[t + 1] = np.minimum(np.maximum(n_main[t + 1], 0), njam_main_time[t])

    # -----------------------------------------------------
    # 후처리: density, speed
    # -----------------------------------------------------
    density_main = n_main[:-1] / (DX * main_lanes_by_cell[None, :])     # veh/km/lane
    density_on = np.zeros((total_steps, 0))

    flow_main_cell = np.zeros_like(density_main)
    speed_main = np.zeros_like(density_main)

    for t in range(total_steps):
        flow_main_cell[t, 0] = inflow_main[t]

        for i in range(1, n_main_cells - 1):
            if i == merge_top_idx:
                flow_main_cell[t, i] = merge_top_main[t] + merge_top_on[t]
            elif i == merge_bottom_idx:
                flow_main_cell[t, i] = merge_bottom_main[t] + merge_bottom_on[t]
            elif i == diverge_idx:
                flow_main_cell[t, i] = div_main[t] + div_off[t]
            else:
                flow_main_cell[t, i] = f_main_link[t, i - 1]

        flow_main_cell[t, -1] = outflow_main[t]

    for t in range(total_steps):
        for i in range(n_main_cells):
            k = density_main[t, i]
            if k <= 1e-6:
                speed_main[t, i] = VF
                continue

            # FD-based speed from density only: q = min(free-flow, qmax, congested-flow)
            qmax_lane = qmax_main_time[t, i] / max(1e-9, main_lanes_by_cell[i])
            kj_lane = njam_main_time[t, i] / max(1e-9, DX * main_lanes_by_cell[i])
            q_free = VF * k
            q_cong = max(0.0, (ALPHA * VF) * (kj_lane - k))
            q_fd = min(q_free, qmax_lane, q_cong)
            speed_main[t, i] = np.clip(q_fd / k, 0.0, VF)

    return {
        "time": time,
        "n_main": n_main,
        "n_on": n_on,
        "density_main": density_main,
        "density_on": density_on,
        "flow_main_cell": flow_main_cell,
        "speed_main": speed_main,
        "inflow_main": inflow_main,
        "inflow_on": inflow_on,
        "inflow_on_top": inflow_on_top,
        "inflow_on_bottom": inflow_on_bottom,
        "outflow_main": outflow_main,
        "outflow_off": outflow_off,
        "div_main": div_main,
        "div_off": div_off,
        "merge_top_main": merge_top_main,
        "merge_top_on": merge_top_on,
        "merge_bottom_main": merge_bottom_main,
        "merge_bottom_on": merge_bottom_on,
        "qmax_main_time": qmax_main_time,
        "njam_main_time": njam_main_time,
        "main_lanes_by_cell": main_lanes_by_cell,
        "network": {
            "n_main_cells": n_main_cells,
            "n_on_cells": n_on_cells,
            "diverge_idx": diverge_idx,
            "merge_top_idx": merge_top_idx,
            "merge_bottom_idx": merge_bottom_idx,
            "merge_idx": merge_bottom_idx,
            "incident_cell": incident_cell,
        },
    }


# =========================================================
# 6. 시각화 함수
# =========================================================
def plot_heatmap(data, title, cmap="hot", ylabel="Mainline Cell", cbar_label=""):
    plt.figure(figsize=(12, 4))
    im = plt.imshow(
        data.T,
        origin="lower",
        aspect="auto",
        extent=[0, data.shape[0] * DT, 0, data.shape[1]],
        cmap=cmap
    )
    cbar = plt.colorbar(im)
    if cbar_label:
        cbar.set_label(cbar_label)
    plt.xlabel("Time (hr)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_line(time, series_dict, title, ylabel):
    plt.figure(figsize=(10, 4))
    for label, values in series_dict.items():
        plt.plot(time, values, label=label)
    plt.xlabel("Time (hr)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# =========================================================
# 7. 실행 예시
# =========================================================
if __name__ == "__main__":

    # 1) Off-peak 예시
    res1 = run_ctm(
        T_hours=3.0,
        main_inflow=OFF_PEAK["main_inflow"],
        onramp_top_inflow=OFF_PEAK["onramp_top_inflow"],
        onramp_bottom_inflow=OFF_PEAK["onramp_bottom_inflow"],
        beta_off=OFF_PEAK["beta_off"],
        p_main=OFF_PEAK["p_main"],
        p_on=OFF_PEAK["p_on"],
        with_incident=False
    )

    # 2) Peak + Incident 예시
    res2 = run_ctm(
        T_hours=3.0,
        main_inflow=PEAK_INCIDENT["main_inflow"],
        onramp_top_inflow=PEAK_INCIDENT["onramp_top_inflow"],
        onramp_bottom_inflow=PEAK_INCIDENT["onramp_bottom_inflow"],
        beta_off=PEAK_INCIDENT["beta_off"],
        p_main=PEAK_INCIDENT["p_main"],
        p_on=PEAK_INCIDENT["p_on"],
        with_incident=True,
        incident_start_hour=1.0,
        incident_duration_hour=0.5,
        blocked_lanes=1
    )

    # Density heatmap
    plot_heatmap(
        res2["density_main"],
        "Mainline Density Heatmap (Peak + Incident, veh/km/lane)",
        cmap="hot",
        cbar_label="Density (veh/km/lane)",
    )

    # Speed heatmap
    plot_heatmap(
        res2["speed_main"],
        "Mainline Speed Heatmap (Peak + Incident, km/hr)",
        cmap="coolwarm",
        cbar_label="Speed (km/hr)",
    )

    # Outflow comparison
    plot_line(
        res1["time"],
        {
            "Main downstream outflow (off-peak)": res1["outflow_main"],
            "Main downstream outflow (peak+incident)": res2["outflow_main"],
            "Off-ramp outflow (peak+incident)": res2["outflow_off"],
        },
        "Outflow Comparison",
        "Flow (veh/hr)"
    )
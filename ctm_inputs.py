# Base external demands (veh/hr)
MAIN_INFLOW = 4500.0
ONRAMP_TOP_INFLOW = 1500.0
ONRAMP_BOTTOM_INFLOW = 1500.0
ONRAMP_INFLOW = ONRAMP_BOTTOM_INFLOW  # backward-compatible alias

# Node operation ratios
BETA_OFF = 0.20   # diverge split ratio to off-ramp
P_MAIN = 0.7      # merge priority for mainline
P_ON = 0.3        # merge priority for on-ramp

# Example scenarios used in __main__
OFF_PEAK = {
    "main_inflow": 4800.0,
    "onramp_top_inflow": 1200.0,
    "onramp_bottom_inflow": 1200.0,
    "beta_off": 0.20,
    "p_main": 0.7,
    "p_on": 0.3,
}

PEAK_INCIDENT = {
    "main_inflow": 6500.0,
    "onramp_top_inflow": 1600.0,
    "onramp_bottom_inflow": 1600.0,
    "beta_off": 0.20,
    "p_main": 0.7,
    "p_on": 0.3,
}

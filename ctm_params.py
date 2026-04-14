# Fundamental diagram parameters
QMAX_PER_LANE = 1920.0      # veh/hr/lane
VF = 120.0                  # km/hr
W = 15.0                    # km/hr
KJ_PER_LANE = 150.0         # veh/km/lane

# Discretization
DT = 5.0 / 60.0             # hr (5 min)
DX = VF * DT                # km, CFL condition -> 10 km

# Lane settings
MAIN_LANES = 3
MAIN_LANES_MAX = 4
ONRAMP_LANES = 1

# Derived capacities and jam occupancies
QMAX_MAIN = QMAX_PER_LANE * MAIN_LANES_MAX
QMAX_RAMP = QMAX_PER_LANE * ONRAMP_LANES

KJ_MAIN = KJ_PER_LANE * MAIN_LANES_MAX
KJ_RAMP = KJ_PER_LANE * ONRAMP_LANES

NJAM_MAIN = KJ_MAIN * DX
NJAM_RAMP = KJ_RAMP * DX

ALPHA = W / VF

# Network layout
N_MAIN = 12
N_ON = 2

DIVERGE_IDX = 1             # off-ramp diverge index on mainline
MERGE_TOP_IDX = 10          # upper on-ramp merge index on mainline
MERGE_BOTTOM_IDX = 9        # lower on-ramp merge index on mainline
MERGE_IDX = MERGE_BOTTOM_IDX  # backward-compatible alias

import numpy as np

Z_BIN_CENTRES = np.logspace(-4, np.log10(0.03), 50).round(5)
BIN_WIDTH = (np.log10(Z_BIN_CENTRES)[1] - np.log10(Z_BIN_CENTRES)[0]) / 2
Z_BIN_EDGES = np.logspace(np.log10(Z_BIN_CENTRES[0]) - BIN_WIDTH, np.log10(Z_BIN_CENTRES[-1]) + BIN_WIDTH, 51)
Z_BIN_EDGES[0] = Z_BIN_CENTRES[0]
Z_BIN_EDGES[-1] = Z_BIN_CENTRES[-1]

M1_MIN = { "NSWD": 4, "NSNS": 5, "BHWD": 14, "BHNS": 16, "BHBH": 19}

DCO_TYPES = ["BHBH", "BHNS", "BHWD", "NSNS", "NSWD"]
DCO_COLOURS = dict(zip(DCO_TYPES, ["#0F145D", "#3670D5", "#38C9CC", "#38813B", "#6ACB4A"]))
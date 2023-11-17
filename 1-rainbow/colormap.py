import matplotlib
import matplotlib.pyplot as plt
import numpy as np


# Rescaled colors and ph values
def ph_indicator_cmap():
    ph = np.array([4, 5, 6, 7, 8, 9, 10]) / 14.0
    characteristic_colors = (
        np.array(
            [
                [101, 81, 0],
                [66, 73, 0],
                [38, 63, 8],
                [20, 54, 37],
                [10, 49, 86],
                [26, 72, 80],
                [168, 138, 74],
            ]
        )
        / 255
    )

    # Init colormap
    cdict = {"red": [], "green": [], "blue": []}

    # Fill in data
    cdict["red"].append((0, 0, 0))
    cdict["green"].append((0, 0, 0))
    cdict["blue"].append((0, 0, 0))
    for i in range(len(ph)):
        p = ph[i]
        c = characteristic_colors[i]

        if i == 0:
            # White color on boundary
            cdict["red"].append((p, 0, c[0]))
            cdict["green"].append((p, 0, c[1]))
            cdict["blue"].append((p, 0, c[2]))
        elif i == len(ph) - 1:
            # Black color on boundary
            cdict["red"].append((p, c[0], 1))
            cdict["green"].append((p, c[1], 1))
            cdict["blue"].append((p, c[2], 1))
        else:
            cdict["red"].append((p, c[0], c[0]))
            cdict["green"].append((p, c[1], c[1]))
            cdict["blue"].append((p, c[2], c[2]))
    cdict["red"].append(
        (
            1,
            1,
            1,
        )
    )
    cdict["green"].append((1, 1, 1))
    cdict["blue"].append((1, 1, 1))
    cmap = matplotlib.colors.LinearSegmentedColormap(
        "testCmap", segmentdata=cdict, N=256
    )
    return cmap

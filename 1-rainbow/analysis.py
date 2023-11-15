"""Application of calibrated pH analysis.

"""
# ! ---- IMPORTS ---- !

import json
from pathlib import Path

import darsia
import matplotlib.pyplot as plt
import numpy as np
import skimage

# ! ---- DATA MANAGEMENT ---- !

user = None  # "helene"

# Define single baseline image
if user == "helene":
    baseline_folder = None  # TODO
else:
    baseline_folder = "data/baseline_images"
baseline_path = list(sorted(Path(baseline_folder).glob("*.JPG")))[0]

# Define experiment images
if user == "helene":
    experiment_folder = None  # TODO
else:
    experiment_folder = "data/experiment_images"
experiment_path = list(sorted(Path(experiment_folder).glob("*.JPG")))  # [5:10]

# ! ---- CORRECTION MANAGEMENT ---- !

# Idea: Apply three corrections:
# 1. Drift correction aligning images by simple translation with respect to teh color checker.
# 2. Color correction applying uniform colors in the color checker.
# 3. Curvature correction to crop images to the right rectangular format.
# The order has to be applied in later scripts as well.
# The calibration data is stored in a json file, generated using setup_preprocessing.

# Read the unmodified baseline image on which the preprocessing is defined on.
original_baseline = darsia.imread(baseline_path)

# Read config from json file
if user == "helene":
    f = open(
        Path(
            "/Users/heleneskretting/inf100/darsia-calibration-project/B032/config/preprocessing_2023-10-24_1143.json"
        )
    )
else:
    f = open(Path("config/preprocessing_2023-10-18 1500.json"))
preprocessing = json.load(f)

drift_correction = darsia.DriftCorrection(original_baseline, **preprocessing["drift"])
color_correction = darsia.ColorCorrection(original_baseline, **preprocessing["color"])
curvature_correction = darsia.CurvatureCorrection(config=preprocessing["curvature"])
corrections = [drift_correction, color_correction, curvature_correction]

# ! ---- BASELINE IMAGE ---- !

baseline_image = darsia.imread(baseline_path, transformations=corrections)

# ! ---- CONCENTRATION ANALYSIS ---- !

concentration_options = {
    "base": baseline_image.img_as(float),
    "restoration": darsia.TVD(
        weight=0.025, eps=1e-4, max_num_iter=100, method="isotropic Bregman"
    ),
    "restoration -> model": False,
    "diff option": "plain",
}

# Read config from json file and make compatible with kernel interpolation
calibration_path = Path("config/calibration_2023-11-15 1435.json")  # TODO
f = open(calibration_path)
calibration = json.load(f)
calibration["colors"] = np.array(calibration["colors"])

# Define models (clip values)
kernel_interpolation = darsia.KernelInterpolation(
    darsia.GaussianKernel(gamma=9.73),
    calibration["colors"],
    calibration["ph"],
)
# clip = darsia.ClipModel(**{"min value": 0, "max value": 1})
model = kernel_interpolation

# Define concentration analysis for now without any model (to be defined later).
analysis = darsia.ConcentrationAnalysis(model=model, **concentration_options)

# ! ---- VIZUALIZATION


def comparison_plot(image, concentration, path, subregion=None):
    # Extract subregion
    if subregion is not None:
        c_img = image.subregion(**subregion)
        concentration_img = concentration.subregion(**subregion)
    else:
        c_img = image.copy()
        concentration_img = concentration.copy()

    # Detect physical domain
    domain = concentration_img.domain

    # Visualize output
    # USe figsize maximized window
    fig = plt.figure(figsize=(37, 15))
    fig.suptitle("Original image and resulting pH values")
    ax = plt.subplot(211)
    ax.imshow(skimage.img_as_ubyte(c_img.img), extent=domain)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax = plt.subplot(212)
    im = ax.imshow(concentration_img.img, extent=domain, vmin=4, vmax=8.02)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    cbax = ax.inset_axes([1.1, 0, 0.06, 1], transform=ax.transAxes)
    cb = fig.colorbar(
        im,
        cax=cbax,
        orientation="vertical",
        label="pH",
    )

    # Allow to store plot to file
    plt.savefig(path, dpi=800, transparent=False, bbox_inches="tight")
    # And show on screen
    plt.show()


# Compare full images
if user == "helene":
    plot_path = "/Users/heleneskretting/inf100/darsia-calibration-project/results"
else:
    plot_path = "results"

# ! ---- SERIES ANALYSIS ---- !

# Goal: Track ph values over time - NOTE: There is no correlation between ph and concentration
for i, path in enumerate(experiment_path):

    # Print info
    print(f"Analyze image {path} ({i} / {len(experiment_path)})")

    # Read image
    image = darsia.imread(
        path, transformations=corrections, reference_date=baseline_image.date
    )

    # Extract pH
    ph = analysis(image.img_as(float))

    # Store image to file.
    comparison_plot(image, ph, plot_path + "/" + "ph_" + str(path.stem) + ".png")

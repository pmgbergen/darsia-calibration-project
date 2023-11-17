"""Preprocessing and setup of kernel-based concentration analysis.

This file is supposed to be a template for the BSc/DarSIA calibration project.

Images used for calibration:
    B050/DSC06608.JPG
    B050/DSC07208.JPG
    B050/DSC07332.JPG
    B050/DSC07461.JPG
    B050/DSC07587.JPG
    B050/DSC07709.JPG
    B050/DSC07833.JPG

"""
# ! ---- IMPORTS ---- !

import json
from datetime import datetime
from pathlib import Path

import darsia
import matplotlib.pyplot as plt
import numpy as np
import skimage

# ! ---- DATA MANAGEMENT ---- !

user = "helene"

# Define single baseline image
if user == "helene":
    baseline_folder = (
        "/Users/heleneskretting/inf100/darsia-calibration-project/B050/baseline_image"
    )
else:
    baseline_folder = "data/baseline_images"
baseline_path = list(sorted(Path(baseline_folder).glob("*.JPG")))[0]

# Define calibration image(s)
if user == "helene":
    calibration_folder = "/Users/heleneskretting/inf100/darsia-calibration-project/B050/calibration_images"
else:
    calibration_folder = "data/calibration_images"
calibration_path = list(sorted(Path(calibration_folder).glob("*.JPG")))[6]
num_calibration_images = len(calibration_path)

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
config = json.load(f)

drift_correction = darsia.DriftCorrection(original_baseline, **config["drift"])
color_correction = darsia.ColorCorrection(original_baseline, **config["color"])
curvature_correction = darsia.CurvatureCorrection(config=config["curvature"])
corrections = [drift_correction, color_correction, curvature_correction]

# ! ---- PREPROCESSED IMAGES ---- !

baseline_image = darsia.imread(baseline_path, transformations=corrections)
calibration_image = []
for i in range(num_calibration_images):
    calibration_image.append(
        darsia.imread(calibration_path[i], transformations=corrections)
    )

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


# MAIN STEP: Augment data with additional expert knowledge
additional_colors = np.array([
    [[0.597, 0.276, 0.036],
    [0.517, 0.22, 0.039],
    [0.538, 0.224, 0.06],
    [0.548, 0.237, 0.041],
    [0.488, 0.21, 0.039]]
)
additional_ph_value = 5 * [9]

calibration["colors"] = np.concatenate(calibration["colors"], additional_colors)
calibration["ph"] = calibration["ph"] + additional_ph_value

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

# ! ---- QUICK TEST ---- !

test_image = calibration_image[-1]

# Finally, apply the (full) concentration analysis to analyze the test image
concentration = concentration_analysis(test_image.img_as(float))

# Store solution to file
Path("results").mkdir(exist_ok=True)
np.save(Path("results/rainbow.npy"), concentration.img)

# ! ---- VIZUALIZATION

def comparison_plot(concentration, path, subregion=None):
    # Extract subregion
    if subregion is not None:
        c_img = test_image.subregion(**subregion)
        concentration_img = concentration.subregion(**subregion)
    else:
        c_img = test_image.copy()
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
    plot_path = "/Users/heleneskretting/inf100/darsia-calibration-project/results/calibration_rainbow.png"
else:
    plot_path = "results/rainbow.png"
comparison_plot(concentration, plot_path)

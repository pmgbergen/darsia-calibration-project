"""Preprocessing and setup of kernel-based concentration analysis.

This file is supposed to be a template for the BSc/DarSIA calibration project.

Calibration images used:
Folder: \\klient.uib.no\FELLES\LAB-IT\IFT\resfys\FluidFlower\FF AB original data\Bilbo 030-on VTC chemistry series\B032 CO2 BTB 0.14 mM\230510_203400_B032_injection_15s

DSC46907.JPG
DSC47341.JPG

"""
# ! ---- IMPORTS ---- !

import json
from datetime import datetime
from pathlib import Path

import darsia
import matplotlib.pyplot as plt
import skimage
import numpy as np

# ! ---- DATA MANAGEMENT ---- !

# Define single baseline image
baseline_folder = "baseline_image"
baseline_path = list(sorted(Path(baseline_folder).glob("*.JPG")))[0]

# Define calibration image(s)
calibration_folder = "calibration_image"
calibration_path = list(sorted(Path(calibration_folder).glob("*.JPG")))[0]

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
f = open(Path("config/preprocessing_2023-10-24_1143.json"))
config = json.load(f)

drift_correction = darsia.DriftCorrection(original_baseline, **config["drift"])
color_correction = darsia.ColorCorrection(original_baseline, **config["color"])
curvature_correction = darsia.CurvatureCorrection(config=config["curvature"])
corrections = [drift_correction, color_correction, curvature_correction]

# ! ---- PREPROCESSED IMAGES ---- !

baseline_image = darsia.imread(baseline_path, transformations=corrections)
calibration_image = darsia.imread(calibration_path, transformations=corrections)

# ! ---- CONCENTRATION CALIBRATION ---- !

concentration_options = {
    "base": baseline_image.img_as(float),
    "restoration -> model": False,
    "diff option": "plain",
}

restoration = darsia.TVD(
    weight=0.025, eps=1e-4, max_num_iter=100, method="isotropic Bregman"
)

# Predefine concentration analysis for now without any model (to be defined later).
co2_aq_analysis = darsia.ConcentrationAnalysis(**concentration_options)
co2_g_analysis = darsia.ConcentrationAnalysis(**concentration_options)

# The goal is to define ome ROIs for which physical information is known.
# One possibility is to use a GUI for interactive use. This option can
# be activated on demand. For testing purposes this example by default
# uses a pre-defined sample selection.
interactive_calibration = True
if interactive_calibration:
    # Same but under the use of a graphical user interface.
    # Ask user to provide characteristic regions with expected concentration values
    assistant = darsia.BoxSelectionAssistant(calibration_image, width=100)
    samples = assistant()
    concentrations_co2_aq = [
        float(x)
        for x in input("Enter corresponding CO2(aq) concentration values\n").split(", ")
    ]
    concentrations_co2_g = [
        float(x)
        for x in input("Enter corresponding CO2(g) concentration values\n").split(", ")
    ]
    assert len(samples) == len(concentrations_co2_aq)
    assert len(samples) == len(concentrations_co2_g)
else:
    # NOTE: This calibration set only works for some Bilbo experiments.
    samples = [
        (slice(2150, 2250, None), slice(4841, 4941, None)),
        (slice(2459, 2559, None), slice(4075, 4175, None)),
        (slice(971, 1071, None), slice(4399, 4499, None)),
    ]
    concentrations_co2_aq = [1.0, 0.0, 0.0]
    concentrations_co2_g = [0.0, 0.0, 1.0]

# Now add kernel interpolation as model trained by the extracted information.
smooth_RGB = co2_aq_analysis(calibration_image.img_as(float)).img
colours_RGB = darsia.extract_characteristic_data(
    signal=smooth_RGB, samples=samples, verbosity=True, surpress_plot=True
)

# Collect calibration data
calibration_config = {
    "colors": colours_RGB,
    "concentrations_co2_aq": concentrations_co2_aq,
    "concentrations_co2_g": concentrations_co2_g,
}
# Make json compatible
calibration_config["colors"] = calibration_config["colors"].tolist()

# Store config to file and use current datetime as ID
date = datetime.now().strftime("%Y-%m-%d %H%M")
Path("config").mkdir(exist_ok=True)
with open(Path(f"config/calibration_{date}.json"), "w") as output:
    json.dump(calibration_config, output)

# ! ---- QUICK TEST ---- !

kernel_interpolation_co2_aq = darsia.KernelInterpolation(
    darsia.GaussianKernel(gamma=9.73), colours_RGB, concentrations_co2_aq
)
kernel_interpolation_co2_g = darsia.KernelInterpolation(
    darsia.GaussianKernel(gamma=9.73), colours_RGB, concentrations_co2_g
)
clip = darsia.ClipModel(**{"min value": 0, "max value": 1})
co2_aq_analysis.restoration = restoration
co2_g_analysis.restoration = restoration
co2_aq_analysis.model = darsia.CombinedModel([kernel_interpolation_co2_aq, clip])
co2_g_analysis.model = darsia.CombinedModel([kernel_interpolation_co2_g, clip])

# Finally, apply the (full) concentration analysis to analyze the test image
co2_aq_concentration_image = co2_aq_analysis(calibration_image.img_as(float)).img
co2_g_concentration_image = co2_g_analysis(calibration_image.img_as(float)).img

# Visualize output
fig = plt.figure()
fig.suptitle("Original image and resulting concentrations")
ax = plt.subplot(311)
ax.imshow(skimage.img_as_ubyte(calibration_image.img))
ax = plt.subplot(312)
im_co2_aq = ax.imshow(co2_aq_concentration_image)
cb_co2_aq = fig.colorbar(
    im_co2_aq, orientation="vertical", label="concentration CO2(aq) [%]"
)
ax = plt.subplot(313)
im_co2_g = ax.imshow(co2_g_concentration_image)
cb_co2_g = fig.colorbar(
    im_co2_g, orientation="vertical", label="concentration CO2(g) [%]"
)
plt.show()

# Store solution to file
Path("results_ingvild").mkdir(exist_ok=True)
np.save(Path("results_ingvild/concentration_co2_g.npy"), co2_g_concentration_image)
np.save(Path("results_ingvild/concentration_co2_aq.npy"), co2_aq_concentration_image)
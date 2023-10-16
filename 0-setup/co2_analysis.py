"""Preprocessing and setup of kernel-based concentration analysis.

Applied to: TODO add description

This file is supposed to be a template for the BSc/DarSIA calibration project.

"""

# ! ---- IMPORTS ---- !

from pathlib import Path

import darsia
import json

import numpy as np
import matplotlib.pyplot as plt
import skimage

# ! ---- DATA MANAGEMENT ---- !

# Define single baseline image
baseline_folder = "data/baseline_images"
baseline_path = list(sorted(Path(baseline_folder).glob("*.JPG")))[0]

# Define calibration image(s)
calibration_folder = "data/calibration_images"
calibration_path = list(sorted(Path(calibration_folder).glob("*.JPG")))[0]

# Define experiment images
experiment_folder = "data/experiment_images"
experiment_path = list(sorted(Path(experiment_folder).glob("*.JPG")))[5:10]

# ! ---- UNMODIFIED BASELINE IMAGE ---- !
original_baseline = darsia.imread(baseline_path)

# ! ---- CORRECTION MANAGEMENT ---- !

# Read config from json file
f = open(Path("config/preprocessing.json"))
config = json.load(f)

# First correction - drift correction: This image will be used as reference to
# align other images through pure translation wrt some chosen ROI - e.g.
# the color checker.
drift_correction = darsia.DriftCorrection(original_baseline, **config["drift"])

# Second correction - curvature correction: Crop image mainly. It is
# based on the unmodifed baseline image. All images are assumed to be
# aligned with that one.

# Read the instructions of the assistant in the terminal
curvature_correction = darsia.CurvatureCorrection(config=config["curvature"])

# ! ---- PREPROCESSED IMAGES ---- !

baseline_image = darsia.imread(
    baseline_path, transformations=[drift_correction, curvature_correction]
)

calibration_image = darsia.imread(
    calibration_path, transformations=[drift_correction, curvature_correction]
)

# ! ---- MAIN CONCENTRATION ANALYSIS AND CALIBRATION ---- !

# Predefine concentration analysis for now without any model (to be defined later).
co2_aq_analysis = darsia.ConcentrationAnalysis(
    base=baseline_image.img_as(float),
    restoration=darsia.TVD(
        weight=0.025, eps=1e-4, max_num_iter=100, method="isotropic Bregman"
    ),
    **{"diff option": "plain"},
)
co2_g_analysis = darsia.ConcentrationAnalysis(
    base=baseline_image.img_as(float),
    restoration=darsia.TVD(
        weight=0.025, eps=1e-4, max_num_iter=100, method="isotropic Bregman"
    ),
    **{"diff option": "plain"},
)

# The goal is to define ome ROIs for which physical information is known.
# One possibility is to use a GUI for interactive use. This option can
# be activated on demand. For testing purposes this example by default
# uses a pre-defined sample selection.
if True:
    samples = [(slice(2150, 2250, None), slice(4841, 4941, None)), (slice(2459, 2559, None), slice(4075, 4175, None)), (slice(971, 1071, None), slice(4399, 4499, None))]
    concentrations_co2_aq = [1., 0., 0.]
    concentrations_co2_g = [0., 0., 1.]
else:
    # Same but under the use of a graphical user interface.
    # Ask user to provide characteristic regions with expected concentration values
    assistant = darsia.BoxSelectionAssistant(calibration_image)
    samples = assistant()
    co2_aq_concentrations = [float(x) for x in input("Enter corresponding CO2(aq) concentration values\n").split(', ')] 
    co2_g_concentrations = [float(x) for x in input("Enter corresponding CO2(g) concentration values\n").split(', ')] 
    assert len(samples) == len(concentrations_co2_aq) 
    assert len(samples) == len(concentrations_co2_g) 

# Now add kernel interpolation as model trained by the extracted information.
smooth_RGB = co2_aq_analysis(calibration_image.img_as(float)).img
colours_RGB = darsia.extract_characteristic_data(
    signal=smooth_RGB, samples=samples, verbosity=True, surpress_plot=True
)
kernel_interpolation_co2_aq = darsia.KernelInterpolation(
    darsia.GaussianKernel(gamma=9.73), colours_RGB, concentrations_co2_aq
)
kernel_interpolation_co2_g = darsia.KernelInterpolation(
    darsia.GaussianKernel(gamma=9.73), colours_RGB, concentrations_co2_g
)
clip = darsia.ClipModel(**{"min value": 0, "max value": 1})
co2_aq_analysis.model = darsia.CombinedModel([kernel_interpolation_co2_aq, clip])
co2_g_analysis.model = darsia.CombinedModel([kernel_interpolation_co2_g, clip])

# Finally, apply the (full) concentration analysis to analyze the test image
co2_aq_concentration_image = co2_aq_analysis(calibration_image.img_as(float)).img
co2_g_concentration_image = co2_g_analysis(calibration_image.img_as(float)).img

# ! ----- VISUALISATION ---- !

fig = plt.figure()
fig.suptitle("Original image and resulting concentrations")
ax = plt.subplot(311)
ax.imshow(skimage.img_as_ubyte(calibration_image.img))
ax.set_ylabel("vertical pixel")
ax.set_xlabel("horizontal pixel")
ax = plt.subplot(312)
ax.imshow(co2_aq_concentration_image)
ax.set_ylabel("vertical pixel")
ax.set_xlabel("horizontal pixel")
ax = plt.subplot(313)
ax.imshow(co2_g_concentration_image)
ax.set_ylabel("vertical pixel")
ax.set_xlabel("horizontal pixel")
plt.show()

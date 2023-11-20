"""Preprocessing and setup of kernel-based concentration analysis.

This file is supposed to be a template for the BSc/DarSIA calibration project.

"""
# ! ---- IMPORTS ---- !

import json
from pathlib import Path

import darsia
import matplotlib.pyplot as plt
import numpy as np

# ! ---- DATA MANAGEMENT ---- !

# Define single baseline image
baseline_folder = "images"
baseline_path = list(sorted(Path(baseline_folder).glob("*.JPG")))[0]

# Define experiment images
experiment_folder = "mass_images"
experiment_path = list(sorted(Path(experiment_folder).glob("*.JPG")))

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
f = open(Path("config\preprocessing_2023-10-20 1237.json"))
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
f = open(Path("config\calibration_2023-10-20 1318.json"))
calibration = json.load(f)
calibration["colors"] = np.array(calibration["colors"])

# Define models (clip values)
kernel_interpolation_co2_aq = darsia.KernelInterpolation(
    darsia.GaussianKernel(gamma=9.73),
    calibration["colors"],
    calibration["concentrations_co2_aq"],
)
kernel_interpolation_co2_g = darsia.KernelInterpolation(
    darsia.GaussianKernel(gamma=9.73),
    calibration["colors"],
    calibration["concentrations_co2_g"],
)
clip = darsia.ClipModel(**{"min value": 0, "max value": 1})
co2_aq_model = darsia.CombinedModel([kernel_interpolation_co2_aq, clip])
co2_g_model = darsia.CombinedModel([kernel_interpolation_co2_g, clip])

# Define concentration analysis for now without any model (to be defined later).
co2_aq_analysis = darsia.ConcentrationAnalysis(
    model=co2_aq_model, **concentration_options
)
co2_g_analysis = darsia.ConcentrationAnalysis(
    model=co2_g_model, **concentration_options
)

# ! ---- SERIES ANALYSIS ---- !

# Goal: Track co2 mass based on concentrations only
# Apply simple formula (as in benchmark): m = rho_g * s + c * (1-s),
# where rho_g = 2 kg * m**(-3), s denotes the volumetric concentration of CO2(g)
# with values between 0 and 1, and c denotes the mass concentration of CO2(aq).

time = []
mass = []

for i, path in enumerate(experiment_path):

    # Print info
    print(f"Analyze image {path} ({i} / {len(experiment_path)})")

    # Read image
    image = darsia.imread(
        path, transformations=corrections, reference_date=baseline_image.date
    )

    # Extract concentrations
    co2_aq = co2_aq_analysis(image.img_as(float))
    co2_g = co2_g_analysis(image.img_as(float))

    # Integrate over area
    shape_metadata = image.shape_metadata()
    porosity = 0.44  # [-]
    depth = 0.02  # m
    geometry = darsia.ExtrudedPorousGeometry(
        porosity=porosity, depth=depth, **shape_metadata
    )

    volume_co2_aq = geometry.integrate(co2_aq)
    volume_co2_g = geometry.integrate(co2_g)

    # Define mass distribution
    rho_co2_aq = 1  # kg * m**(-3) [assume same density as water]
    rho_co2_g = 1.8  # kg * m**(-3)
    mass_co2 = volume_co2_aq * rho_co2_aq + volume_co2_g * rho_co2_g

    # Store informations
    time.append(image.time)
    mass.append(mass_co2)

# Plot CO2 mass evolution
fig = plt.figure()
fig.suptitle("CO2 mass evolution")
plt.plot(time, mass)
plt.xlabel("time [s]")
plt.ylabel("mass [kg]")

plt.show()

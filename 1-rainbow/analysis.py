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

user = "ingvild"  # "helene"

# Define single baseline image
if user == "helene":
    baseline_folder = None  # TODO
elif user == "ingvild":
    baseline_folder = r"C:\Users\Bruker\Documents\GitHub\darsia-calibration-project\1-rainbow\co2_images"  # TODO

else:
    baseline_folder = "data/baseline_images"
baseline_path = list(sorted(Path(baseline_folder).glob("*.JPG")))[0]

# Define experiment images
if user == "helene":
    experiment_folder = None  # TODO
if user == "ingvild":
    experiment_folder = r"C:\Users\Bruker\Documents\GitHub\darsia-calibration-project\1-rainbow\co2_images"  # TODO
else:
    experiment_folder = "data/experiment_images"
experiment_path = list(sorted(Path(experiment_folder).glob("*.JPG")))[6:7]

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
elif user == "ingvild":
    f = open(
        Path(
            r"C:\Users\Bruker\Documents\GitHub\darsia-calibration-project\1-rainbow\config\preprocessing_2023-11-16_1505_02.json"
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
calibration_path = Path(r"C:\Users\Bruker\Documents\GitHub\darsia-calibration-project\1-rainbow\config\calibration_2023-11-16_1418_01.json")  # TODO
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


def comparison_plot(image, concentration, density, path, subregion=None):
    # Extract subregion
    if subregion is not None:
        c_img = image.subregion(**subregion)
        concentration_img = concentration.subregion(**subregion)
        density_img = density.subregion(**subregion)
    else:
        c_img = image.copy()
        concentration_img = concentration.copy()
        density_img = density.copy()

    # Detect physical domain
    domain = concentration_img.domain

    # Visualize output
    # USe figsize maximized window
    fig = plt.figure(figsize=(37, 15))
    fig.suptitle("Original image and resulting pH values")
    ax = plt.subplot(311)
    ax.imshow(skimage.img_as_ubyte(c_img.img), extent=domain)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax = plt.subplot(312)
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
    ax = plt.subplot(313)
    im = ax.imshow(density_img.img, extent=domain, vmin=0, vmax=10)  # TODO
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    cbax = ax.inset_axes([1.1, 0, 0.06, 1], transform=ax.transAxes)
    cb = fig.colorbar(
        im,
        cax=cbax,
        orientation="vertical",
        label="density (g/m**3)",
    )

    # Allow to store plot to file
    plt.savefig(path, dpi=800, transparent=False, bbox_inches="tight")
    # And show on screen
    plt.show()


# Compare full images
if user == "helene":
    plot_path = "/Users/heleneskretting/inf100/darsia-calibration-project/results"
elif user == "ingvild":
    plot_path = r"C:\Users\Bruker\Documents\GitHub\darsia-calibration-project\1-rainbow\results"
else:
    plot_path = "results"

# ! ---- SERIES ANALYSIS ---- !

# Goal: Track ph values over time - NOTE: There is no correlation between ph and concentration
time = []  # in seconds
mass = []  # in gramms
for i, path in enumerate(experiment_path):

    # Print info
    print(f"Analyze image {path} ({i} / {len(experiment_path)})")

    # Read image
    image = darsia.imread(
        path, transformations=corrections, reference_date=baseline_image.date
    )

    # Extract pH
    ph = analysis(image.img_as(float))

    # Convert from pH to density in mol / m**(-3) and g / m**(-3)
    data_ph = [4, 5, 6, 7, 8]
    data_mM = [33, 4.3, 1.4, 1.0, 0.0]
    density_mM = ph.copy()
    density_mM.img = np.interp(ph.img, data_ph, data_mM)
    density_CO2 = 44.01  # g / mol # TODO
    density_g = density_mM.copy()
    density_g.img *= density_CO2

    # Compute total mass in g over time
    shape_metadata = image.shape_metadata()
    porosity = 0.44  # [-]
    depth = 0.02  # m
    geometry = darsia.ExtrudedPorousGeometry(
        porosity=porosity, depth=depth, **shape_metadata
    )
    mass_co2 = geometry.integrate(density_g)

    # Store informations
    time.append(image.time)
    mass.append(mass_co2)

    print(f"Time: {time}")
    print(f"Mass: {mass}")

    # Store image to file.
    comparison_plot(
        image, ph, density_g, plot_path + "/" + "ph_" + str(path.stem) + ".png"
    )

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

user = "ingvild"

# Define single baseline image
if user == "helene":
    baseline_folder = (
        "/Users/heleneskretting/inf100/darsia-calibration-project/B050/baseline_image" 
    )
elif user == "ingvild":
    baseline_folder = (
        r"C:\Users\Bruker\Documents\GitHub\darsia-calibration-project\1-rainbow\rainbow_images"
    )
else:
    baseline_folder = "data/baseline_images"
baseline_path = list(sorted(Path(baseline_folder).glob("*.JPG")))[0]

# Define calibration image(s)
if user == "helene":
    calibration_folder = "/Users/heleneskretting/inf100/darsia-calibration-project/B050/calibration_image"
elif user == "ingvild":
    calibration_folder = r"C:\Users\Bruker\Documents\GitHub\darsia-calibration-project\1-rainbow\rainbow_images"
else:
    calibration_folder = "data/calibration_images"
calibration_path = list(sorted(Path(calibration_folder).glob("*.JPG")))[
    0:7
]

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
elif user == "ingvild":
    f = open(
        Path(
            r"C:\Users\Bruker\Documents\GitHub\darsia-calibration-project\B033\config\preprocessing_2023-10-24_1306.json"
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
concentration_analysis = darsia.ConcentrationAnalysis(**concentration_options)

# The goal is to define ome ROIs for which physical information is known.
# One possibility is to use a GUI for interactive use. This option can
# be activated on demand. For testing purposes this example by default
# uses a pre-defined sample selection.
interactive_calibration = True
point_selection_image = calibration_image[-1]
if interactive_calibration:
    # Same but under the use of a graphical user interface.
    # Ask user to provide characteristic regions with expected concentration values
    assistant = darsia.BoxSelectionAssistant(point_selection_image, width=100)
    samples = assistant()
else:
    samples = None


# TODO: Enter the correct concentrations for the calibration images
ph = [10, 4.01,  4.52, 5, 6.03,7, 8.02]
assert len(calibration_image) == len(ph), "Input not correct."

# Now add kernel interpolation as model trained by the extracted information.
all_colours = []
ph_RGB = []
for i in range(num_calibration_images):
    # Fetch calibration images
    image = calibration_image[i]

    # Smooth image
    smooth_RGB = concentration_analysis(image.img_as(float)).img

    # Fetch characteristic colors from samples
    colours_RGB = darsia.extract_characteristic_data(
        signal=smooth_RGB, samples=samples, verbosity=True, surpress_plot=True
    )

    # Choose unique colors
    unique_colours_RGB = np.unique(np.round(colours_RGB, decimals=5), axis=0)
    num_unique_colours = unique_colours_RGB.shape[0]
    all_colours.append(unique_colours_RGB)

    # Assign concentration values to all samples
    ph_RGB = ph_RGB + num_unique_colours * [ph[i]]
colours_RGB = np.concatenate(all_colours)

# Collect calibration data
calibration_config = {
    "colors": colours_RGB,
    "ph": ph_RGB,
}
# Make json compatible
calibration_config["colors"] = calibration_config["colors"].tolist()

# Store config to file and use current datetime as ID
date = datetime.now().strftime("%Y-%m-%d %H%M")
Path("config").mkdir(exist_ok=True)
with open(Path(f"config/calibration_{date}.json"), "w") as output:
    json.dump(calibration_config, output)

# ! ---- DEFINE CONCENTRATION ANALYSIS ---- !

print(len(colours_RGB), colours_RGB)
print(len(ph_RGB), ph_RGB)

kernel_interpolation = darsia.KernelInterpolation(
    darsia.GaussianKernel(gamma=9.73), colours_RGB, ph_RGB
)
# clip = darsia.ClipModel(**{"min value": 0, "max value": 1})
# concentration_analysis.model = darsia.CombinedModel([kernel_interpolation, clip])
concentration_analysis.model = kernel_interpolation
concentration_analysis.restoration = restoration

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
    im = ax.imshow(concentration_img.img, extent=domain, vmin=3, vmax=11)
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

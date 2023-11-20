import numpy as np
from pathlib import Path
import darsia
import matplotlib.pyplot as plt
import skimage
import json

"""
Images used for calibration:
Folder: \\klient.uib.no\FELLES\LAB-IT\IFT\resfys\FluidFlower\FF AB original data\Bilbo 030-on VTC chemistry series\B033 CO2 BTB 0.75 mM

DSC46000.JPG
DSC46422.JPG

"""

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
f = open(Path("config\preprocessing_2023-10-24_1306.json"))
config = json.load(f)

drift_correction = darsia.DriftCorrection(original_baseline, **config["drift"])
color_correction = darsia.ColorCorrection(original_baseline, **config["color"])
curvature_correction = darsia.CurvatureCorrection(config=config["curvature"])
corrections = [drift_correction, color_correction, curvature_correction]

# ! ---- PREPROCESSED IMAGES ---- !

baseline_image = darsia.imread(baseline_path, transformations=corrections)
calibration_image = darsia.imread(calibration_path, transformations=corrections)

# First calibration data
co2_g_03 = np.load(Path("results_ingvild\concentration_co2_g_03.npy"))
co2_aq_03 = np.load(Path("results_ingvild\concentration_co2_aq_03.npy"))

# Second calibration data
co2_g_04 = np.load(Path("results_ingvild\concentration_co2_g_04.npy"))
co2_aq_04 = np.load(Path("results_ingvild\concentration_co2_aq_04.npy"))

# Difference of concentrations, simple method
#co2_g_diff = np.abs(co2_g_03 - co2_g_04)
#co2_aq_diff = np.abs(co2_aq_03 - co2_aq_04)

# Difference of concentrations, integral method
#co2_g_diff = np.sum(np.abs(co2_g_03 - co2_g_04)) / (np.sum(co2_g_03 + co2_aq_03))
#co2_aq_diff = np.sum(np.abs(co2_aq_03 - co2_aq_04)) / (np.sum(co2_aq_03 + co2_aq_03))

# Difference of concentrations, volume method
co2_g_diff = (np.sum(co2_g_03) - np.sum(co2_g_04)) / (np.sum(co2_g_03 + co2_aq_03))
co2_aq_diff = (np.sum(co2_aq_03) - np.sum(co2_aq_04)) / (np.sum(co2_g_03 + co2_aq_03))


#Physical domain
domain = (0.0, 0.899, 0.0, 0.55)
'''
# Visual comparison
fig = plt.figure(figsize=(37, 15))
fig.suptitle("Original image and resulting concentrations")
ax = plt.subplot(311)
ax.imshow(skimage.img_as_ubyte(calibration_image.img), extent=domain)
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax = plt.subplot(312)
im_co2_aq = ax.imshow(co2_aq_diff * 100, vmin=0, vmax=100)
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
cbax = ax.inset_axes([1.1, 0, 0.06, 1], transform=ax.transAxes)
cb_co2_aq = fig.colorbar(
    im_co2_aq, cax=cbax, orientation="vertical", label="concentration difference CO2(aq) [%]"
)
ax = plt.subplot(313)
im_co2_g = ax.imshow(co2_g_diff * 100, vmin=0, vmax=100)
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
cbax = ax.inset_axes([1.1, 0, 0.06, 1], transform=ax.transAxes)
cb_co2_g = fig.colorbar(
    im_co2_g, cax=cbax, orientation="vertical", label="concentration difference CO2(g) [%]"
)
'''
# Allow to store plot to file
#plt.savefig("comparisons\img.png", dpi=800, transparent=False, bbox_inches="tight")

#plt.show()
print('integral aq', np.abs(co2_aq_diff))
print('integral g', np.abs(co2_g_diff))
print('integral sum', np.abs(co2_aq_diff)+np.abs(co2_g_diff))

#print('volum aq', np.abs(co2_aq_diff))
#print('volum g', np.abs(co2_g_diff))
#print('volum sum', np.abs(co2_aq_diff)+np.abs(co2_g_diff))




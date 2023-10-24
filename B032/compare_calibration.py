import numpy as np
from pathlib import Path
import darsia

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
f = open(Path("config/preprocessing_2023-10-23 1918.json"))
config = json.load(f)

drift_correction = darsia.DriftCorrection(original_baseline, **config["drift"])
color_correction = darsia.ColorCorrection(original_baseline, **config["color"])
curvature_correction = darsia.CurvatureCorrection(config=config["curvature"])
corrections = [drift_correction, color_correction, curvature_correction]

# ! ---- PREPROCESSED IMAGES ---- !

baseline_image = darsia.imread(baseline_path, transformations=corrections)
calibration_image = darsia.imread(calibration_path, transformations=corrections)

# Ingvild data
co2_g_ingvild = np.load(Path("results_ingvild/concentration_co2_g.npy"))
co2_aq_ingvild = np.load(Path("results_ingvild/concentration_co2_aq.npy"))

# Helene data
co2_g_helene = np.load(Path("results_helene/concentration_co2_g.npy"))
co2_aq_helene = np.load(Path("results_helene/concentration_co2_aq.npy"))

# Difference of concentrations
co2_g_diff = co2_g_ingvild - co2_g_helene
co2_aq_diff = co2_aq_ingvild - co2_aq_helene

# Visual comparison
fig = plt.figure()
fig.suptitle("Original image and resulting concentrations")
ax = plt.subplot(311)
ax.imshow(skimage.img_as_ubyte(calibration_image.img))
ax = plt.subplot(312)
im_co2_aq = ax.imshow(co2_aq_diff * 100)
cb_co2_aq = fig.colorbar(
    im_co2_aq, orientation="vertical", label="concentration CO2(aq) [%]"
)
ax = plt.subplot(313)
im_co2_g = ax.imshow(co2_g_diff * 100)
cb_co2_g = fig.colorbar(
    im_co2_g, orientation="vertical", label="concentration CO2(g) [%]"
)
plt.show()
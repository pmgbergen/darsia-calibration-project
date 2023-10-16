"""Setup of preprocessing.

This file is supposed to be a template for the BSc/DarSIA calibration project.

"""

# ! ---- IMPORTS ---- !

from pathlib import Path

import darsia
import json

# ! ---- DATA MANAGEMENT ---- !

# Define single baseline image
baseline_folder = "data/baseline_images"
baseline_path = list(sorted(Path(baseline_folder).glob("*.JPG")))[0]

# ! ---- UNMODIFIED BASELINE IMAGE ---- !
original_baseline = darsia.imread(baseline_path)

# ! ---- CORRECTION MANAGEMENT ---- !

# Start with empty config file.
config = {}

# ! ---- DARSIA IMAGES ---- !

# First correction - drift correction: This image will be used as reference to
# align other images through pure translation wrt some chosen ROI - e.g.
# the color checker.

# Define ROI corresponding to colorchecker
# Use assistant
point_selector = darsia.PointSelectionAssistant(original_baseline)
config["drift"] = {}
config["drift"]["roi"] = point_selector()
# Later use: drift_correction = darsia.DriftCorrection(baseline_darsia_image, **config["drift"])

# Second correction - curvature correction: Crop image mainly. It is
# based on the unmodifed baseline image. All images are assumed to be
# aligned with that one.

# Read the instructions of the assistant in the terminal
crop_assistant = darsia.CropAssistant(original_baseline)
config["curvature"] = crop_assistant()
# Later use: curvature_correction = darsia.CurvatureCorrection(config=config["crop"])

# Before storing, need to make dict json serializable
config["drift"]["roi"] = config["drift"]["roi"].tolist()

# Store config to file
with open(Path("config/preprocessing.json"), 'w') as output:
    json.dump(config, output)

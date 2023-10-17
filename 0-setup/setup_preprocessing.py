"""Setup of preprocessing.

This file is supposed to be a template for the BSc/DarSIA calibration project.

"""

# ! ---- IMPORTS ---- !

import json
from datetime import datetime
from pathlib import Path

import darsia

# ! ---- DATA MANAGEMENT ---- !

# Define single baseline image
baseline_folder = "data/baseline_images"
baseline_path = list(sorted(Path(baseline_folder).glob("*.JPG")))[0]

# ! ---- UNMODIFIED BASELINE IMAGE ---- !
original_baseline = darsia.imread(baseline_path)

# ! ---- CORRECTION MANAGEMENT ---- !

# Idea: Apply three corrections:
# 1. Drift correction aligning images by simple translation with respect to teh color checker.
# 2. Color correction applying uniform colors in the color checker.
# 3. Curvature correction to crop images to the right rectangular format.
# The order has to be applied in later scripts as well.

# Start with empty config file.
config = {}

# 1. Drift correction: Define ROI corresponding to colorchecker
point_selector = darsia.PointSelectionAssistant(original_baseline)
config["drift"] = {}
config["drift"]["roi"] = point_selector()
# Later use: drift_correction = darsia.DriftCorrection(baseline_darsia_image, **config["drift"])

# 2. Color correction: Mark the four corners of the color checker
point_selector = darsia.PointSelectionAssistant(original_baseline)
config["color"] = {}
config["color"]["roi"] = point_selector()
# Later use: color_correction = darsia.ColorCorrection(**config["drift"])

# 3. Curvature correction: Crop image mainly. It is based on the unmodifed baseline image. All
# later images are assumed to be aligned with that one. Bulge effects are neglected.
crop_assistant = darsia.CropAssistant(original_baseline)
config["curvature"] = crop_assistant()
# Later use: curvature_correction = darsia.CurvatureCorrection(config=config["crop"])

# ! ---- STORE CONFIG TO FILE ---- !

# Before storing, need to make dict json serializable
config["drift"]["roi"] = config["drift"]["roi"].tolist()
config["color"]["roi"] = config["color"]["roi"].tolist()

# Store config to file and use current datetime as ID
date = datetime.now().strftime("%Y-%m-%d %H:%M")
with open(Path(f"config/preprocessing_{date}.json"), "w") as output:
    json.dump(config, output)

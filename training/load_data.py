import os
from datetime import datetime
from datetime import timedelta
import numpy as np
import numpy.random

import strawberry
import training_util

# Global image only
GLOBAL_ONLY = 0
# Direct and global image stacked together
STACK_ID_IG = 1
# Direct / (Direct + Global + 1e-10)
ID_IG_RATIO = 2
# Epipolar delay profiles, each normalized so area under curve is 1
EPIPOLAR = 3
# Point array patches
PA_PATCH = 4
# Radial average of point array patches
PA_PATCH_RADIAL_AVG = 5
# SIRI AC Images
SIRI_AC = 6
# SIRI AC/DC Images
SIRI_AC_DC = 7
# Approximate direct/global images from epipolar captures
# If using single channel, use channels 25 and 30
EPIPOLAR_PAIR = 8
# Epipolar images without direct channels (24,25,26)
EPIPOLAR_INDIRECT = 9
# Direct/Global and epipolar delay images combined
EPIPOLAR_DG_STACK = 10
# Epipolar single shot captures
EPIPOLAR_SINGLE_SHOT = 11
# Direct/global single shot captures
DG_SINGLE_SHOT = 12
# RGB single shot captures
RGB = 13
# SIRI single shot
SIRI_SINGLE_SHOT = 14
# PA single shot - same as PA_PATCH, but evaluation uses a single image
PA_SINGLE_SHOT = 15
# Epilines - same as EPIPOLAR_SINGLE_SHOT, but evaluation uses every epiline capture
EPILINES = 16

# Point arrays, # captures used for inference
PA_1 = 21
PA_2 = 22
PA_3 = 23
PA_4 = 24


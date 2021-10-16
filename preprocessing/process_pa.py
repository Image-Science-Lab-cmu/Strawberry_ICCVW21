import numpy as np
import os
import util
from PIL import Image

# Reads captures and returns corrected captures
def process_pa(base_path, mask_path, num_capture, num_rotation, rows,
               cols, white_ratio, keep_channel, crop_params):
    path = os.path.join(base_path, str(num_capture)+"-r"+str(num_rotation), "pa")
    raw_captures = util.read_ordered_captures(os.path.join(path, "captures"), rows, cols)
    captures = util.normalize_imgs(raw_captures)

    Iwhite, Idark = util.read_ref_imgs(path, rows, cols)
    mask = np.asarray(Image.open(mask_path))

    captures = captures[:,:,keep_channel,:]
    Iwhite = Iwhite[:,:,keep_channel]
    Idark = Idark[:,:,keep_channel]

    Iwhite /= white_ratio

    captures = util.illumination_correction(captures, Iwhite, Idark, 1)
    captures = util.crop_captures(captures, crop_params)

    for i in range(captures.shape[2]):
        captures[:,:,i] = captures[:,:,i] * mask

    return captures


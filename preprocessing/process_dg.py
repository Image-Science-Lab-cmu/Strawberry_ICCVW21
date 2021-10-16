import numpy as np
import os
from PIL import Image

import util

def dg_separation(Imin, Imax, a, b):
    Id = np.zeros(Imin.shape)
    Ig = np.zeros(Imin.shape)

    if a == 0.5 and b == 0:
        Ig = Imin * 2
        Id = Imax - Imin
    else:
        a = np.array([[1, a+b*(1-a)], [b, 1-a+a*b]])
        for i in range(0, len(Imin)):
            b = np.array([Imax[i], Imin[i]])
            Id[i], Ig[i] = np.linalg.solve(a, b)


    return Id, Ig

def process_dg(base_path, mask_path, num_capture, num_rotation, rows, cols,
               keep_channel, a, b, crop_params, white_ratio=1, spectralon_reflectance=1):
    path = os.path.join(base_path, str(num_capture)+"-r"+str(num_rotation), "dg")

    raw_captures = util.read_imgs(os.path.join(path, "captures"), rows, cols)
    captures = util.normalize_imgs(raw_captures)

    Iwhite, Idark = util.read_ref_imgs(path, rows, cols)

    Iwhite = Iwhite / white_ratio

    captures = captures[:,:,keep_channel,:]
    Iwhite = Iwhite[:,:,keep_channel]
    Idark = Idark[:,:,keep_channel]

    Imin = np.amin(captures, axis=2)
    Imax = np.amax(captures, axis=2)

    # Uniform illumination correction
    Imin = util.illumination_correction(Imin, Iwhite, Idark, spectralon_reflectance)
    Imax = util.illumination_correction(Imax, Iwhite, Idark, spectralon_reflectance)
    captures_corr = util.illumination_correction(captures, Iwhite, Idark, spectralon_reflectance).astype(np.float32)

    # Load mask
    mask = np.asarray(Image.open(mask_path))

    # Crop images
    Imin = util.crop_captures(Imin, crop_params)
    Imax = util.crop_captures(Imax, crop_params)

    # Mask images
    Imin *= mask
    Imax *= mask

    Id, Ig = dg_separation(Imin, Imax, a, b)

    return Id, Ig, captures_corr


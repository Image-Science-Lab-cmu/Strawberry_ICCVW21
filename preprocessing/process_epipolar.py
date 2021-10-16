import numpy as np
import os
import util
from PIL import Image
import h5py

def center_T(T, peak_width, max_index):
    T_centered = np.zeros((T.shape[0], T.shape[1], peak_width), dtype=np.float32)

    for r in range(0, T.shape[0]):
        for c in range(0, T.shape[1]):
            rang = np.arange(max_index[r,c]+int(np.ceil(-peak_width/2)), max_index[r,c]+int(np.ceil(peak_width/2)))
            index_of_zero = np.where(rang == 0)
            if len(index_of_zero[0]) == 0:
                start = 0
            else:
                start = index_of_zero[0][0]

            rang = rang[np.where((rang >= 0) & (rang < T.shape[2]))]
            T_centered[r,c,start:start+len(rang)] = T[r,c,rang]

    return T_centered

# Find T_centered from epipolar captures path
# Masks the computed images based on the mask at path mask_path
def process_epipolar(base_path, mask_path, num_capture, num_rotation, rows,
                     cols, white_ratio, peak_width, keep_channel, crop_params,
                     spectralon_reflectance=1):
    path = os.path.join(base_path, str(num_capture)+"-r"+str(num_rotation), "epipolar")

    # Load captures
    raw_captures = util.read_ordered_captures(os.path.join(path, "captures"), rows, cols)
    captures = util.normalize_imgs(raw_captures)

    Iwhite_captures = util.read_imgs(os.path.join(path, "Iwhite"), rows, cols)
    Idark_captures = util.read_imgs(os.path.join(path, "Idark"), rows, cols)

    Iwhite = util.normalize_imgs(Iwhite_captures) / white_ratio
    Iwhite = np.mean(Iwhite, axis=3)
    Idark = np.mean(Idark_captures.astype('float64') / 255, axis=3)

    # Load mask
    mask = np.asarray(Image.open(mask_path))

    # Crop captures
    captures = util.crop_captures(captures, crop_params)
    Iwhite = util.crop_captures(Iwhite, crop_params)
    Idark = util.crop_captures(Idark, crop_params)

    # Save only the red channel
    captures = captures[:,:,keep_channel,:]
    Iwhite = Iwhite[:,:,keep_channel]
    Idark = Idark[:,:,keep_channel]

    # Uniform illumination correction
    captures_corr = util.illumination_correction(captures, Iwhite, Idark, spectralon_reflectance)

    # Mask corrected captures
    for i in range(0, captures_corr.shape[2]):
        captures_corr[:,:,i] *= mask

    # Compute and center T
    T = np.zeros((captures_corr.shape[0], captures_corr.shape[1], captures_corr.shape[2]*3), dtype=np.float32)
    for i in range(0,3):
        T[:,:,i*captures_corr.shape[2]:(i+1)*captures_corr.shape[2]] = captures_corr

    max_index = np.argmax(T, axis=2)
    max_index = max_index + captures_corr.shape[2]
    T_centered = center_T(T, peak_width, max_index)

    return T_centered, captures_corr.astype(np.float32)


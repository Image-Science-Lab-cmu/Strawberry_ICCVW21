import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

def center_crop(imgs, crop_size):
    """
    Crop imgs (NxHxW or HxW) to crop_size
    """
    assert (imgs.ndim == 3) or (imgs.ndim == 2)

    if imgs.ndim == 3:
        r_dim, c_dim = (imgs.shape[1], imgs.shape[2])
    else:
        r_dim, c_dim = imgs.shape

    start_r = r_dim // 2 - crop_size[0] // 2
    end_r = start_r + crop_size[0]
    start_c = c_dim // 2 - crop_size[1] // 2
    end_c = start_c + crop_size[1]

    if imgs.ndim == 3:
        return imgs[:,start_r:end_r,start_c:end_c]
    else:
        return imgs[start_r:end_r,start_c:end_c]

def normalize_imgs(imgs):
    return imgs.astype(np.float32) / 255

# Read ordered capture files whose name include an integer at the beginning
# Only implemented for num_frames = 1
# Ex. 10-1.png
def read_ordered_captures(path, rows, cols):
    files = os.listdir(path)
    files.sort(key=lambda s: int(s.split("-")[0]))
    imgs = np.zeros((rows, cols, 3, len(files)), dtype=np.uint8)
    i = 0
    for f in files:
        imgs[:,:,:,i] = np.asarray(Image.open(os.path.join(path, f)))
        i += 1
    return imgs

# Read siri capture files
# Only implemented for num_frames = 1
# Ex f2p1-1.png
def read_siri_captures(path, rows, cols, num_phases, num_freqs, num_orientations):
    imgs = np.zeros((rows, cols, 3, num_phases, num_freqs, num_orientations), dtype=np.uint8)
    for o in range(0, num_orientations):
        for f in range(0, num_freqs):
            for p in range(0, num_phases):
                file_name = "f" + str(f+1) + "p" + str(p+1) + "o" + str(o+1) + "-1.png"
                imgs[:,:,:,p,f,o] = np.asarray(Image.open(os.path.join(path, file_name)))
    return imgs

# Read unordered captures
def read_imgs(path, rows, cols):
    files = os.listdir(path)
    imgs = np.zeros((rows, cols, 3, len(files)), dtype=np.uint8)
    i = 0
    for f in files:
        imgs[:,:,:,i] = np.asarray(Image.open(os.path.join(path, f)))
        i += 1
    return imgs

# Read unordered reference image captures Iwhite, Idark from their
# respective folders.
# Average the captures together to obtain Iwhite, Idark
def read_ref_imgs(path, rows, cols):
    Iwhite_captures = read_imgs(os.path.join(path, "Iwhite"), rows, cols)
    Idark_captures = read_imgs(os.path.join(path, "Idark"), rows, cols)
    Iwhite = np.mean(normalize_imgs(Iwhite_captures), axis=3)
    Idark = np.mean(normalize_imgs(Idark_captures), axis=3)
    return Iwhite, Idark

# Apply uniform illumination correction to an array of captures
# NaN and Inf values are set to 0
def illumination_correction(captures, Iwhite, Idark, spectralon_reflectance=1):
    captures_corr = np.zeros(captures.shape)

    # Ignore divide-by-zero and NaN warnings
    with (np.errstate(divide='ignore', invalid='ignore')):
        if captures_corr.ndim == 2:
            captures_corr = (captures - Idark) * spectralon_reflectance / (Iwhite - Idark)
        else:
            for i in range(0, captures.shape[-1]):
                captures_corr[...,i] = (captures[...,i] - Idark) * spectralon_reflectance / (Iwhite - Idark)

    captures_corr[np.isnan(captures_corr)] = 0
    captures_corr[np.isinf(captures_corr)] = 0
    return captures_corr

# imshow a 2d grayscale image in range [0,1]
def imshow2g(img):
    plt.imshow(img, cmap="gray", vmin=0, vmax=1)
    plt.show()

# Crops images with crop_parameters = (start_r, end_r, start_c, end_c)
def crop_captures(imgs, crop_parameters):
    start_r = crop_parameters[0]
    end_r = crop_parameters[1]
    start_c = crop_parameters[2]
    end_c = crop_parameters[3]
    return imgs[start_r:end_r, start_c:end_c, ...]

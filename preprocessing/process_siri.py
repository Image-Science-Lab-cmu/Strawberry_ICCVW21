import numpy as np
import util
import os
from PIL import Image

# Three-phase demodulation
def tpd(captures):
    assert(captures.shape[2] == 3) # 3 phases
    Idc = np.mean(captures, axis=2)
    Iac = np.sqrt(2)/3 * np.sqrt( \
            np.square(captures[:,:,0] - captures[:,:,1]) \
            + np.square(captures[:,:,1]-captures[:,:,2]) \
            + np.square(captures[:,:,2] - captures[:,:,0]))
    return Idc, Iac

def process_siri(base_path, mask_path, num_capture, num_rotation, rows, 
                 cols, keep_channel, num_phases, num_freqs, 
                 num_orientations, white_ratio=1, spectralon_reflectance=1):
    path = os.path.join(base_path, "%d-r%d" % (num_capture, num_rotation), "siri")
    raw_captures = util.read_siri_captures( \
            os.path.join(path, "captures"), rows, cols, num_phases, \
            num_freqs, num_orientations)
    captures = util.normalize_imgs(raw_captures)

    Iwhite, Idark = util.read_ref_imgs(path, rows, cols)

    Iwhite = Iwhite / white_ratio

    captures = captures[:,:,keep_channel,:,:,:]
    Iwhite = Iwhite[:,:,keep_channel]
    Idark = Idark[:,:,keep_channel]

    mask = np.asarray(Image.open(mask_path))

    captures_corr = util.illumination_correction(captures.reshape((rows,cols,num_phases*num_freqs*num_orientations)),
                                                 Iwhite, Idark, spectralon_reflectance)
    captures_corr = captures_corr.reshape((rows, cols, num_phases, num_freqs, num_orientations))

    Idc = np.zeros((rows, cols, num_freqs, num_orientations))
    Iac = np.zeros((rows, cols, num_freqs, num_orientations))

    for o in range(num_orientations):
        for f in range(num_freqs):
            Idc[:,:,f,o], Iac[:,:,f,o] = tpd(captures_corr[:,:,:,f,o])
            Idc[:,:,f,o] *= mask
            Iac[:,:,f,o] *= mask

    return Idc, Iac


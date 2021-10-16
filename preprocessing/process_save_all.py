# Created directory structure:
# processed/21-Sep-2020/0-r0/label.txt
# processed/21-Sep-2020/0-r0/strawberry.npz
# processed/21-Sep-2020/0-r0/patches.pkl
#
#   strawberry.npz:
#       T_centered
#       Id
#       Ig
#       Idc
#       Iac
#       pa
#       spectralon_reflectances
#
#   patches.pkl: dict of patches for each patch size key
#
#   epilines.pkl: list of np array of epilines
#

import numpy as np
import scipy.ndimage
import os
import h5py
import csv
from datetime import datetime
from PIL import Image
import pickle

import process_epipolar
import process_dg
import process_pa
import process_siri
import util

# Save the label (days until decay, positive for edible fruit) in a text file
# Save processed captures in a single numpy zip file
# spectralon_reflectances = [epipolar_sr, dg_sr, siri_sr]
def save_label_and_data(save_path_base, capture_date, decay_dates,
                        T_centered, Id, Ig, dg_captures,
                        Idc, Iac, strawberry_index, num_rotation, pa, mask,
                        epilines, epilines_mask,
                        epilines_per_capture, spectralon_reflectances):

    # Round label to nearest 12 hour increment
    datediff = capture_date - decay_dates[strawberry_index]
    label = round(datediff.total_seconds() / 3600 / 12) / 2

    # Save class labels
    save_path = os.path.join(save_path_base, str(strawberry_index)+"-r"+str(num_rotation))
    os.makedirs(save_path, exist_ok=True)

    with open(os.path.join(save_path, "label.txt"), 'w') as f:
        f.write("%f" % label)

    with open(os.path.join(save_path, "strawberry.npz"), 'wb') as f:
        np.savez(f, T_centered=T_centered, Id=Id, Ig=Ig,
                 dg_captures=dg_captures, Idc=Idc, Iac=Iac, pa=pa, epilines=epilines,
                 epilines_mask=epilines_mask,
                 mask=mask,
                 epilines_per_capture=epilines_per_capture,
                 spectralon_reflectances=spectralon_reflectances)

    save_patches(strawberry_index, num_rotation, mask)

    if os.path.isfile(os.path.join(save_path, "epilines.pkl")):
        os.remove(os.path.join(save_path, "epilines.pkl"))
        print("Deleting epilines.pkl")

# Returns a list of all patches completely inside the mask
# Each pask is a tuple of (row, column) or top left corner
def all_patches_in_mask(mask, patch_size):
    (r_max, c_max) = mask.shape
    patches = []
    for r in range(r_max - patch_size):
        for c in range(c_max - patch_size):
            patch = (r, c)
            patch_mask = mask[r:r+patch_size, c:c+patch_size]
            if patch_mask.all():
                patches.append(patch)

    return patches

# Save dictionary of patches of common sizes
def save_patches(strawberry_index, num_rotation, mask):
    patches = {}
    patch_sizes = [32, 40, 48, 60, 64, 128, 256]
    for s in patch_sizes:
        patches[s] = all_patches_in_mask(mask, s)

    save_path = os.path.join(save_path_base, str(strawberry_index)+"-r"+str(num_rotation))
    with open(os.path.join(save_path, "patches.pkl"), 'wb') as f:
        pickle.dump(patches, f)


# Crop parameters in form (r_start, r_end, c_start, c_end) from MATLAB's
# bounding box (indices start at 1)
def crop_params_from_box(bb):
    r_start = bb[1]-1
    r_end = r_start + bb[3]
    c_start = bb[0]-1
    c_end = c_start + bb[2]
    return (r_start, r_end, c_start, c_end)

# Returns whether or not that capture on the given day exists
def capture_exists(num_capture, capture_date, last_capture_dates):
    s0 = num_capture * 2
    s1 = num_capture * 2 + 1
    return strawberry_exists(s0, capture_date, last_capture_dates) or strawberry_exists(s1, capture_date, last_capture_dates)

# Returns whether or not the strawberry ([0,33]) exists on the given day
def strawberry_exists(num_strawberry, capture_date, last_capture_dates):
    return last_capture_dates[num_strawberry] >= capture_date

# Gets the average spectralon reflectance (double in [0,1]) from the masked
# region in the white calibration image
def spectralon_reflectance(spectralon_mask_path, Iwhite_path, rows, cols, keep_channel):
    Iwhite_captures = util.read_imgs(Iwhite_path, rows, cols)
    Iwhite = util.normalize_imgs(Iwhite_captures)
    Iwhite = np.mean(Iwhite, axis=3)

    mask = np.asarray(Image.open(spectralon_mask_path))
    Iwhite = Iwhite[:,:,keep_channel] * mask
    return Iwhite[np.nonzero(Iwhite)].mean()


def get_capture_dates(path):
    d = np.loadtxt(path,  dtype={'names': ('date', 'temp', 'rh'),
                                 'formats': ('U20', 'f4', 'f4')},
                   delimiter=",")

    dates = [datetime.strptime(t[0], "%m/%d/%Y %H:%M:%S") for t in d]
    return dates

def get_decay_dates(decay_table_path, capture_dates):
    decay_dates = {}
    last_capture_dates = {}

    with open(decay_table_path, 'r') as f:
        datetime_fmt = "%m/%d/%Y %H:%M:%S"
        csv_reader = csv.reader(f, delimiter=',')
        s = 0
        for (i, row) in enumerate(csv_reader):
            if i == 0:
                continue

            strawberry_index = int(row[0])
            decay_dates[s] = datetime.strptime(row[1], datetime_fmt)
            assert decay_dates[s] in capture_dates

            last_capture_dates[s] = datetime.strptime(row[2], datetime_fmt)
            assert last_capture_dates[s] in capture_dates

            s += 1

    return decay_dates, last_capture_dates

# Rotates captures (HxWxN) so the epilines are horizontal
def rotate_captures(captures):
    img = captures[:,:,captures.shape[2]//2]
    theta = np.arange(0, 180, 1)
    f = np.zeros(len(theta))
    idx = range(len(theta))

    for i in idx:
        t = theta[i]
        img_rot = scipy.ndimage.rotate(img, t, axes=(0,1))
        row_sum = np.sum(img_rot, axis=1)
        f[theta[i]] = np.max(row_sum)

    angle = np.argmax(f)
    if angle > 90:
        angle -= 180

    rotated_captures = scipy.ndimage.rotate(captures, angle, axes=(0,1))
    return rotated_captures, angle

def rotate_imgs_small_angle(imgs):
    """
    Rotate NxHxW images, each channel separately
    """
    rotated_captures = np.zeros(imgs.shape)
    angles = np.zeros(imgs.shape[0])

    for img_i in range(imgs.shape[0]):
        theta = np.arange(-5, 5, .1)
        f = np.zeros(len(theta))
        idx = range(len(theta))

        for i in idx:
            t = theta[i]
            img_rot = scipy.ndimage.rotate(imgs[img_i,:,:], t, axes=(0,1))
            row_sum = np.sum(img_rot, axis=1)
            f[i] = np.max(row_sum)

        angle_i = np.argmax(f)
        angle = theta[angle_i]

        angles[img_i] = angle

        rotated_capture = scipy.ndimage.rotate(imgs[img_i,:,:], angle, axes=(0,1))
        rotated_captures[img_i,:,:] = util.center_crop(rotated_capture, (imgs.shape[1], imgs.shape[2]))

    return rotated_captures, angles

# Extracts patches centered on the epilines from the input captures (HxWxN)
# returns list of arrays M x line_height x W, M: number of epilines
def extract_epilines(captures_rotated, mask_rotated, line_height):
    all_line_patches = None
    all_rotated_masks = None
    epilines_per_capture = np.zeros(captures_rotated.shape[2])

    for img_i in range(captures_rotated.shape[2]):
        img = captures_rotated[:,:,img_i]

        row_sum = np.sum(img, axis=1)

        row_sum_nms = np.zeros(row_sum.shape)
        nms_win_size = 21
        for i in range(nms_win_size//2, len(row_sum) - nms_win_size//2):
            window = row_sum[i-nms_win_size//2:i+nms_win_size//2+1]
            if row_sum[i] < np.max(window):
                row_sum_nms[i] = 0
            else:
                row_sum_nms[i] = row_sum[i]


        peaks = np.where(row_sum_nms > 2)[0]

        line_height_before_rot = int(line_height * 1.5)
        if line_height_before_rot % 2 == 0:
            line_height_before_rot += 1

        line_patches = np.ones((len(peaks), line_height_before_rot, img.shape[1])) * -1 # N x H x W epilines
        line_patch_masks = np.zeros((len(peaks), line_height_before_rot, img.shape[1])) # N x H x W masks for each epiline
        idx = []
        for i, l in enumerate(peaks):
            line_patch_curr = img[l-line_height_before_rot//2:l+line_height_before_rot//2+1,:]
            mask_line_patch_curr = mask_rotated[l-line_height_before_rot//2:l+line_height_before_rot//2+1,:]

            if line_patch_curr.shape == (line_height_before_rot, img.shape[1]):
                line_patches[i,:,:] = line_patch_curr
                line_patch_masks[i,:,:] = mask_line_patch_curr
                idx.append(i)

        # Drop epilines with size issues that are close to the edge
        idx = np.asarray(idx, dtype=int)
        line_patches = line_patches[idx,:,:]
        line_patch_masks = line_patch_masks[idx,:,:]

        line_patches_rotated, angles = rotate_imgs_small_angle(line_patches)

        # Rotate masks by small angle
        for i in range(len(angles)):
            mask_curr = line_patch_masks[i,:,:]
            mask_curr_rotated = scipy.ndimage.rotate(mask_curr, angles[i])
            line_patch_masks[i,:,:] = util.center_crop(mask_curr_rotated, (line_patch_masks.shape[1], line_patch_masks.shape[2]))

        # Crop line_patches, line_patch_masks to line_height
        line_patches_rotated = util.center_crop(line_patches_rotated, (line_height, line_patches_rotated.shape[2]))
        line_patch_masks = util.center_crop(line_patch_masks, (line_height, line_patch_masks.shape[2]))

        if all_line_patches is None:
            all_line_patches = line_patches_rotated
            all_rotated_masks = line_patch_masks
        else:
            all_line_patches = np.concatenate((all_line_patches, line_patches_rotated), axis=0)
            all_rotated_masks = np.concatenate((all_rotated_masks, line_patch_masks), axis=0)

        epilines_per_capture[img_i] = line_patches_rotated.shape[0]

    return all_line_patches, all_rotated_masks, epilines_per_capture

if __name__ == "__main__":

    ###################################
    # Init

    rows = 1024
    cols = 1280
    peak_width = 51
    keep_channel = 0
    base = "/path/to/dataset"
    a = 0.5
    b = 0

    epiline_height = 51

    # White ratios in white calibration images
    # also found in white_ratios.txt
    epipolar_white = 0.01
    pa_white = 0.01
    dg_white = 0.5
    siri_white = 0.5

    siri_num_phases = 3
    siri_num_freqs = 5
    siri_num_orientations = 2

    # Load dates & tables from text files
    capture_dates = get_capture_dates(os.path.join(base, "fridge.txt"))

    decay_table_path = os.path.join("decay-dates.txt")
    decay_dates, last_capture_dates = get_decay_dates(decay_table_path, capture_dates)


    # Process each strawberry capture
    for c in capture_dates:
        print("")
        print(c)
        print("-----------")

        for num_capture in range(17):
            if not capture_exists(num_capture, c, last_capture_dates):
                continue

            for num_rotation in range(2):
                print("%d-r%d" % (num_capture, num_rotation))

                capture_date_str = c.strftime("%m-%d-%Y %H %M")

                base_path = os.path.join(base, capture_date_str)
                mask_base_path = os.path.join(base, "masks", capture_date_str, str(num_capture)+"-r"+str(num_rotation))
                mask_path = os.path.join(mask_base_path, "mask.png")
                spectralon_mask_path = os.path.join(mask_base_path, "spectralon_mask.png")
                mask_bbs_path = os.path.join(base, "masks", capture_date_str, "bounds.mat")
                save_path_base = os.path.join(base, "processed/", capture_date_str)

                crop_params = (0, rows, 0, cols)

                # Save spectralon reflectances in a separate array
                epipolar_sr = spectralon_reflectance(spectralon_mask_path,
                         os.path.join(base_path, "%d-r%d" % (num_capture, num_rotation), "epipolar/Iwhite/"),
                         rows, cols, keep_channel)
                dg_sr = spectralon_reflectance(spectralon_mask_path,
                         os.path.join(base_path, "%d-r%d" % (num_capture, num_rotation), "dg/Iwhite/"),
                         rows, cols, keep_channel)
                siri_sr = spectralon_reflectance(spectralon_mask_path,
                         os.path.join(base_path, "%d-r%d" % (num_capture, num_rotation), "siri/Iwhite/"),
                         rows, cols, keep_channel)

                spectralon_reflectances = np.zeros(3)
                spectralon_reflectances[0] = epipolar_sr
                spectralon_reflectances[1] = dg_sr
                spectralon_reflectances[2] = siri_sr


                # Epipolar captures
                T_centered, epipolar_captures = process_epipolar.process_epipolar(base_path,
                        mask_path, num_capture, num_rotation, rows, cols, epipolar_white,
                        peak_width, keep_channel, crop_params, spectralon_reflectance=1)

                # Direct/global separation
                Id, Ig, dg_captures = process_dg.process_dg(base_path, mask_path,
                        num_capture, num_rotation, rows, cols, keep_channel,
                        a, b, crop_params, white_ratio=dg_white, spectralon_reflectance=1)

                # SIRI
                Idc, Iac = process_siri.process_siri(base_path, mask_path,
                        num_capture, num_rotation, rows, cols, keep_channel,
                        siri_num_phases, siri_num_freqs, siri_num_orientations,
                        white_ratio=siri_white, spectralon_reflectance=1)

                # Point-array captures
                pa = process_pa.process_pa(base_path, mask_path, num_capture, num_rotation,
                                           rows, cols, epipolar_white, keep_channel, crop_params)


                # Find bounding_boxes
                bbs = None
                with h5py.File(mask_bbs_path, 'r') as f:
                    bbs = np.transpose(np.array(f['bbs']), (3,2,1,0))
                bounding_boxes = np.round(bbs[:,:,num_rotation,num_capture])
                bounding_boxes = bounding_boxes.astype(int) - 1

                # Load mask
                mask = np.asarray(Image.open(mask_path))

                strawberry_index_left = num_capture * 2
                strawberry_index_right = num_capture * 2 + 1

                s_left_exists = strawberry_exists(strawberry_index_left, c, last_capture_dates)
                s_right_exists = strawberry_exists(strawberry_index_right, c, last_capture_dates)

                # Only save data and label if the strawberry is in the capture
                if (s_left_exists):
                    bounding_box = crop_params_from_box(bounding_boxes[:,0])

                    epipolar_captures_left = util.crop_captures(epipolar_captures, bounding_box)
                    T_centered_left = util.crop_captures(T_centered, bounding_box)
                    Id_left = util.crop_captures(Id, bounding_box)
                    Ig_left = util.crop_captures(Ig, bounding_box)
                    dg_captures_left = np.moveaxis(util.crop_captures(dg_captures, bounding_box), 2, 0)
                    Idc_left = util.crop_captures(Idc, bounding_box)
                    Iac_left = util.crop_captures(Iac, bounding_box)
                    pa_left = util.crop_captures(pa, bounding_box)

                    mask_left = util.crop_captures(mask, bounding_box)

                    epipolar_captures_rotated, angle = rotate_captures(epipolar_captures_left)
                    mask_left_rotated = scipy.ndimage.rotate(mask_left, angle)
                    epilines_left, epilines_mask_left, epilines_per_capture = \
                        extract_epilines(epipolar_captures_rotated, mask_left_rotated, epiline_height)

                    save_label_and_data(save_path_base, c, decay_dates,
                                        T_centered_left,
                                        Id_left, Ig_left, dg_captures_left,
                                        Idc_left, Iac_left,
                                        strawberry_index_left, num_rotation,
                                        pa_left, mask_left,
                                        epilines_left, epilines_mask_left,
                                        epilines_per_capture,
                                        spectralon_reflectances)

                if (s_right_exists):
                    if not s_left_exists:
                        bounding_box = crop_params_from_box(bounding_boxes[:,0])
                    else:
                        bounding_box = crop_params_from_box(bounding_boxes[:,1])

                    epipolar_captures_right = util.crop_captures(epipolar_captures, bounding_box)
                    T_centered_right = util.crop_captures(T_centered, bounding_box)
                    Id_right = util.crop_captures(Id, bounding_box)
                    Ig_right = util.crop_captures(Ig, bounding_box)
                    dg_captures_right = np.moveaxis(util.crop_captures(dg_captures, bounding_box), 2, 0)
                    Idc_right = util.crop_captures(Idc, bounding_box)
                    Iac_right = util.crop_captures(Iac, bounding_box)
                    pa_right = util.crop_captures(pa, bounding_box)

                    mask_right = util.crop_captures(mask, bounding_box)

                    epipolar_captures_rotated, angle = rotate_captures(epipolar_captures_right)
                    mask_right_rotated = scipy.ndimage.rotate(mask_right, angle)
                    epilines_right, epilines_mask_right, epilines_per_capture = \
                        extract_epilines(epipolar_captures_rotated, mask_right_rotated, epiline_height)

                    save_label_and_data(save_path_base, c, decay_dates,
                                        T_centered_right,
                                        Id_right, Ig_right, dg_captures_right,
                                        Idc_right, Iac_right,
                                        strawberry_index_right, num_rotation,
                                        pa_right, mask_right,
                                        epilines_right, epilines_mask_right,
                                        epilines_per_capture,
                                        spectralon_reflectances)


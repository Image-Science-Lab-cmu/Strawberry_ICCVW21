import os
from datetime import datetime
import numpy as np
import torch
import csv
from PIL import Image
import pickle
import math
import h5py
import cv2

# Import methods from preprocessing for loading bounding bxo
import sys
sys.path.insert(1, "../preprocessing")
import util
from process_save_all import crop_params_from_box

import load_data

# Switch to drop strawberries whose labels are x.5
SKIP_HALF_DAYS = True

NUM_STRAWBERRIES = 34
NUM_ROTATIONS = 2

MAX_LABEL = 5
MIN_LABEL = -8.5

if SKIP_HALF_DAYS:
    NUM_CLASSES = math.floor(MAX_LABEL - MIN_LABEL) + 1
else:
    NUM_CLASSES = int((MAX_LABEL - MIN_LABEL)*2 + 1)

EPIPOLAR_NUM_CHANNELS = 41
EPIPOLAR_INDIRECT_NUM_CHANNELS = 48
SIRI_AC_NUM_CHANNELS = 10
SIRI_AC_DC_NUM_CHANNELS = 20
SIRI_NUM_FREQS = 5

NUM_PA_CAPTURES = 4

# Load all processed data into a single Strawberry object
# Raises exception if the processed strawberry does not exist
# strawberry_index [0..33]
# num_rotation [0,1]
class Strawberry:
    dataset_path = "/path/to/dataset"
    capture_dates = None
    inedible_dates = None
    last_capture_dates = None
    weights = None

    def __init__(self, strawberry_index, num_rotation, capture_date, preload_feature=None):
        if Strawberry.capture_dates == None:
            capture_dates = get_capture_dates(os.path.join(Strawberry.dataset_path, "fridge.txt"))
            decay_dates, last_capture_dates = get_decay_dates(os.path.join(Strawberry.dataset_path, "decay-dates.txt"), capture_dates)
            Strawberry.capture_dates = capture_dates
            Strawberry.decay_dates = decay_dates
            Strawberry.last_capture_dates = last_capture_dates
            Strawberry.weights = get_weights(os.path.join(Strawberry.dataset_path, "weights.csv"))

        if not strawberry_exists(strawberry_index, capture_date, Strawberry.last_capture_dates):
            raise Exception("Strawberry does not exist")

        self.strawberry_index = strawberry_index
        self.num_rotation = num_rotation
        self.capture_date = capture_date

        capture_date_str = capture_date.strftime("%m-%d-%Y %H %M")
        strawberry_dir = str(strawberry_index) + "-r" + str(num_rotation)
        self.label = self.load_label(strawberry_dir, capture_date_str)

        if self.label > MAX_LABEL or self.label < MIN_LABEL:
            raise Exception("Ignoring labels > %f and < %f" % (MAX_LABEL, MIN_LABEL))
        if SKIP_HALF_DAYS and round(self.label) != self.label:
            raise Exception("Skipping strawberries with fractional labels")


        self.npz_path = os.path.join(Strawberry.dataset_path, "processed",
                                     capture_date_str, strawberry_dir,
                                     "strawberry.npz")
        self.pa_points_path = os.path.join(Strawberry.dataset_path, "processed",
                                           capture_date_str, strawberry_dir,
                                           "pa_points.mat")
        self.patches_path = os.path.join(Strawberry.dataset_path, "processed",
                                         capture_date_str, strawberry_dir,
                                         "patches.pkl")
        # Path to bounding box matrix
        self.mask_bbs_path = os.path.join(Strawberry.dataset_path, "masks", capture_date_str, "bounds.mat")

        self.raw_capture_path = os.path.join(Strawberry.dataset_path, capture_date_str,
                                             "%d-r%d" % (self.strawberry_index // 2, self.num_rotation))

        # Load spectralon reflectances
        with np.load(self.npz_path) as f:
            self.epipolar_sr = f["spectralon_reflectances"][0]
            self.dg_sr = f["spectralon_reflectances"][1]
            self.siri_sr = f["spectralon_reflectances"][2]


        self.T_centered = None
        self.Id = None
        self.Ig = None
        self.Id_Ig_ratio = None
        self.Idc = None
        self.Iac = None
        self.siri_stack = None
        self.pa = None
        self.mask = None
        self.patches = None

        # Single shot data
        self.epilines = None
        self.epilines_mask = None
        self.epilines_per_capture = None
        self.dg_captures = None
        self.rgb = None
        self.siri_captures = None

        # (r_start, r_end, c_start, c_end) of strawberry in raw captures
        self.bounding_box = None

        if preload_feature is not None:
            self._load_patches_()

        if preload_feature == load_data.GLOBAL_ONLY or preload_feature == load_data.ID_IG_RATIO or preload_feature == load_data.STACK_ID_IG:
            self._load_dg_()
        elif preload_feature == load_data.EPIPOLAR or \
            preload_feature == load_data.EPIPOLAR_PAIR or \
            preload_feature == load_data.EPIPOLAR_INDIRECT:
            self._load_T_centered_()
        elif preload_feature == load_data.SIRI_AC:
            self._load_Iac_()
        elif preload_feature == load_data.SIRI_AC_DC:
            self._load_Iac_()
            self._load_Idc_()
        elif preload_feature == load_data.PA_PATCH:
            self._load_pa_()
        elif preload_feature == load_data.EPIPOLAR_DG_STACK:
            self._load_dg_()
            self._load_T_centered_()
        elif preload_feature == load_data.EPIPOLAR_SINGLE_SHOT:
            self._load_epilines_()
        elif preload_feature == load_data.DG_SINGLE_SHOT:
            self._load_dg_single_shot_()
        elif preload_feature == load_data.RGB:
            self._load_rgb_()
        elif preload_feature == load_data.SIRI_SINGLE_SHOT:
            self._load_siri_single_shot_()

    def get_T_centered(self):
        if self.T_centered is None:
            self._load_T_centered_()
        return self.T_centered

    def get_Id(self):
        if self.Id is None:
            self._load_dg_()
        return self.Id

    def get_Ig(self):
        if self.Id is None:
            self._load_dg_()
        return self.Ig

    def get_Id_Ig_ratio(self):
        if self.Id is None:
            self._load_dg_()
        return self.Id_Ig_ratio

    def get_Idc(self):
        if self.Idc is None:
            self._load_Idc_()
        return self.Idc

    def get_Iac(self):
        if self.Iac is None:
            self._load_Iac_()
        return self.Iac

    def get_siri_stack(self):
        if self.siri_stack is None:
            self._load_siri_stack_()
        return self.siri_stack

    def get_pa(self):
        if self.pa is None:
            self._load_pa_()
        return self.pa

    def get_mask(self):
        if self.mask is None:
            self._load_mask_()
        return self.mask

    def get_patches(self):
        if self.patches is None:
            self._load_patches_()
        return self.patches

    def get_epilines(self):
        if self.epilines is None:
            self._load_epilines_()
        return self.epilines

    def get_epilines_mask(self):
        if self.epilines is None:
            self._load_epilines_()
        return self.epilines_mask

    def get_epilines_per_capture(self):
        if self.epilines_per_capture is None:
            self._load_epilines_per_capture_()
        return self.epilines_per_capture

    def get_dg_captures(self):
        if self.dg_captures is None:
            self._load_dg_single_shot_()
        return self.dg_captures

    def get_rgb(self):
        if self.rgb is None:
            self._load_rgb_()
        return self.rgb

    def get_siri_captures(self):
        if self.siri_captures is None:
            self._load_siri_single_shot_()
        return self.siri_captures

    # Returns the spectralon reflectance ([0,1]) for the given
    # feature num
    def get_spectralon_for_feature(self, feature_num):
        if feature_num == load_data.GLOBAL_ONLY or \
           feature_num == load_data.STACK_ID_IG or \
           feature_num == load_data.ID_IG_RATIO:
            return self.dg_sr
        elif feature_num == load_data.EPIPOLAR or \
             feature_num == load_data.EPIPOLAR_PAIR or \
             feature_num == load_data.EPIPOLAR_INDIRECT or \
             feature_num == load_data.PA_PATCH:
             return self.epipolar_sr
        elif feature_num == load_data.SIRI_AC or \
             feature_num == load_data.SIRI_AC_DC:
             return self.siri_sr
        else:
            return None


    # On-the-fly data loading
    def _load_T_centered_(self):
        with np.load(self.npz_path) as loaded_files:
            self.T_centered = loaded_files["T_centered"].astype(np.float32)

        if self.T_centered.shape[2] != EPIPOLAR_NUM_CHANNELS:
            cut_size = (self.T_centered.shape[2] - EPIPOLAR_NUM_CHANNELS) // 2
            self.T_centered = self.T_centered[:,:,cut_size:-cut_size]

    def _load_dg_(self):
        with np.load(self.npz_path) as loaded_files:
            self.Id = loaded_files["Id"]
            self.Ig = loaded_files["Ig"]
            self.Id_Ig_ratio = (self.Id / (self.Id + self.Ig + 1e-10)).astype(np.float32)
            self.Id = self.Id.astype(np.float32)
            self.Ig = self.Ig.astype(np.float32)

    def _load_Idc_(self):
        with np.load(self.npz_path) as loaded_files:
            self.Idc = loaded_files["Idc"].astype(np.float32)

    def _load_Iac_(self):
        with np.load(self.npz_path) as loaded_files:
            self.Iac = loaded_files["Iac"].astype(np.float32)

    def _load_siri_stack_(self):
        if self.Iac is None:
            self._load_Iac_()
            self._load_Idc_()

        Iac = self.get_Iac()
        Iac = Iac.reshape(Iac.shape[0], Iac.shape[1], 10).transpose(2, 0, 1)
        Idc = self.get_Idc()
        Idc = Idc.reshape(Idc.shape[0], Idc.shape[1], 10).transpose(2, 0, 1)
        self.siri_stack = np.concatenate((Iac, Idc), axis=0)

    def _load_pa_(self):
        with np.load(self.npz_path) as loaded_files:
            self.pa = loaded_files["pa"].astype(np.float32)

        with h5py.File(self.pa_points_path, 'r') as f:
            # pa_points: 4x2xN, ignoire pa_points == -1
            self.pa_points = np.round(np.transpose(np.array(f['pa_points']))).astype(int)

    def _load_mask_(self):
        with np.load(self.npz_path) as loaded_files:
            self.mask = loaded_files["mask"]

    def _load_patches_(self):
        with open(self.patches_path, 'rb') as f:
            self.patches = pickle.load(f)

    def _load_epilines_(self):
        with np.load(self.npz_path) as loaded_files:
            self.epilines = loaded_files["epilines"].astype(np.float32)
            self.epilines_mask = loaded_files["epilines_mask"] > 0.5

    def _load_epilines_per_capture_(self):
        with np.load(self.npz_path) as loaded_files:
            self.epilines_per_capture = loaded_files["epilines_per_capture"].astype(np.int32)

    def _load_dg_single_shot_(self):
        with np.load(self.npz_path) as loaded_files:
            self.dg_captures = loaded_files["dg_captures"].astype(np.float32)

    def _load_bounding_box_(self):
        with h5py.File(self.mask_bbs_path, 'r') as f:
            bbs = np.transpose(np.array(f['bbs']), (3,2,1,0))
        bounding_boxes = np.round(bbs[:, :, self.num_rotation, self.strawberry_index // 2])
        bounding_boxes = bounding_boxes.astype(int) - 1

        # Determine which bounding box belongs to this strawberry
        strawberry_on_left = None
        if self.strawberry_index % 2 == 0:
            strawberry_index_left = self.strawberry_index
            strawberry_index_right = self.strawberry_index + 1
            strawberry_on_left = True
        else:
            strawberry_index_left = self.strawberry_index - 1
            strawberry_index_right = self.strawberry_index
            strawberry_on_left = False

        s_left_exists = strawberry_exists(strawberry_index_left, self.capture_date,
                                          Strawberry.last_capture_dates)
        s_right_exists = strawberry_exists(strawberry_index_right, self.capture_date,
                                           Strawberry.last_capture_dates)

        if strawberry_on_left:
            assert s_left_exists
            self.bounding_box = crop_params_from_box(bounding_boxes[:,0])
        else:
            assert s_right_exists
            if not s_left_exists:
                self.bounding_box = crop_params_from_box(bounding_boxes[:,0])
            else:
                self.bounding_box = crop_params_from_box(bounding_boxes[:,1])

    def _load_rgb_(self):
        # Load bounding box
        if self.bounding_box is None:
            self._load_bounding_box_()

        Iwhite, Idark = util.read_ref_imgs(os.path.join(self.raw_capture_path, "dg"), 1024, 1280)
        Idark *= 2

        rgb = Iwhite - Idark
        rgb = rgb[self.bounding_box[0]:self.bounding_box[1],
                  self.bounding_box[2]:self.bounding_box[3]] * self.get_mask()[:,:,None]
        rgb = np.clip(rgb, 0, 1) # ignore specular highlights in dark image
        rgb = np.moveaxis(rgb, 2, 0) # RGB image is 3xHxW
        self.rgb = np.ascontiguousarray(rgb)

    def _load_siri_single_shot_(self):
        """
        Loads SIRI captures (stack of 5 images) to self.siri_captures
        Uses horizontal sinusoids at 1 phase shift
        """
        # Load bounding box
        if self.bounding_box is None:
            self._load_bounding_box_()

        keep_channel = 0

        siri_path = os.path.join(self.raw_capture_path, "siri")
        Iwhite, Idark = util.read_ref_imgs(siri_path, 1024, 1280)
        Iwhite = Iwhite[:,:,keep_channel]
        Idark = Idark[:,:,keep_channel]
        Iwhite /= 2

        phase = 1 # Use first phase shift
        orientation = 2 # Use horizontal sinusoids

        siri_captures = np.zeros((SIRI_NUM_FREQS, 1024, 1280), dtype=np.uint8)
        for i in range(SIRI_NUM_FREQS):
            img_path = os.path.join(siri_path, "captures", "f%dp%do%d-1.png" % (i+1, phase, orientation))
            siri_captures[i,:,:] = np.asarray(Image.open(img_path))[:,:,keep_channel]

        siri_captures = util.normalize_imgs(siri_captures)
        with np.errstate(divide='ignore', invalid='ignore'):
            siri_captures = (siri_captures - Idark[None,:,:]) / (Iwhite[None,:,:] - Idark[None,:,:])
        siri_captures[np.isnan(siri_captures) | np.isinf(siri_captures)] = 0
        siri_captures = siri_captures[:,self.bounding_box[0]:self.bounding_box[1],
                                      self.bounding_box[2]:self.bounding_box[3]]
        siri_captures *= self.get_mask()

        self.siri_captures = np.ascontiguousarray(siri_captures)

    def load_label(self, strawberry_dir, capture_date_str):
        path = os.path.join(Strawberry.dataset_path, "processed",
                            capture_date_str, strawberry_dir, "label.txt")

        with open(path, 'r') as f:
            label = float(f.read())

        return label

    # T_centered flattened with zero vectors removed
    # Each row is a single pixel with length peak_width
    def T_centered_non_zero(self):
        (rows, cols, peak_width) = self.get_T_centered().shape
        T_centered_non_zero = np.reshape(self.get_T_centered(), (rows*cols, peak_width))
        nonzero_rows = np.any(T_centered_non_zero, axis=1)
        return T_centered_non_zero[nonzero_rows]

    # Returns the label as a one-hot vector
    def one_hot_label(self):
        v = np.zeros(NUM_CLASSES)
        v[label_to_index(self.label)] = 1
        return v

    # Returns label as a one-hot vector with gaussian pdf around label
    # sigma: standard deviation
    def one_hot_gaussian_label(self, sigma):
        one_hot_label = self.one_hot_label()
        return add_gaussian(one_hot_label[np.newaxis,:], sigma)

    # Returns a 2D array of patches given the feature_num defined in
    # load_data
    # All patches are vectorized
    def imgs_from_patches(self, patches, patch_size, feature_num):
        assert(feature_num == load_data.GLOBAL_ONLY or
                feature_num == load_data.STACK_ID_IG or
                feature_num == load_data.ID_IG_RATIO or
                feature_num == load_data.EPIPOLAR)

        patch_imgs = None
        if feature_num == load_data.EPIPOLAR:
            patch_imgs = np.zeros((len(patches), patch_size*patch_size*self.get_T_centered().shape[2]), dtype=np.float32)
        else:
            l = patch_size * patch_size
            if feature_num == load_data.STACK_ID_IG:
                l *= 2
            patch_imgs = np.zeros((len(patches), l), dtype=np.float32)

        i = 0
        for patch in patches:
            assert(patch[0] <= self.Ig.shape[0] - patch_size)
            assert(patch[1] <= self.Ig.shape[1] - patch_size)

            patch_curr = None

            if feature_num == load_data.GLOBAL_ONLY:
                patch_curr = self.get_Ig()[patch[0]:patch[0]+patch_size,
                    patch[1]:patch[1]+patch_size]
            elif feature_num == load_data.STACK_ID_IG:
                Ig_patch = self.get_Ig()[patch[0]:patch[0]+patch_size,
                    patch[1]:patch[1]+patch_size]
                Id_patch = self.get_Id()[patch[0]:patch[0]+patch_size,
                    patch[1]:patch[1]+patch_size]
                patch_curr = np.concatenate((Ig_patch, Id_patch), axis=1)
            elif feature_num == load_data.ID_IG_RATIO:
                patch_curr = self.get_Id_Ig_ratio()[patch[0]:patch[0]+patch_size,
                    patch[1]:patch[1]+patch_size]
            elif feature_num == load_data.EPIPOLAR:
                patch_curr = self.get_T_centered()[patch[0]:patch[0]+patch_size,
                    patch[1]:patch[1]+patch_size,:]

            assert(patch_curr is not None)
            patch_imgs[i,:] = patch_curr.ravel()

            i += 1

        return patch_imgs

    # Returns an array of patches given the feature_num defined in
    # load_data
    # Patches are not vectorized
    def patches_no_vectorize(self, patches, patch_size, feature_num):
        patch_imgs = np.zeros((len(patches), num_channels_for_feature(feature_num),
                               patch_size, patch_size),
                              dtype=np.float32)

        for i, patch in enumerate(patches):
            if feature_num == load_data.GLOBAL_ONLY:
                patch_imgs[i,0,:,:] = self.get_Ig()[patch[0]:patch[0]+patch_size,
                    patch[1]:patch[1]+patch_size]
            elif feature_num == load_data.STACK_ID_IG:
                patch_imgs[i,0,:,:] = self.get_Ig()[patch[0]:patch[0]+patch_size,
                    patch[1]:patch[1]+patch_size]
                patch_imgs[i,1,:,:] = self.get_Id()[patch[0]:patch[0]+patch_size,
                    patch[1]:patch[1]+patch_size]
            elif feature_num == load_data.ID_IG_RATIO:
                patch_imgs[i,0,:,:] = self.get_Id_Ig_ratio()[patch[0]:patch[0]+patch_size,
                    patch[1]:patch[1]+patch_size]
            elif feature_num == load_data.EPIPOLAR:
                patch_imgs[i,:,:,:] = self.get_T_centered()[patch[0]:patch[0]+patch_size,
                    patch[1]:patch[1]+patch_size,:].transpose((2, 0, 1))
            elif feature_num == load_data.SIRI_AC:
                Iac = self.get_Iac()
                Iac = Iac.reshape(Iac.shape[0], Iac.shape[1], SIRI_NUM_CHANNELS_AC).transpose(2, 0, 1)
                patch_imgs[i,:,:,:] = Iac[:,patch[0]:patch[0]+patch_size,
                    patch[1]:patch[1]+patch_size]
            elif feature_num == load_data.SIRI_AC_DC:
                patch_imgs[i,:,:,:] = self.get_siri_stack()[:,patch[0]:patch[0]+patch_size,
                    patch[1]:patch[1]+patch_size]
            elif feature_num == load_data.EPIPOLAR_PAIR:
                channels = np.arange(EPIPOLAR_NUM_CHANNELS)
                # First image: average of channel 20 and 30
                patch_imgs[i,0,:,:] = (self.get_T_centered()[patch[0]:patch[0]+patch_size,
                    patch[1]:patch[1]+patch_size, 30] + \
                    self.get_T_centered()[patch[0]:patch[0]+patch_size, \
                    patch[1]:patch[1]+patch_size, 20]).squeeze() / 2
                # Second image: direct epipolar channel
                patch_imgs[i,1,:,:] = self.get_T_centered()[patch[0]:patch[0]+patch_size,
                    patch[1]:patch[1]+patch_size, EPIPOLAR_NUM_CHANNELS // 2]
            elif feature_num == load_data.EPIPOLAR_INDIRECT:
                channels = np.arange(EPIPOLAR_NUM_CHANNELS)
                patch_imgs[i,:,:,:] = self.get_T_centered()[patch[0]:patch[0]+patch_size,
                    patch[1]:patch[1]+patch_size, (channels != 24) & \
                                    (channels != 25) & (channels != 26)].transpose((2,0,1))
            elif feature_num == load_data.EPIPOLAR_DG_STACK:
                Ig = self.get_Ig()
                Id = self.get_Id()
                T = self.get_T_centered()

                Ig_patch = Ig[patch[0]:patch[0]+patch_size,
                              patch[1]:patch[1]+patch_size,None]
                Id_patch = Id[patch[0]:patch[0]+patch_size,
                              patch[1]:patch[1]+patch_size,None]
                T_patch = T[patch[0]:patch[0]+patch_size,
                            patch[1]:patch[1]+patch_size,:]

                patch_imgs[i,:,:,:] = np.concatenate((Ig_patch, Id_patch, T_patch), axis=2)
            elif feature_num == load_data.RGB:
                patch_imgs[i,:,:,:] = self.get_rgb()[:,patch[0]:patch[0]+patch_size,
                                                     patch[1]:patch[1]+patch_size]
            elif feature_num == load_data.SIRI_SINGLE_SHOT:
                patch_imgs[i,:,:,:] = self.get_siri_captures()[:,patch[0]:patch[0]+patch_size,
                                                     patch[1]:patch[1]+patch_size]
            else:
                assert False

        return patch_imgs

    # Gets matrix of all point array patches
    # patch_size must be odd
    def pa_patches(self, patch_size, num_capture=-1):
        assert(patch_size % 2 == 1)
        half_patch_size = patch_size // 2

        patches = []
        pa = self.get_pa()

        if num_capture != -1:
            pa_idx = [num_capture]
        else:
            pa_idx = list(range(pa.shape[2]))

        for i in pa_idx:
            for j in range(self.pa_points.shape[2]):
                pt = self.pa_points[i,:,j].squeeze()

                if pt[0] == -1: # Ignore (-1, -1)
                    break

                # Skip patch if not completely in the mask
                if not np.all(self.get_mask()[pt[0]-half_patch_size:pt[0]+half_patch_size+1,
                                              pt[1]-half_patch_size:pt[1]+half_patch_size+1]) or \
                    pt[0] < half_patch_size or pt[0] + half_patch_size + 1 > self.get_mask().shape[0] or \
                    pt[1] < half_patch_size or pt[1] + half_patch_size + 1 > self.get_mask().shape[1]:
                    continue

                patch = pa[pt[0]-half_patch_size:pt[0]+half_patch_size+1,
                           pt[1]-half_patch_size:pt[1]+half_patch_size+1,i]

                # normalize to center
                center_size = 9
                r_start = (patch.shape[0] // 2) - (center_size // 2)
                r_end = r_start + center_size
                c_start = (patch.shape[1] // 2) - (center_size // 2)
                c_end = c_start + center_size
                patch = patch / np.mean(patch[r_start:r_end, c_start:c_end])

                patches.append(patch)

        return np.asarray(patches)


    # Return weight of strawberry divided by weight on day it went inedible
    def weight_ratio(self):
        i_today = Strawberry.capture_dates.index(self.capture_date)
        i_0 = i_today + (label_to_index(0) - label_to_index(self.label))
        return Strawberry.weights[self.strawberry_index, i_today] / Strawberry.weights[self.strawberry_index, i_0]



    def __str__(self):
        return "%s, s: %d, r: %d, label: %f" % (self.capture_date.strftime("%m/%d/%Y %H:%M:%S"),
                                                self.strawberry_index, self.num_rotation,
                                                self.label)



# Copy from process_save_all.py
def get_capture_dates(path=None):
    if path is None:
        path = os.path.join(Strawberry.dataset_path, "fridge.txt")
    d = np.loadtxt(path,  dtype={'names': ('date', 'temp', 'rh'),
                                 'formats': ('U20', 'f4', 'f4')},
                   delimiter=",")

    dates = [datetime.strptime(t[0], "%m/%d/%Y %H:%M:%S") for t in d]
    return dates

# Copy from process_save_all.py
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

def strawberry_exists(strawberry_index, capture_date, last_capture_dates):
    return last_capture_dates[strawberry_index] >= capture_date


# Load strawberry weights
# weights: NUM_STRAWBERRIES x M
# weights == -1 is invalid
def get_weights(path=None):
    if path is None:
        path = os.path.join(Strawberry.dataset_path, "weights.csv")
    return np.loadtxt(path, delimiter=",")[:,1:]

# Convert label to index in one-hot-vector
# Label can be float, numpy array, or pytorch tensor
def label_to_index(label):
    if SKIP_HALF_DAYS:
        if isinstance(label, np.ndarray):
            return (label - math.ceil(MIN_LABEL)).astype(int)
        elif isinstance(label, torch.Tensor):
            return (label - math.ceil(MIN_LABEL)).int()
        else:
            return int(label - math.ceil(MIN_LABEL))
    else:
        if isinstance(label, np.ndarray):
            return ((label * 2) - (MIN_LABEL * 2)).astype(int)
        elif isinstance(label, torch.Tensor):
            return ((label * 2) - (MIN_LABEL * 2)).int()
        else:
            return int((label * 2) - (MIN_LABEL * 2))

# Convert index in one-hot-vector to floating-point label
# index may be a numpy array, torch tensor, or integer
def index_to_label(index):
    if SKIP_HALF_DAYS:
        return index + math.ceil(MIN_LABEL)
    else:
        if isinstance(index, np.ndarray):
            return index.astype(float) / 2 + MIN_LABEL
        elif isinstance(index, torch.Tensor):
            return index.float() / 2 + MIN_LABEL
        else:
            return float(index) / 2 + MIN_LABEL

def num_channels_for_feature(feature_num):
    if feature_num == load_data.GLOBAL_ONLY:
        return 1
    elif feature_num == load_data.STACK_ID_IG:
        return 2
    elif feature_num == load_data.ID_IG_RATIO:
        return 1
    elif feature_num == load_data.EPIPOLAR:
        return EPIPOLAR_NUM_CHANNELS
    elif feature_num == load_data.PA_PATCH:
        return 1
    elif feature_num == load_data.SIRI_AC:
        return SIRI_AC_NUM_CHANNELS
    elif feature_num == load_data.SIRI_AC_DC:
        return SIRI_AC_DC_NUM_CHANNELS
    elif feature_num == load_data.EPIPOLAR_PAIR:
        return 2
    elif feature_num == load_data.EPIPOLAR_INDIRECT:
        return EPIPOLAR_NUM_CHANNELS - 3
    elif feature_num == load_data.EPIPOLAR_DG_STACK:
        return EPIPOLAR_NUM_CHANNELS + 2
    elif feature_num == load_data.EPIPOLAR_SINGLE_SHOT:
        return 1
    elif feature_num == load_data.DG_SINGLE_SHOT:
        return 1
    elif feature_num == load_data.RGB:
        return 3
    elif feature_num == load_data.SIRI_SINGLE_SHOT:
        return SIRI_NUM_FREQS
    elif feature_num == load_data.PA_SINGLE_SHOT:
        return 1
    elif feature_num == load_data.EPILINES:
        return 1

    elif feature_num == load_data.PA_1 or feature_num == load_data.PA_2 or \
        feature_num == load_data.PA_3 or feature_num == load_data.PA_4:
        return 1
    else:
        assert False

# Adds gaussian noise centered around each true label
# # labels: matrix of one-hot vectors
# # sigma: standard deviation
def add_gaussian(labels, sigma):
    assert labels.shape[1] == NUM_CLASSES
    labels_with_gaussian = np.zeros(labels.shape)
    for i in range(labels.shape[0]):
        label = np.argmax(labels[i,:])
        gaussian_x = np.arange(NUM_CLASSES) - label
        gaussian = 1/(sigma * np.sqrt(2*np.pi))*np.exp(-gaussian_x*gaussian_x/(2*sigma*sigma))
        gaussian = gaussian / np.sum(gaussian)
        labels_with_gaussian[i,:] = gaussian
    return labels_with_gaussian

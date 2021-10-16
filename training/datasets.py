import torch
import torch.utils.data as data
import torchvision
import numpy as np
import random
import os
import scipy.ndimage

import strawberry
from strawberry import Strawberry
import load_data
import training_util


def center_crop(patch, size):
    """
    Crops patch to the center with given size

    patch: N x H x W
    size: H', W'
    """

    start_r = (patch.shape[1] // 2) - (size[0] // 2)
    end_r = start_r + size[0]
    start_c = (patch.shape[2] // 2) - (size[1] // 2)
    end_c = start_c + size[1]

    return patch[:, start_r:end_r, start_c:end_c]

def random_rotation(patch, max_rotation, final_patch_size):
    patch_r = torch.zeros(patch.shape[0], final_patch_size,
                          final_patch_size)
    rot_angle = np.random.randint(-max_rotation, max_rotation+1)
    patch_r = torch.from_numpy(scipy.ndimage.rotate(patch, rot_angle, axes=(1,2)))

    return center_crop(patch_r, (final_patch_size, final_patch_size))


class PatchDataset(data.Dataset):
    """
    Creates a dataset with a random subset of patches from each strawberry
    If patches_per_strawberry == -1, it uses every possible patch inside
    the mask
    sigma: std of gaussian to add to ground truth label
    data_augmentation: adds gaussian noise and random horizontal flip to each
        patch
    use_spectralon: whether or not to include spectralon reflectance as a final
        channel in the feature vector
    return_weight: __getitem__ returns weight ratio as a third item
    fraction_in_mask: fraction of patch required to be in the mask
    max_rotation: Max rotation in degrees to rotate patch if using data
        augmentation
    normalize: normalize patches to precomputed dataset mean and std

    output: N x patch_size x patch_size, where N is number of channels (possibly
            including spectralon reflectance as the final channel)
    """
    def __init__(self, strawberries, patch_size, total_patches=1e6,
                 feature_num=load_data.EPIPOLAR, sigma=0,
                 data_augmentation=False, noise_std=0,
                 use_spectralon=False, return_weight=False,
                 fraction_in_mask=1, max_rotation=0,
                 normalize=False, allow_overlap=True):
        assert(feature_num != load_data.PA_PATCH)

        self.strawberries = strawberries
        self.patch_size = patch_size
        self.feature_num = feature_num
        self.sigma = sigma
        self.data_augmentation = data_augmentation
        self.use_spectralon = use_spectralon
        self.return_weight = return_weight
        self.max_rotation = max_rotation

        self.original_patch_size = patch_size
        if data_augmentation:
            # Std of 0-mean gaussian noise
            self.noise_std = noise_std

            if max_rotation != 0:
                # Transforms for random rotations
                self.original_patch_size = int(np.ceil(patch_size * 1.5))

        self.final_patch_size = patch_size

        # setup normalization
        self.normalize = normalize
        if normalize:
            self.dataset_mean, self.dataset_std = normalization_for_feature(self.feature_num)

        # label frequencies and patch sampling
        if total_patches == -1:
            self.patches_per_strawberry = torch.ones(strawberry.NUM_CLASSES) * -1
        else:
            labels = torch.zeros(len(strawberries))
            for (i, s) in enumerate(strawberries):
                labels[i] = s.label
            unique_labels, label_cnts = torch.unique(labels, return_counts=True)
            counts = label_cnts.double()
            self.patches_per_strawberry = torch.zeros(strawberry.NUM_CLASSES)
            for (u, c) in zip(unique_labels, label_cnts):
                i = strawberry.label_to_index(u)
                self.patches_per_strawberry[i] = total_patches / (len(unique_labels) * c)

        # load patches
        if allow_overlap:
            all_patches = [training_util.all_patches_in_mask(s, \
                    self.original_patch_size, fraction_in_mask=fraction_in_mask) \
                    for s in self.strawberries]
        else:
            assert fraction_in_mask == 1
            all_patches = [training_util.all_patches_no_overlap(s, \
                    self.original_patch_size) \
                    for s in self.strawberries]

        if total_patches == -1:
            self.patches = all_patches
        else:
            self.patches = []
            for p, s in zip(all_patches, self.strawberries):
                subsampled_patches = [p[i] for i in random.sample(range(len(p)), self._patches_per_strawberry_(s))]
                self.patches.append(subsampled_patches)

        self.num_patches_per_strawberry = torch.tensor([len(p) for p in self.patches])
        self.cumsum_patches = torch.cumsum(self.num_patches_per_strawberry, dim=0)


    def __getitem__(self, index):
        strawberry_index = torch.nonzero(self.cumsum_patches - index > 0, as_tuple=False)[0].item()
        if strawberry_index == 0:
            prev_sum = 0
        else:
            prev_sum = self.cumsum_patches[strawberry_index - 1].item()
        patch_index = index - prev_sum

        if self.data_augmentation:
            patch_size_to_load = self.original_patch_size
        else:
            patch_size_to_load = self.final_patch_size

        patch = self.strawberries[strawberry_index].patches_no_vectorize(
                    [self.patches[strawberry_index][patch_index]],
                    patch_size_to_load, self.feature_num)
        patch = torch.from_numpy(patch).type(torch.FloatTensor).squeeze()
        if self.feature_num == 0:
            patch = patch[None,:,:]
        assert(patch.ndim == 3)

        if self.normalize:
            patch -= self.dataset_mean[:,None,None]
            patch /= self.dataset_std[:,None,None]

        if self.data_augmentation:
            patch = patch.detach().clone()
            # Random horizontal flip with probability 0.5
            if torch.rand(1) > 0.5:
                patch = patch.flip(1)

            gaussian_noise = torch.randn_like(patch) * self.noise_std
            patch += gaussian_noise

            if self.max_rotation > 0:
                patch = random_rotation(patch, self.max_rotation, self.final_patch_size)
                if self.feature_num == load_data.RGB:
                    torch.clamp(patch, 0, 1)

        if self.use_spectralon:
            sr = self.strawberries[strawberry_index].get_spectralon_for_feature(self.feature_num)
            patch = torch.cat((patch, torch.ones((1, self.final_patch_size, self.final_patch_size), dtype=torch.float32) * sr), dim=0)

        label = self.strawberries[strawberry_index].one_hot_label()
        if self.sigma != 0:
            label = training_util.add_gaussian(label[np.newaxis,:], self.sigma)

        label = torch.from_numpy(label).type(torch.FloatTensor).squeeze()

        if self.return_weight:
            return patch, label, self.strawberries[strawberry_index].weight_ratio()
        else:
            return patch, label

    def __len__(self):
        # total number of patches
        return self.cumsum_patches[-1]

    # Get total number of patches to use from strawberry si (depends on label)
    def _patches_per_strawberry_(self, s):
        i = strawberry.label_to_index(s.label)
        return int(self.patches_per_strawberry[i])

    def set_data_augmentation(self, data_augmentation):
        self.data_augmentation = data_augmentation

class PAPatchDataset(data.Dataset):
    """
    Creates a dataset for point-array patches.
    """
    def __init__(self, strawberries, patch_size, feature_num, sigma=0,
                 data_augmentation=False, max_rotation=0, noise_std=0, use_spectralon=False):
        self.final_patch_size = patch_size
        self.feature_num = feature_num
        self.data_augmentation = data_augmentation
        self.sigma = sigma
        self.use_spectralon = use_spectralon
        self.max_rotation = max_rotation

        self.sr = []
        self.labels = [] # Label as index in one-hot vector

        self.original_patch_size = patch_size
        if data_augmentation:
            # Std of 0-mean gaussian noise
            self.noise_std = noise_std

            if max_rotation != 0:
                # Transforms for random rotations
                self.original_patch_size = int(np.ceil(patch_size * 1.5))
                if self.original_patch_size % 2 == 0:
                    self.original_patch_size += 1

        self.final_patch_size = patch_size

        for i, s in enumerate(strawberries):
            patches = torch.from_numpy(s.pa_patches(self.original_patch_size)).type(torch.FloatTensor)

            if i == 0:
                self.patches = patches
            else:
                self.patches = torch.cat((self.patches, patches), dim=0)
            for j in range(patches.shape[0]):
                self.sr.append(s.epipolar_sr)
                self.labels.append(strawberry.label_to_index(s.label))

        self.sr = torch.tensor(self.sr)
        self.labels = torch.tensor(self.labels, dtype=torch.int)


    def __getitem__(self, index):
        patch = self.patches[index,:,:]
        patch = patch[None,:,:]

        if self.data_augmentation:
            patch = patch.detach().clone()
            # Random horizontal flip with probability 0.5
            #if torch.rand(1) > 0.5:
                #patch = patch.flip(1)

            gaussian_noise = torch.randn_like(patch) * self.noise_std
            patch += gaussian_noise

            if self.max_rotation > 0:
                patch = random_rotation(patch, self.max_rotation, self.final_patch_size)
        elif self.final_patch_size != self.original_patch_size:
            # Dataset initialized with data augementation, but data augmentation is False
            patch = center_crop(patch, (self.final_patch_size, self.final_patch_size))

        if self.use_spectralon:
            sr = self.sr[index]
            patch = torch.cat((patch, torch.ones(1, self.final_patch_size, self.final_patch_size, dtype=torch.float32) * sr), dim=0)

        label = np.zeros(strawberry.NUM_CLASSES)
        label[self.labels[index]] = 1
        if self.sigma != 0:
            label = training_util.add_gaussian(label[np.newaxis,:], self.sigma)
        label = torch.from_numpy(label).type(torch.FloatTensor).squeeze()

        return patch, label


    def __len__(self):
        return self.patches.shape[0]

    def set_data_augmentation(self, data_augmentation):
        self.data_augmentation = data_augmentation


class EpilineDataset(data.Dataset):
    """
    Single dataset to retrieve patches from a single epiline and its
    corresponding mask

    epiline: HxW
    epiline_mask: Hx@
    """
    def __init__(self, epiline, epiline_mask, patch_size, label,
                 data_augmentation=False, jitter_dist=2, allow_overlap=False):
        self.epiline = epiline
        self.patch_size = patch_size
        self.label = label
        self.data_augmentation = data_augmentation
        self.jitter_dist = jitter_dist
        self.patches = [] # list of column of top-left corner

        if isinstance(epiline_mask, torch.Tensor):
            epiline_mask = np.asarray(epiline_mask)

        step = 1 if allow_overlap else patch_size
        for c in range(0, epiline.shape[1] - patch_size, step):
            if np.all(epiline_mask[:, c:c+patch_size]):
                self.patches.append(c)

    def __get_patches__(self):
        return self.patches

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, index):
        patch_c = self.patches[index]
        patch = self.epiline[:, patch_c:patch_c+self.patch_size]

        # center crop
        start_r = (patch.shape[0] // 2) - (self.patch_size // 2)
        end_r = start_r + self.patch_size
        start_c = (patch.shape[1] // 2) - (self.patch_size // 2)
        end_c = start_c + self.patch_size
        patch = patch[None, start_r:end_r, start_c:end_c].detach().clone()

        # normalize patch to average intensity of epiline
        center_size = 7
        start_r = (patch.shape[1] // 2) - (center_size // 2)
        end_r = start_r + center_size
        patch = patch / patch[:, start_r:end_r, :].mean()

        # data augmentation of random jitter and flip
        if self.data_augmentation:
            current_jitter = np.random.randint(-self.jitter_dist, self.jitter_dist+1)
            start_r += current_jitter
            end_r += current_jitter

            # Randomly flip left/right
            if torch.rand(1) > 0.5:
                patch = patch.flip(1)

        return patch, self.label


class EpipolarSingleShotDataset(data.Dataset):
    """
    Dataset for predicting strawberry freshness patches in a single capture
    with projected epilines
    """
    def __init__(self, strawberries, patch_size, sigma, jitter_dist=2,
                 data_augmentation=False, allow_overlap=False):
        self.epiline_datasets = []
        for s in strawberries:
            epilines = torch.from_numpy(s.get_epilines()).float()
            epilines_mask = s.get_epilines_mask()
            label = torch.from_numpy(s.one_hot_gaussian_label(sigma)).float().squeeze()
            for i in range(epilines.shape[0]):
                current_epiline_dataset = \
                    EpilineDataset(epilines[i,:,:], epilines_mask[i,:,:],
                                   patch_size, label,
                                   data_augmentation=data_augmentation,
                                   jitter_dist=jitter_dist,
                                   allow_overlap=allow_overlap)
                self.epiline_datasets.append(current_epiline_dataset)

        self.nested_dataset = data.ConcatDataset(self.epiline_datasets)

    def __len__(self):
        return len(self.nested_dataset)

    def __getitem__(self, index):
        return self.nested_dataset[index]

    def set_data_augmentation(self, data_augmentation):
        for D in self.epiline_datasets:
            D.data_augmentation = data_augmentation

class PatchesInImageDataset(data.Dataset):
    """
    Dataset that samples overlapping patches from a single HxW image
    """
    def __init__(self, img, patches, patch_size, label):
        assert img.ndim == 2
        self.img = img
        self.patches = patches
        self.patch_size = patch_size
        self.label = label

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, index):
        start_r = self.patches[index][0]
        end_r = start_r + self.patch_size
        start_c = self.patches[index][1]
        end_c = start_c + self.patch_size

        return self.img[None, start_r:end_r, start_c:end_c], self.label

class PatchesInImageStackDataset(data.Dataset):
    """
    Dataset that samples patches from a NxHxW image,
    where the image channel is randomly chosen each time
    """
    def __init__(self, img, patches, patch_size, label):
        self.img = img
        self.patches = patches
        self.patch_size = patch_size
        self.label = label

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, index):
        num_channels = self.img.shape[0]
        i = np.random.randint(num_channels)

        start_r = self.patches[index][0]
        end_r = start_r + self.patch_size
        start_c = self.patches[index][1]
        end_c = start_c + self.patch_size

        return self.img[i, start_r:end_r, start_c:end_c][None,:,:], self.label

class DGSingleShotDataset(data.Dataset):
    def __init__(self, strawberries, patch_size, sigma, allow_overlap=False):
        self.tile_datasets = []
        for s in strawberries:
            if allow_overlap:
                all_patches = training_util.all_patches_in_mask(s, patch_size)
            else:
                all_patches = training_util.all_patches_no_overlap(s, patch_size)
            dg_captures = s.get_dg_captures()
            label = torch.from_numpy(s.one_hot_gaussian_label(sigma)).float().squeeze()
            tile_dataset = PatchesInImageStackDataset(dg_captures, all_patches, patch_size, label)
            self.tile_datasets.append(tile_dataset)

        self.nested_dataset = data.ConcatDataset(self.tile_datasets)

    def __len__(self):
        return len(self.nested_dataset)

    def __getitem__(self, index):
        return self.nested_dataset[index]

    def set_data_augmentation(self, data_augmentation):
        pass

def _dataloader_mean_std_(dataloader):
    count = 0

    for img, _ in dataloader:
        if count == 0:
            mean_sum = torch.zeros(img.shape[1])
            std_sum = torch.zeros(img.shape[1])

        img = img.view(img.shape[0], img.shape[1], -1)
        mean = torch.mean(img, dim=2)
        std = torch.std(img, dim=2)

        mean_sum += torch.sum(mean, dim=0)
        std_sum += torch.sum(std, dim=0)

        count += img.shape[0]

    return mean_sum, std_sum, count

# Returns the dataset mean and standard deviation for the given
# training and validation dataloaders
# returns:
#  dataset_mean: Nx1
#  dataset_std: Nx1
def normalization_for_dataset(train_dataloader, val_dataloader):
    train_mean_sum, train_std_sum, train_cnt = _dataloader_mean_std_(train_dataloader)
    val_mean_sum, val_std_sum, val_cnt = _dataloader_mean_std_(train_dataloader)

    dataset_mean = (train_mean_sum + val_mean_sum) / (train_cnt + val_cnt)
    dataset_std = (train_std_sum + val_std_sum) / (train_cnt + val_cnt)

    return dataset_mean, dataset_std


# Returns dataset_mean, dataset_std for given feature
def normalization_for_feature(feature_num, normalization_dir="normalization"):
    if feature_num == load_data.EPIPOLAR:
        with open(os.path.join(normalization_dir, "epipolar_normalization.npz"), 'rb') as f:
            files = np.load(f)
            dataset_mean = torch.from_numpy(files['epipolar_mean'])
            dataset_std = torch.from_numpy(files['epipolar_std'])

    elif feature_num == load_data.EPIPOLAR_INDIRECT:
        with open(os.path.join(normalization_dir, "epipolar_normalization.npz"), 'rb') as f:
            files = np.load(f)
            channels = torch.arange(strawberry.EPIPOLAR_NUM_CHANNELS)
            dataset_mean = torch.from_numpy(files['epipolar_mean'][(channels != 24) & \
                        (channels != 25) & (channels != 26)])
            dataset_std = torch.from_numpy(files['epipolar_std'][(channels != 24) & (channels != 25) & (channels != 26)])

    elif feature_num == load_data.STACK_ID_IG or feature_num == load_data.GLOBAL_ONLY:
        with open(os.path.join(normalization_dir, "dg_normalization.npz"), 'rb') as f:
            files = np.load(f)
            if feature_num == load_data.STACK_ID_IG:
                dataset_mean = torch.tensor([float(files['global_mean']), float(files['direct_mean'])])
                dataset_std = torch.tensor([float(files['global_std']), float(files['direct_std'])])
            elif feature_num == load_data.GLOBAL_ONLY:
                dataset_mean = torch.tensor([float(files['global_mean'])])
                dataset_std = torch.tensor([float(files['global_std'])])

    elif feature_num == load_data.EPIPOLAR_DG_STACK:
        with open(os.path.join(normalization_dir, "epipolar_normalization.npz"), 'rb') as f:
            files = np.load(f)
            epipolar_mean = torch.from_numpy(files['epipolar_mean'])
            epipolar_std = torch.from_numpy(files['epipolar_std'])
        with open(os.path.join(normalization_dir, "dg_normalization.npz"), 'rb') as f:
            files = np.load(f)
            dg_mean = torch.tensor([float(files['global_mean']), float(files['direct_mean'])])
            dg_std = torch.tensor([float(files['global_std']), float(files['direct_std'])])

        dataset_mean = np.concatenate([dg_mean, epipolar_mean], axis=0)
        dataset_std = np.concatenate([dg_std, epipolar_std], axis=0)

    elif feature_num == load_data.SIRI_AC:
        with open(os.path.join(normalization_dir, "siri_normalization.npz"), 'rb') as f:
            files = np.load(f)
            dataset_mean = torch.from_numpy(files['siri_ac_mean'])
            dataset_std = torch.from_numpy(files['siri_ac_std'])

    elif feature_num == load_data.SIRI_AC_DC:
        with open(os.path.join(normalization_dir, "siri_normalization.npz"), 'rb') as f:
            files = np.load(f)
            Iac_mean = torch.from_numpy(files['siri_ac_mean'])
            Iac_std = torch.from_numpy(files['siri_ac_std'])
            Idc_mean = torch.from_numpy(files['siri_dc_mean'])
            Idc_std = torch.from_numpy(files['siri_dc_std'])

        dataset_mean = torch.cat([Iac_mean, Idc_mean], dim=0)
        dataset_std = torch.cat([Iac_std, Idc_std], dim=0)

    else:
        assert(False)

    return dataset_mean, dataset_std

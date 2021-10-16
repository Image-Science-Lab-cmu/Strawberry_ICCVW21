import numpy as np
import numpy.random
import random
import strawberry
from sklearn.metrics import confusion_matrix

# Returns the nearest-K accuracy for predicted labels
# A label is accurate if  abs(ground_truth - predict) <= K
def nearest_k_accuracy(Y_true, Y_predicted, K):
    diff = np.abs(Y_true - Y_predicted)
    return np.sum(diff <= K) / len(diff)

# Returns a list of tuples, where each tuple gives row_start and column_start
# for each patch.
# Patches are unique and must be completely inside the masked region
# If num_patches == -1, returns all possible patches
def random_patches(s, patch_size, num_patches, allow_overlap=True, fraction_in_mask=1):
    if allow_overlap:
        all_patches = all_patches_in_mask(s, patch_size, fraction_in_mask=fraction_in_mask)
    else:
        all_patches = all_patches_no_overlap(s, patch_size)

    if num_patches == -1 or num_patches > len(all_patches):
        return all_patches

    assert(num_patches <= len(all_patches))

    return [all_patches[i] for i in random.sample(range(len(all_patches)), num_patches)]

# Returns a list of tuples of every patch completely inside the strawberry's
# mask
# fraction_in_mask: fraction of pixels that must be in the masked region
def all_patches_in_mask(s, patch_size, fraction_in_mask=1):
    if fraction_in_mask == 1:
        common_patches = s.get_patches()
        if patch_size in common_patches.keys():
            return common_patches[patch_size]

    patches = []

    (r_max, c_max) = s.get_mask().shape
    for r in range(r_max - patch_size):
        for c in range(c_max - patch_size):
            if not s.get_mask()[r,c]:
                continue
            patch = (r, c)
            patch_mask = s.get_mask()[patch[0]:patch[0]+patch_size,patch[1]:patch[1]+patch_size]
            if np.sum(patch_mask) >= patch_size * patch_size * fraction_in_mask:
                patches.append(patch)

    return patches

# Returns a list of all patches in the mask that do not overlap
def all_patches_no_overlap(s, patch_size):
    patches = []

    (r_max, c_max) = s.get_mask().shape
    for r in range(0, r_max - patch_size, patch_size):
        for c in range(0, c_max - patch_size, patch_size):
            if not s.get_mask()[r,c]:
                continue
            patch = (r, c)
            patch_mask = s.get_mask()[patch[0]:patch[0]+patch_size,patch[1]:patch[1]+patch_size]
            if patch_mask.all():
                patches.append(patch)

    return patches

# Converts a column vector of labels to one-hot vectors
def labels_to_onehot(labels):
    vs = np.zeros((labels.shape[0], strawberry.NUM_CLASSES))

    for i in range(labels.shape[0]):
        label = labels[i]
        vs[i,strawberry.label_to_index(label)] = 1

    return vs

# Adds gaussian noise centered around each true label
# labels: matrix of one-hot vectors
# sigma: standard deviation
def add_gaussian(labels, sigma):
    assert(labels.shape[1] == strawberry.NUM_CLASSES)
    labels_with_gaussian = np.zeros(labels.shape)
    for i in range(labels.shape[0]):
        label = np.argmax(labels[i,:])
        gaussian_x = np.arange(strawberry.NUM_CLASSES) - label
        gaussian = 1/(sigma * np.sqrt(2*np.pi))*np.exp(-gaussian_x*gaussian_x/(2*sigma*sigma))
        gaussian = gaussian / np.sum(gaussian)
        labels_with_gaussian[i,:] = gaussian
    return labels_with_gaussian

def print_metrics(Y, Y_hat, num_k):
    k_accuracy = np.zeros(num_k)
    for k in range(num_k):
        k_accuracy[k] = nearest_k_accuracy(Y, Y_hat, k)
    print("Nearest-K accuracy")
    print(k_accuracy)
    print()
    print("Confusion Matrix")
    print(confusion_matrix(Y, Y_hat))


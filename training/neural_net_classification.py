import os
import time
import pathlib
from time import perf_counter
import datetime
from pathlib import Path
import argparse

from sklearn import model_selection
from torch.utils.data.dataset import Subset
from datasets import EpilineDataset, PatchesInImageDataset
import numpy as np
import psutil
import pickle
import random
import io

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torch.utils.data
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms

import sklearn.metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
import seaborn

import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import training_util
import load_data
import strawberry
import nn_classifiers
import datasets
from datasets import PatchDataset, PAPatchDataset, \
                     EpipolarSingleShotDataset, DGSingleShotDataset

torch.set_default_tensor_type('torch.FloatTensor')

GPU_DEBUG = False

device = None # GPU device

train_val_dataset = None # dataset used to switch data augmentation on/off
inference_time = 0


# Sets pytorch, numpy and python random seeds to given seed
def reset_random(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# Returns a tensor of the confusion matrix C (numpy array) as an image
def confusion_matrix_image(C, b_acc):
    edibility_days = strawberry.index_to_label(np.arange(strawberry.NUM_CLASSES))

    plt.figure(figsize=(11,11))
    sns.heatmap(C / np.sum(C, axis=1)[:,None], annot=True, linewidths=.5,
                square=True, cmap = 'Blues_r');
    plt.ylabel('Actual label');
    plt.xlabel('Predicted label');
    all_sample_title = 'Balanced Accuracy Score: {0}'.format(b_acc)
    plt.title(all_sample_title, size = 15);
    plt.show()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    cm_img = Image.open(buf)
    cm_img = torchvision.transforms.ToTensor()(cm_img)

    return cm_img


def print_metrics(Y_true, Y_predicted, output_file=None, conf_file=None, name=None,
                  writer=None):
    num_k = 3
    k_accuracy = np.zeros(num_k)
    k_accuracy_edible = np.zeros(num_k)

    Y_true_edible = Y_true[Y_true < 0]
    Y_predicted_edible = Y_predicted[Y_true < 0]

    for k in range(num_k):
        k_accuracy[k] = training_util.nearest_k_accuracy(Y_true, Y_predicted, k)
        k_accuracy_edible[k] = training_util.nearest_k_accuracy(Y_true_edible, Y_predicted_edible, k)

    # Negative: Edible, Positive: Inedible
    # label 0 => inedible
    num_true_positive = np.sum((Y_predicted >= 0) & (Y_true >= 0))
    num_false_positive = np.sum((Y_predicted >= 0) & (Y_true < 0))
    num_true_negative = np.sum((Y_predicted < 0) & (Y_true < 0))
    num_false_negative = np.sum((Y_predicted < 0) & (Y_true >= 0))

    C = confusion_matrix([strawberry.label_to_index(i) for i in Y_true],
                         [strawberry.label_to_index(i) for i in Y_predicted])

    b_acc = sklearn.metrics.balanced_accuracy_score(Y_true, Y_predicted)
    b_acc_edible = sklearn.metrics.balanced_accuracy_score(Y_true_edible, Y_predicted_edible)

    print("All Strawberries")
    print("N = %d" % len(Y_true))
    print("Balanced accuracy")
    print(b_acc)
    print("Nearest-K accuracy")
    print(np.round(k_accuracy, 3))
    print()
    print("Edible Strawberries")
    print("N = %d" % len(Y_true_edible))
    print("Balanced Accuracy")
    print(b_acc_edible)
    print("Nearest-K accuracy for only edible strawberries")
    print(np.round(k_accuracy_edible, 3))
    print()
    print("Negative: Edible, Positive: Inedible")
    print("True Positives:", num_true_positive)
    print("False Positives:", num_false_positive)
    print("True Negatives:", num_true_negative)
    print("False Negatives:", num_false_negative)
    print()
    print("Confusion Matrix")
    print(C)

    if output_file is not None:
        with open(output_file, 'a') as f:
            f.write(name + "\n")
            f.write("-----------\n")
            f.write("Balanced accuracy\n")
            f.write(str(b_acc) + "\n")
            f.write("Nearest-K accuracy for all strawberries\n")
            f.write(str(np.round(k_accuracy, 3)) + "\n")
            f.write("Nearest-K accuracy for only edible strawberries\n")
            f.write(str(np.round(k_accuracy_edible, 3)) + "\n")
            f.write("\n")
            f.write("Negative: Edible, Positive: Inedible\n")
            f.write("True Positives: %d\n" % num_true_positive)
            f.write("False Positives: %d\n" % num_false_positive)
            f.write("True Negatives: %d\n" % num_true_negative)
            f.write("False Negatives: %d\n" % num_false_negative)
            f.write("\n")
            f.write("Confusion Matrix\n")
            f.write(str(confusion_matrix(strawberry.label_to_index(Y_true), strawberry.label_to_index(Y_predicted))) + "\n")
            f.write("\n\n")

    if conf_file is not None:
        with open(conf_file, 'wb') as f:
            np.save(f, C)

    if writer is not None:
        cm_img = confusion_matrix_image(C, b_acc)
        writer.add_image(name, cm_img)

# Given training and testing indices, load all strawberry
# objects into training and testing list
def load_strawberries(train_idx, val_idx, test_idx, feature_num):
    capture_dates = strawberry.get_capture_dates()
    train_s = []
    val_s = []
    test_s = []

    for c in capture_dates:
        for si in range(strawberry.NUM_STRAWBERRIES):
            if si not in train_idx and si not in val_idx and si not in test_idx:
                continue
            for ri in range(strawberry.NUM_ROTATIONS):
                try:
                    s = strawberry.Strawberry(si, ri, c, feature_num)
                except:
                    continue

                if si in train_idx:
                    train_s.append(s)
                elif si in val_idx:
                    val_s.append(s)
                elif si in test_idx:
                    test_s.append(s)

    return train_s, val_s, test_s

def forward_pass_on_dataset(dataset, model, device, batch_size=256):
    num_workers = 4
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers,
                            pin_memory=True, drop_last=False)

    # Forward pass on GPU
    model.eval()
    global inference_time

    with torch.no_grad():
        predictions = torch.zeros(len(dataset), strawberry.NUM_CLASSES,
                                  device=device)
        p_i = 0
        for i, target in enumerate(dataloader):
            img = target[0]
            if img.device != device:
                img = img.to(device)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            model_time = time.time()
            output = model(img)
            model_time = time.time() - model_time
            inference_time += model_time

            predictions[p_i:p_i+img.shape[0],:] = output
            p_i += img.shape[0]

    return predictions.cpu()

def label_from_predictions(predictions, average_pdfs):
    num_patches = predictions.shape[0]
    # If average pdfs, average patch pdfs and then perform argmax
    if average_pdfs:
        mean_prediction = torch.sum(predictions, dim=0) / num_patches
        predicted_index = torch.argmax(mean_prediction).item()
    else:
        # Majority vote of patch classifications
        prediction_indices = torch.argmax(predictions, dim=1)
        predicted_index = torch.argmax(torch.bincount(prediction_indices)).item()

    return strawberry.index_to_label(predicted_index)

def label_from_dataset(dataset, model, device, average_pdfs):
    predictions = forward_pass_on_dataset(dataset, model, device)
    return label_from_predictions(predictions, average_pdfs)

# Predicts strawberry's class by taking the majority vote of each patch's
# predicted label
def predict_strawberry(s, model, num_patches, patch_size, feature_num, num_in_channels,
                       average_pdfs, use_spectralon=False, fraction_in_mask=1,
                       dataset_mean=0, dataset_std=1, allow_overlap=False):
    assert dataset_mean == 0
    assert dataset_std == 1
    assert not use_spectralon

    if feature_num == load_data.DG_SINGLE_SHOT:
        assert num_patches == -1
        dg_captures = torch.from_numpy(s.get_dg_captures()).float().to(device)
        if not allow_overlap:
            patches = training_util.all_patches_no_overlap(s, patch_size)
        else:
            patches = training_util.all_patches_in_mask(s, patch_size,
                                                        fraction_in_mask=fraction_in_mask)
        predicted_labels = np.zeros(dg_captures.shape[0], dtype=np.float32)
        capture_i = np.random.randint(0, dg_captures.shape[0])
        dataset = PatchesInImageDataset(dg_captures[capture_i,:,:], patches, patch_size, 0)
        predicted_label = label_from_dataset(dataset, model, device, average_pdfs)
        return np.asarray([predicted_label])

    elif feature_num == load_data.EPIPOLAR_SINGLE_SHOT:
        epilines = torch.from_numpy(s.get_epilines()).float().to(device)
        epilines_mask = s.get_epilines_mask()
        epilines_per_capture = s.get_epilines_per_capture()

        # Only examine the first capture
        current_epiline = 0
        current_epiline_datasets = []
        for j in range(epilines_per_capture[0]):
            current_epiline_img = epilines[current_epiline,:,:]
            current_epiline_mask = epilines_mask[current_epiline,:,:]
            current_epiline += 1
            current_epiline_dataset = \
                EpilineDataset(current_epiline_img, current_epiline_mask,
                                patch_size, 0, data_augmentation=False,
                                allow_overlap=allow_overlap)
            if len(current_epiline_dataset) > 0:
                current_epiline_datasets.append(current_epiline_dataset)
        if len(current_epiline_datasets) == 0:
            return []
        dataset = ConcatDataset(current_epiline_datasets)
        predicted_label = label_from_dataset(dataset, model, device, average_pdfs)

        return np.asarray([predicted_label])

    elif feature_num == load_data.PA_SINGLE_SHOT:
        predicted_labels = np.zeros(strawberry.NUM_PA_CAPTURES)
        for i in range(strawberry.NUM_PA_CAPTURES):
            patch_imgs = s.pa_patches(patch_size, num_capture=i)[:,None,:,:]
            patch_imgs = torch.from_numpy(patch_imgs).float().to(device)

            dataset = TensorDataset(patch_imgs)
            predicted_labels[i] = label_from_dataset(dataset, model, device, average_pdfs)

        return predicted_labels

    elif feature_num == load_data.EPILINES:
        dataset = EpipolarSingleShotDataset([s], patch_size, 0,
                                            data_augmentation=False,
                                            allow_overlap=allow_overlap)
        predicted_label = label_from_dataset(dataset, model, device, average_pdfs)

        return np.asarray([predicted_label])

    elif feature_num == load_data.PA_1 or feature_num == load_data.PA_2 or \
        feature_num == load_data.PA_3 or feature_num == load_data.PA_4:

        import itertools

        predicted_labels = []
        for subset_captures in itertools.combinations(range(4), feature_num - 20):
            patch_imgs = None
            for i in subset_captures:
                current_pa_patches = s.pa_patches(patch_size, num_capture=i)[:,None,:,:]
                if patch_imgs is None:
                    patch_imgs = current_pa_patches
                else:
                    patch_imgs = np.concatenate((patch_imgs, current_pa_patches), axis=0)
            num_patches = patch_imgs.shape[0]
            patch_imgs = torch.from_numpy(patch_imgs).float().to(device)

            dataset = TensorDataset(patch_imgs)
            predicted_label = label_from_dataset(dataset, model, device, average_pdfs)
            predicted_labels.append(predicted_label)

        return np.asarray(predicted_labels)

    else:
        if feature_num == load_data.PA_PATCH:
            patch_imgs = s.pa_patches(patch_size)[:,None,:,:]
            num_patches = patch_imgs.shape[0]
        else:
            if num_patches == -1:
                patches = training_util.random_patches(s, patch_size, num_patches, allow_overlap=allow_overlap, fraction_in_mask=fraction_in_mask)
            else:
                patches = training_util.random_patches(s, patch_size, num_patches, allow_overlap=True, fraction_in_mask=fraction_in_mask)
            num_patches = len(patches)
            patch_imgs = s.patches_no_vectorize(patches, patch_size, feature_num)

        if use_spectralon:
            sr = s.get_spectralon_for_feature(feature_num)
            sr_channel = np.ones((num_patches, 1, patch_size, patch_size), dtype=np.float32) * sr
            patch_imgs = np.concatenate((patch_imgs, sr_channel), axis=1)

        patch_imgs = torch.from_numpy(patch_imgs).float().to(device)

        dataset = TensorDataset(patch_imgs)
        predicted_label = label_from_dataset(dataset, model, device, average_pdfs)

        return np.asarray([predicted_label])



def test_on_strawberries(model, num_patches, patch_size, feature_num, num_in_channels,
                         train_s, val_s, test_s, random_seed,
                         average_pdfs=False, use_spectralon=False, verbose=False,
                         fraction_in_mask=1, runs_dir="", normalize=False, writer=None):

    if len(train_s) > 0:
        train_true_labels, train_predicted_labels = \
            get_strawberry_predictions(model, num_patches, patch_size, feature_num,
                                       num_in_channels, train_s, random_seed,
                                       average_pdfs=average_pdfs,
                                       use_spectralon=use_spectralon,
                                       fraction_in_mask=fraction_in_mask,
                                       normalize=normalize)
        print("Train Strawberry Metrics")
        print_metrics(train_true_labels, train_predicted_labels,
                      output_file=os.path.join(runs_dir, "strawberry_metrics.txt"),
                      conf_file=os.path.join(runs_dir, "train_s_conf.npy"),
                      name="Train Strawberry Metrics", writer=writer)
        print()
        print()

    if len(val_s) > 0:
        val_true_labels, val_predicted_labels = \
            get_strawberry_predictions(model, num_patches, patch_size, feature_num,
                                       num_in_channels, val_s, random_seed,
                                       average_pdfs=average_pdfs,
                                       use_spectralon=use_spectralon,
                                       fraction_in_mask=fraction_in_mask,
                                       normalize=normalize)
        print("Val Strawberry Metrics")
        print_metrics(val_true_labels, val_predicted_labels,
                      output_file=os.path.join(runs_dir, "strawberry_metrics.txt"),
                      conf_file=os.path.join(runs_dir, "val_s_conf.npy"),
                      name="Val Strawberry Metrics", writer=writer)
        print()
        print()

    if len(test_s) > 0:
        test_true_labels, test_predicted_labels = \
            get_strawberry_predictions(model, num_patches, patch_size, feature_num,
                                       num_in_channels, test_s, random_seed,
                                       average_pdfs=average_pdfs,
                                       use_spectralon=use_spectralon,
                                       fraction_in_mask=fraction_in_mask,
                                       normalize=normalize)
        print("Test Strawberry Metrics")
        print_metrics(test_true_labels, test_predicted_labels,
                      output_file=os.path.join(runs_dir, "strawberry_metrics.txt"),
                      conf_file=os.path.join(runs_dir, "test_s_conf.npy"),
                      name="Test Strawberry Metrics", writer=writer)


def get_strawberry_predictions(model, num_patches, patch_size, feature_num, num_in_channels,
                            strawberries, random_seed, average_pdfs=False,
                            use_spectralon=False, fraction_in_mask=1, normalize=False,
                            allow_overlap=False):
    """
    Computes strawberry predictions on the given strawberries.
    Each prediction uses non-overlapping patches.
    """
    assert not normalize

    predicted_labels = []
    true_labels = []

    global inference_time
    inference_time = 0

    for i, s in enumerate(strawberries):
        current_predictions = predict_strawberry(s, model, num_patches,
                                             patch_size, feature_num,
                                             num_in_channels, average_pdfs,
                                             use_spectralon=use_spectralon,
                                             fraction_in_mask=fraction_in_mask,
                                             allow_overlap=allow_overlap)
        for j in range(len(current_predictions)):
            predicted_labels.append(current_predictions[j])
            true_labels.append(s.label)

    print("get_strawberry_predictions inference_time: %f seconds" % inference_time)

    return np.asarray(true_labels), np.asarray(predicted_labels)

def get_predictions(patches, predictions, patch_size, s):
    """
    Given patches: list of tuples, predictions: Nx14 predictions, returns
    a map: HxWx14 of prediction in the mask, the average 1x14 prediction, and
    an HxW image of patch predictions
    """
    prediction_map = np.zeros((s.get_mask().shape[0], s.get_mask().shape[1], strawberry.NUM_CLASSES))

    for i in range(predictions.shape[0]):
        p = patches[i]
        ri = p[0]
        ci = p[1]
        prediction_map[ri + (patch_size // 2),ci + (patch_size // 2),:] = predictions[i,:]

    # Argmax ( Sum ( Patch PDF's ))
    average_prediction = torch.sum(predictions, dim=0) / predictions.shape[0]

    # Majority vote ( Argmax ( Patch PDF's ))
    patch_predicted_labels = np.argmax(prediction_map, axis=2)

    return prediction_map, average_prediction, patch_predicted_labels

def predictions_on_pa(s, patch_imgs, pa_idx, patch_size, model, device):
    dataset = TensorDataset(torch.from_numpy(patch_imgs).float()[:,None,:,:])
    predictions = forward_pass_on_dataset(dataset, model, device)
    average_prediction = predictions.mean(0)

    prediction_map = np.zeros((s.get_mask().shape[0], s.get_mask().shape[1], strawberry.NUM_CLASSES))

    # Create prediction map where point arrays are projected
    pa = s.get_pa()
    half_patch_size = patch_size // 2
    prediction_i = 0

    for i in pa_idx:
        for j in range(s.pa_points.shape[2]):
            pt = s.pa_points[i,:,j].squeeze()
            if pt[0] == -1:
                break

            # Skip patch if not completely in the mask
            if not np.all(s.get_mask()[pt[0]-half_patch_size:pt[0]+half_patch_size+1,
                                          pt[1]-half_patch_size:pt[1]+half_patch_size+1]) or \
                    pt[0] < half_patch_size or pt[0] + half_patch_size + 1 > s.get_mask().shape[0] or \
                    pt[1] < half_patch_size or pt[1] + half_patch_size + 1 > s.get_mask().shape[1]:
                continue

            patch = pa[pt[0]-half_patch_size:pt[0]+half_patch_size+1,
                       pt[1]-half_patch_size:pt[1]+half_patch_size+1,i]
            prediction_map[pt[0]-half_patch_size:pt[0]+half_patch_size+1,
                           pt[1]-half_patch_size:pt[1]+half_patch_size+1,:] = \
                predictions[prediction_i,:]

            prediction_i += 1


    patch_predicted_labels = np.argmax(prediction_map, axis=2)
    return prediction_map, average_prediction, patch_predicted_labels, predictions


# Runs patch-wise predictions on every possible patch, returning
# a map of patch PDF's for each patch location (row, column)
# returns prediction_map: num_rows x num_cols x num_classes
#         average_prediction: num_classes x 1
#         patch_predicted_labels: num_rows x num_cols x 1
#         predictions: N x 13, each 13-dim patch prediction
def visualize_predictions(model, s, patch_size, feature_num, fraction_in_mask=1,
                          dataset_mean=0, dataset_std=1, device=None):
    assert dataset_mean == 0 and dataset_std == 1
    if feature_num == load_data.DG_SINGLE_SHOT:
        dg_captures = torch.from_numpy(s.get_dg_captures()).float()
        patches = training_util.all_patches_in_mask(s, patch_size,
                                                    fraction_in_mask=fraction_in_mask)

        prediction_maps = np.zeros((dg_captures.shape[0], s.get_mask().shape[0], s.get_mask().shape[1], strawberry.NUM_CLASSES))
        average_predictions = np.zeros((dg_captures.shape[0], strawberry.NUM_CLASSES))
        patch_predicted_labels = np.zeros((dg_captures.shape[0], s.get_mask().shape[0], s.get_mask().shape[1]))
        predictions = np.zeros((dg_captures.shape[0], len(patches), strawberry.NUM_CLASSES))

        for i in range(dg_captures.shape[0]):
            current_capture = dg_captures[i,:,:]
            current_dataset = PatchesInImageDataset(current_capture, patches, patch_size, 0)
            current_predictions = forward_pass_on_dataset(current_dataset, model, device, batch_size=256)

            current_map, current_avg_prediction, current_patch_predicted_labels = \
                get_predictions(patches, current_predictions, patch_size, s)

            prediction_maps[i,:,:,:] = current_map
            average_predictions[i,:] = current_avg_prediction
            patch_predicted_labels[i,:,:] = current_patch_predicted_labels
            predictions[i,:,:] = current_predictions

        return prediction_maps, average_predictions, patch_predicted_labels, \
            predictions

    elif feature_num == load_data.EPIPOLAR_SINGLE_SHOT:
        epilines = torch.from_numpy(s.get_epilines()).float()
        epilines_mask = s.get_epilines_mask()
        epilines_per_capture = s.get_epilines_per_capture()

        prediction_maps = []
        average_predictions = []
        patch_predicted_labels = []
        all_predictions = []

        current_epiline = 0
        cum_num_epilines = 0
        for i in range(len(epilines_per_capture)):
            current_num_epilines = epilines_per_capture[i]
            current_epiline_datasets = []
            for j in range(current_num_epilines):
                current_epiline_img = epilines[current_epiline,:,:]
                current_epiline_mask = epilines_mask[current_epiline,:,:]
                current_epiline += 1
                current_epiline_dataset = EpilineDataset(current_epiline_img, current_epiline_mask,
                                                         patch_size, 0,
                                                         data_augmentation=False,
                                                         allow_overlap=False)
                if len(current_epiline_dataset) > 0:
                    current_epiline_datasets.append(current_epiline_dataset)
            if len(current_epiline_datasets) == 0:
                continue

            current_dataset = ConcatDataset(current_epiline_datasets)
            predictions = forward_pass_on_dataset(current_dataset, model, device)
            prediction_map = np.zeros((current_epiline_mask.shape[0] * current_num_epilines,
                                       current_epiline_mask.shape[1], strawberry.NUM_CLASSES))
            prediction_i = 0

            for di, D in enumerate(current_epiline_datasets):
                current_patches = D.__get_patches__()
                for p in current_patches:
                    offset = (current_epiline_mask.shape[0] - patch_size) // 2
                    prediction_map[di*current_epiline_mask.shape[0]+offset:(di+1)*current_epiline_mask.shape[0]-offset,
                                   p:p+patch_size,:] = predictions[prediction_i,:][None,None,:]
                    prediction_i += 1

            prediction_maps.append(prediction_map)
            average_predictions.append(predictions.mean(0))
            patch_predicted_labels.append(prediction_map.argmax(2))
            all_predictions.append(predictions)
            cum_num_epilines += current_num_epilines

        return prediction_maps, average_predictions, patch_predicted_labels, all_predictions

    elif feature_num == load_data.PA_SINGLE_SHOT:
        prediction_maps = []
        average_predictions = np.zeros(strawberry.NUM_PA_CAPTURES, strawberry.NUM_CLASSES)
        patch_predicted_labels = []
        predictions = []

        for ci in strawberry.NUM_PA_CAPTURES:
            pa_patches_curr = s.pa_patches(patch_size, num_capture=ci)
            prediction_map, average_prediction, patch_predicted_labels_curr, predictions_curr = \
                predictions_on_pa(s, pa_patches_curr, [ci], patch_size, model, device)

            prediction_maps.append(prediction_map)
            average_predictions[ci,:] = average_prediction
            patch_predicted_labels.append(patch_predicted_labels_curr)
            predictions.append(predictions_curr)

        return prediction_maps, average_predictions, patch_predicted_labels, predictions

    elif feature_num == load_data.PA_PATCH:
        patch_imgs = s.pa_patches(patch_size)

        prediction_map, average_prediction, patch_predicted_labels, predictions = \
            predictions_on_pa(s, patch_imgs, list(range(strawberry.NUM_PA_CAPTURES)), patch_size,
                              model, device)

        return prediction_map, average_prediction, patch_predicted_labels, predictions

    else:
        patches = training_util.all_patches_in_mask(s, patch_size,
                                                    fraction_in_mask=fraction_in_mask)
        patch_imgs = s.patches_no_vectorize(patches, patch_size, feature_num)
        patch_imgs = torch.from_numpy(patch_imgs).float()
        dataset = TensorDataset(patch_imgs)

        predictions = forward_pass_on_dataset(dataset, model, device)

        prediction_map, average_prediction, patch_predicted_labels = \
            get_predictions(patches, predictions, patch_size, s)

        return prediction_map, average_prediction, patch_predicted_labels, predictions

def patch_predictions(model, dataloader, batch_size):
    predictions = None
    ground_truths = None

    with torch.no_grad():
        model.eval()
        for i, sample in enumerate(dataloader):
            input, target = sample[0], sample[1]
            input = input.to(device)
            output = model(input)

            predicted_labels_curr = strawberry.index_to_label(torch.argmax(output, dim=1)).cpu()
            ground_truths_curr = strawberry.index_to_label(torch.argmax(target, dim=1))

            if predictions is None:
                predictions = predicted_labels_curr
                ground_truths = ground_truths_curr
            else:
                predictions = torch.cat((predictions, predicted_labels_curr), dim=0)
                ground_truths = torch.cat((ground_truths, ground_truths_curr), dim=0)

    return ground_truths, predictions

# Forward pass with all patches
def test_on_patches(model, patch_size, feature_num, batch_size,
                    train_dataloader, val_dataloader, test_dataloader,
                    runs_dir="", writer=None):
    model.eval()

    train_gt, train_predictions = patch_predictions(model, train_dataloader, batch_size)
    val_gt, val_predictions = patch_predictions(model, val_dataloader, batch_size)
    test_gt, test_predictions = patch_predictions(model, test_dataloader, batch_size)

    print("Train Patch Accuracies")
    print_metrics(train_gt.numpy(), train_predictions.numpy(),
                  output_file=os.path.join(runs_dir, "patch_metrics.txt"),
                  conf_file=os.path.join(runs_dir, "train_patch_conf.npy"),
                  name="Train Patch Metrics", writer=writer)
    print()
    print()
    print("Val Patch Accuracies")
    print_metrics(val_gt.numpy(), val_predictions.numpy(),
                  output_file=os.path.join(runs_dir, "patch_metrics.txt"),
                  conf_file=os.path.join(runs_dir, "val_patch_conf.npy"),
                  name="Val Patch Metrics", writer=writer)
    print()
    print()
    print("Test Patch Accuracies")
    print_metrics(test_gt.numpy(), test_predictions.numpy(),
                  output_file=os.path.join(runs_dir, "patch_metrics.txt"),
                  conf_file=os.path.join(runs_dir, "test_patch_conf.npy"),
                  name="Test Patch Metrics", writer=writer)
    print()
    print()


def print_gpu_usage(help_str, device):
    '''
    Print current GPU usage
    '''
    if GPU_DEBUG:
        print(help_str)
        print('Allocated:', round(torch.cuda.memory_allocated(device)/1024**2,1), 'MB')
        print('Cached:   ', round(torch.cuda.memory_cached(device)/1024**2,1), 'MB')
        print()

def get_args():
    parser = argparse.ArgumentParser(description="PyTorch Neural Net Classification")
    parser.add_argument("feature_num", type=int, help="load_data.py feature number")
    parser.add_argument("patch_size", type=int, help="Patch size in px")
    parser.add_argument("total_patches", type=int, default=500000, help="Total number of patches accros training/validation/testing")
    parser.add_argument("patches_per_prediction", type=int, help="Patches per strawberry to make prediction with")
    parser.add_argument("--batch_size", type=int, default=512, help="Mini-batch size")
    parser.add_argument("--num_epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--num_feature_maps", type=int, default=512, help="Number of feature maps")
    parser.add_argument("--sigma", type=float, default=1.5, help="Standard deviation of gaussian to add to ground truth labels")
    parser.add_argument("--name", type=str, default=None, help="Name of neural network for saving runs and models")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of DataLoader workers")
    parser.add_argument("--noise_std", type=float, default=0, help="Std of gaussian noise in data augmentation")
    parser.add_argument("--batch_norm", action="store_true", help="Use batch normalization after conv layers")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--spectralon", action="store_true", help="Include spectralon reflectance as final channel")
    parser.add_argument("--fraction_in_mask", type=float, default=1, help="Fraction patch in mask")
    parser.add_argument("--num_splits", type=int, default=0, help="Number of splits for k-fold cross validation")
    parser.add_argument("--normalize", action="store_true", help="Normalize the dataset to mean 0 and std 0")
    parser.add_argument("--no_save_every_epoch", action="store_true", default=False, help="Do not save trained model after every epoch")
    parser.add_argument("--combine_train_val_berries", action="store_true", default=False, help="Combine training and validation strawberries, drawining non-overlapping patches")
    parser.add_argument("--kfold", action="store_true", default=False, help="Use K-Fold cross validation")

    args = parser.parse_args()
    print(args)
    print("")
    return args



def load_datasets(args, train_s, val_s, test_s):
    # Define dataset and dataloader
    total_strawberries = len(train_s) + len(val_s) + len(test_s)
    train_patches = round(args.total_patches * len(train_s) / total_strawberries)
    val_patches = round(args.total_patches * len(val_s) / total_strawberries)
    test_patches = round(args.total_patches * len(test_s) / total_strawberries)

    if args.total_patches == -1:
        train_patches = -1
        val_patches = -1
        test_patches = -1

    if args.feature_num == load_data.PA_PATCH:
        train_val_s = []
        for s in train_s:
            train_val_s.append(s)
        for s in val_s:
            train_val_s.append(s)

        train_val_dataset = PAPatchDataset(strawberries=train_val_s,
                patch_size=args.patch_size,
                feature_num=args.feature_num,
                sigma=args.sigma, data_augmentation=True, max_rotation=20,
                noise_std=args.noise_std, use_spectralon=args.spectralon)
        train_size = int(0.8 * len(train_val_dataset))
        val_size = len(train_val_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(train_val_dataset,
                (train_size, val_size))

        if len(test_s) > 0:
            test_dataset = PAPatchDataset(strawberries=test_s,
                    patch_size=args.patch_size,
                    feature_num=args.feature_num,
                    sigma=args.sigma, data_augmentation=False,
                    use_spectralon=args.spectralon)
        else:
            test_dataset = None

    elif args.feature_num == load_data.EPIPOLAR_SINGLE_SHOT:
        print("Drawing patches without any overlap")
        if args.combine_train_val_berries:
            print("Combining training and validation strawberries")
            train_val_s = []
            for s in train_s:
                train_val_s.append(s)
            for s in val_s:
                train_val_s.append(s)
            if train_patches == -1:
                train_val_patches = -1
            else:
                train_val_patches = train_patches + val_patches
            train_val_dataset = EpipolarSingleShotDataset(train_val_s, args.patch_size,
                                                          args.sigma, jitter_dist=2,
                                                          data_augmentation=True,
                                                          allow_overlap=False)

            train_size = int(0.8 * len(train_val_dataset))
            val_size = len(train_val_dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(train_val_dataset,
                    (train_size, val_size))
        else:
            train_dataset = EpipolarSingleShotDataset(train_s, args.patch_size,
                                                      args.sigma, jitter_dist=2,
                                                      data_augmentation=True,
                                                      allow_overlap=False)
            if len(val_s) != 0:
                val_dataset = EpipolarSingleShotDataset(val_s, args.patch_size,
                                                        args.sigma,
                                                        data_augmentation=False,
                                                        allow_overlap=False)
            else:
                val_dataset = None

        if len(test_s) != 0:
            test_dataset = EpipolarSingleShotDataset(test_s, args.patch_size,
                                                     args.sigma,
                                                     data_augmentation=False,
                                                     allow_overlap=False)
        else:
            test_dataset = None

    elif args.feature_num == load_data.DG_SINGLE_SHOT:
        print("Drawing patches without any overlap")
        if args.combine_train_val_berries:
            print("Combining training and validation strawberries")
            train_val_s = []
            for s in train_s:
                train_val_s.append(s)
            for s in val_s:
                train_val_s.append(s)
            train_val_dataset = DGSingleShotDataset(train_val_s, args.patch_size,
                                                    args.sigma, allow_overlap=False)

            train_size = int(0.8 * len(train_val_dataset))
            val_size = len(train_val_dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(train_val_dataset,
                    (train_size, val_size))
        else:
            train_dataset = DGSingleShotDataset(train_s, args.patch_size,
                                                args.sigma, allow_overlap=False)
            if len(val_s) != 0:
                val_dataset = DGSingleShotDataset(val_s, args.patch_size,
                                                args.sigma, allow_overlap=False)
            else:
                val_dataset = None

        if len(test_s) != 0:
            test_dataset = DGSingleShotDataset(test_s, args.patch_size,
                                               args.sigma, allow_overlap=False)

        else:
            test_dataset = None

    else:
        print("Drawing patches without any overlap")
        if args.combine_train_val_berries:
            print("Combining training and validation strawberries")
            train_val_s = []
            for s in train_s:
                train_val_s.append(s)
            for s in val_s:
                train_val_s.append(s)
            if train_patches == -1:
                train_val_patches = -1
            else:
                train_val_patches = train_patches + val_patches
            train_val_dataset = PatchDataset(strawberries=train_val_s,
                                             patch_size=args.patch_size,
                                             total_patches=train_val_patches,
                                             feature_num=args.feature_num,
                                             sigma=args.sigma, data_augmentation=True,
                                             noise_std=args.noise_std, use_spectralon=args.spectralon,
                                             fraction_in_mask=args.fraction_in_mask,
                                             normalize=args.normalize, max_rotation=30,
                                             allow_overlap=False)
            train_size = int(0.8 * len(train_val_dataset))
            val_size = len(train_val_dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(train_val_dataset,
                    (train_size, val_size))
        else:
            train_dataset = PatchDataset(strawberries=train_s,
                                         patch_size=args.patch_size,
                                         total_patches=train_patches,
                                         feature_num=args.feature_num,
                                         sigma=args.sigma, data_augmentation=True,
                                         noise_std=args.noise_std, use_spectralon=args.spectralon,
                                         fraction_in_mask=args.fraction_in_mask,
                                         normalize=args.normalize, max_rotation=30,
                                         allow_overlap=False)

            val_dataset = PatchDataset(strawberries=val_s,
                                       patch_size=args.patch_size,
                                       total_patches=val_patches,
                                       feature_num=args.feature_num,
                                       sigma=args.sigma, data_augmentation=False,
                                       use_spectralon=args.spectralon,
                                       fraction_in_mask=args.fraction_in_mask,
                                       normalize=args.normalize, max_rotation=0,
                                       allow_overlap=False)

        if len(test_s) != 0:
            test_dataset = PatchDataset(strawberries=test_s,
                                        patch_size=args.patch_size,
                                        total_patches=test_patches,
                                        feature_num=args.feature_num,
                                        sigma=args.sigma, data_augmentation=False,
                                        use_spectralon=args.spectralon,
                                        fraction_in_mask=args.fraction_in_mask,
                                        normalize=args.normalize, max_rotation=0,
                                        allow_overlap=False)
        else:
            test_dataset = None


    return train_dataset, val_dataset, test_dataset

def load_dataloaders(args, train_dataset, val_dataset, test_dataset):
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size,
                                shuffle=True,
                                num_workers=args.num_workers,
                                pin_memory=True, drop_last=True)
    if test_dataset is not None:
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers = args.num_workers,
                                     pin_memory=True, drop_last=True)
    else:
        test_dataloader = None

    return train_dataloader, val_dataloader, test_dataloader,


# Returns loss (based on criterion) and overall patch accuracy
# Disables data augmentation on train_val_dataset global
def loss(model, criterion, dataloader):
    model.eval()
    if train_val_dataset is not None:
        if isinstance(train_val_dataset, Subset):
            train_val_dataset.dataset.set_data_augmentation(False)
        else:
            train_val_dataset.set_data_augmentation(False)

    num_patches = 0
    num_correct = 0
    with torch.no_grad():
        loss_epoch = 0.0
        for i, sample in enumerate(dataloader):
            input, target = sample[0], sample[1]
            input = input.to(device)
            target = target.to(device)
            output = model(input)

            predicted_label = torch.argmax(output, dim=1)
            ground_truth = torch.argmax(target, dim=1)

            num_correct += torch.sum(predicted_label == ground_truth).item()
            num_patches += input.shape[0]

            loss = criterion(output, target)
            loss_epoch += loss.item() / target.shape[1]

        loss_epoch /= num_patches

    if train_val_dataset is not None:
        if isinstance(train_val_dataset, Subset):
            train_val_dataset.dataset.set_data_augmentation(True)
        else:
            train_val_dataset.set_data_augmentation(True)

    return loss_epoch, num_correct * 100 / num_patches


# Trains the model and returns the model and validation loss corresponding
# to the epoch with the best validation loss
def train_model(args, model, criterion, train_dataloader,
                val_dataloader, test_dataloader, writer,
                name_prefix=""):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
    #print("Using scheduler with step size 40")
    scheduler = None

    # Save model information
    if args.name == None:
        args.name = f'nn_patch_{args.patch_size}'
    save_dir = os.path.join('saved_models', args.name)
    Path(save_dir).mkdir(parents=True, exist_ok=True)


    ## Training
    print_every_nth_epoch = 1

    min_loss = float('inf') # Best validation loss tracker
    max_acc = -1 # Best validation accuracy tracker

    start_time = perf_counter()
    print(f'Training started at {time.asctime(time.localtime())}')

    # Go through each epoch
    num_iter = 0
    for epoch in range(1,args.num_epochs+1):
        epoch_time = perf_counter()

        # Training minibatch loop
        model.train()
        train_loss_epoch = 0.0

        train_num_correct = 0
        train_num_patches = 0
        for i, sample in enumerate(train_dataloader):
            input, target = sample[0], sample[1]
            input = input.to(device)
            target = target.to(device)
            output = model(input)

            train_loss = criterion(output, target)

            train_loss_epoch += train_loss.item() / target.shape[1]
            writer.add_scalar('IterLoss/train', train_loss / input.shape[0], num_iter)

            # Update training accuracy
            predicted_labels = torch.argmax(output, dim=1)
            ground_truth = torch.argmax(target, dim=1)
            train_num_correct += torch.sum(predicted_labels == ground_truth).item()
            train_num_patches += input.shape[0]

            # Backprop for each bin
            model.zero_grad()
            train_loss.backward()
            optimizer.step()
            num_iter = num_iter + 1

        train_loss_epoch /= train_num_patches
        train_acc_epoch = train_num_correct * 100 / train_num_patches

        # torch.cuda.empty_cache()
        print_gpu_usage(f'After training epoch {epoch}', device)

        # Scheduler for epoch, not bin
        if scheduler is not None:
            scheduler.step()

        # Validation minibatch loop
        val_loss_epoch, val_acc_epoch = loss(model, criterion, val_dataloader)

        # torch.cuda.empty_cache()
        print_gpu_usage(f'After validation epoch {epoch}', device)

        if epoch == 1 or epoch % print_every_nth_epoch == 0 or epoch == args.num_epochs:
            print("Epoch: %d, Train loss: %.6f, Train acc: %.4f%%, Val loss: %.6f, Val acc: %.4f%%" % (epoch, train_loss_epoch, train_acc_epoch, val_loss_epoch, val_acc_epoch))
            print("Epoch time: %f minutes" % ((perf_counter() - epoch_time) / 60))

        writer.add_scalar('Loss/train', train_loss_epoch, epoch)
        writer.add_scalar('Acc/train', train_acc_epoch, epoch)
        writer.add_scalar('Loss/validation', val_loss_epoch, epoch)
        writer.add_scalar('Acc/validation', val_acc_epoch, epoch)

        if val_loss_epoch < min_loss:
            min_loss = val_loss_epoch
            epoch_best_loss = epoch
            torch.save(model.state_dict(), os.path.join(save_dir, '%sbest_loss.pth' % name_prefix))

        if val_acc_epoch > max_acc:
            max_acc = val_acc_epoch
            epoch_best_acc = epoch
            torch.save(model.state_dict(), os.path.join(save_dir, '%sbest_acc.pth') % name_prefix)

        if not args.no_save_every_epoch:
            torch.save(model.state_dict(), os.path.join(save_dir, "%sepoch-%d.pth" % (name_prefix, epoch)))

    end_time = perf_counter()
    print()
    print(f'Training ended at {time.asctime(time.localtime())}')
    print(f'Time taken is {(end_time-start_time):.2f} seconds')
    print("Lowest validation loss in epoch %d" % epoch_best_loss)
    print("Highest validation accuracy in epoch %d" % epoch_best_acc)
    print()
    print()
    # train ends

    model.load_state_dict(torch.load(os.path.join(save_dir, '%sbest_loss.pth' % name_prefix)))

    return model, min_loss, max_acc

# Given training and validation indices [0,34), return list of
# training/validation strawberries
def split_by_idx(strawberries, train_idx, val_idx):
    train_s = []
    val_s = []
    for s in strawberries:
        if s.strawberry_index in train_idx:
            train_s.append(s)
        elif s.strawberry_index in val_idx:
            val_s.append(s)

    return train_s, val_s


def main_cross_validation(args, train_val_s, random_seed):
    """
    Splits strawberries into 24 training strawberries and 10 validation strawberries
    Non-overlapping patches are sampled from these strawberries without class-balancing
    to create training and validation sets

    Saves best validation patch loss, accuracy, and strawberry accuracy for each fold
    """

    train_idx = np.arange(strawberry.NUM_STRAWBERRIES)

    train_size = 24
    train_folds = dict()
    val_folds = dict()

    min_loss = float('inf')
    best_model = None
    min_loss_fold = -1
    save_dir = os.path.join('saved_models', args.name)

    # Loss, accuracy statistics
    patch_losses = np.zeros(args.num_splits)
    patch_accs = np.zeros(args.num_splits)
    # Balanced accuracy for validation strawberry predictions
    strawberry_accs = np.zeros(args.num_splits)

    for i in range(args.num_splits):
        # Gen train_split_idx, val_split_idx
        train_idx_split = np.random.choice(train_idx, size=train_size, replace=False)
        train_idx_curr = set(train_idx[train_idx_split])
        val_idx_curr = set(train_idx) - set(train_idx_curr)
        print("Fold %d" % i)
        print("------------")
        print("train_idx:", train_idx_curr)
        print("val_idx:", val_idx_curr)
        train_folds[i] = train_idx_curr
        val_folds[i] = val_idx_curr


        train_s, val_s = split_by_idx(train_val_s, train_idx_curr, val_idx_curr)

        train_dataset, val_dataset, _ = load_datasets(args, train_s, val_s, [])
        train_dataloader, val_dataloader, test_dataloader = load_dataloaders(args,
                train_dataset, val_dataset, None)

        # Run single model
        model, val_loss, val_acc, tensorboard_dir, num_in_channels, average_pdf, writer = \
            train_single_model(args, train_dataloader, val_dataloader, test_dataloader,
                               name_prefix="fold_%d_" % i)
        # Compute validation strawberry predictions
        s_true, s_predicted = get_strawberry_predictions(model, args.patches_per_prediction,
                                    args.patch_size, args.feature_num, num_in_channels,
                                    val_s, random_seed, average_pdfs=False,
                                    use_spectralon=args.spectralon,
                                    fraction_in_mask=args.fraction_in_mask,
                                    normalize=args.normalize)

        # Save loss, accuracy metrics for validation strawberries
        patch_losses[i] = val_loss
        patch_accs[i] = val_acc
        strawberry_accs[i] = sklearn.metrics.balanced_accuracy_score(s_true, s_predicted)

        print("Patch Loss: %f" % patch_losses[i])
        print("Patch Acc: %f" % patch_accs[i])
        print("Strawberry Acc: %f" % strawberry_accs[i])

        if val_loss < min_loss:
            min_loss = val_loss
            min_loss_fold = i
            best_model = model
            best_train_idx = train_idx_curr
            best_val_idx = val_idx_curr

        print()
        print()

    print("Best validation loss of %f in fold %d" % (min_loss, min_loss_fold))

    assert(best_model is not None)

    # Only save best model from cross-validation
    torch.save(best_model.state_dict(), os.path.join(save_dir, 'best_model.pth'))

    # Save fold indices and best fold
    with open(os.path.join(save_dir, 'train_folds.pkl'), 'wb') as f:
        pickle.dump(train_folds, f)
    with open(os.path.join(save_dir, 'val_folds.pkl'), 'wb') as f:
        pickle.dump(val_folds, f)
    with open(os.path.join(save_dir, 'best_fold.pkl'), 'wb') as f:
        pickle.dump(min_loss_fold, f)

    metrics = (patch_losses, patch_accs, strawberry_accs)
    with open(os.path.join(save_dir, 'cross_val_metrics.pkl'), 'wb') as f:
        pickle.dump(metrics, f)

    return best_model, tensorboard_dir, num_in_channels, average_pdf, \
           best_train_idx, best_val_idx, writer, metrics

def main_k_fold_cross_val(args, train_val_s):
    global train_val_dataset
    train_val_dataset, _, _ = load_datasets(args, train_val_s, [], [])
    all_idx = np.arange(len(train_val_dataset))

    save_dir = os.path.join('saved_models', args.name)

    kf = sklearn.model_selection.KFold(n_splits=args.num_splits,
                                       shuffle=True)
    best_loss = float('inf')
    best_fold = -1
    best_model = None

    for i, (train_idx, val_idx) in enumerate(kf.split(all_idx)):
        train_dataset = Subset(train_val_dataset, train_idx)
        val_dataset = Subset(train_val_dataset, val_idx)

        print("Fold %d" % i)
        train_dataloader, val_dataloader, _ = load_dataloaders(args,
                train_dataset, val_dataset, None)

        # Run single model
        model, val_loss, val_acc, tensorboard_dir, num_in_channels, average_pdf, writer = \
            train_single_model(args, train_dataloader, val_dataloader, None,
                               name_prefix="fold_%d_" % i)
        if val_loss < best_loss:
            best_loss = val_loss
            best_fold = i
            best_model = model
        print("Validation loss: %f" % val_loss)
        print()

    assert best_model is not None
    print()
    print("Best validation of loss of %.3f in split %d" % (best_loss, best_fold))

    torch.save(best_model.state_dict(), os.path.join(save_dir, 'best_model.pth'))

def train_single_model(args, train_dataloader, val_dataloader, test_dataloader,
                       name_prefix=""):
    ###### Setup
    ## Define model
    num_in_channels = strawberry.num_channels_for_feature(args.feature_num)

    if args.spectralon:
        num_in_channels += 1

    model = nn_classifiers.PatchClassifier(num_in_channels,
                                                strawberry.NUM_CLASSES,
                                                args.num_feature_maps,
                                                batch_norm=args.batch_norm,
                                                k=3, dropout_p=0)

    # MSE or BCE criterion, depending on if using gaussian pdf on ground truth
    if args.sigma != 0:
        criterion = torch.nn.MSELoss(reduction='sum')
        average_pdfs = True
    else:
        criterion = torch.nn.BCELoss(reduction='sum')
        average_pdfs = False


    # Move model to GPU before creating optimizers
    global device
    device = torch.device('cuda:0')
    model = model.to(device)
    # For multi-GPU training
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    criterion = criterion.to(device)

    # Tensorboard
    tensorboard_dir = os.path.join('runs', args.name)
    writer = SummaryWriter(log_dir=tensorboard_dir)


    ##### Training
    # Returns model with lowest validation loss
    model, val_loss, val_acc = train_model(args, model, criterion, train_dataloader,
                        val_dataloader, test_dataloader, writer,
                        name_prefix=name_prefix)

    writer.close()

    return model, val_loss, val_acc, tensorboard_dir, num_in_channels, average_pdfs, writer

def main():
    ## Constants
    # Random seeds for dataloading and testing
    random_seed = 20210223

    # Origianl train/val/test
    train_idx = {0, 1, 2, 3, 4, 5, 6, 9, 10, 12, 14, 15, 16, 17, 19, 20, 22, 23, 25, 26, 27, 28, 29, 30}
    val_idx = {18, 21, 24}
    test_idx = {32, 33, 7, 8, 11, 13, 31}


    ## Init
    reset_random(random_seed)
    args = get_args()


    ## Cross-validation
    if args.num_splits != 0:
        if args.kfold:
            # K-Fold cross validation on the fixed set
            args.no_save_every_epoch = True
            train_val_idx = train_idx.union(val_idx)
            train_val_s, _, _ = load_strawberries( \
                train_val_idx, [], [], args.feature_num)
            main_k_fold_cross_val(args, train_val_s)
        else:
            # Load strawberries
            train_val_idx = train_idx.union(val_idx)
            train_val_s, _, _ = load_strawberries( \
                train_val_idx, [], [], args.feature_num)

            # Run cross-validation on all strawberries (including test strawberries)
            args.combine_train_val_berries = False
            model, tensorboard_dir, num_in_channels, average_pdfs, \
                train_idx, val_idx, writer, metrics = \
                main_cross_validation(args, train_val_s, random_seed)
            patch_losses, patch_accs, strawberry_accs = metrics

            print("patch loss v. epoch:", patch_losses)
            print("patch acc v. epoch:", patch_accs)
            print("strawberry acc v. epoch", strawberry_accs)

    ## Train single model
    else:
        train_s, val_s, test_s = load_strawberries(train_idx, val_idx, test_idx, args.feature_num)
        train_dataset, val_dataset, test_dataset = load_datasets(args, train_s, val_s, test_s)
        print("Training patches:", len(train_dataset) + len(val_dataset))
        return
        train_dataloader, val_dataloader, test_dataloader = \
            load_dataloaders(args, train_dataset, val_dataset, test_dataset)
        model, _, _, tensorboard_dir, num_in_channels, average_pdfs, writer = \
            train_single_model(args, train_dataloader, val_dataloader, test_dataloader)

        torch.cuda.empty_cache()
        model.eval()
        with torch.no_grad():
            test_on_patches(model, args.patch_size, args.feature_num, args.batch_size,
                            train_dataloader, val_dataloader, test_dataloader,
                            runs_dir=tensorboard_dir, writer=writer)

            import sys
            sys.exit(0)
            test_on_strawberries(model, args.patches_per_prediction, args.patch_size,
                                 args.feature_num, num_in_channels,
                                 train_s, val_s, test_s, random_seed,
                                 average_pdfs=average_pdfs,
                                 use_spectralon=args.spectralon,
                                 fraction_in_mask=args.fraction_in_mask,
                                 runs_dir=tensorboard_dir, normalize=args.normalize,
                                 writer=writer)

if __name__ == "__main__":
    main()

import os

import numpy as np
from scipy.spatial import KDTree
from skimage.morphology import binary_erosion
from scipy.ndimage import zoom
import torch
from medpy import metric

def count_params(model):
    param_num = sum(p.numel() for p in model.parameters())
    return param_num / 1e6

#---------------ACDC metric cal---------------------------------
def test_single_volume(image, label, net, classes, patch_size=256):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size / x, patch_size / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = net(input)
            if isinstance(out, list) or isinstance(out, tuple):
                out = out[0]
            out = torch.argmax(torch.softmax(out, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size, y / patch_size), order=0)
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase_volume(
            prediction == i, label == i))

    PA = np.sum(prediction == label) / label.size

    return metric_list, PA

def calculate_metric_percase_volume(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        jaccard = metric.binary.jc(pred, gt)  
        asd = metric.binary.asd(pred, gt)    
        return dice, hd95, jaccard, asd
    else:
        return 0, 0, 0, 0

#--------------2d RGB metric cal---------------------
def cal_metric_pixel_2D(image, label, net, classes, patch_size=256):

    image = image.cpu().detach().numpy()
    label = label.cpu().detach().numpy()
    prediction = np.zeros_like(label)

    for ind in range(image.shape[0]):
        slice = image[ind]
        x, y = slice.shape[1], slice.shape[2]
        slice = zoom(slice, (1, patch_size / x, patch_size / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(0).float().cuda()

        with torch.no_grad():
            out = net(input)
            out = out[0] if isinstance(out, (list, tuple)) else out
            out = torch.argmax(torch.softmax(out, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size, y / patch_size), order=0)
            prediction[ind] = pred

    PA = np.sum(prediction == label) / label.size

    metric_list = []

    for cls in range(1, classes):
        pred_mask = (prediction == cls).astype(np.uint8)
        gt_mask = (label == cls).astype(np.uint8)

        intersection = np.sum(pred_mask * gt_mask)
        pred_sum = np.sum(pred_mask)
        gt_sum = np.sum(gt_mask)

        if pred_sum + gt_sum == 0:
            dice = 1.0
        else:
            dice = (2. * intersection) / (pred_sum + gt_sum)

        union = pred_sum + gt_sum - intersection
        if union == 0:
            jaccard = 1.0
        else:
            jaccard = intersection / union

        hd95, asd = compute_hd95_asd_2d(pred_mask, gt_mask)

        metric_list.append([dice, jaccard, hd95, asd])

    return metric_list, PA


def compute_hd95_asd_2d(pred_mask, gt_mask, return_diagonal=False):

    pred_mask = pred_mask.astype(np.uint8) if pred_mask.dtype != np.uint8 else pred_mask
    gt_mask = gt_mask.astype(np.uint8) if gt_mask.dtype != np.uint8 else gt_mask

    pred_contour = np.argwhere(binary_erosion(pred_mask) ^ pred_mask)
    gt_contour = np.argwhere(binary_erosion(gt_mask) ^ gt_mask)

    if len(pred_contour) == 0 or len(gt_contour) == 0:
        if return_diagonal:
            diagonal = np.sqrt(pred_mask.shape[0] ** 2 + pred_mask.shape[1] ** 2)
            return diagonal, diagonal
        return np.nan, np.nan

    gt_tree = KDTree(gt_contour)
    pred_tree = KDTree(pred_contour)

    d_pred_to_gt, _ = gt_tree.query(pred_contour)
    d_gt_to_pred, _ = pred_tree.query(gt_contour)

    hd95 = max(np.percentile(d_pred_to_gt, 95), np.percentile(d_gt_to_pred, 95))

    asd = (np.mean(d_pred_to_gt) + np.mean(d_gt_to_pred)) / 2

    return hd95, asd

from scipy.stats import t

def compute_confidence_interval(data, confidence=0.95):
    data = np.array(data)
    data = data[~np.isnan(data)]

    if len(data) == 0:
        return np.nan, np.nan, np.nan, np.nan

    mean_val = np.nanmean(data)
    std_val = np.nanstd(data, ddof=1)
    n = len(data)
    if n > 1:
        t_critical = t.ppf((1 + confidence) / 2, df=n - 1)
        margin_of_error = t_critical * (std_val / np.sqrt(n))
    else:
        margin_of_error = 0.0

    ci_lower = mean_val - margin_of_error
    ci_upper = mean_val + margin_of_error

    return mean_val, std_val, ci_lower, ci_upper


#---------------3d----------------
import h5py
from collections import OrderedDict
from tqdm import tqdm
import nibabel as nib
import pandas as pd
import math
import torch.nn.functional as F

def test_all_case(
    net,
    image_list,
    num_classes,
    patch_size=(112, 112, 80),
    stride_xy=18,
    stride_z=4,
    save_result=True,
    test_save_path=None,
    preproc_fn=None,
):
    total_metric = 0.0
    metric_dict = OrderedDict()
    metric_dict["name"] = list()
    metric_dict["dice"] = list()
    metric_dict["jaccard"] = list()
    metric_dict["asd"] = list()
    metric_dict["95hd"] = list()
    for image_path in tqdm(image_list):
    # for image_path in image_list:
        case_name = image_path.split("/")[-2]
        id = image_path.split("/")[-1]
        h5f = h5py.File(image_path, "r")
        image = h5f["image"][:]
        label = h5f["label"][:]
        if preproc_fn is not None:
            image = preproc_fn(image)
        prediction, score_map = test_single_case(
            net, image, stride_xy, stride_z, patch_size, num_classes=num_classes
        )

        if np.sum(prediction) == 0:
            single_metric = (0, 0, 0, 0)
        else:
            single_metric = calculate_metric_percase(prediction, label[:])
            metric_dict["name"].append(case_name)
            metric_dict["dice"].append(single_metric[0])
            metric_dict["jaccard"].append(single_metric[1])
            metric_dict["asd"].append(single_metric[2])
            metric_dict["95hd"].append(single_metric[3])
            # print(metric_dict)

        total_metric += np.asarray(single_metric)

        if save_result:
            test_save_path_temp = os.path.join(test_save_path, case_name)
            if not os.path.exists(test_save_path_temp):
                os.makedirs(test_save_path_temp)
            nib.save(
                nib.Nifti1Image(prediction.astype(np.float32), np.eye(4)),
                test_save_path_temp + "/" + id + "_pred.nii.gz",
            )
            nib.save(
                nib.Nifti1Image(image[:].astype(np.float32), np.eye(4)),
                test_save_path_temp + "/" + id + "_img.nii.gz",
            )
            nib.save(
                nib.Nifti1Image(label[:].astype(np.float32), np.eye(4)),
                test_save_path_temp + "/" + id + "_gt.nii.gz",
            )
    avg_metric = total_metric / len(image_list)
    if save_result:
        metric_csv = pd.DataFrame(metric_dict)
        metric_csv.to_csv(test_save_path + "/metric.csv", index=False)
    print("average metric is {}".format(avg_metric))

    return avg_metric

def test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes=1):
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0] - w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1] - h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2] - d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad // 2, w_pad - w_pad // 2
    hl_pad, hr_pad = h_pad // 2, h_pad - h_pad // 2
    dl_pad, dr_pad = d_pad // 2, d_pad - d_pad // 2
    if add_pad:
        image = np.pad(
            image,
            [(wl_pad, wr_pad), (hl_pad, hr_pad), (dl_pad, dr_pad)],
            mode="constant",
            constant_values=0,
        )
    ww, hh, dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes,) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy * x, ww - patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y, hh - patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd - patch_size[2])
                test_patch = image[
                    xs : xs + patch_size[0],
                    ys : ys + patch_size[1],
                    zs : zs + patch_size[2],
                ]
                test_patch = np.expand_dims(
                    np.expand_dims(test_patch, axis=0), axis=0
                ).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()
                y1 = net(test_patch)
                # CRKD, URPC, MLRPL
                if isinstance(y1, list) or isinstance(y1, tuple):
                    y1 = y1[0]

                y = F.softmax(y1, dim=1)
                y = y.cpu().data.numpy()
                y = y[0, :, :, :, :]
                score_map[
                    :,
                    xs : xs + patch_size[0],
                    ys : ys + patch_size[1],
                    zs : zs + patch_size[2],
                ] = (
                    score_map[
                        :,
                        xs : xs + patch_size[0],
                        ys : ys + patch_size[1],
                        zs : zs + patch_size[2],
                    ]
                    + y
                )
                cnt[
                    xs : xs + patch_size[0],
                    ys : ys + patch_size[1],
                    zs : zs + patch_size[2],
                ] = (
                    cnt[
                        xs : xs + patch_size[0],
                        ys : ys + patch_size[1],
                        zs : zs + patch_size[2],
                    ]
                    + 1
                )
    score_map = score_map / np.expand_dims(cnt, axis=0)
    label_map = np.argmax(score_map, axis=0)
    if add_pad:
        label_map = label_map[
            wl_pad : wl_pad + w, hl_pad : hl_pad + h, dl_pad : dl_pad + d
        ]
        score_map = score_map[
            :, wl_pad : wl_pad + w, hl_pad : hl_pad + h, dl_pad : dl_pad + d
        ]
    return label_map, score_map

def calculate_metric_percase(pred, gt):
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    hd = metric.binary.hd95(pred, gt)
    asd = metric.binary.asd(pred, gt)

    return dice, jc, hd, asd


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
        jaccard = metric.binary.jc(pred, gt)  # Jaccard指数
        asd = metric.binary.asd(pred, gt)     # 平均表面距离
        return dice, hd95, jaccard, asd
    else:
        return 0, 0, 0, 0

#--------------2d RGB metric cal---------------------
def cal_metric_pixel_2D(image, label, net, classes, patch_size=256):
    '''
    2d版本的指标计算

    参数:
        image: 输入图像，形状为 [B, C, H, W] 的torch张量
        label: 真实标签，形状为 [B, H, W] 的torch张量
        net: 用于预测的神经网络模型
        classes: 类别数量（包括背景0类）
        patch_size: 模型输入尺寸，默认为256

    返回:
        metric_list: 每个非0类的指标列表，格式为[[Dice,Jaccard,HD95,ASD], ...]
        PA: 整体像素准确率
    '''
    # 将数据转为numpy并初始化预测结果
    image = image.cpu().detach().numpy()
    label = label.cpu().detach().numpy()
    prediction = np.zeros_like(label)

    # 逐片预测并调整大小
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

    # 计算PA
    PA = np.sum(prediction == label) / label.size

    # 初始化指标列表（只计算非0类）
    metric_list = []

    # 计算每个非0类的指标
    for cls in range(1, classes):
        # 获取当前类别的二值mask
        pred_mask = (prediction == cls).astype(np.uint8)
        gt_mask = (label == cls).astype(np.uint8)

        # 计算Dice和Jaccard
        intersection = np.sum(pred_mask * gt_mask)
        pred_sum = np.sum(pred_mask)
        gt_sum = np.sum(gt_mask)

        # Dice计算
        if pred_sum + gt_sum == 0:
            dice = 1.0
        else:
            dice = (2. * intersection) / (pred_sum + gt_sum)

        # Jaccard计算
        union = pred_sum + gt_sum - intersection
        if union == 0:
            jaccard = 1.0
        else:
            jaccard = intersection / union

        # 计算HD95和ASD
        hd95, asd = compute_hd95_asd_2d(pred_mask, gt_mask)

        # 将当前类别的指标添加到列表中
        metric_list.append([dice, jaccard, hd95, asd])

    return metric_list, PA


def compute_hd95_asd_2d(pred_mask, gt_mask, return_diagonal=False):
    """
    计算2D图像的HD95和ASD（优化合并版本）

    参数:
        pred_mask: 预测分割掩码（0-1二值矩阵）
        gt_mask: 真实分割掩码（0-1二值矩阵）
        return_diagonal: 是否在空掩码时返回图像对角线长度

    返回:
        hd95, asd 或 (nan, nan)/(diagonal, diagonal)
    """
    # 验证输入
    pred_mask = pred_mask.astype(np.uint8) if pred_mask.dtype != np.uint8 else pred_mask
    gt_mask = gt_mask.astype(np.uint8) if gt_mask.dtype != np.uint8 else gt_mask

    # 提取轮廓点
    pred_contour = np.argwhere(binary_erosion(pred_mask) ^ pred_mask)
    gt_contour = np.argwhere(binary_erosion(gt_mask) ^ gt_mask)

    # 处理空掩码情况
    if len(pred_contour) == 0 or len(gt_contour) == 0:
        if return_diagonal:
            diagonal = np.sqrt(pred_mask.shape[0] ** 2 + pred_mask.shape[1] ** 2)
            return diagonal, diagonal
        return np.nan, np.nan

    # 构建KDTree
    gt_tree = KDTree(gt_contour)
    pred_tree = KDTree(pred_contour)

    # 计算双向距离
    d_pred_to_gt, _ = gt_tree.query(pred_contour)
    d_gt_to_pred, _ = pred_tree.query(gt_contour)

    # 计算HD95
    hd95 = max(np.percentile(d_pred_to_gt, 95), np.percentile(d_gt_to_pred, 95))

    # 计算ASD
    asd = (np.mean(d_pred_to_gt) + np.mean(d_gt_to_pred)) / 2

    return hd95, asd

from scipy.stats import t

def compute_confidence_interval(data, confidence=0.95):
    """
    计算给定数据的均值、标准差、95%置信区间
    输入:
        data: 包含数值的列表或 NumPy 数组（允许 NaN 值）
        confidence: 置信度（默认 0.95）
    输出:
        mean_val: 均值
        std_val: 标准差
        ci_lower: 置信区间下限
        ci_upper: 置信区间上限
    """
    # 转换为 NumPy 数组，并过滤 NaN
    data = np.array(data)
    data = data[~np.isnan(data)]  # 移除 NaN

    if len(data) == 0:
        return np.nan, np.nan, np.nan, np.nan  # 数据为空时返回 NaN

    # 计算均值和标准差（样本标准差，ddof=1）
    mean_val = np.nanmean(data)
    std_val = np.nanstd(data, ddof=1)  # ddof=1 表示样本标准差（无偏估计）

    # 计算置信区间（使用 t 分布，适用于小样本）
    n = len(data)
    if n > 1:
        # 使用 t 分布（更准确）
        t_critical = t.ppf((1 + confidence) / 2, df=n - 1)
        margin_of_error = t_critical * (std_val / np.sqrt(n))
    else:
        # 单个样本无法计算置信区间
        margin_of_error = 0.0

    ci_lower = mean_val - margin_of_error
    ci_upper = mean_val + margin_of_error

    return mean_val, std_val, ci_lower, ci_upper



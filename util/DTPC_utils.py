import numpy as np
import torch
from torch.nn import functional as F
import torch.nn as nn
from timm.models.layers import trunc_normal_
from einops import rearrange, repeat
import torch.distributed as dist
from copy import deepcopy

#-------------------Confidence-Adaptive_Cutmix module-----------------------
def Confidence_Adaptive_Cutmix(
        x_u_s, pred_u_w,
        prob=0.5, min_ratio=0.4, max_ratio=0.7,
        ratio_1=0.3, ratio_2=1/0.3,
        softmax=False
):
    b, c, h, w = pred_u_w.shape

    if softmax:
        pred_u_w = F.softmax(pred_u_w, dim=1)

    logits_u, label_u = torch.max(pred_u_w, dim=1)

    batch_conf_u = cal_conf(pred_u_w)

    best_index = torch.argmax(batch_conf_u, dim=0)

    x_bg, label_bg, logits_bg = x_u_s, label_u, logits_u
    x_fore, label_fore, logits_fore = x_u_s, label_u, logits_u

    # 1. 取出表现最好的样本对应的 seg_l, x_l, gt, 将最好的样本复制 b 次
    best_x_bg = x_bg[best_index].unsqueeze(0).repeat(b, 1, 1, 1)  # 取出后增加一个维度，变成 1 x 3 x h x w
    best_label_bg = label_bg[best_index].repeat(b, 1, 1)  # 取出后增加一个维度，变成 1 x h x w
    best_logits_bg = logits_bg[best_index].repeat(b, 1, 1)  # 取出后增加一个维度，变成 1 x h x w

    # 2. 初始化 x_mix, label_mix 和 logits_mix
    x_mix = torch.clone(x_fore)
    label_mix = torch.clone(label_fore)
    logits_mix = torch.clone(logits_fore)

    # 3. x_u作为前景，x_l作为背景，按照 前景:背景=0.7 进行cutmix

    # 随机触发 CutMix
    if np.random.rand() < prob:
        # 生成前景和背景的掩码
        mask = torch.zeros((b, h, w), device=x_u_s.device)
        for i in range(b):
            # 随机生成宽高比
            aspect_ratio = np.random.uniform(ratio_1, ratio_2)

            # 随机生成剪裁区域的宽度和高度
            cut_width = np.random.uniform(min_ratio * w, max_ratio * w)
            cut_height = cut_width / aspect_ratio  # 根据宽高比调整高度

            # 确保剪裁区域不会超出图像边界
            cut_width = int(min(cut_width, w))
            cut_height = int(min(cut_height, h))

            # 随机生成剪裁区域的起始点
            x_start = np.random.randint(0, w - cut_width + 1)
            y_start = np.random.randint(0, h - cut_height + 1)

            # 设置剪裁区域的掩码
            mask[i, y_start:y_start + cut_height, x_start:x_start + cut_width] = 1.0  # 前景区域

        # 混合图像
        x_mix = mask.unsqueeze(1) * x_fore + (1 - mask.unsqueeze(1)) * best_x_bg

        # 混合标签矩阵
        label_mix = mask * label_fore + (1 - mask) * best_label_bg

        # 混合置信度矩阵
        logits_mix = mask * logits_fore + (1 - mask) * best_logits_bg

    return x_mix, label_mix, logits_mix

def cal_conf(pred, softmax=False):
    if softmax:
        pred = F.softmax(pred, dim=1)
    logits, label = torch.max(pred, dim=1)
    entropy = -torch.sum(pred * torch.log(pred + 1e-10), dim=1) # [b, h, w]
    entropy = entropy * logits * (label != 0).float()
    num_classes = pred.shape[1]
    entropy /= (np.log(num_classes) + 1e-10)
    confidence = 1.0 - entropy
    confidence = confidence.mean(dim=[1, 2])  # list[batchsize]
    return confidence


#---------------Dynamic Adaptive Threshold---------------------
class DAT:
    def __init__(self, num_classes, momentum=0.99, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes
        self.m = momentum
        self.prob_conf = torch.ones((self.num_classes)) / self.num_classes
        self.time_p = self.prob_conf.mean()
        self.prob_max = torch.ones((self.num_classes)) / self.num_classes
        self.prob_std = torch.ones((self.num_classes))

    @torch.no_grad()
    def update(self, max_probs, max_idx, eps=1e-10):

        # 计算每个类别的置信度均值、方差
        conf_max_list = []
        conf_mean_list = []
        conf_std_list = []
        probs_mean = torch.mean(max_probs)
        for cls in range(self.num_classes):
            cls_map = (max_idx == cls)
            pred_conf_cls_all = max_probs[cls_map]
            if pred_conf_cls_all.numel() != 0:
                cls_mean_conf = torch.mean(pred_conf_cls_all)
                cls_max_conf = pred_conf_cls_all.max()
                if pred_conf_cls_all.numel() > 1:
                    cls_std_conf = torch.var(pred_conf_cls_all, unbiased=True)
                else:
                    cls_std_conf = torch.var(pred_conf_cls_all, unbiased=False)
            else:
                cls_mean_conf = self.prob_conf[cls]
                cls_max_conf = self.prob_max[cls]
                cls_std_conf = self.prob_std[cls]
            conf_mean_list.append(cls_mean_conf)
            conf_max_list.append(cls_max_conf)
            conf_std_list.append(cls_std_conf)

        conf_mean = torch.tensor(conf_mean_list).to(max_probs.device)
        conf_max = torch.tensor(conf_max_list).to(max_probs.device)
        conf_std = torch.tensor(conf_std_list).to(max_probs.device)

        self.prob_max = self.prob_max * self.m + (1 - self.m) * conf_max
        weight = self.prob_max / torch.max(self.prob_max)

        self.prob_conf = self.prob_conf * self.m + (1 - self.m) * conf_mean * weight

    @torch.no_grad()
    def masking(self, logits, label, iters):
        if not self.prob_conf.is_cuda:
            self.prob_conf = self.prob_conf.to(logits.device)
            self.prob_max = self.prob_max.to(logits.device)
            self.prob_std = self.prob_std.to(logits.device)
        if not self.time_p.is_cuda:
            self.time_p = self.time_p.to(logits.device)

        max_probs, max_idx = logits, label
        if iters == 0:
            for cls in range(self.num_classes):
                cls_map = (max_idx == cls)
                pred_conf_cls_all = max_probs[cls_map]
                if pred_conf_cls_all.numel() != 0:
                    # top-25%
                    self.prob_conf[cls] = torch.quantile(pred_conf_cls_all, 0.75)
        else:
            self.update(max_probs, max_idx)
        cls_thresholds = self.prob_conf[max_idx]
        mask = max_probs.ge(cls_thresholds)
        valid_pix_num = mask.sum().item()
        mask = mask.to(max_probs.dtype)

        mask_ratio = torch.tensor(valid_pix_num / max_probs.numel())
        tau = self.prob_conf.mean()

        return mask, mask_ratio, tau

#----------Prototype Contrastive Learning module---------------

def proto_contrastive_learning(rep, pred, valid_pixel, temp=0.07,
                         num_queries=256, num_negatives=512, sample_times=10):

    device = pred.device
    num_segments = pred.shape[1]

    # permute representation for indexing: batch x im_h x im_w x feature_channel
    pred_soft = torch.softmax(pred, dim=1)
    pred = pred.permute(0, 2, 3, 1)  # [2B, H, W, C]
    feat = rep.permute(0, 2, 3, 1)  # [2B, H, W, D]

    # compute prototype (class mean representation) for each class across all valid pixels
    seg_all_list = []
    seg_hard_list = []
    seg_num_list = []
    seg_proto_list = []
    for i in range(num_segments):
        valid_pixel_seg = valid_pixel[:, i]  # select binary mask for i-th class  # [2B,H,W]
        if valid_pixel_seg.sum() == 0:  # not all classes would be available in a mini-batch
            continue

        prob_seg = pred_soft[:, i, :, :]  # [2B,H,W], 类别置信度矩阵
        # 以上90%分位数作为阈值，大于则为proto，小于则为hard_pix
        strong_threshold = torch.quantile(prob_seg, 0.90)
        rep_proto_mask = (prob_seg > strong_threshold) * valid_pixel_seg.bool()  # [2B,H,W], select proto queries
        rep_mask_hard = (prob_seg < strong_threshold) * valid_pixel_seg.bool()  # [2B,H,W], select hard queries

        proto = feat[rep_proto_mask.bool()] # [N1, D]
        seg_proto_list.append(proto)

        feat_hard = feat[rep_mask_hard.bool()] # [N2, D]
        seg_hard_list.append(feat_hard)

        feat_all = feat[valid_pixel_seg.bool()] # [N3, D]
        seg_all_list.append(feat_all)

        seg_num_list.append(int(valid_pixel_seg.sum().item()))  # valid_pixel_seg.sum()：当前类别下像素之和 [N, ]

    # 不能过滤空的列表，只能维持现状。
    feat_proto_list = process_proto_list(seg_proto_list, num_queries, need_global=True) # [C, sample_times+1, D]
    query_list = process_proto_list(seg_hard_list, num_queries, need_global=True) # [C, sample_times, D]

    # intra loss
    # plan 1: 继续使用 regularization_loss
    # inner = [tensor.mean(dim=0, keepdim=True) for tensor in feat_proto_list]
    # inner = [tensor for tensor in inner if tensor.numel() > 0]  # 仅保留非空元素
    # if len(inner) <= 1:
    #     loss_inner = torch.tensor(0.0).to(device)
    # else:
    #     inner = torch.stack(inner, dim=0).squeeze(1)  # [C, d]
    #     loss_inner = regularization_loss(inner)

    # plan 2: 使用 supcon
    loss_inner = supcon_contrastive_loss(feat_proto_list)

    # compute regional contrastive loss
    if len(seg_num_list) <= 1:
        # in some rare cases, a small mini-batch might only contain 1 or no semantic class
        return torch.tensor(0.0).to(device), torch.tensor(0.0).to(device)
    else:
        reco_loss = torch.tensor(0.0).to(device)
        valid_seg = len(seg_num_list)  # N

        for i in range(valid_seg):
            # 跳过空的查询或键
            if i >= len(query_list) or i >= len(feat_proto_list) or query_list[i].numel() == 0 or feat_proto_list[i].numel() == 0:
                continue

            query_cls = query_list[i]  # [sample_times, d]
            key_cls = feat_proto_list[i]  # [sample_times+1, d]

            # apply negative key sampling (with no gradients)
            with torch.no_grad():
                query_cls = nn.functional.normalize(query_cls, dim=1)  # [sample_times, d]
                key_cls = nn.functional.normalize(key_cls, dim=1)  # [sample_times+1, d]


                cur_cls = query_cls
                other_cls = [] # 排除query_list中的当前i类
                for j in range(valid_seg):
                    if j != i and j < len(query_list) and query_list[j].numel() > 0:
                        other_cls.append(query_list[j])

                if not other_cls:  # 如果没有其他有效类别
                    continue

                similarities = []
                for proto in other_cls:
                    if proto.size(1) == cur_cls.size(1):  # 确保维度匹配
                        sim = F.cosine_similarity(cur_cls, proto, dim=1).mean()  # 计算均值相似度
                        similarities.append(sim)

                if not similarities:  # 如果没有有效的相似度计算
                    continue

                similarities = torch.tensor(similarities).to(device)

                # 对余弦相似度进行 Softmax 归一化，得到概率分布
                probs = F.softmax(similarities, dim=0)

                # 计算从每个类别中抽取的向量数量
                sample_counts = (probs * num_negatives).round().int()

                # 确保总数等于 num_negatives
                while sample_counts.sum() != num_negatives:
                    diff = num_negatives - sample_counts.sum()
                    idx = torch.argmax(probs)  # 在概率最大的类别上调整
                    sample_counts[idx] += diff

                # 从其他类别中抽取向量
                other_cls_queue = [] # seg_all_list中排除当前类
                for j in range(valid_seg):
                    if j != i and j < len(seg_all_list) and seg_all_list[j].numel() > 0:
                        other_cls_queue.append(seg_all_list[j])

                negative_vectors = []
                for k, count in enumerate(sample_counts):
                    if count > 0 and k < len(other_cls_queue):
                        proto = other_cls_queue[k]
                        available = proto.size(0)
                        if available == 0:
                            continue
                        count = min(count, available)  # 确保不超过可用数量
                        indices = torch.randperm(available)[:count]
                        selected_vectors = proto[indices]
                        negative_vectors.append(selected_vectors)

                if not negative_vectors:  # 如果没有抽到负样本
                    continue

                queue_neg = torch.cat(negative_vectors, dim=0)  # [num_negatives, D]
                if queue_neg.numel() == 0:  # 如果最终队列为空
                    continue

                queue_neg = nn.functional.normalize(queue_neg, dim=1)  # [num_negatives, D], 负类向量列表
                queue_neg = queue_neg.permute(1, 0) # [D, num_negatives]

                if query_cls.size(0) != key_cls.size(0):
                    N = query_cls.size(0)
                    M = key_cls.size(0) - N
                    # 随机采样 M 行（可重复采样）
                    indices = torch.randint(0, N, (M,))
                    sampled_rows = query_cls[indices, :]  # 形状 [M, D]
                    query_cls = torch.cat([query_cls, sampled_rows], dim=0)  # 形状 [N+M, D]

                # positive logits: Nx1
                l_pos = torch.einsum("nd,nd->n", [query_cls, key_cls]).unsqueeze(-1)  # [n, 1]
                # negative logits: NxK
                l_neg = torch.einsum("nd,dk->nk", [query_cls, queue_neg.clone().detach()])
                # logits: Nx(1+K)
                logits = torch.cat([l_pos, l_neg], dim=1)
                # apply temperature
                logits /= temp
                # labels: positive key indicators
                labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

                reco_loss = reco_loss + F.cross_entropy(logits, labels)

        if valid_seg > 0:
            return reco_loss / valid_seg, loss_inner / valid_seg
        else:
            return torch.tensor(0.0).to(device), torch.tensor(0.0).to(device)

def label_one_hot_encoder(input_tensor, n_classes):
    '''
    当前类别预测正确时，像素编码为1
    '''
    tensor_list = []
    for i in range(n_classes):
        temp_prob = input_tensor == i * torch.ones_like(input_tensor)
        tensor_list.append(temp_prob)
    output_tensor = torch.cat(tensor_list, dim=1)
    return output_tensor.float()

def regularization_loss(bases):
    k, c = bases.size()
    if k==0:
        print(k)
    loss_all = 0
    num = 0
    for i in range(k - 1):
        for j in range(i + 1, k):
            num += 1
            simi = F.cosine_similarity(bases[i].unsqueeze(0), bases[j].unsqueeze(0).detach(), dim=1)
            simi = F.relu(simi)
            loss_all += simi ** 2
    loss_all = loss_all / num

    return loss_all

def supcon_contrastive_loss(out_list, temperature=0.07):
    """
    基于SupCon的对比损失计算，拉近同类特征，拉开不同类特征

    参数:
        out_list: List[Tensor], 每个元素为[sample_times+1, d]的特征矩阵
        temperature: float, 温度系数

    返回:
        loss: Tensor, 对比损失值
    """
    # 过滤掉空的特征矩阵
    device = out_list[0].device

    valid_features = [feat for feat in out_list if feat.numel() > 0]
    if len(valid_features) < 2:  # 至少需要两个类别才能计算对比损失
        return torch.tensor(0.0, device=out_list[0].device if len(out_list) > 0 else 'cpu')

    # 准备对比学习所需数据
    features = torch.cat(valid_features, dim=0)  # [total_samples, d]
    num_classes = len(valid_features)
    samples_per_class = [feat.shape[0] for feat in valid_features]

    # 创建标签 (每个样本属于哪个类别)
    labels = torch.cat([
        torch.full((samples,), class_id, dtype=torch.long, device=features.device)
        for class_id, samples in enumerate(samples_per_class)
    ])  # [total_samples]

    # 归一化特征
    features = F.normalize(features, p=2, dim=-1)

    # 计算相似度矩阵
    # similarity_matrix = torch.matmul(features, features.T) / temperature  # [total_samples, total_samples]
    # compute logits
    anchor_dot_contrast = torch.div(
        torch.matmul(features, features.T),
        temperature)  # 公式中的分子和分母部分。
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()  # 减去最大值以确保数值稳定性。


    # 创建掩码矩阵 (标识哪些样本属于同一类)
    mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()  # [total_samples, total_samples]

    # 排除自身对比
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(mask.size(0)).view(-1, 1).to(device),
        0
    )  # 生成一个掩码矩阵，用于排除自对比的情况。对角线元素为0
    mask = mask * logits_mask  # 更新掩码，将自对比掩蔽掉。对角线元素为0。

    # 计算对比损失
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True))

    # 只计算正样本对的平均对数概率
    mean_log_prob_pos = (mask * log_prob).sum(dim=1) / mask.sum(dim=1)

    # 最终损失
    loss = -mean_log_prob_pos.mean()

    return loss

def process_proto_list(seg_proto_list, sample_times, need_global=False):
    """
    处理原型列表，对每个类别的原型特征进行采样和全局特征拼接

    参数:
        seg_proto_list: List[Tensor], 长度为C，每个元素为[N_i, d]的特征矩阵
        sample_times: int, 需要采样的局部特征数量

    返回:
        out_list: List[Tensor], 长度为C，每个元素为[sample_times+1, d]的特征矩阵
    """
    out_list = []
    d = seg_proto_list[0].shape[1] if len(seg_proto_list) > 0 else 0  # 特征维度

    for proto in seg_proto_list:
        # 情况1: 当N=0时，跳过该元素
        if proto.shape[0] == 0:
            out_list.append(proto)  # 保持原样
            continue

        N = proto.shape[0]

        # 情况2: 当N < sample_times时，复制扩充
        if N < sample_times:
            # 先采样所有可用样本
            indices = torch.arange(N)
            # 然后随机选择补充样本
            extra_indices = torch.randint(0, N, (sample_times - N,))
            indices = torch.cat([indices, extra_indices])
        else:
            # 情况3: 正常随机采样
            indices = torch.randperm(N)[:sample_times]

        # 采样局部特征
        local_feats = proto[indices]  # [sample_times, d]

        if need_global:
            # 计算全局特征 (均值池化)
            global_feat = proto.mean(dim=0, keepdim=True)  # [1, d]
            # 拼接局部和全局特征
            combined = torch.cat([local_feats, global_feat], dim=0)  # [sample_times+1, d]
        else:
            combined = local_feats

        out_list.append(combined)

    return out_list


class ProjectionHead(nn.Module):
    """PyTorch version of projection_head"""

    def __init__(self,
                 num_input_channels,
                 num_projection_channels=256,
                 num_projection_layers=3,
                 ):
        super().__init__()
        self.layers = nn.ModuleList()

        # Intermediate layers
        for _ in range(num_projection_layers - 1):
            self.layers.append(nn.Conv2d(num_input_channels, num_projection_channels, kernel_size=1))
            self.layers.append(nn.BatchNorm2d(num_projection_channels))
            self.layers.append(nn.ReLU(inplace=True))
            num_input_channels = num_projection_channels

        # Final layer
        self.final_conv = nn.Conv2d(num_input_channels, num_projection_channels, kernel_size=1)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.final_conv(x)
        return F.normalize(x, p=2, dim=-1)

if __name__ == '__main__':
    # proto_test = torch.rand(12, 256)
    # loss = regularization_loss(proto_test)
    # seg = torch.rand(2, 2, 64, 64)
    # feat = torch.rand(2, 256, 64, 64)
    # model = proto_SA()
    # out = model(feat, seg)

    # 模拟数据
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    C, d = 5, 256
    seg_proto_list = [
        torch.randn(10, d, device=device),  # 类别0
        torch.randn(3, d, device=device),  # 类别1
        torch.randn(0, d, device=device),  # 类别2 (空)
        torch.randn(7, d, device=device),  # 类别3
        torch.randn(4, d, device=device)  # 类别4
    ]
    sample_times = 5

    # 测试原型处理
    out_list = process_proto_list(seg_proto_list, sample_times)
    for i, out in enumerate(out_list):
        print(f"类别 {i}: 输入形状 {seg_proto_list[i].shape} -> 输出形状 {out.shape}")

    # 测试对比损失
    loss = supcon_contrastive_loss(out_list)
    print(f"\n对比损失值: {loss.item():.4f}")



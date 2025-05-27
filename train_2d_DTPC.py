import argparse
import logging
import os
import shutil
import sys
from copy import deepcopy

import numpy as np
import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

from dataset.mydataset import (MyDataset, WeakStrongAugment_mocoV2)
from torchvision import transforms
import random
import torch.nn.functional as F
from util import DTPC_utils, losses

from networks.net_factory import net_factory
from util.utils import (count_params, cal_metric_pixel_2D, compute_confidence_interval)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()

parser.add_argument("--config", type=str,
                    default="configs/isic2018.yaml")
# model name and save path
parser.add_argument('--method', default="DTPC", type=str,
                    help='')
parser.add_argument('--model', type=str, default='unet_DTPC',
                    help='')
# label rat
parser.add_argument('--label-rat', default='1_4', type=str, 
                    help='')
# DTPC arams
parser.add_argument('--temp', type=float, default=0.07)
parser.add_argument('--num_queries', type=int, default=256)
parser.add_argument('--num_negatives', type=int, default=512)
parser.add_argument('--cl-size', type=int, default=64, help='cl_size')
# seed
parser.add_argument("--seed", type=int, default=1337, help="random seed")
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
args = parser.parse_args()

cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)


def getLog(args, snapshot_path):
    logging.basicConfig(filename=snapshot_path + "/logging.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))


def patients_to_slices(dataset, patiens_rat):
    ref_dict = None
    if "ISIC2018" in dataset:
        ref_dict = {"1_4": 518, "1_8": 259, "1_16": 129}
    elif "KvasirSEG" in dataset:
        ref_dict = {"1_4": 200, "1_8": 100, "1_16": 50}
    else:
        print("Error")
    return ref_dict[patiens_rat]


def get_current_consistency_weight(epoch):
    consistency = cfg['semi']['consistency']
    consistency_rampup = cfg['semi']['consistency_rampup']
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return consistency * sigmoid_rampup(epoch, consistency_rampup)


def update_ema_variables(model, model_teacher, ema_decay, iters):
    ema_decay = min(1 - 1 / (iters + 1), ema_decay, )
    # update weight
    for param_train, param_eval in zip(model.parameters(), model_teacher.parameters()):
        param_eval.data = param_eval.data * ema_decay + param_train.data * (1 - ema_decay)
    # update bn
    for buffer_train, buffer_eval in zip(model.buffers(), model_teacher.buffers()):
        buffer_eval.data = buffer_eval.data * ema_decay + buffer_train.data * (1 - ema_decay)


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def main(args):
    # save model and logging
    exp = "exp/{}/{}".format(
        cfg['data']['dataset'], args.method)

    snapshot_path = "{}/{}/{}_labeled".format(
        exp, args.model, args.label_rat)

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    current_file = __file__
    shutil.copyfile(current_file, os.path.join(snapshot_path, os.path.basename(current_file)))

    getLog(args, snapshot_path)
    logging.info("Configuration settings:\n%s", yaml.dump(cfg))

    evens_path = snapshot_path + '/log'
    if not os.path.exists(evens_path):
        os.makedirs(evens_path)
    writer = SummaryWriter(evens_path)

    dataset = cfg['data']['dataset']
    root_path = cfg['data']['root_path']
    crop_size = cfg['data']['crop_size']
    in_ch = cfg['data']['in_chns']

    # train params
    base_lr = cfg['train']['base_lr']
    num_classes = cfg['train']['num_classes']
    batch_size = cfg['train']['batch_size']
    epochs = cfg['train']['epochs']

    # semi params
    ema_decay = cfg['semi']['ema_decay']

    labeled_id_path = "splits/{}/{}/labeled.txt".format(
        dataset, args.label_rat)
    unlabeled_id_path = "splits/{}/{}/unlabeled.txt".format(
        dataset, args.label_rat)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)


    model = net_factory(net_type=args.model, in_chns=in_ch,
                        class_num=num_classes)

    optimizer = SGD(model.parameters(), lr=base_lr,
                    momentum=0.9, weight_decay=0.0001)

    logging.info('Total params: {:.1f}M\n'.format(count_params(model)))

    model.cuda()

    # teacher model initial
    model_teacher = deepcopy(model)
    model_teacher.cuda()
    for p in model_teacher.parameters():
        p.requires_grad = False

    with torch.no_grad():
        for t_params, s_params in zip(model_teacher.parameters(), model.parameters()):
            t_params.data = s_params.data

    trainset_u = MyDataset(dataset, root_path, 'train_u',
                           crop_size, unlabeled_id_path,
                           transform=transforms.Compose([WeakStrongAugment_mocoV2(crop_size)]),
                           )
    trainset_l = MyDataset(dataset, root_path, 'train_l',
                           crop_size, labeled_id_path, nsample=len(trainset_u.ids),
                           transform=transforms.Compose([WeakStrongAugment_mocoV2(crop_size)]),
                           )
    valset = MyDataset(dataset, root_path, 'val')

    labeled_slice = patients_to_slices(dataset, args.label_rat)
    slices = len(trainset_u) + labeled_slice

    logging.info('Total silices is: {}, labeled slices is: {}\n'.format(
        slices, labeled_slice))

    trainloader_l = DataLoader(trainset_l, batch_size=batch_size,
                               pin_memory=True, num_workers=4, drop_last=True,
                               worker_init_fn=worker_init_fn,
                               )
    trainloader_u = DataLoader(trainset_u, batch_size=batch_size,
                                pin_memory=True, num_workers=4, drop_last=True,
                                worker_init_fn=worker_init_fn,
                                )
    valloader = DataLoader(valset, batch_size=1, num_workers=1,
                           shuffle=False,)

    total_iters = len(trainloader_u) * epochs
    logging.info('Total iters is: {}\n'.format(total_iters))
    previous_best_dice, previous_best_acc = 0.0, 0.0
    epoch = -1
    iters = 0

    if os.path.exists(os.path.join(snapshot_path, 'latest.pth')):
        checkpoint = torch.load(os.path.join(snapshot_path, 'latest.pth'))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        previous_best_dice = checkpoint['previous_best_dice']
        previous_best_acc = checkpoint['previous_best_acc']

        logging.info('************ Load from checkpoint at epoch %i\n' % epoch)

    celoss_l = nn.CrossEntropyLoss(ignore_index=255, reduction='none').cuda()
    dice_loss = losses.DiceLoss(n_classes=num_classes).cuda()

    # DAT
    get_mask = DTPC_utils.DAT(num_classes=num_classes, momentum=0.99)

    proj_head = DTPC_utils.ProjectionHead(
        num_input_channels=16,
        num_projection_channels=256,
    ).cuda()

    for epoch in range(epoch + 1, epochs):
        logging.info(
            '===========> Epoch: {:}, LR: {:.5f}, \033[31m Previous best {} dice: {:.2f}, Overall Accuracy: {:.2f}\033[0m'.format(
                epoch, optimizer.param_groups[0]['lr'], dataset, previous_best_dice, previous_best_acc))

        loader = zip(trainloader_l, trainloader_u)

        for i, ((sample_l), (sample_u)) in enumerate(loader):

            img_x, mask_x = sample_l["image"], sample_l["label"]
            img_x, mask_x = img_x.cuda(), mask_x.cuda()

            img_u_w, img_u_s = sample_u["image_weak"], sample_u["image_strong"]
            img_u_w, img_u_s = img_u_w.cuda(), img_u_s.cuda()

            num_lb, num_ulb = img_x.shape[0], img_u_w.shape[0]

            with torch.no_grad():
                model_teacher.eval()
                pred_u_w, feat_u_w = model_teacher(img_u_w.detach())
                pred_u_w_soft = F.softmax(pred_u_w, dim=1)

            model.train()
            
            #----------CAC------------
            x_mix, label_mix, logits_mix = (DTPC_utils.
            Confidence_Adaptive_Cutmix(
                img_u_s, pred_u_w_soft
            ))
            label_mix = label_mix.long()

            preds, feats = model(torch.cat((img_x, x_mix)))
            preds_x = preds[:num_lb]
            preds_x_soft = torch.softmax(preds_x, dim=1)

            pre_mix = preds[num_lb:]

            feats = F.interpolate(feats, size=(args.cl_size, args.cl_size), mode='nearest')

            # supevised loss
            entry = celoss_l(preds_x, mask_x.long())
            loss_ce = entry.mean()
            loss_dice = dice_loss(preds_x_soft, mask_x.long().unsqueeze(1).float(), ignore=255)
            loss_x = 0.5 * (loss_ce + loss_dice)

            #----------DAT------------
            mask_mix, mask_ratio_mix, tau_mix = get_mask.masking(logits_mix, label_mix, iters)
            loss_dat = F.cross_entropy(pre_mix, label_mix, ignore_index=255, reduction='none') * mask_mix
            loss_dat = loss_dat.mean()

            #----------PCL------------
            cls_thresholds = get_mask.prob_conf[label_mix]
            mask_u_weak = logits_mix.ge(cls_thresholds).to(logits_mix.dtype)
            mask_u_weak = F.interpolate(mask_u_weak.unsqueeze(dim=1),
                                        size=feats.shape[2:], mode='nearest')
            valid_pix_u = F.interpolate(DTPC_utils.label_one_hot_encoder(label_mix.unsqueeze(1), num_classes),
                                        size=feats.shape[2:], mode='nearest')

            valid_pix_u = valid_pix_u * mask_u_weak
            valid_pix_l = F.interpolate(DTPC_utils.label_one_hot_encoder(mask_x.unsqueeze(1), num_classes),
                                        size=feats.shape[2:], mode='nearest')
            valid_pix_all = torch.cat((valid_pix_l, valid_pix_u))
            # 2、get preds = [2B, C, H, W]
            preds_s = torch.cat((preds_x, pre_mix))
            preds_s = F.interpolate(preds_s, size=feats.shape[2:], mode='nearest')
            # 3、get feats SA = [2B, D, H, W]
            reps_s = proj_head(feats)

            loss_intra, loss_inter = DTPC_utils.proto_contrastive_learning(
                reps_s, preds_s, valid_pix_all,
                temp=args.temp,
                num_queries=args.num_queries, num_negatives=args.num_negatives
            )

            consistency_weight = get_current_consistency_weight(iters // 150)

            loss_consis = loss_dat
            loss_pcl = (loss_intra + loss_inter) / 2

            loss = loss_x + loss_consis * consistency_weight + loss_pcl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr = base_lr * (1 - iters / total_iters) ** 0.9

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # update teacher model with EMA
            with torch.no_grad():
                update_ema_variables(model, model_teacher, ema_decay, iters)

            iters = epoch * len(trainloader_u) + i

            writer.add_scalar('train/loss_all', loss.item(), iters)
            writer.add_scalar('train/loss_x', loss_x.item(), iters)
            writer.add_scalar('train/loss_ce', loss_ce.item(), iters)
            writer.add_scalar('train/loss_dice', loss_dice.item(), iters)
            writer.add_scalar('train/loss_consis', loss_consis.item(), iters)
            writer.add_scalar('train/loss_dat', loss_dat.item(), iters)
            writer.add_scalar('train/tau_mix', tau_mix.item(), iters)
            writer.add_scalar('train/mask_ratio_mix', mask_ratio_mix.item(), iters)
            writer.add_scalar('train/loss_pcl', loss_pcl.item(), iters)
            writer.add_scalar('train/loss_intra', loss_intra.item(), iters)
            writer.add_scalar('train/loss_inter', loss_inter.item(), iters)

            if (i % (len(trainloader_u) // 8) == 0):
                logging.info(
                    'Iters: %d, Total loss: %f, Loss x: %f, Loss x ce: %f, Loss x dice: %f, '
                    'Loss consis: %f, Loss mix: %f, tau_mix: %f, mask ratio mix: %f, '
                    'Loss PCL: %f, Loss intra: %f, Loss inter: %f ' % (
                        iters, loss.item(), loss_x.item(), loss_ce.item(), loss_dice.item(),
                        loss_consis.item(), loss_dat.item(), tau_mix.item(), mask_ratio_mix.item(),
                        loss_pcl.item(), loss_intra.item(), loss_inter.item()
                    ))

        model.eval()
        metric_list = 0.0
        m_dice_list = []
        m_jaccard_list = []
        pa_list = []
        m_hd95_list = []
        m_asd_list = []
        for i, (image, target) in enumerate(valloader):
            metric_i, PA = cal_metric_pixel_2D(
                image, target, model,
                classes=cfg['train']['num_classes'],
                patch_size=cfg['data']['crop_size'])

            dice = [row[0] for row in metric_i]
            jaccard = [row[1] for row in metric_i]
            hd95 = [row[2] for row in metric_i]
            asd = [row[3] for row in metric_i]

            metric_list += np.array(metric_i)
            m_dice_list.append(np.mean(dice))
            m_jaccard_list.append(np.mean(jaccard))
            m_hd95_list.append(np.mean(hd95))
            m_asd_list.append(np.mean(asd))
            pa_list.append(PA)

        metric_list = metric_list / len(valset)

        dice_list = [row[0] for row in metric_list]
        jaccard_list = [row[1] for row in metric_list]
        hd95_list = [row[2] for row in metric_list]
        asd_list = [row[3] for row in metric_list]

        m_dice = np.nanmean(dice_list)
        m_jaccard = np.nanmean(jaccard_list)
        m_hd95 = np.nanmean(hd95_list)
        m_asd = np.nanmean(asd_list)
        m_pa = np.nanmean(pa_list)

        for class_i in range(num_classes - 1):
            writer.add_scalar('info/val_{}_dice'.format(class_i + 1),
                              metric_list[class_i, 0], iters)
            writer.add_scalar('info/val_{}_jac'.format(class_i + 1),
                              metric_list[class_i, 1], iters)
            writer.add_scalar('info/val_{}_hd95'.format(class_i + 1),  # 新增
                              metric_list[class_i, 2], iters)
            writer.add_scalar('info/val_{}_asd'.format(class_i + 1),  # 新增
                              metric_list[class_i, 3], iters)

        is_best = m_dice * 100.0 > previous_best_dice
        previous_best_dice = max(m_dice * 100.0, previous_best_dice)

        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'previous_best_dice': previous_best_dice,
            'previous_best_acc': previous_best_acc * 100.0,
        }
        if is_best:
            logging.info('***** \033[31m best eval! \033[0m *****')

            previous_best_acc = m_pa * 100.0
            save_mode_path = os.path.join(snapshot_path,
                                          'epoch_{}_dice_{}_jac_{}_pa_{}.pth'.format(
                                              epoch, round(previous_best_dice, 2),
                                              round(m_jaccard * 100.0, 2), round(previous_best_acc, 2)))
            torch.save(checkpoint, save_mode_path)
            torch.save(checkpoint, os.path.join(snapshot_path, 'best.pth'))

        torch.save(checkpoint, os.path.join(snapshot_path, 'latest.pth'))

        logging.info('***** \033[33m eval! \033[0m *****')

        dsc, std_dsc, ci_lower_dsc, ci_upper_dsc = compute_confidence_interval(m_dice_list)  # 计算每张图像的mean_DSC标准差和95%CI
        jac, std_jac, ci_lower_jac, ci_upper_jac = compute_confidence_interval(m_jaccard_list)
        hd95, std_hd95, ci_lower_hd95, ci_upper_hd95 = compute_confidence_interval(m_hd95_list)
        asd, std_asd, ci_lower_asd, ci_upper_asd = compute_confidence_interval(m_asd_list)
        pa, std_pa, ci_lower_pa, ci_upper_pa = compute_confidence_interval(pa_list)

        logging.info(
            'iteration %d : DSC : %.2f Jac : %.2f HD95 : %.2f ASD : %.2f PA : %.2f' % (
                iters, dsc * 100.0, jac * 100.0, hd95, asd, pa * 100.0))

        logging.info(f"dsc: {dsc * 100.0:.2f}, dsc_std: ({std_dsc:.2f}), "
                     f"95% CI: ({ci_lower_dsc * 100.0:.2f}, {ci_upper_dsc * 100.0:.2f})")
        logging.info(f"jac: {jac * 100.0:.2f}, jac_std: ({std_jac:.2f}), "
                     f"95% CI: ({ci_lower_jac * 100.0:.2f}, {ci_upper_jac * 100.0:.2f})")
        logging.info(f"hd95: {hd95:.2f}, hd95_std: ({std_hd95:.2f}), "
                     f"95% CI: ({ci_lower_hd95:.2f}, {ci_upper_hd95:.2f})")
        logging.info(f"asd: {asd:.2f}, asd_std: ({std_asd:.2f}), "
                     f"95% CI: ({ci_lower_asd:.2f}, {ci_upper_asd:.2f})")
        logging.info(f"pa: {pa * 100.0:.2f}, PA_std: ({std_pa:.2f}), "
                     f"95% CI: ({ci_lower_pa * 100.0:.2f}, {ci_upper_pa * 100.0:.2f})")


if __name__ == '__main__':

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    main(args)

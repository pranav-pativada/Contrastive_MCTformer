import math
import sys
from typing import Iterable
from loss import *
from PIL import Image

import torch
import torch.nn.functional as F
import utils
import code

from sklearn.metrics import average_precision_score
import numpy as np
import cv2
import os
from pathlib import Path

def train_one_epoch(args, model: torch.nn.Module, data_loader: Iterable,
                    optimizer: torch.optim.Optimizer, device: torch.device,
                    epoch: int, loss_scaler, max_norm: float = 0,
                    set_training_mode=True):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    criterion = [SimMaxLoss(metric='cos', alpha=0.25).cuda(), SimMinLoss(metric='cos').cuda(),
                 SimMaxLoss(metric='cos', alpha=0.25).cuda()]

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        patch_outputs = None
        with torch.cuda.amp.autocast():
            output = model(samples)
            cls_logits, cams, patch_features = output[0:3]
            if len(output) == 5:
                patch_outputs = output[4]
            if len(cams.shape) > 3:
                cams = cams.reshape(cams.shape[0], cams.shape[1], cams.shape[2] * cams.shape[-1])
            
            cams = torch.sigmoid(cams)
            multi_label_soft_margin_loss = F.multilabel_soft_margin_loss(cls_logits, targets)
            metric_logger.update(cls_loss=multi_label_soft_margin_loss.item())
            
            if args.clr_loss:
                b, c, n = cams.shape
                dim = patch_features.shape[-1] # b, n, d
                code.interact(local=dict(globals(), **locals()))
                
                
                clr_loss = 0


                # 
                fg_feats = (torch.matmul(cams, patch_features) / (n)).reshape(b, dim * c)
                bg_feats = (torch.matmul(1-cams, patch_features) / (n)).reshape(b, dim * c)
                clr_loss = criterion[0](fg_feats) + criterion[1](fg_feats, bg_feats) + criterion[2](bg_feats)
                metric_logger.update(clr_loss=clr_loss.item())
                loss = (args.cls_weight * multi_label_soft_margin_loss) + (args.clr_weight * clr_loss)
            else:
                loss = multi_label_soft_margin_loss
            
            if  patch_outputs is not None:
                ploss = F.multilabel_soft_margin_loss(patch_outputs, targets)
                metric_logger.update(pat_loss=ploss.item())
                loss = loss + ploss

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(args, data_loader, model, device):
    cls_criterion = torch.nn.MultiLabelSoftMarginLoss()
    clr_criterion = [SimMaxLoss(metric='cos', alpha=0.25).cuda(), SimMinLoss(metric='cos').cuda(), SimMaxLoss(metric='cos', alpha=0.25).cuda()]
    mAP = []
    mIOU = None
    start_iter, end_iter = 1, 60

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    gt_path = 'voc12/data/SegmentationClass/'
    # switch to evaluation mode
    model.eval()

    for images, target, names in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        
        if args.mIOU_metric:
            gts = [Image.open(os.path.join(gt_path, name + '.png')) for name in names]
        
        batch_size = images.shape[0]
        
        with torch.cuda.amp.autocast():
            output, predictions, patch_features = model(images)[0:3]
            cls_loss = cls_criterion(output, target)
            predictions = torch.sigmoid(predictions)
            
            if args.clr_loss:
                b, c, n = predictions.shape
                dim = patch_features.shape[-1]
                    
                fg_feats = (torch.matmul(predictions, patch_features) / (n)).reshape(b, dim * c)
                bg_feats = (torch.matmul(1-predictions, patch_features) / (n)).reshape(b, dim * c)
                clr_loss = clr_criterion[0](fg_feats) + clr_criterion[1](fg_feats, bg_feats) + clr_criterion[2](bg_feats)
                loss = (args.cls_weight * cls_loss) + (args.clr_weight * clr_loss)
            else:
                loss = cls_loss
            
            output = torch.sigmoid(output)
            
            mAP_list = compute_mAP(target, output)
            mAP = mAP + mAP_list 
            metric_logger.meters['mAP'].update(np.mean(mAP_list), n=batch_size)
            
            if args.mIOU_metric:
                if len(predictions.shape) == 3:
                    predictions = torch.reshape(predictions, [batch_size, c, int(n ** 0.5), int(n ** 0.5)])
            
                mIOU_list = compute_mIOU(gts, predictions, target, start_iter, end_iter)
                if mIOU is None: 
                    mIOU = mIOU_list
                else:
                    mIOU = np.vstack((mIOU, mIOU_list))
                
                for iter in range(mIOU.shape[1]):
                    metric_logger.meters[f'mIOU@{iter}'].update(np.mean(mIOU_list, axis=0)[iter], n=b)
        
        metric_logger.update(cls_loss=cls_loss.item())
        if args.clr_loss:
            metric_logger.update(clr_loss=clr_loss.item())
        metric_logger.update(loss=loss.item())
     
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    
    if args.mIOU_metric:
        best_mIOU = 0.0
        for key, val in stats.items():
            if 'mIOU' in key:
                best_mIOU = max(best_mIOU, val)
        
        stats = {k: val for k, val in stats.items() if 'mIOU' not in k}
        stats['best_mIOU'] = best_mIOU
        print('* mAP {mAP.global_avg:.3f} | loss {losses.global_avg:.3f} | best_mIOU {best_mIOU}%'
                    .format(mAP=metric_logger.mAP, losses=metric_logger.loss, best_mIOU = round(stats['best_mIOU'] * 100, 2)))
    else:
        print('* mAP {mAP.global_avg:.3f} | loss {losses.global_avg:.3f}'
                    .format(mAP=metric_logger.mAP, losses=metric_logger.loss))
    
    return stats

def compute_mAP(labels, outputs):
    y_true = labels.cpu().numpy()
    y_pred = outputs.cpu().numpy()
    AP = []
    for i in range(y_true.shape[0]):
        if np.sum(y_true[i]) > 0:
            ap_i = average_precision_score(y_true[i], y_pred[i])
            AP.append(ap_i)
            # print(ap_i)
    return AP
    
def compute_mIOU(gts, predictions, targets, start_iter, end_iter):
    assert len(gts) == predictions.shape[0]
    size = len(gts)
    mIOUs = np.zeros((size, end_iter - start_iter+1))
    batch = len(gts)
    for i in range(batch):
            gt = np.array(gts[i])
            ori_w, ori_h = gt.shape
            
            cams = predictions[i]
            classes, pw, ph = cams.shape
            cams = cams.reshape(1, classes, pw, ph)
            cams = F.interpolate(cams, size=(ori_w, ori_h), mode='bilinear', align_corners=False)[0]
            cams = cams.cpu().numpy() # include gt?
            cam_dict = {}
            for c in range(classes):
                if (targets[i, c] > 0):
                    cam = cams[c]
                    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
                    cam_dict[c] = cam
            
            tensor = np.zeros((classes+1, ori_w, ori_h), np.float32)
            for key in cam_dict.keys():
                tensor[key+1] = cam_dict[key]
            for k in range(start_iter, end_iter):
                th = k/100.0
                tensor[0,:,:] = th
                predict = np.argmax(tensor, axis=0).astype(np.uint8)
                cal = gt<255
                mask = (predict==gt) * cal
                
                IOU = []
                for c in range(classes+1):
                    P_val = (np.sum((predict==c)*cal))
                    T_val = (np.sum((gt==c)*cal))
                    TP_val = (np.sum((gt==c)*mask))
                    IOU.append(TP_val / (T_val + P_val - TP_val + 1e-10))
                
                mIOUs[i, k] = np.mean(np.array(IOU))
    
    return mIOUs
                

@torch.no_grad()
def generate_attention_maps_ms(data_loader, model, device, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Generating attention maps:'
    if args.attention_dir is not None:
        Path(args.attention_dir).mkdir(parents=True, exist_ok=True)
    if args.cam_npy_dir is not None:
        Path(args.cam_npy_dir).mkdir(parents=True, exist_ok=True)

    # switch to evaluation mode
    model.eval()

    img_list = open(os.path.join(args.img_list, 'train_aug_id.txt')).readlines()
    index = 0
    for image_list, target in metric_logger.log_every(data_loader, 10, header):
    # for iter, (image_list, target) in enumerate(data_loader):
        images1 = image_list[0].to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        batch_size = images1.shape[0]
        img_name = img_list[index].strip()
        index += 1

        img_temp = images1.permute(0, 2, 3, 1).detach().cpu().numpy()
        orig_images = np.zeros_like(img_temp)
        orig_images[:, :, :, 0] = (img_temp[:, :, :, 0] * 0.229 + 0.485) * 255.
        orig_images[:, :, :, 1] = (img_temp[:, :, :, 1] * 0.224 + 0.456) * 255.
        orig_images[:, :, :, 2] = (img_temp[:, :, :, 2] * 0.225 + 0.406) * 255.

        w_orig, h_orig = orig_images.shape[1], orig_images.shape[2]
        # w, h = images1.shape[2] - images1.shape[2] % args.patch_size, images1.shape[3] - images1.shape[3] % args.patch_size
        # w_featmap = w // args.patch_size
        # h_featmap = h // args.patch_size


        with torch.cuda.amp.autocast():
            cam_list = []
            vitattn_list = []
            cam_maps = None
            for s in range(len(image_list)):
                images = image_list[s].to(device, non_blocking=True)
                w, h = images.shape[2] - images.shape[2] % args.patch_size, images.shape[3] - images.shape[3] % args.patch_size
                w_featmap = w // args.patch_size
                h_featmap = h // args.patch_size

                if 'MCTformerV1' in args.model:
                    outputs = model(images, return_att=True, n_layers=args.layer_index)
                    output, cls_attentions, patch_features, patch_attn = outputs
                    cls_attentions = cls_attentions.reshape(batch_size, args.nb_classes, w_featmap, h_featmap)
                    patch_attn = torch.sum(patch_attn, dim=0)

                elif 'MCTformerV2' in args.model:
                    outputs = model(images, return_att=True, n_layers=args.layer_index, attention_type=args.attention_type)
                    output, cls_attentions, patch_features, patch_attn, patch_logits = outputs
                    patch_attn = torch.sum(patch_attn, dim=0)


                if args.patch_attn_refine:
                    cls_attentions = torch.matmul(patch_attn.unsqueeze(1), cls_attentions.view(cls_attentions.shape[0],cls_attentions.shape[1], -1, 1)).reshape(cls_attentions.shape[0],cls_attentions.shape[1], w_featmap, h_featmap)

                cls_attentions = F.interpolate(cls_attentions, size=(w_orig, h_orig), mode='bilinear', align_corners=False)[0]
                cls_attentions = cls_attentions.cpu().numpy() * target.clone().view(args.nb_classes, 1, 1).cpu().numpy()

                if s % 2 == 1:
                    cls_attentions = np.flip(cls_attentions, axis=-1)
                cam_list.append(cls_attentions)
                vitattn_list.append(cam_maps)

            sum_cam = np.sum(cam_list, axis=0)
            sum_cam = torch.from_numpy(sum_cam)
            sum_cam = sum_cam.unsqueeze(0).to(device)

            output = torch.sigmoid(output)

        if args.visualize_cls_attn:
            for b in range(images.shape[0]):
                if (target[b].sum()) > 0:
                    cam_dict = {}
                    for cls_ind in range(args.nb_classes):
                        if target[b,cls_ind]>0:
                            cls_score = format(output[b, cls_ind].cpu().numpy(), '.3f')

                            cls_attention = sum_cam[b,cls_ind,:]

                            cls_attention = (cls_attention - cls_attention.min()) / (cls_attention.max() - cls_attention.min() + 1e-8)
                            cls_attention = cls_attention.cpu().numpy()

                            cam_dict[cls_ind] = cls_attention

                            if args.attention_dir is not None:
                                fname = os.path.join(args.attention_dir, img_name + '_' + str(cls_ind) + '_' + str(cls_score) + '.png')
                                show_cam_on_image(orig_images[b], cls_attention, fname)

                    if args.cam_npy_dir is not None:
                        np.save(os.path.join(args.cam_npy_dir, img_name + '.npy'), cam_dict)

                    if args.out_crf is not None:
                        for t in [args.low_alpha, args.high_alpha]:
                            orig_image = orig_images[b].astype(np.uint8).copy(order='C')
                            crf = _crf_with_alpha(cam_dict, t, orig_image)
                            folder = args.out_crf + ('_%s' % t)
                            if not os.path.exists(folder):
                                os.makedirs(folder)
                            np.save(os.path.join(folder, img_name + '.npy'), crf)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    return


def _crf_with_alpha(cam_dict, alpha, orig_img):
    from psa.tool.imutils import crf_inference
    v = np.array(list(cam_dict.values()))
    bg_score = np.power(1 - np.max(v, axis=0, keepdims=True), alpha)
    bgcam_score = np.concatenate((bg_score, v), axis=0)
    crf_score = crf_inference(orig_img, bgcam_score, labels=bgcam_score.shape[0])

    n_crf_al = dict()

    n_crf_al[0] = crf_score[0]
    for i, key in enumerate(cam_dict.keys()):
        n_crf_al[key + 1] = crf_score[i + 1]

    return n_crf_al


def show_cam_on_image(img, mask, save_path):
    img = np.float32(img) / 255.
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + img
    cam = cam / np.max(cam)
    cam = np.uint8(255 * cam)
    cv2.imwrite(save_path, cam)
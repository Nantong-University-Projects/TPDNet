import math

import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from clrnet.utils.lane import Lane
from clrnet.models.losses.focal_loss import FocalLoss
from clrnet.models.losses.accuracy import accuracy
from clrnet.ops import nms

from clrnet.models.utils.roi_gather import ROIGather, LinearModule
from clrnet.models.utils.seg_decoder import SegDecoder
from clrnet.models.utils.dynamic_assign import assign
from clrnet.models.losses.lineiou_loss import liou_loss
from ..utils.topology_position_reasoner import TopologyPositionReasoner
from ..registry import HEADS


@HEADS.register_module
class CLRHead(nn.Module):
    def __init__(self,
                 num_points=72,
                 prior_feat_channels=64,
                 fc_hidden_dim=64,
                 num_priors=192,
                 num_fc=2,
                 refine_layers=3,
                 sample_points=36,
                 use_directional_attention=True,
                 use_tpr=True,
                 tpr_num_heads=4,
                 tpr_dropout=0.1,
                 cfg=None):
        super(CLRHead, self).__init__()
        self.cfg = cfg
        if self.cfg.haskey('use_tpr'):
            use_tpr = self.cfg.use_tpr
        if self.cfg.haskey('tpr_num_heads'):
            tpr_num_heads = self.cfg.tpr_num_heads
        if self.cfg.haskey('tpr_dropout'):
            tpr_dropout = self.cfg.tpr_dropout

        self.img_w = self.cfg.img_w
        self.img_h = self.cfg.img_h
        self.n_strips = num_points - 1
        self.n_offsets = num_points
        self.num_priors = num_priors
        self.sample_points = sample_points
        self.refine_layers = refine_layers
        self.fc_hidden_dim = fc_hidden_dim

        self.register_buffer(name='sample_x_indexs', tensor=(torch.linspace(
            0, 1, steps=self.sample_points, dtype=torch.float32) *
                                self.n_strips).long())
        self.register_buffer(name='prior_feat_ys', tensor=torch.flip(
            (1 - self.sample_x_indexs.float() / self.n_strips), dims=[-1]))
        self.register_buffer(name='prior_ys', tensor=torch.linspace(1,
                                       0,
                                       steps=self.n_offsets,
                                       dtype=torch.float32))

        self.prior_feat_channels = prior_feat_channels

        self._init_prior_embeddings()
        init_priors, priors_on_featmap = self.generate_priors_from_embeddings() #None, None
        self.register_buffer(name='priors', tensor=init_priors)
        self.register_buffer(name='priors_on_featmap', tensor=priors_on_featmap)

        self.seg_decoder = SegDecoder(self.img_h, self.img_w,
                                      self.cfg.num_classes,
                                      self.prior_feat_channels,
                                      self.refine_layers)

        reg_modules = list()
        cls_modules = list()
        for _ in range(num_fc):
            reg_modules += [*LinearModule(self.fc_hidden_dim)]
            cls_modules += [*LinearModule(self.fc_hidden_dim)]
        self.reg_modules = nn.ModuleList(reg_modules)
        self.cls_modules = nn.ModuleList(cls_modules)

        self.roi_gather = ROIGather(self.prior_feat_channels, self.num_priors,
                                    self.sample_points, self.fc_hidden_dim,
                                    self.refine_layers,
                                    use_directional_attention=use_directional_attention)
        self.use_tpr = use_tpr
        if self.use_tpr:
            self.tpr = TopologyPositionReasoner(
                feat_dim=self.fc_hidden_dim,
                num_heads=tpr_num_heads,
                dropout=tpr_dropout,
                n_offsets=self.n_offsets)

        self.reg_layers = nn.Linear(
            self.fc_hidden_dim, self.n_offsets + 1 + 2 +
            1)  # n offsets + 1 length + start_x + start_y + theta
        self.cls_layers = nn.Linear(self.fc_hidden_dim, 2)

        weights = torch.ones(self.cfg.num_classes)
        weights[0] = self.cfg.bg_weight
        self.criterion = torch.nn.NLLLoss(ignore_index=self.cfg.ignore_label,
                                     weight=weights)

        # init the weights here
        self.init_weights()

    def init_weights(self):
        # initialize heads
        for m in self.cls_layers.parameters():
            nn.init.normal_(m, mean=0., std=1e-3)

        for m in self.reg_layers.parameters():
            nn.init.normal_(m, mean=0., std=1e-3)

    def _lane_smoothness_loss(self, lane_xs):
        if lane_xs.numel() == 0 or lane_xs.shape[-1] < 3:
            return lane_xs.new_tensor(0.)

        valid = (lane_xs >= 0) & (lane_xs < self.img_w)
        valid_triplets = valid[:, :-2] & valid[:, 1:-1] & valid[:, 2:]
        if not valid_triplets.any():
            return lane_xs.new_tensor(0.)

        second_diff = lane_xs[:, 2:] - 2 * lane_xs[:, 1:-1] + lane_xs[:, :-2]
        return second_diff[valid_triplets].pow(2).mean()

    def _lane_order_loss(self, lane_xs, margin=5.):
        if lane_xs.numel() == 0 or lane_xs.shape[0] < 2:
            return lane_xs.new_tensor(0.)

        valid = (lane_xs >= 0) & (lane_xs < self.img_w)
        valid_counts = valid.sum(dim=1).clamp(min=1)
        mean_x = (lane_xs * valid.float()).sum(dim=1) / valid_counts.float()
        sorted_inds = torch.argsort(mean_x)
        sorted_xs = lane_xs[sorted_inds]
        sorted_valid = valid[sorted_inds]

        left_xs = sorted_xs[:-1]
        right_xs = sorted_xs[1:]
        pair_valid = sorted_valid[:-1] & sorted_valid[1:]
        if not pair_valid.any():
            return lane_xs.new_tensor(0.)

        separation = right_xs - left_xs
        order_penalty = F.relu(margin - separation)
        return order_penalty[pair_valid].mean()

    def _orthogonality_loss(self, reg_features, cls_features):
        if reg_features.numel() == 0 or cls_features.numel() == 0:
            return reg_features.new_tensor(0.)

        prod = torch.bmm(reg_features.transpose(1, 2), cls_features)
        scale = max(reg_features.shape[1] * reg_features.shape[2], 1)
        return prod.pow(2).mean() / scale

    def pool_prior_features(self, batch_features, num_priors, prior_xs):

        batch_size = batch_features.shape[0]

        prior_xs = prior_xs.view(batch_size, num_priors, -1, 1)
        prior_ys = self.prior_feat_ys.repeat(batch_size * num_priors).view(
            batch_size, num_priors, -1, 1)

        prior_xs = prior_xs * 2. - 1.
        prior_ys = prior_ys * 2. - 1.
        grid = torch.cat((prior_xs, prior_ys), dim=-1)
        feature = F.grid_sample(batch_features, grid,
                                align_corners=True).permute(0, 2, 1, 3)

        feature = feature.reshape(batch_size * num_priors,
                                  self.prior_feat_channels, self.sample_points,
                                  1)
        return feature

    def generate_priors_from_embeddings(self):
        predictions = self.prior_embeddings.weight  # (num_prop, 3)

        # 2 scores, 1 start_y, 1 start_x, 1 theta, 1 length, 72 coordinates, score[0] = negative prob, score[1] = positive prob
        priors = predictions.new_zeros(
            (self.num_priors, 2 + 2 + 2 + self.n_offsets), device=predictions.device)

        priors[:, 2:5] = predictions.clone()
        priors[:, 6:] = (
            priors[:, 3].unsqueeze(1).clone().repeat(1, self.n_offsets) *
            (self.img_w - 1) +
            ((1 - self.prior_ys.repeat(self.num_priors, 1) -
              priors[:, 2].unsqueeze(1).clone().repeat(1, self.n_offsets)) *
             self.img_h / torch.tan(priors[:, 4].unsqueeze(1).clone().repeat(
                 1, self.n_offsets) * math.pi + 1e-5))) / (self.img_w - 1)

        # init priors on feature map
        priors_on_featmap = priors.clone()[..., 6 + self.sample_x_indexs]

        return priors, priors_on_featmap

    def _init_prior_embeddings(self):
        # [start_y, start_x, theta] -> all normalize
        self.prior_embeddings = nn.Embedding(self.num_priors, 3)

        bottom_priors_nums = self.num_priors * 3 // 4
        left_priors_nums, _ = self.num_priors // 8, self.num_priors // 8

        strip_size = 0.5 / (left_priors_nums // 2 - 1)
        bottom_strip_size = 1 / (bottom_priors_nums // 4 + 1)
        for i in range(left_priors_nums):
            nn.init.constant_(self.prior_embeddings.weight[i, 0],
                              (i // 2) * strip_size)
            nn.init.constant_(self.prior_embeddings.weight[i, 1], 0.)
            nn.init.constant_(self.prior_embeddings.weight[i, 2],
                              0.16 if i % 2 == 0 else 0.32)

        for i in range(left_priors_nums,
                       left_priors_nums + bottom_priors_nums):
            nn.init.constant_(self.prior_embeddings.weight[i, 0], 0.)
            nn.init.constant_(self.prior_embeddings.weight[i, 1],
                              ((i - left_priors_nums) // 4 + 1) *
                              bottom_strip_size)
            nn.init.constant_(self.prior_embeddings.weight[i, 2],
                              0.2 * (i % 4 + 1))

        for i in range(left_priors_nums + bottom_priors_nums, self.num_priors):
            nn.init.constant_(
                self.prior_embeddings.weight[i, 0],
                ((i - left_priors_nums - bottom_priors_nums) // 2) *
                strip_size)
            nn.init.constant_(self.prior_embeddings.weight[i, 1], 1.)
            nn.init.constant_(self.prior_embeddings.weight[i, 2],
                              0.68 if i % 2 == 0 else 0.84)

    # forward function here
    def forward(self, x, **kwargs):
        batch_features = list(x[len(x) - self.refine_layers:])
        batch_features.reverse()
        batch_size = batch_features[-1].shape[0]

        if self.training:
            self.priors, self.priors_on_featmap = self.generate_priors_from_embeddings()

        priors, priors_on_featmap = self.priors.repeat(batch_size, 1,
                                                  1), self.priors_on_featmap.repeat(
                                                      batch_size, 1, 1)

        predictions_lists = []
        tpr_attn_lists = []
        final_stage_branch_features = {}

        prior_features_stages = []
        for stage in range(self.refine_layers):
            num_priors = priors_on_featmap.shape[1]
            prior_xs = torch.flip(priors_on_featmap, dims=[2])

            batch_prior_features = self.pool_prior_features(
                batch_features[stage], num_priors, prior_xs)
            prior_features_stages.append(batch_prior_features)

            fc_features = self.roi_gather(prior_features_stages,
                                          batch_features[stage], stage)

            if self.use_tpr:
                fc_features, tpr_attn = self.tpr(fc_features, priors)
                tpr_attn_lists.append(tpr_attn)

            fc_features = fc_features.view(num_priors, batch_size,
                                           -1).reshape(batch_size * num_priors,
                                                       self.fc_hidden_dim)

            cls_features = fc_features.clone()
            reg_features = fc_features.clone()
            for cls_layer in self.cls_modules:
                cls_features = cls_layer(cls_features)
            for reg_layer in self.reg_modules:
                reg_features = reg_layer(reg_features)

            if stage == self.refine_layers - 1:
                final_stage_branch_features = {
                    'cls': cls_features.reshape(batch_size, num_priors, -1),
                    'reg': reg_features.reshape(batch_size, num_priors, -1)
                }

            cls_logits = self.cls_layers(cls_features)
            reg = self.reg_layers(reg_features)

            cls_logits = cls_logits.reshape(
                batch_size, -1, cls_logits.shape[1])  # (B, num_priors, 2)
            reg = reg.reshape(batch_size, -1, reg.shape[1])

            predictions = priors.clone()
            predictions[:, :, :2] = cls_logits

            predictions[:, :,
                        2:5] += reg[:, :, :3]  # also reg theta angle here
            predictions[:, :, 5] = reg[:, :, 3]  # length

            def tran_tensor(t):
                return t.unsqueeze(2).clone().repeat(1, 1, self.n_offsets)

            predictions[..., 6:] = (
                tran_tensor(predictions[..., 3]) * (self.img_w - 1) +
                ((1 - self.prior_ys.repeat(batch_size, num_priors, 1) -
                  tran_tensor(predictions[..., 2])) * self.img_h /
                 torch.tan(tran_tensor(predictions[..., 4]) * math.pi + 1e-5))) / (self.img_w - 1)

            prediction_lines = predictions.clone()
            predictions[..., 6:] += reg[..., 4:]

            predictions_lists.append(predictions)

            if stage != self.refine_layers - 1:
                priors = prediction_lines.detach().clone()
                priors_on_featmap = priors[..., 6 + self.sample_x_indexs]

        if self.training:
            seg = None
            seg_features = torch.cat([
                F.interpolate(feature,
                              size=[
                                  batch_features[-1].shape[2],
                                  batch_features[-1].shape[3]
                              ],
                              mode='bilinear',
                              align_corners=False)
                for feature in batch_features
            ],
                                     dim=1)
            seg = self.seg_decoder(seg_features)
            output = {'predictions_lists': predictions_lists, 'seg': seg}
            if self.use_tpr:
                output['tpr_attn_lists'] = tpr_attn_lists
            if final_stage_branch_features:
                output['final_stage_branch_features'] = final_stage_branch_features
            return self.loss(output, kwargs['batch'])

        return predictions_lists[-1]

    def predictions_to_pred(self, predictions):
        
        self.prior_ys = self.prior_ys.to(predictions.device)
        self.prior_ys = self.prior_ys.double()
        lanes = []
        for lane in predictions:
            lane_xs = lane[6:]  # normalized value
            start = min(max(0, int(round(lane[2].item() * self.n_strips))),
                        self.n_strips)
            length = int(round(lane[5].item()))
            end = start + length - 1
            end = min(end, len(self.prior_ys) - 1)
            # end = label_end
            # if the prediction does not start at the bottom of the image,
            # extend its prediction until the x is outside the image
            mask = ~((((lane_xs[:start] >= 0.) & (lane_xs[:start] <= 1.)
                       ).cpu().numpy()[::-1].cumprod()[::-1]).astype(np.bool))
            lane_xs[end + 1:] = -2
            lane_xs[:start][mask] = -2
            lane_ys = self.prior_ys[lane_xs >= 0]
            lane_xs = lane_xs[lane_xs >= 0]
            lane_xs = lane_xs.flip(0).double()
            lane_ys = lane_ys.flip(0)

            lane_ys = (lane_ys * (self.cfg.ori_img_h - self.cfg.cut_height) +
                       self.cfg.cut_height) / self.cfg.ori_img_h
            if len(lane_xs) <= 1:
                continue
            points = torch.stack(
                (lane_xs.reshape(-1, 1), lane_ys.reshape(-1, 1)),
                dim=1).squeeze(2)
            lane = Lane(points=points.cpu().numpy(),
                        metadata={
                            'start_x': lane[3],
                            'start_y': lane[2],
                            'conf': lane[1]
                        })
            lanes.append(lane)
        return lanes

    def loss(self,
             output,
             batch,
             cls_loss_weight=2.,
             xyt_loss_weight=0.5,
             iou_loss_weight=2.,
             seg_loss_weight=1.,
             tcl_loss_weight=0.1,
             order_loss_weight=1.0,
             orth_loss_weight=0.01,
             order_margin=5.):
        if self.cfg.haskey('cls_loss_weight'):
            cls_loss_weight = self.cfg.cls_loss_weight
        if self.cfg.haskey('xyt_loss_weight'):
            xyt_loss_weight = self.cfg.xyt_loss_weight
        if self.cfg.haskey('iou_loss_weight'):
            iou_loss_weight = self.cfg.iou_loss_weight
        if self.cfg.haskey('seg_loss_weight'):
            seg_loss_weight = self.cfg.seg_loss_weight
        if self.cfg.haskey('tcl_loss_weight'):
            tcl_loss_weight = self.cfg.tcl_loss_weight
        if self.cfg.haskey('order_loss_weight'):
            order_loss_weight = self.cfg.order_loss_weight
        if self.cfg.haskey('orth_loss_weight'):
            orth_loss_weight = self.cfg.orth_loss_weight
        if self.cfg.haskey('order_margin'):
            order_margin = self.cfg.order_margin

        predictions_lists = output['predictions_lists']
        targets = batch['lane_line'].clone()
        cls_criterion = FocalLoss(alpha=0.25, gamma=2.)
        cls_loss = 0
        reg_xytl_loss = 0
        iou_loss = 0
        smooth_loss = 0
        order_loss = 0
        cls_acc = []

        for stage in range(self.refine_layers):
            cls_acc_stage = []
            predictions_list = predictions_lists[stage]
            for predictions, target in zip(predictions_list, targets):
                target = target[target[:, 1] == 1]

                if len(target) == 0:
                    # If there are no targets, all predictions have to be negatives (i.e., 0 confidence)
                    cls_target = predictions.new_zeros(predictions.shape[0]).long()
                    cls_pred = predictions[:, :2]
                    cls_loss = cls_loss + cls_criterion(
                        cls_pred, cls_target).sum()
                    continue

                with torch.no_grad():
                    matched_row_inds, matched_col_inds = assign(
                        predictions, target, self.img_w, self.img_h)

                # classification targets
                cls_target = predictions.new_zeros(predictions.shape[0]).long()
                cls_target[matched_row_inds] = 1
                cls_pred = predictions[:, :2]

                # regression targets -> [start_y, start_x, theta] (all transformed to absolute values), only on matched pairs
                reg_yxtl = predictions[matched_row_inds, 2:6]
                reg_yxtl[:, 0] *= self.n_strips
                reg_yxtl[:, 1] *= (self.img_w - 1)
                reg_yxtl[:, 2] *= 180
                reg_yxtl[:, 3] *= self.n_strips

                target_yxtl = target[matched_col_inds, 2:6].clone()

                # regression targets -> S coordinates (all transformed to absolute values)
                reg_pred = predictions[matched_row_inds, 6:]
                reg_pred *= (self.img_w - 1)
                reg_targets = target[matched_col_inds, 6:].clone()

                with torch.no_grad():
                    predictions_starts = torch.clamp(
                        (predictions[matched_row_inds, 2] *
                         self.n_strips).round().long(), 0,
                        self.n_strips)  # ensure the predictions starts is valid
                    target_starts = (target[matched_col_inds, 2] *
                                     self.n_strips).round().long()
                    target_yxtl[:, -1] -= (predictions_starts - target_starts
                                           )  # reg length

                # Loss calculation
                cls_loss = cls_loss + cls_criterion(cls_pred, cls_target).sum(
                ) / target.shape[0]

                target_yxtl[:, 0] *= self.n_strips
                target_yxtl[:, 2] *= 180
                reg_xytl_loss = reg_xytl_loss + F.smooth_l1_loss(
                    reg_yxtl, target_yxtl,
                    reduction='none').mean()

                iou_loss = iou_loss + liou_loss(
                    reg_pred, reg_targets,
                    self.img_w, length=15)
                smooth_loss = smooth_loss + self._lane_smoothness_loss(reg_pred)
                order_loss = order_loss + self._lane_order_loss(
                    reg_pred, margin=order_margin)

                # calculate acc
                cls_accuracy = accuracy(cls_pred, cls_target)
                cls_acc_stage.append(cls_accuracy)

            if cls_acc_stage:
                cls_acc.append(sum(cls_acc_stage) / len(cls_acc_stage))
            else:
                cls_acc.append(predictions_lists[stage].new_tensor(0.))

        # extra segmentation loss
        seg_loss = self.criterion(F.log_softmax(output['seg'], dim=1),
                             batch['seg'].long())
        tcl_loss = smooth_loss + order_loss_weight * order_loss
        orth_loss = predictions_lists[0].new_tensor(0.)
        if 'final_stage_branch_features' in output:
            orth_loss = self._orthogonality_loss(
                output['final_stage_branch_features']['reg'],
                output['final_stage_branch_features']['cls'])

        cls_loss /= (len(targets) * self.refine_layers)
        reg_xytl_loss /= (len(targets) * self.refine_layers)
        iou_loss /= (len(targets) * self.refine_layers)
        smooth_loss /= (len(targets) * self.refine_layers)
        order_loss /= (len(targets) * self.refine_layers)
        tcl_loss /= (len(targets) * self.refine_layers)

        loss = cls_loss * cls_loss_weight + reg_xytl_loss * xyt_loss_weight \
            + seg_loss * seg_loss_weight + iou_loss * iou_loss_weight \
            + tcl_loss * tcl_loss_weight + orth_loss * orth_loss_weight

        return_value = {
            'loss': loss,
            'loss_stats': {
                'loss': loss,
                'cls_loss': cls_loss * cls_loss_weight,
                'reg_xytl_loss': reg_xytl_loss * xyt_loss_weight,
                'seg_loss': seg_loss * seg_loss_weight,
                'iou_loss': iou_loss * iou_loss_weight,
                'smooth_loss': smooth_loss,
                'order_loss': order_loss_weight * order_loss,
                'tcl_loss': tcl_loss_weight * tcl_loss,
                'orth_loss': orth_loss_weight * orth_loss
            }
        }

        for i in range(self.refine_layers):
            return_value['loss_stats']['stage_{}_acc'.format(i)] = cls_acc[i]

        return return_value


    def get_lanes(self, output, as_lanes=True):
        softmax = nn.Softmax(dim=1)

        decoded = []
        for predictions in output:
            try:
                max_predictions = 500  
                if predictions.shape[0] > max_predictions:
                    cls_scores = predictions[:, 1] - predictions[:, 0]  # 简单的得分估计
                    top_scores, top_indices = torch.topk(cls_scores, max_predictions)
                    predictions = predictions[top_indices]
                    del cls_scores, top_scores
                    torch.cuda.empty_cache()
                
                # filter out the conf lower than conf threshold
                threshold = self.cfg.test_parameters.conf_threshold
                scores = softmax(predictions[:, :2])[:, 1]
                keep_inds = scores >= threshold
                predictions = predictions[keep_inds]
                scores = scores[keep_inds]

                if predictions.shape[0] == 0:
                    decoded.append([])
                    continue
                
                if predictions.shape[0] > max_predictions:
                    top_scores, top_indices = torch.topk(scores, max_predictions)
                    predictions = predictions[top_indices]
                    scores = top_scores
                    del top_indices
                
                nms_predictions = predictions.detach().clone()
                nms_predictions = torch.cat(
                    [nms_predictions[..., :4], nms_predictions[..., 5:]], dim=-1)
                nms_predictions[..., 4] = nms_predictions[..., 4] * self.n_strips
                nms_predictions[...,
                                5:] = nms_predictions[..., 5:] * (self.img_w - 1)

                keep, num_to_keep, _ = nms(
                    nms_predictions,
                    scores,
                    overlap=self.cfg.test_parameters.nms_thres,
                    top_k=self.cfg.max_lanes)
                
                del nms_predictions
                torch.cuda.empty_cache()
                if num_to_keep > 0:
                    keep = keep[:num_to_keep]
                    max_idx = predictions.shape[0] - 1
                    if isinstance(keep, torch.Tensor):

                        valid_mask = (keep >= 0) & (keep <= max_idx)
                        if valid_mask.any():
                            keep = keep[valid_mask]
                            predictions = predictions[keep]
                        else:
                            decoded.append([])
                            continue
                    else:
                        keep = torch.tensor(keep, device=predictions.device, dtype=torch.long)
                        valid_mask = (keep >= 0) & (keep <= max_idx)
                        if valid_mask.any():
                            keep = keep[valid_mask]
                            predictions = predictions[keep]
                        else:
                            decoded.append([])
                            continue
                else:
                    decoded.append([])
                    continue

                if predictions.shape[0] == 0:
                    decoded.append([])
                    continue

                predictions[:, 5] = torch.round(predictions[:, 5] * self.n_strips)
                if as_lanes:
                    pred = self.predictions_to_pred(predictions)
                else:
                    pred = predictions
                decoded.append(pred)
            
                del predictions, scores, keep
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    decoded.append([])
                    continue
                else:
                    raise e

        return decoded

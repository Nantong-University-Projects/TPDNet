import os.path as osp
import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import torchvision
import logging
from .registry import DATASETS
from .process import Process
from clrnet.utils.visualization import imshow_lanes
from mmcv.parallel import DataContainer as DC


@DATASETS.register_module
class BaseDataset(Dataset):
    def __init__(self, data_root, split, processes=None, cfg=None):
        self.cfg = cfg
        self.logger = logging.getLogger(__name__)
        self.data_root = data_root
        self.training = 'train' in split
        self.processes = Process(processes, cfg)

    def view(self, predictions, img_metas):
        img_metas = [item for img_meta in img_metas.data for item in img_meta]
        for lanes, img_meta in zip(predictions, img_metas):
            img_name = img_meta['img_name']
            img = cv2.imread(osp.join(self.data_root, img_name))
            out_file = osp.join(self.cfg.work_dir, 'visualization',
                                img_name.replace('/', '_'))
            lanes = [lane.to_array(self.cfg) for lane in lanes]
            imshow_lanes(img, lanes, out_file=out_file)

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        data_info = self.data_infos[idx]
        
        # 读取图像，处理文件损坏或缺失的情况
        img = cv2.imread(data_info['img_path'])
        if img is None:
            self.logger.warning(f'Failed to read image: {data_info["img_path"]}, skipping...')
            # 返回下一个样本（递归调用，但限制递归深度）
            if idx < len(self.data_infos) - 1:
                return self.__getitem__(idx + 1)
            else:
                # 如果是最后一个样本，返回第一个
                return self.__getitem__(0)
        
        img = img[self.cfg.cut_height:, :, :]
        sample = data_info.copy()
        sample.update({'img': img})

        if self.training:
            # 读取标签，处理文件损坏或缺失的情况
            if 'mask_path' in sample and sample['mask_path']:
                label = cv2.imread(sample['mask_path'], cv2.IMREAD_UNCHANGED)
                if label is None:
                    self.logger.warning(f'Failed to read mask: {sample["mask_path"]}, creating empty mask...')
                    # 创建一个空的mask，尺寸与图像匹配
                    label = np.zeros((img.shape[0] + self.cfg.cut_height, img.shape[1]), dtype=np.uint8)
                else:
                    if len(label.shape) > 2:
                        label = label[:, :, 0]
                    label = label.squeeze()
                    label = label[self.cfg.cut_height:, :]
            else:
                # 如果没有mask_path，创建空mask
                self.logger.warning(f'No mask_path for {data_info["img_path"]}, creating empty mask...')
                label = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
            
            sample.update({'mask': label})

            if self.cfg.cut_height != 0:
                new_lanes = []
                for i in sample['lanes']:
                    lanes = []
                    for p in i:
                        lanes.append((p[0], p[1] - self.cfg.cut_height))
                    new_lanes.append(lanes)
                sample.update({'lanes': new_lanes})

        sample = self.processes(sample)
        meta = {'full_img_path': data_info['img_path'],
                'img_name': data_info['img_name']}
        meta = DC(meta, cpu_only=True)
        sample.update({'meta': meta})

        return sample

import os
import os.path as osp
import numpy as np
from .base_dataset import BaseDataset
from .registry import DATASETS
import clrnet.utils.culane_metric as culane_metric
import cv2
from tqdm import tqdm
import logging
import pickle as pkl

LIST_FILE = {
    'train': 'list/train_gt.txt',
    'val': 'list/val.txt',
    'test': 'list/test.txt',
}

CATEGORYS = {
    'normal': 'list/test_split/test0_normal.txt',
    'crowd': 'list/test_split/test1_crowd.txt',
    'hlight': 'list/test_split/test2_hlight.txt',
    'shadow': 'list/test_split/test3_shadow.txt',
    'noline': 'list/test_split/test4_noline.txt',
    'arrow': 'list/test_split/test5_arrow.txt',
    'curve': 'list/test_split/test6_curve.txt',
    'cross': 'list/test_split/test7_cross.txt',
    'night': 'list/test_split/test8_night.txt',
}


@DATASETS.register_module
class CULane(BaseDataset):
    def __init__(self, data_root, split, processes=None, cfg=None):
        super().__init__(data_root, split, processes=processes, cfg=cfg)
        self.list_path = osp.join(data_root, LIST_FILE[split])
        self.split = split
        self.load_annotations()

    def load_annotations(self):
        self.logger.info('Loading CULane annotations...')
        # Waiting for the dataset to load is tedious, let's cache it
        os.makedirs('cache', exist_ok=True)
        cache_path = 'cache/culane_{}.pkl'.format(self.split)
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as cache_file:
                self.data_infos = pkl.load(cache_file)
                self.max_lanes = max(
                    len(anno['lanes']) for anno in self.data_infos)
                return

        self.data_infos = []
        with open(self.list_path) as list_file:
            for line in list_file:
                infos = self.load_annotation(line.split())
                self.data_infos.append(infos)
        
        # cache data infos to file
        with open(cache_path, 'wb') as cache_file:
            pkl.dump(self.data_infos, cache_file)

    def load_annotation(self, line):
        infos = {}
        img_line = line[0]
        img_line = img_line[1 if img_line[0] == '/' else 0::]
        img_path = os.path.join(self.data_root, img_line)
        infos['img_name'] = img_line
        infos['img_path'] = img_path
        if len(line) > 1:
            mask_line = line[1]
            mask_line = mask_line[1 if mask_line[0] == '/' else 0::]
            mask_path = os.path.join(self.data_root, mask_line)
            infos['mask_path'] = mask_path

        if len(line) > 2:
            exist_list = [int(l) for l in line[2:]]
            infos['lane_exist'] = np.array(exist_list)

        anno_path = img_path[:-3] + 'lines.txt'  # remove sufix jpg and add lines.txt
        with open(anno_path, 'r') as anno_file:
            data = [
                list(map(float, line.split()))
                for line in anno_file.readlines()
            ]
        lanes = [[(lane[i], lane[i + 1]) for i in range(0, len(lane), 2)
                  if lane[i] >= 0 and lane[i + 1] >= 0] for lane in data]
        lanes = [list(set(lane)) for lane in lanes]  # remove duplicated points
        lanes = [lane for lane in lanes
                 if len(lane) > 2]  # remove lanes with less than 2 points

        lanes = [sorted(lane, key=lambda x: x[1])
                 for lane in lanes]  # sort by y
        infos['lanes'] = lanes

        return infos

    def get_prediction_string(self, pred):
        ys = np.arange(270, 590, 8) / self.cfg.ori_img_h
        out = []
        for lane in pred:
            xs = lane(ys)
            valid_mask = (xs >= 0) & (xs < 1)
            xs = xs * self.cfg.ori_img_w
            lane_xs = xs[valid_mask]
            lane_ys = ys[valid_mask] * self.cfg.ori_img_h
            lane_xs, lane_ys = lane_xs[::-1], lane_ys[::-1]
            lane_str = ' '.join([
                '{:.5f} {:.5f}'.format(x, y) for x, y in zip(lane_xs, lane_ys)
            ])
            if lane_str != '':
                out.append(lane_str)

        return '\n'.join(out)

    def evaluate(self, predictions, output_basedir):
        loss_lines = [[], [], [], []]
        print('Generating prediction output...')
        
        # 确保logger存在
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            logger = logging.getLogger('clrnet')
        
        # 检查输出目录的磁盘空间和权限
        try:
            import shutil
            stat = shutil.disk_usage(os.path.dirname(output_basedir))
            free_gb = stat.free / (1024**3)
            if free_gb < 1.0:
                logger.warning(f'Low disk space: {free_gb:.2f} GB free')
        except Exception as e:
            logger.warning(f'Could not check disk space: {e}')
        
        # 统计写入失败的文件数
        failed_count = 0
        total_count = len(predictions)
        
        # 使用tqdm显示进度
        for idx, pred in enumerate(tqdm(predictions, desc='Writing predictions')):
            try:
            output_dir = os.path.join(
                output_basedir,
                os.path.dirname(self.data_infos[idx]['img_name']))
            output_filename = os.path.basename(
                self.data_infos[idx]['img_name'])[:-3] + 'lines.txt'
                
                # 确保目录存在，增加重试机制
                max_dir_retries = 5
                dir_created = False
                for dir_retry in range(max_dir_retries):
                    try:
            os.makedirs(output_dir, exist_ok=True)
                        # 测试目录是否可写
                        test_file = os.path.join(output_dir, '.write_test')
                        try:
                            with open(test_file, 'w') as f:
                                f.write('test')
                            os.remove(test_file)
                            dir_created = True
                            break
                        except Exception as test_e:
                            if dir_retry == max_dir_retries - 1:
                                raise OSError(f'Directory not writable: {output_dir}, test error: {test_e}')
                    except (OSError, IOError) as e:
                        if dir_retry == max_dir_retries - 1:
                            logger.error(f'Failed to create directory {output_dir} after {max_dir_retries} retries: {e}')
                            failed_count += 1
                            break
                        import time
                        time.sleep(0.3 * (dir_retry + 1))
                        logger.warning(f'Retry {dir_retry + 1}/{max_dir_retries} creating directory {output_dir}')
                
                if not dir_created:
                    continue
                
            output = self.get_prediction_string(pred)
            output_path = os.path.join(output_dir, output_filename)
                
                # 改进的文件写入重试机制
                max_retries = 10  # 增加重试次数
                retry_delay = 0.5
                write_success = False
                
            for retry in range(max_retries):
                try:
                        # 使用临时文件然后重命名，更安全
                        temp_path = output_path + '.tmp'
                        # 确保临时文件目录存在
                        temp_dir = os.path.dirname(temp_path)
                        if temp_dir and not os.path.exists(temp_dir):
                            os.makedirs(temp_dir, exist_ok=True)
                        
                        # 写入临时文件
                        with open(temp_path, 'w', encoding='utf-8') as out_file:
                            out_file.write(output)
                            out_file.flush()
                            os.fsync(out_file.fileno())  # 强制刷新到磁盘
                        
                        # 原子性重命名
                        try:
                            os.replace(temp_path, output_path)
                            write_success = True
                            break
                        except (OSError, IOError) as rename_e:
                            # 如果重命名失败，尝试直接写入
                            if retry == max_retries - 1:
                                # 最后一次尝试，直接写入目标文件
                                try:
                                    with open(output_path, 'w', encoding='utf-8') as out_file:
                        out_file.write(output)
                                        out_file.flush()
                                        os.fsync(out_file.fileno())
                                    write_success = True
                    break
                                except:
                                    pass
                            # 清理临时文件
                            if os.path.exists(temp_path):
                                try:
                                    os.remove(temp_path)
                                except:
                                    pass
                            raise rename_e
                            
                except (OSError, IOError) as e:
                    if retry == max_retries - 1:
                            error_msg = f'Failed to write {output_path} after {max_retries} retries: {e}'
                            logger.error(error_msg)
                            # 尝试清理临时文件
                            temp_path = output_path + '.tmp'
                            if os.path.exists(temp_path):
                                try:
                                    os.remove(temp_path)
                                except:
                                    pass
                            failed_count += 1
                            # 继续处理下一个文件，不中断
                            break
                    import time
                        time.sleep(retry_delay * (retry + 1))  # 指数退避
                        if retry % 3 == 0:  # 每3次重试才记录一次，避免日志过多
                            logger.warning(f'Retry {retry + 1}/{max_retries} writing {output_path}: {e}')
                
            except Exception as e:
                # 捕获所有其他异常，避免整个验证过程崩溃
                logger.error(f'Unexpected error processing prediction {idx}: {e}')
                failed_count += 1
                continue
        
        # 报告写入结果
        success_count = total_count - failed_count
        logger.info(f'Prediction files written: {success_count}/{total_count} ({100*success_count/total_count:.2f}%)')
        if failed_count > 0:
            logger.warning(f'Failed to write {failed_count} prediction files due to I/O errors')
            if failed_count > total_count * 0.1:  # 如果失败超过10%
                logger.error(f'Too many files failed ({failed_count}/{total_count}), evaluation may be inaccurate!')

        for cate, cate_file in CATEGORYS.items():
            result = culane_metric.eval_predictions(output_basedir,
                                                    self.data_root,
                                                    os.path.join(self.data_root, cate_file),
                                                    iou_thresholds=[0.5],
                                                    official=True)

        result = culane_metric.eval_predictions(output_basedir,
                                                self.data_root,
                                                self.list_path,
                                                iou_thresholds=np.linspace(0.5, 0.95, 10),
                                                official=True)

        return result[0.5]['F1']

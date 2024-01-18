import argparse
import os
from ops import utils
from metrics import auroc

import torch
import random
import numpy as np
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(seed)

def experiment(inference_output_dir_path, mask_image_dir_path, delimiter='---'):
    # read
    inf_raw_image_list, gt_image_list, inf_image_path_list = utils.prepare_inf_gt_images(inference_output_dir_path, mask_image_dir_path)

    # auroc by max score
    instance_auroc_max, reliable_good_ratio_max, detail_dict_max, label_score_dict_max, anomaly_min_score_max = auroc.instance_auroc_max(inf_raw_image_list, gt_image_list, inf_image_path_list)
    print('auroc_by_max_score.csv')
    print(f'"instance_auroc", "reliable_good_ratio", "anomaly_min_score"')
    print(f'{instance_auroc_max:.4f}, {reliable_good_ratio_max:.4f}, {anomaly_min_score_max:.4f}')
    print(delimiter)
    print(f'detail_by_max_score.csv')
    print(f'"image_path", "label", "score"')
    for gt_label, inf_image_path, inf_raw in zip(label_score_dict_max['gt_labels'], label_score_dict_max['inf_image_path_list'], label_score_dict_max['inf_raw_list']):
        label = 'anomaly' if gt_label else 'good'
        print(f'"{inf_image_path}", "{label}", {inf_raw:.4f}')
    print(delimiter)

    # auroc by mean score
    instance_auroc_mean, reliable_good_ratio_mean, detail_dict_mean, label_score_dict_mean, anomaly_min_score_mean = auroc.instance_auroc_mean(inf_raw_image_list, gt_image_list, inf_image_path_list)
    print('auroc_by_mean_score.csv')
    print(f'"instance_auroc", "reliable_good_ratio", "anomaly_min_score"')
    print(f'{instance_auroc_mean:.4f}, {reliable_good_ratio_mean:.4f}, {anomaly_min_score_mean:.4f}')
    print(delimiter)

    print(f'detail_by_mean_score.csv')
    print(f'"image_path", "label", "score"')
    for gt_label, inf_image_path, inf_raw in zip(label_score_dict_mean['gt_labels'], label_score_dict_mean['inf_image_path_list'], label_score_dict_mean['inf_raw_list']):
        label = 'anomaly' if gt_label else 'good'
        print(f'"{inf_image_path}", "{label}", {inf_raw:.4f}')
    print(delimiter)

    # pixelwise auroc
    full_pixel_auroc, detail_dict = auroc.full_pixel_auroc(inf_raw_image_list, gt_image_list)
    anomaly_pixel_auroc, detail_dict = auroc.anomaly_pixel_auroc(inf_raw_image_list, gt_image_list)

    auroc_metric_list, detail_dict_list = auroc.anomaly_detail_pixel_auroc(inf_raw_image_list, gt_image_list,
                                                                           inf_image_path_list)
    print(f'pixelwise_auroc.csv')
    print(f'"full_pixel_auroc", "anomaly_pixel_auroc"')
    print(f'{full_pixel_auroc:.4f}, {anomaly_pixel_auroc:.4f}')
    print(delimiter)

    print(f'anomaly_auroc_metric_list.csv')
    print(f'"image_path", "anomaly_pixel_auroc"')
    for auroc_metric in auroc_metric_list:
        print(f'{auroc_metric[0]}, {auroc_metric[1]:.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='experiment')
    parser.add_argument('--inference_output_dir_path', type=str, default='/tmp/output/test_inf')
    parser.add_argument('--mask_image_dir_path', type=str, default=os.path.join(os.path.dirname(__file__), 'sample_dataset/ground_truth'))
    args = parser.parse_args()

    args.inference_output_dir_path = os.path.expanduser(args.inference_output_dir_path)
    args.mask_image_dir_path = os.path.expanduser(args.mask_image_dir_path)

    experiment(args.inference_output_dir_path, args.mask_image_dir_path)
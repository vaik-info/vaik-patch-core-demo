import glob
import os
import json
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter


def decode(anomaly_scores, image_shape, pool_size):
    anomaly_scores = np.reshape(anomaly_scores, (anomaly_scores.shape[0],) + (1, image_shape[0] // pool_size, image_shape[1] // pool_size))
    anomaly_scores = F.interpolate(torch.tensor(anomaly_scores), size=image_shape, mode='bilinear', align_corners=False)
    result_anomaly_scores_list = []
    for anomaly_score in anomaly_scores:
        anomaly_score = anomaly_score[0].numpy()
        result_anomaly_scores_list.append(gaussian_filter(anomaly_score, sigma=np.std(anomaly_score), radius=int(max(image_shape)*0.01)))
    return np.stack(result_anomaly_scores_list, axis=0)


def convert_min_max_normalize_images(anomaly_scores_raw_images):
    anomaly_min_max_normalize_images = []
    min_raw_list = []
    max_raw_list = []
    for anomaly_scores_raw_image in anomaly_scores_raw_images:
        min_raw = np.min(anomaly_scores_raw_image)
        max_raw = np.max(anomaly_scores_raw_image)
        anomaly_min_max_normalize_images.append((anomaly_scores_raw_image - min_raw) / (max_raw - min_raw))
        min_raw_list.append(min_raw)
        max_raw_list.append(max_raw)
    return anomaly_min_max_normalize_images, min_raw_list, max_raw_list


def convert_rgb_images(anomaly_min_max_normalize_images):
    anomaly_rgb_images = []
    for anomaly_min_max_normalize_image in anomaly_min_max_normalize_images:
        anomaly_rgb_image = np.zeros(anomaly_min_max_normalize_image.shape[:2] + (3,), dtype=np.uint8)
        anomaly_image = np.clip(anomaly_min_max_normalize_image * 255, 0, 255).astype('uint8')
        anomaly_rgb_image[:, :, 0] = anomaly_image
        anomaly_rgb_images.append(anomaly_rgb_image)
    return anomaly_rgb_images


def convert_min_max_revert_images(anomaly_min_max_normalize_images, min_raw_list, max_raw_list):
    anomaly_min_max_revert_images = []
    for anomaly_min_max_normalize_image, min_raw, max_raw in zip(anomaly_min_max_normalize_images, min_raw_list,
                                                                 max_raw_list):
        anomaly_min_max_revert_images.append((anomaly_min_max_normalize_image / 255. * (max_raw - min_raw)) + min_raw)
    return anomaly_min_max_revert_images


def prepare_inf_gt_images(inference_output_dir_path, test_image_dir_path):
    # read
    inf_image_path_list = sorted(glob.glob(os.path.join(inference_output_dir_path, '**/*.png'), recursive=True))
    inf_image_list = []
    gt_image_list = []
    json_dict_list = []
    for inf_image_path in inf_image_path_list:
        file_prefix = os.path.splitext(os.path.basename(inf_image_path))[0].split('___')[0]
        json_path = os.path.splitext(inf_image_path)[0] + '.json'
        with open(json_path, 'r') as f:
            json_dict = json.load(f)

        inf_image_list.append(np.asarray(Image.open(inf_image_path))[:, :, 0])
        if json_dict["dir_name"] != 'good':
            gt_image_path = \
                glob.glob(os.path.join(test_image_dir_path, f'{json_dict["dir_name"]}/{file_prefix}*.png'))[0]
            gt_image_list.append(np.asarray(Image.open(gt_image_path).convert('L').resize((inf_image_list[-1].shape[1], inf_image_list[-1].shape[0]))))
        else:
            gt_image_list.append(np.zeros(inf_image_list[-1].shape, dtype=inf_image_list[-1].dtype))
        json_dict_list.append(json_dict)
    # Revert
    inf_raw_image_list = convert_min_max_revert_images(inf_image_list,
                                                       [json_dict['min_raw'] for json_dict in json_dict_list],
                                                       [json_dict['max_raw'] for json_dict in json_dict_list])
    return inf_raw_image_list, gt_image_list, inf_image_path_list


def prepare_gt_images(inf_image_path_list, test_image_dir_path, input_image_shape):
    # read
    gt_image_list = []
    for inf_image_path in inf_image_path_list:
        file_prefix = os.path.splitext(os.path.basename(inf_image_path))[0]
        category = inf_image_path.split('/')[-2]
        if  category != 'good':
            gt_image_path = \
                glob.glob(os.path.join(test_image_dir_path, f'{category}/{file_prefix}*.png'))[0]
            pil_image = Image.open(gt_image_path).convert('L')
            pil_image = pil_image.resize((input_image_shape[0], input_image_shape[1]))
            gt_image_list.append(np.asarray(pil_image))
        else:
            pil_image = Image.open(inf_image_path).convert('L')
            pil_image = pil_image.resize((input_image_shape[0], input_image_shape[1]))
            gt_image_list.append(np.zeros(np.asarray(pil_image).shape, dtype=np.uint8))
    return gt_image_list

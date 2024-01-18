import argparse
import os
import glob
import copy
import gc

from PIL import Image
import torch
from tqdm import tqdm
import numpy as np
import json

from model.pretrain_classification import WideResnet502Model
from model.feature_extractor import FeatureExtractor
from model.feature_patch_maker import PatchMaker
from model.memory_bank import FaissNearestNeighbour
from ops.utils import decode, convert_min_max_normalize_images, convert_rgb_images

import time

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

def patch(device, image_list, preprocessing_dim, aggregate_dims, pretrain_model_path):
    # Extract feature
    if os.path.exists(pretrain_model_path):
        checkpoint = torch.load(pretrain_model_path)
        num_classes = checkpoint['num_classes']
        model = WideResnet502Model(num_classes)
        model.load_state_dict(checkpoint['state_dict'])
        backbone = model.backbone
        print(f'Load: at {pretrain_model_path}')
    else:
        print(f'Load: at default')
        backbone = None

    feature_extractor_model = FeatureExtractor(device, backbone)
    pool_ratio = feature_extractor_model.pool_ratio

    features_list = []
    for image in tqdm(image_list, desc='feature_extractor_model(image)'):
        features = feature_extractor_model(image)
        features_list.append(copy.deepcopy(features))

    # Release gpu memory
    del feature_extractor_model
    gc.collect()
    torch.cuda.empty_cache()

    # Patch feature
    patch_maker_model = PatchMaker(device, preprocessing_dim, aggregate_dims,
                                   (features_list[0][sorted(list(features_list[0].keys()))[0]].shape[1],
                                    features_list[0][sorted(list(features_list[0].keys()))[1]].shape[1]))
    patch_feature_list = []
    for features in tqdm(features_list, desc='patch_process(feature)'):
        patch_feature = patch_maker_model.patch((features[sorted(list(features.keys()))[0]],
                                                 features[sorted(list(features.keys()))[1]]))
        patch_feature_list.append(patch_feature)
    return patch_feature_list, pool_ratio
def inference(input_faiss_path, input_json_path, test_image_dir_path, output_dir_path, pretrain_model_path):
    start = time.time()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Read json
    with open(input_json_path, 'r') as f:
        json_dict = json.load(f)
        good_n_total = json_dict['good_n_total']
        input_image_shape = (json_dict['input_image_height'], json_dict['input_image_width'])
        preprocessing_dim = json_dict['preprocessing_dim']
        aggregate_dims = json_dict['aggregate_dims']

    # Prepare image
    image_path_list = glob.glob(os.path.join(test_image_dir_path, '**/*.png'), recursive=True)
    image_list = [Image.open(image_path).convert('RGB').resize((input_image_shape[1], input_image_shape[0])) for image_path in image_path_list]

    # Prepare patch
    patch_features, pool_ratio = patch(device, image_list, preprocessing_dim, aggregate_dims, pretrain_model_path)

    # Prepare Memory Block
    memory_block_model = FaissNearestNeighbour()
    memory_block_model.load(input_faiss_path, good_n_total)

    # Inference Memory Block
    anomaly_score_list = []
    for patch_feature in tqdm(patch_features, desc='memory_block_model.predict()'):
        anomaly_score_list.append(memory_block_model.predict(np.expand_dims(patch_feature, axis=0)))

    print(f'{len(image_path_list)/(time.time()-start)}[images/s]')
    # Dump
    for image_path, image, anomaly_score in zip(image_path_list, image_list, anomaly_score_list):
        anomaly_scores_raw_images = decode(anomaly_score, (image.size[1], image.size[0]), pool_ratio)
        anomaly_min_max_normalize_images, min_raw_list, max_raw_list = convert_min_max_normalize_images(anomaly_scores_raw_images)
        anomaly_rgb_images = convert_rgb_images(anomaly_min_max_normalize_images)

        anomaly_rgb_image, min_raw, max_raw = anomaly_rgb_images[0], min_raw_list[0], max_raw_list[0]
        sub_dir_name = image_path.split('/')[-2]
        output_file_name = os.path.join(output_dir_path, os.path.splitext(os.path.basename(image_path))[0])
        output_image_path = f'{output_file_name}.png'
        Image.fromarray(anomaly_rgb_image).save(output_image_path, quality=100, subsampling=0)
        json_dict = {'dir_name': sub_dir_name, 'raw_image_path': image_path, 'min_raw': float(min_raw), 'max_raw': float(max_raw)}
        output_json_path = f'{output_file_name}.json'
        with open(output_json_path, 'w') as f:
            json.dump(json_dict, f, ensure_ascii=False, indent=4, sort_keys=True, separators=(',', ': '))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='inference')
    parser.add_argument('--input_faiss_path', type=str, default='/tmp/output/model.faiss')
    parser.add_argument('--input_json_path', type=str, default='/tmp/output/model.json')
    parser.add_argument('--test_image_dir_path', type=str, default=os.path.join(os.path.dirname(__file__),
                                                                                'sample_dataset/test'))
    parser.add_argument('--output_dir_path', type=str, default='/tmp/output/test_inf')
    parser.add_argument('--pretrain_model_path', type=str, default='/tmp/pretrain_output/latest.pth')
    args = parser.parse_args()

    args.input_faiss_path = os.path.expanduser(args.input_faiss_path)
    args.input_json_path = os.path.expanduser(args.input_json_path)
    args.test_image_dir_path = os.path.expanduser(args.test_image_dir_path)
    args.output_dir_path = os.path.expanduser(args.output_dir_path)
    args.pretrain_model_path = os.path.expanduser(args.pretrain_model_path)

    os.makedirs(args.output_dir_path, exist_ok=True)

    inference(args.input_faiss_path, args.input_json_path, args.test_image_dir_path, args.output_dir_path, args.pretrain_model_path)

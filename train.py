import glob
import os
import argparse
import copy
import gc

from PIL import Image
from tqdm import tqdm
import torch
import numpy as np
import json

from model.pretrain_classification import WideResnet502Model
from model.feature_extractor import FeatureExtractor
from model.feature_patch_maker import PatchMaker
from model.feature_sampler import ApproximateGreedyCoresetSampler
from model.memory_bank import FaissNearestNeighbour

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
        backbone = None
        print(f'Load: at default')

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
    patch_features = np.concatenate(patch_feature_list, axis=0)
    return patch_features, pool_ratio


def train(train_good_image_dir_path, input_image_shape, preprocessing_dim, aggregate_dims,
          percentage, number_of_starting_points, dimension_to_project_features_to,
          output_model_dir_path, pretrain_model_path):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Prepare image
    good_image_path_list = glob.glob(os.path.join(train_good_image_dir_path, '**/*.png'), recursive=True)
    good_image_list = [Image.open(good_image_path).convert('RGB').resize((input_image_shape[1], input_image_shape[0])) for good_image_path in good_image_path_list]

    # Prepare patch
    good_patch_features, pool_ratio = patch(device, good_image_list, preprocessing_dim, aggregate_dims, pretrain_model_path)

    # Sampling feature
    sampler = ApproximateGreedyCoresetSampler(percentage, device, number_of_starting_points,
                                              dimension_to_project_features_to)
    good_patch_feature_samples = sampler.run(good_patch_features)

    # Prepare Memory Block
    memory_block_model = FaissNearestNeighbour()
    # Train Memory Block
    memory_block_model.train(good_patch_feature_samples)

    # Save Memory Block
    output_faiss_file_path = os.path.join(output_model_dir_path, 'model.faiss')
    memory_block_model.save(output_faiss_file_path)

    # Save meta info
    output_json_file_path = os.path.join(output_model_dir_path, 'model.json')
    with open(f'{os.path.splitext(output_json_file_path)[0]}.json', 'w') as f:
        json_dict = {}
        if memory_block_model.good_n_total is not None:
            json_dict['good_n_total'] = memory_block_model.good_n_total
        json_dict['input_image_height'] = input_image_shape[0]
        json_dict['input_image_width'] = input_image_shape[1]
        json_dict['preprocessing_dim'] = preprocessing_dim
        json_dict['aggregate_dims'] = aggregate_dims
        json.dump(json_dict, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--train_good_image_dir_path', type=str,
                        default=os.path.join(os.path.dirname(__file__), 'sample_dataset/train/good'))
    parser.add_argument('--input_image_height', type=int, default=512)
    parser.add_argument('--input_image_width', type=int, default=512)
    parser.add_argument('--preprocessing_dim', type=int, default=1024)
    parser.add_argument('--aggregate_dims', type=int, default=1024)
    parser.add_argument('--percentage', type=float, default=0.25)
    parser.add_argument('--number_of_starting_points', type=int, default=10)
    parser.add_argument('--dimension_to_project_features_to', type=int, default=128)
    parser.add_argument('--output_model_dir_path', type=str, default='/tmp/output')
    parser.add_argument('--pretrain_model_path', type=str, default='/tmp/pretrain_output/latest.pth')

    args = parser.parse_args()

    args.train_good_image_dir_path = os.path.expanduser(args.train_good_image_dir_path)
    args.output_model_dir_path = os.path.expanduser(args.output_model_dir_path)
    args.pretrain_model_path = os.path.expanduser(args.pretrain_model_path)

    os.makedirs(args.output_model_dir_path, exist_ok=True)

    train(args.train_good_image_dir_path,
          (args.input_image_height, args.input_image_width) , args.preprocessing_dim, args.aggregate_dims,
          args.percentage, args.number_of_starting_points, args.dimension_to_project_features_to,
          args.output_model_dir_path, args.pretrain_model_path)

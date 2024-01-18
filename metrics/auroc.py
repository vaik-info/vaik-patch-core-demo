from sklearn import metrics
import numpy as np


def instance_auroc_mean(inf_raw_image_list, gt_image_list, inf_image_path_list):
    gt_labels = [np.max(gt_image) > 125 for gt_image in gt_image_list]
    inf_raw_mean_list = [np.mean(inf_raw_image) for inf_raw_image in inf_raw_image_list]
    return instance_auroc(gt_labels, inf_raw_mean_list, inf_image_path_list)

def instance_auroc_max(inf_raw_image_list, gt_image_list, inf_image_path_list):
    gt_labels = [np.max(gt_image) > 125 for gt_image in gt_image_list]
    inf_raw_max_list = [np.max(inf_raw_image) for inf_raw_image in inf_raw_image_list]
    return instance_auroc(gt_labels, inf_raw_max_list, inf_image_path_list)

def instance_auroc(gt_labels, inf_raw_list, inf_image_path_list):
    fpr, tpr, thresholds = metrics.roc_curve(
        gt_labels, inf_raw_list
    )
    auroc_metric = metrics.roc_auc_score(
        gt_labels, inf_raw_list
    )
    sort_indexes = np.argsort(inf_raw_list).tolist()
    gt_labels = [gt_labels[sort_index] for sort_index in sort_indexes]
    inf_raw_list = [inf_raw_list[sort_index] for sort_index in sort_indexes]
    inf_image_path_list = [inf_image_path_list[sort_index] for sort_index in sort_indexes]

    anomaly_min_score = None
    for gt_index, gt_label in enumerate(gt_labels):
        if gt_label:
            anomaly_min_score = inf_raw_list[gt_index]
            break

    return auroc_metric, gt_index / len([ gt_label for gt_label in gt_labels if not gt_label]), {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds}, {'gt_labels': gt_labels, 'inf_raw_list': inf_raw_list, 'inf_image_path_list': inf_image_path_list}, anomaly_min_score

def full_pixel_auroc(inf_raw_image_list, gt_image_list):
    gt_image_array = np.concatenate([(gt_image.flatten() > 125).astype(np.uint8) for gt_image in gt_image_list])
    inf_raw_image_array = np.concatenate([inf_raw_image.flatten() for inf_raw_image in inf_raw_image_list])
    fpr, tpr, thresholds = metrics.roc_curve(
        gt_image_array, inf_raw_image_array
    )
    auroc_metric = metrics.roc_auc_score(
        gt_image_array, inf_raw_image_array
    )
    return auroc_metric, {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds}


def anomaly_pixel_auroc(inf_raw_image_list, gt_image_list):
    gt_labels = [np.max(gt_image) > 125 for gt_image in gt_image_list]
    gt_anomaly_indexes = np.asarray(gt_labels) == True
    gt_image_list = [gt_image for index, gt_image in enumerate(gt_image_list) if gt_anomaly_indexes[index]]
    inf_raw_image_list = [inf_raw_image for index, inf_raw_image in enumerate(inf_raw_image_list) if
                          gt_anomaly_indexes[index]]
    return full_pixel_auroc(inf_raw_image_list, gt_image_list)


def anomaly_detail_pixel_auroc(inf_raw_image_list, gt_image_list, inf_image_path_list):
    gt_labels = [np.max(gt_image) > 125 for gt_image in gt_image_list]
    gt_anomaly_indexes = np.asarray(gt_labels) == True
    gt_image_list = [gt_image for index, gt_image in enumerate(gt_image_list) if gt_anomaly_indexes[index]]
    inf_raw_image_list = [inf_raw_image for index, inf_raw_image in enumerate(inf_raw_image_list) if
                          gt_anomaly_indexes[index]]
    inf_image_path_list = [inf_image_path for index, inf_image_path in enumerate(inf_image_path_list) if
                           gt_anomaly_indexes[index]]

    auroc_metric_list, detail_list = [], []
    for inf_raw_image, gt_image, inf_image_path in zip(inf_raw_image_list, gt_image_list, inf_image_path_list):
        auroc_metric, detail_dict = full_pixel_auroc([inf_raw_image], [gt_image])
        auroc_metric_list.append((inf_image_path, auroc_metric))
        detail_list.append(detail_dict)
    return auroc_metric_list, detail_list

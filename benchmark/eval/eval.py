import numpy as np
import os
import torch
import cv2
import csv
from metric import * 
import metric
import argparse
from tqdm import tqdm
import json


device = 'cuda'
eval_metrics = [
    "abs_relative_difference",
    "rmse_linear",
    "delta1_acc",
    # "squared_relative_difference",
    # "rmse_log",
    # "log10",
    # "delta2_acc",
    # "delta3_acc",
    # "i_rmse",
    # "silog_rmse",
]


def depth2disparity(depth, return_mask=False):
    if isinstance(depth, torch.Tensor):
        disparity = torch.zeros_like(depth)
    elif isinstance(depth, np.ndarray):
        disparity = np.zeros_like(depth)
    non_negtive_mask = depth > 0
    disparity[non_negtive_mask] = 1.0 / depth[non_negtive_mask]
    if return_mask:
        return disparity, non_negtive_mask
    else:
        return disparity


def resize_images(images, new_size):
    resized_images = np.empty(
        (images.shape[0], new_size[0], new_size[1], images.shape[3])
    )

    for i, image in enumerate(images):
        if image.shape[2]==1:
            resized_images[i] = cv2.resize(image, (new_size[1], new_size[0]))[..., None]
        else:
            resized_images[i] = cv2.resize(image, (new_size[1], new_size[0]))

    return resized_images
    

def eval_single(
    pred_disp_path, 
    gt_disp_path, 
    seq_len=98, 
    domain='depth', 
    method_type="ours",
    dataset_max_depth="70"
):
    # load data
    gt_disp = np.load(gt_disp_path)['disparity'] \
        if 'disparity' in np.load(gt_disp_path).files else \
        np.load(gt_disp_path)['arr_0']  # (t, 1, h, w)

    if method_type=="ours":
        pred_disp = np.load(pred_disp_path)['depth'] # (t, h, w)
    if method_type=="depth_anything":
        pred_disp = np.load(pred_disp_path)['disparity'] # (t, h, w)
    
    # seq_len
    if pred_disp.shape[0] < seq_len:
         seq_len = pred_disp.shape[0]

    # preprocess
    pred_disp = resize_images(pred_disp[..., None], (gt_disp.shape[-2], gt_disp.shape[-1])) # (t, h, w)
    pred_disp = pred_disp[..., 0] # (t, h, w)
    pred_disp = pred_disp[:seq_len]
    gt_disp = gt_disp[:seq_len, 0] # (t, h, w)

    # valid mask
    valid_mask = np.logical_and(
            (gt_disp > 1e-3), 
            (gt_disp < dataset_max_depth)
        )
    pred_disp = np.clip(pred_disp, a_min=1e-3, a_max=None) 
    pred_disp_masked = pred_disp[valid_mask].reshape((-1, 1))
    
    # choose evaluation domain
    DOMAIN = domain
    if DOMAIN=='disp':
        # align in real disp, calc in disp
        gt_disp_maksed = gt_disp[valid_mask].reshape((-1, 1)).astype(np.float64)
    elif DOMAIN=='depth':
        # align in disp = 1/depth, calc in depth
        gt_disp_maksed = 1. / (gt_disp[valid_mask].reshape((-1, 1)).astype(np.float64) + 1e-8)
    else:
        pass


    # calc scale and shift
    _ones = np.ones_like(pred_disp_masked)
    A = np.concatenate([pred_disp_masked, _ones], axis=-1)
    X = np.linalg.lstsq(A, gt_disp_maksed, rcond=None)[0]
    scale, shift = X # gt = scale * pred + shift
    
    # align
    aligned_pred = scale * pred_disp + shift
    aligned_pred = np.clip(aligned_pred, a_min=1e-3, a_max=None) 


    # align in real disp, calc in disp
    if DOMAIN=='disp':
        pred_depth = aligned_pred
        gt_depth = gt_disp
    # align in disp = 1/depth, calc in depth
    elif DOMAIN=='depth':
        pred_depth = depth2disparity(aligned_pred)
        gt_depth = gt_disp
    else:
        pass

    # metric evaluation, clip to dataset min max
    pred_depth = np.clip(
            pred_depth, a_min=1e-3, a_max=dataset_max_depth
        )

    # evaluate metric 
    sample_metric = []
    metric_funcs = [getattr(metric, _met) for _met in eval_metrics]

    # Evaluate 
    sample_metric = []
    pred_depth_ts = torch.from_numpy(pred_depth).to(device)
    gt_depth_ts = torch.from_numpy(gt_depth).to(device)
    valid_mask_ts = torch.from_numpy(valid_mask).to(device)

    n = valid_mask.sum((-1, -2))
    valid_frame = (n > 0)
    pred_depth_ts = pred_depth_ts[valid_frame]
    gt_depth_ts = gt_depth_ts[valid_frame]
    valid_mask_ts = valid_mask_ts[valid_frame]

    for met_func in metric_funcs:
        _metric_name = met_func.__name__
        _metric = met_func(pred_depth_ts, gt_depth_ts, valid_mask_ts).item()
        sample_metric.append(_metric)

    return sample_metric



if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seq_len", 
        type=int,
        default=50,
        help="Max video frame length for evaluation."
    )

    parser.add_argument(
        "--domain",
        type=str,
        default="depth",
        choices=["depth", "disp"],
        help="Domain of metric calculation."
    )

    parser.add_argument(
        "--method_type",
        type=str,
        default="ours",
        choices=["ours", "depth_anything"],
        help="Choose the methods."
    )

    parser.add_argument(
        "--dataset_max_depth",
        type=int,
        default=70,
        help="Dataset max depth clip."
    )

    parser.add_argument(
        "--pred_disp_root",
        type=str,
        default="./demo_output",
        help="Predicted output directory."
    )

    parser.add_argument(
        "--gt_disp_root",
        type=str,
        required=True,
        help="GT depth directory."
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Choose the datasets."
    )

    parser.add_argument(
        "--meta_path",
        type=str,
        required=True,
        help="Path of test dataset csv file."
    )


    args = parser.parse_args()

    SEQ_LEN = args.seq_len
    method_type = args.method_type
    if method_type == "ours":
        pred_disp_root = os.path.join(args.pred_disp_root, f'results_{args.dataset}')
    else:
        # pred_disp_root = args.pred_disp_root
        pred_disp_root = os.path.join(args.pred_disp_root, f'results_{args.dataset}')
    domain = args.domain
    dataset_max_depth = args.dataset_max_depth
    saved_json_path = os.path.join(args.pred_disp_root, f"results_{args.dataset}.json")

    meta_path = args.meta_path

    assert method_type in ["depth_anything", "ours"], "Invalid method type, must be in ['depth_anything', 'ours']"
    assert domain in ["depth", "disp"], "Invalid domain type, must be in ['depth', 'disp']"
        
    with open(meta_path, mode="r", encoding="utf-8") as csvfile:
        csv_reader = csv.DictReader(csvfile)
        samples = list(csv_reader)

    # iterate all cases
    results_all = []
    for i, sample in enumerate(tqdm(samples)):
        gt_disp_path = os.path.join(args.gt_disp_root, samples[i]['filepath_disparity'])
        if method_type=="ours":
            pred_disp_path = os.path.join(pred_disp_root, samples[i]['filepath_disparity'])
            pred_disp_path = pred_disp_path.replace("disparity", "rgb_left")
        
        if method_type=="depth_anything":
            pred_disp_path = os.path.join(pred_disp_root, samples[i]['filepath_disparity'])
            pred_disp_path = pred_disp_path.replace("disparity", "rgb_left_depth")
        
        results_single = eval_single(
            pred_disp_path, 
            gt_disp_path, 
            seq_len=SEQ_LEN, 
            domain=domain, 
            method_type=method_type, 
            dataset_max_depth=dataset_max_depth
        )

        results_all.append(results_single)

    # avarage
    final_results =  np.array(results_all)
    final_results_mean = np.mean(final_results, axis=0)
    print("")

    # save mean to json
    result_dict = { 'name': method_type }
    for i, metric in enumerate(eval_metrics):
        result_dict[metric] = final_results_mean[i]
        print(f"{metric}: {final_results_mean[i]:04f}")

    # save each case to json
    for i, results in enumerate(results_all):
        result_dict[samples[i]['filepath_disparity']] = results

    # write json
    with open(saved_json_path, 'w') as f:
        json.dump(result_dict, f, indent=4)
    print("")
    print(f"Evaluation results json are saved to {saved_json_path}")
    

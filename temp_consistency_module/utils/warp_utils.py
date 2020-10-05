import re
import os
import numpy as np
import os.path as osp

import json
from tqdm import tqdm
from skimage import io
from scipy.ndimage import binary_fill_holes
import pycocotools.mask as maskUtils

import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt


def compute_mask_IU(mask_gt, mask_pred):
    assert(mask_pred.shape[-2:] == mask_gt.shape[-2:])
    temp = (mask_gt * mask_pred)
    intersection = temp.sum()
    union = ((mask_gt + mask_pred) - temp).sum()
    return intersection, union

def load_json(filename):
    with open(filename, "r") as f:
        data = json.load(f)
    return data

def uncrop_image(img, org_h=1080,
               org_w=1920, h_start=28,
               w_start=320, h=1024, w=1280):
            h_end, w_end = h + h_start, w + w_start
            large_img = np.zeros((org_h, org_w), dtype=np.uint8)
            large_img[h_start:h_end, w_start:w_end] = img
            return large_img

def crop_image(image, h_start=28, w_start=320, h=1024, w=1280):
    image = image[h_start : h_start + h, w_start : w_start + w]
    return image

def extract_candidates(img_name, coco_anns, coco_preds, name_list, pred_img_ids, min_score=0.75):
    cands = []
    idx = [i for i, name in enumerate(name_list) if name == img_name]
    if len(idx) > 0:
        idx = idx[0]
    else:
        return cands
    this_data = coco_anns["images"][idx]
    image_id = this_data["id"]
    (matching_idx,) = np.nonzero(pred_img_ids == image_id)
    for p_idx in matching_idx:
        pred = coco_preds[p_idx]
        if pred["score"] >= min_score:
            segm = pred["segmentation"]
            segm_mask = maskUtils.decode(segm).astype(np.float)
            segm_mask = np.expand_dims(segm_mask, axis=2)
            cands.append(segm_mask)
    return cands


def calculate_distances(descriptor_insts, candidates, inst_order, num_inst, descriptor_type='gradient'):   
     # distances is a matrix num cands x num instances
    distances = np.full((candidates.shape[2], num_inst), np.inf)
    if descriptor_type == 'gradient':
        gradients = descriptor_insts
        for g in range(gradients.shape[2]):
            gradient = gradients[:, :, g]
            inst_idx = inst_order[g]
            for c in range(candidates.shape[2]):
                cand = candidates[:, :, c]
                distances[c, inst_idx] = np.mean(gradient[cand == 255])
    elif descriptor_type == 'com':
        com_insts = descriptor_insts
        for inst in range(len(com_insts)):
            com_inst = com_insts[inst]
            inst_idx = inst_order[inst]
            com_cands = calculate_descriptor(candidates, descriptor_type=descriptor_type)
            for c in range(len(com_cands)):
                com_cand = com_cands[c]
                distances[c, inst_idx] = math.sqrt( ((com_inst[0]-com_cand[0])**2)+((com_inst[1]-com_cand[1])**2) )
    else:
        print('Exception')
    return distances

def warp_ann(ann, flow, padding_mode='zeros'):
    """Warp an image or feature map with optical flow
    Args:
        image (Tensor): size (n, c, h, w)
        flow (Tensor): size (n, 2, h, w), values range from -1 to 1 (relevant to image width or height)
        padding_mode (str): 'zeros' or 'border'
    Returns:
        Tensor: warped image or feature map
    """
    # w/o this normalization, warping DOES NOT work
    _, h, w = flow.shape
    flow[0, :, :] /= w
    flow[1, :, :] /= h

    ann = np.expand_dims(ann, axis=0)
    # remove padding from optical flow
    if ann.shape[:1] != flow.shape[:1]:
        flow = flow[:, :ann.shape[1], :]
    assert ann.shape[1] == flow.shape[1]
    ann = np.expand_dims(ann, axis=0)
    flow = flow.unsqueeze(0).cpu()
    ann = torch.tensor(ann)

    n, _, h, w = ann.size()
    x_ = torch.arange(w).view(1, -1).expand(h, -1)
    y_ = torch.arange(h).view(-1, 1).expand(-1, w)
    grid = torch.stack([x_, y_], dim=0).float()
    grid = grid.unsqueeze(0).expand(n, -1, -1, -1)
    grid[:, 0, :, :] = 2 * grid[:, 0, :, :] / (w - 1) - 1
    grid[:, 1, :, :] = 2 * grid[:, 1, :, :] / (h - 1) - 1
    grid += 2 * flow
    grid = grid.permute(0, 2, 3, 1)

    warped_ann = F.grid_sample(ann.float(), grid, padding_mode=padding_mode)
    warped_ann = warped_ann.squeeze(dim=0)
    warped_ann = np.asarray(warped_ann).astype(np.uint8)
    warped_ann = warped_ann.transpose(1, 2, 0)
    return warped_ann


def warp_candidates(cands, flow):
    warped_cands = torch.zeros(cands.shape)
    for idx in range(cands.shape[0]):
        cand = cands[idx, :, :]
        if cand.sum() != 0:
            warped_cands[idx, :, :] = torch.from_numpy(warp_ann(cand, flow).squeeze())
    warped_cands[warped_cands > 0] = 255
    return warped_cands

def warp_bw_annotations(anns, flow):
    # anns is an io.imageCollection where each image is a binary annotation
    # flow is the optical flow for the image
    strel = np.zeros((3,3))
    strel[1,:] = 1; strel[:,1] = 1

    warped_anns = np.zeros((anns[0].shape[0], anns[0].shape[1], len(anns)))
    for idx, ann in enumerate(anns):
        if ann.sum() != 0:
            # handle sequence 7
            if len(ann.shape) > 2:
                ann = ann[:,:,0]
            # remove part annotations and borders
            ann = crop_image(ann > 0)
            warped_ann = warp_ann(ann, flow).squeeze()
            # fill holes in warped ann
            warped_ann = binary_fill_holes(warped_ann, structure=strel)
            warped_ann = uncrop_image(warped_ann.squeeze())
            warped_ann = warped_ann.astype(np.uint8)
            warped_anns[:, :, idx] = warped_ann
    return warped_anns


def calculate_jaccard_matrix(anns, cands):
    # make sure ann and cands are numpy arrays
    anns = np.asarray(anns).astype(np.float32)
    cands = np.asarray(cands).astype(np.float32)
    jaccard_matrix = np.zeros((cands.shape[2], anns.shape[2]))
    for c_idx in range(cands.shape[2]):
        cand = cands[:, :, c_idx]
        for a_idx in range(anns.shape[2]):   
            ann = anns[:, :, a_idx]
            inter, union = compute_mask_IU(ann, cand)
            jaccard_matrix[c_idx, a_idx] = inter/union
    return jaccard_matrix

def jaccard_matrix(anns, cands):
    # make sure ann and cands are numpy arrays
    anns = np.asarray(anns).astype(np.float32)
    cands = np.asarray(cands).astype(np.float32)
    jaccard_matrix = np.zeros((cands.shape[0], anns.shape[0]))
    for c_idx in range(cands.shape[0]):
        cand = cands[c_idx, :, :]
        for a_idx in range(anns.shape[0]):   
            ann = anns[a_idx, :, :]
            inter, union = compute_mask_IU(ann/255, cand/255)
            jaccard_matrix[c_idx, a_idx] = inter/union
    return jaccard_matrix

def match(candidates, warped_cands, min_iou=0.5):
    # calculate jaccard between all cands and anns
    matrix = jaccard_matrix(warped_cands, candidates)
    # TODO: what happens if intersection is zero for only pair?
    # calculate reciprocal pairings
    
    cand_min = np.argmax(matrix, axis=1)
    warped_cand_min = np.argmax(matrix, axis=0)
    warped_cand_idx = np.arange(0, warped_cands.shape[0])

    paired_anns = np.zeros(candidates.shape[0]) > 0
    paired_anns[warped_cand_min] = (cand_min[warped_cand_min] == warped_cand_idx)
    selected_warped_cands = cand_min[paired_anns]
    match = np.full(candidates.shape[0], np.inf)
    match[paired_anns] = selected_warped_cands
    for i, m in enumerate(match):
        if m != np.inf and matrix[i, int(m)] < min_iou:
            match[i] = np.inf

    return torch.from_numpy(match)

def match_candidates(candidates, warped_anns):

    # calculate jaccard between all cands and anns
    jaccard_matrix = calculate_jaccard_matrix(warped_anns, candidates)
    # TODO: what happens if intersection is zero for only pair?
    # calculate reciprocal pairings
    cand_min = np.argmax(jaccard_matrix, axis=1)
    ann_min = np.argmax(jaccard_matrix, axis=0)
    ann_idx = np.arange(0, warped_anns.shape[2])
    paired_anns = (cand_min[ann_min] == ann_idx)
    selected_anns = ann_idx[paired_anns]
    selected_cands = ann_min[selected_anns]
    warped_anns = warped_anns.astype(np.uint8)
    # assign reciprocal pairings, else maintain warped ann
    for i in range(len(selected_anns)):
        warped_anns[:,:,selected_anns[i]] = candidates[:, :, selected_cands[i]].squeeze()
    return warped_anns

def filter_for_videos(img_name, video_list):
    cated_list = " ".join(video_list)
    filtered_frames = re.findall(
    img_name + "_\d+.png", cated_list
    )
    return filtered_frames


def get_video_frames(prefix_name, video_names, num_frames):
    def split_by_numbers(x):
        r = re.compile('(\d+)')
        l = r.split(x)
        return [int(y) if y.isdigit() else y for y in l]
    # get other frames of current video
    frame_names = filter_for_videos(prefix_name, video_names) 
    frame_names = sorted(frame_names, key=split_by_numbers, reverse=True)
    # keep last num_frames 
    frame_names = frame_names[:num_frames]
    return frame_names


def split_anns(anns):
    inst_labels = np.unique(anns)
    # remove background from labels
    inst_labels = inst_labels[1:]
    split_anns = np.zeros((anns.shape[0], anns.shape[1], inst_labels[-1]+1))
    for lbl in inst_labels:
        split_anns[:, :, lbl] = (anns == lbl)
    return split_anns

def generate_ann(anns, frame, video_ann_dir):
    new_annotation = np.zeros((anns.shape[0], anns.shape[1]))
    for i in range(anns.shape[-1]):
        ann = anns[:, :, i].squeeze()
        new_annotation[ann == 1] = i+1
    filename = osp.join(video_ann_dir, 'ann_{}'.format(frame))
    io.imsave(filename, new_annotation.astype(np.uint8))
    return new_annotation

def load_anns(ann_file):
    # get anns of left eye
    ann = io.imread(ann_file)
    if ann.sum()!=0:
        # split anns into one class per channel
        ann = split_anns(ann)
    return ann

def load_all_bw_anns(filenames):
    anns = io.ImageCollection(filenames)
    return anns

def update_annotations(optical_flow, anns, frame, coco_ann_dir, coco_pred_dir, video_ann_dir):
    coco_anns = load_json(coco_ann_dir)
    coco_preds = load_json(coco_pred_dir)
    coco_name_list = [img["file_name"] for img in coco_anns["images"]]
    pred_img_ids = np.asarray([pred["image_id"] for pred in coco_preds])
    candidates = extract_candidates(frame, coco_anns, coco_preds,
                               coco_name_list, pred_img_ids)
    # stop propagating annotations if this frame has no cands
    if len(candidates) == 0:
        new_anns = np.zeros(anns.shape)
        return new_anns
    candidates = np.concatenate(candidates, axis=2)
    # choose new annotations according to IoU
    new_anns = match_candidates(candidates, anns)
    filename = osp.join(video_ann_dir, frame)
    io.imsave(filename, new_anns.astype(np.uint8))
    new_anns = split_anns(new_anns)
    return new_anns
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import warp_utils

H, W = 32, 64


def _cols(mask):
    cols = np.nonzero(np.asarray(mask).sum(0))[0]
    return cols.min(), cols.max()


def test_warp_candidates_does_not_modify_flow():
    cands = torch.zeros(2, H, W)
    cands[0, 10:20, 10:20] = 255
    cands[1, 10:20, 30:40] = 255
    flow = torch.zeros(2, H, W)
    flow[0] = 5.0  # constant 5 px horizontal flow
    flow_before = flow.clone()

    warped = warp_utils.warp_candidates(cands, flow)

    assert torch.equal(flow, flow_before)
    # every candidate is warped with the same flow, i.e. shifted by the same amount
    shift0 = _cols(cands[0])[0] - _cols(warped[0])[0]
    shift1 = _cols(cands[1])[0] - _cols(warped[1])[0]
    assert shift0 > 0
    assert shift0 == shift1
    for i in range(2):
        ref = warp_utils.warp_ann(cands[i], flow_before.clone()).squeeze() > 0
        assert np.array_equal(np.asarray(warped[i]) > 0, ref)


def test_match_keeps_reciprocal_pair_with_duplicate_argmax():
    cur = torch.zeros(1, H, W)
    cur[0, 0:10, 0:10] = 255
    warped = torch.zeros(2, H, W)
    warped[0, 0:10, 0:10] = 255  # IoU 1 with cur[0]
    warped[1, 0:10, 5:15] = 255  # IoU 1/3 with cur[0]
    assert warp_utils.match(cur, warped, min_iou=0.5).tolist() == [0.0]


def test_match_one_to_one_and_unmatched():
    cur = torch.zeros(3, H, W)
    cur[0, 0:10, 0:10] = 255
    cur[1, 0:10, 30:40] = 255
    cur[2, 20:30, 50:60] = 255  # nothing warped lands here
    warped = torch.zeros(2, H, W)
    warped[0, 0:10, 31:41] = 255  # best match of cur[1]
    warped[1, 0:10, 1:11] = 255  # best match of cur[0]
    assert warp_utils.match(cur, warped, min_iou=0.5).tolist() == [1.0, 0.0, np.inf]


def test_match_respects_min_iou():
    cur = torch.zeros(1, H, W)
    cur[0, 0:10, 0:10] = 255
    warped = torch.zeros(1, H, W)
    warped[0, 0:10, 5:15] = 255  # IoU 1/3
    assert warp_utils.match(cur, warped, min_iou=0.5).tolist() == [np.inf]
    assert warp_utils.match(cur, warped, min_iou=0.3).tolist() == [0.0]

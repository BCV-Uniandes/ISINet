import argparse
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import tools


class ToyDataset(object):
    def __init__(self, args, is_cropped, img_dir='images', prev_frames=7, nms=True, maskrcnn_inference=False):
        pass


def _parse(argv):
    module = types.ModuleType('toy_datasets')
    module.ToyDataset = ToyDataset
    parser = argparse.ArgumentParser()
    old_argv = sys.argv
    sys.argv = ['main.py'] + argv  # add_arguments_for_module calls parse_known_args() on sys.argv
    try:
        tools.add_arguments_for_module(parser, module, argument_for_class='inference_dataset', default='ToyDataset',
                                       skip_params=['is_cropped'], parameter_defaults={'nms': True})
        return parser.parse_args(argv)
    finally:
        sys.argv = old_argv


@pytest.mark.parametrize('value, expected', [('True', True), ('true', True), ('1', True),
                                             ('False', False), ('false', False), ('0', False)])
def test_str2bool(value, expected):
    assert tools.str2bool(value) is expected


def test_str2bool_rejects_garbage():
    with pytest.raises(argparse.ArgumentTypeError):
        tools.str2bool('maybe')


def test_bool_flags_parse_false():
    args = _parse(['--inference_dataset_nms', 'False', '--inference_dataset_maskrcnn_inference', 'False'])
    assert args.inference_dataset_nms is False  # from parameter_defaults
    assert args.inference_dataset_maskrcnn_inference is False  # from the class signature


def test_bool_flags_parse_true_and_defaults():
    args = _parse(['--inference_dataset_maskrcnn_inference', 'True'])
    assert args.inference_dataset_maskrcnn_inference is True
    assert args.inference_dataset_nms is True


def test_non_bool_types_unchanged():
    args = _parse(['--inference_dataset_prev_frames', '3', '--inference_dataset_img_dir', 'x'])
    assert args.inference_dataset_prev_frames == 3
    assert args.inference_dataset_img_dir == 'x'

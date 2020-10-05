import re
import os
import json
import glob
import os.path as osp
import numpy as np
from skimage import io
from copy import deepcopy
from shutil import copyfile

import argparse
import warnings
from tqdm import tqdm

import pdb

warnings.filterwarnings('ignore')

def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description='Data organization routine EndoVis 2018 dataset')
    parser.add_argument(
        "--data-dir",
        required=True,
        type=str,
        default="Endovis2018",
        help="path to the data",
    )
    parser.add_argument(
        "--save-dir",
        required=True,
        type=str,
        default="data2018",
        help="path to the save the organized dataset",
    )
    parser.add_argument(
        "--label-path",
        required=True,
        type=str,
        default="final_labels.json",
        help="path to the class definitions and labels to be used ",
    )
    parser.add_argument(
        '--cropped',
        required=False,
        default=False,
        action='store_true',
        help='Crop images')
    parser.add_argument(
        '--test',
        required=False,
        action="store_true",
        default=False,
        help='Organize test images')
    parser.add_argument(
        '--parts',
        required=False,
        default=False,
        action='store_true',
        help='include instrument parts into task')
    parser.add_argument(
        '--organs-suturing',
        required=False,
        default=False,
        action='store_true',
        help='include organs and suturing elements into task')
    parser.add_argument(
        '--stapler',
        required=False,
        default=False,
        action='store_true',
        help='include stapler instrument into task')
    return parser.parse_args()

args = parse_args()
print('Called with args:')
print(args)

data_folders = ['miccai_challenge_2018_release_1', 'miccai_challenge_release_2','miccai_challenge_release_3', 'miccai_challenge_release_4']
val_seqs = ['seq_2','seq_5','seq_9','seq_15']

if not osp.exists(args.save_dir):
    os.makedirs(args.save_dir)

with open(args.label_path) as file:
    labels = json.load(file)

labels = labels[1:] # remove background class

if not args.parts:
    labels = labels[0:14] # remove parts from dictionary

if not args.organs_suturing:
    labels = labels[0:8] # remove organs and suturing elements from dictionary

if not args.stapler:
    labels = labels[0:7] # remove stapler from dictionary

final_labels = deepcopy(labels)    
# get classes present for task
class_ids = []
for i in range(len(final_labels)):
    if 'color' in labels[i].keys():
        final_labels[i].pop('color')
    class_ids.append(final_labels[i]['classid'])
# save modified labels to dataset
with open(osp.join(args.save_dir, 'labels_2017.json'), 'w') as f:
    json.dump(final_labels, f)

def label_ann(mask, labels):
    h, w, _ = mask.shape
    labeled_ann = np.zeros((h, w), dtype=np.uint8)
    for i in range(len(labels)):
        if 'color' in labels[i].keys():
            color_code = labels[i]['color']
            temp = (mask[:,:,0] == color_code[0]) & (mask[:,:,1] == color_code[1]) & (mask[:,:,2] == color_code[2])
            labeled_ann[temp] = labels[i]['classid']
    return labeled_ann

def get_bw_masks(ann, final_name, class_ids):
    for c in class_ids:
        c = c['classid']
        # only create bw ann for non instrument classes
        # include suction and ultrasound instruments
        #if c >= 9 or c == 5 or c == 6:
        if c >= 11 or c == 7 or c == 8:
            bw_mask = (ann == c)
            if bw_mask.sum() > 0:
                bw_filename = final_name.replace('folder', 'binary_annotations')
                bw_filename = bw_filename + '_class{}_inst0.png'.format(c) 
                io.imsave(bw_filename, bw_mask*255)

def filter_for_frame(frame_name, ann_list):
    cated_list = ' '.join(ann_list)
    filtered_anns = re.findall("\w*{0}\w*.png".format(frame_name), cated_list)
    return filtered_anns

def get_class_num(name):
    class_num = re.search(r'class\d+', name).group()
    class_num = np.uint8(class_num.replace('class', ''))
    return class_num
    
##########
if __name__ == '__main__':
    if args.test:
        if not osp.exists(osp.join(args.save_dir, 'test')):
            os.makedirs(osp.join(args.save_dir, 'test', 'images'))
    else:
        if not osp.exists(osp.join(args.save_dir, 'train')):
            os.makedirs(osp.join(args.save_dir,'train','images'))
            os.makedirs(osp.join(args.save_dir,'train','annotations'))

            os.makedirs(osp.join(args.save_dir,'val','images'))
            os.makedirs(osp.join(args.save_dir,'val','annotations'))

            os.makedirs(osp.join(args.save_dir,'train','binary_annotations'))
            os.makedirs(osp.join(args.save_dir,'val','binary_annotations'))
        
    for data_f in tqdm(data_folders):
        inner_folders = os.listdir(osp.join(args.data_dir, data_f, ''))
        inner_folders.sort()
        for in_f in tqdm(inner_folders):
            if osp.isdir(osp.join(args.data_dir, data_f, in_f)):
                instr_label_folder = osp.join(args.data_dir, data_f, in_f, 'instrument_labels_2017')
                instr_ann_names = os.listdir(instr_label_folder)
                if args.test:
                    save_path = osp.join(args.save_dir, 'test')
                else:
                    if any(s in in_f for s in val_seqs): 
                        save_path = osp.join(args.save_dir, 'val')
                    else:
                        save_path = osp.join(args.save_dir, 'train')
                
                image_names = glob.glob(osp.join(args.data_dir, data_f, in_f, 'left_frames', '*.png'))
                for im_name in image_names:
                    basename = osp.basename(im_name)[:-4]
                    final_name = osp.join(save_path, 'folder', ''.join(in_f.split('_')) + '_' + basename)
                    destination = final_name.replace('folder','images') +'.png'
                    copyfile(im_name, destination)
                    if not args.test:
                        ann = io.imread(osp.join(args.data_dir, data_f, in_f, 'labels', basename + '.png'))
                        # create labeled annotation
                        h, w, _ = ann.shape
                        labeled_ann = np.zeros((h, w), dtype=np.uint8)
                        labeled_ann += label_ann(ann, labels)
                        # get and save binary masks
                        get_bw_masks(labeled_ann, final_name, final_labels)
                        # TODO: handle part annotations
                        # add instruments to labeled annotation
                        these_instr_anns = filter_for_frame(basename, instr_ann_names)
                        for instr in these_instr_anns:
                            source = osp.join(instr_label_folder, instr)
                            # label this instrument instance
                            class_num = get_class_num(instr)
                            if class_num in class_ids:
                                instr_im = io.imread(source)
                                # remove colored image
                                if instr_im.ndim == 3:
                                    instr_im = np.amax(instr_im, axis=2)
                                    instr_im[instr_im > 0] = 1
                                # TODO: check this
                                if np.max(instr_im) == 255:
                                    instr_im = np.uint8(instr_im / 255)
                                # save instrument instance to binary anns
                                new_name = '{}_{}'.format(''.join(in_f.split('_')), instr)
                                destination = osp.join(save_path, 'binary_annotations', new_name)
                                io.imsave(destination, instr_im.astype(np.uint8) * 255)
                                labeled_ann[instr_im > 0] = class_num # add instrument to label ann

                        # save labeled annotation
                        destination = final_name.replace('folder','annotations') +'.png'
                        io.imsave(destination, labeled_ann.astype(np.uint8))
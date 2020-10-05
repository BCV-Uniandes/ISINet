import numpy as np
import os
import os.path as osp
import glob
from skimage import io
import argparse
import warnings
from tqdm import tqdm

import pdb

warnings.filterwarnings("ignore")


def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description="Data organization routine EndoVis 2017 dataset")
    parser.add_argument(
        "--data-dir",
        required=True,
        type=str,
        default="Endovis2017/train",
        help="path to the data",
    )
    parser.add_argument(
        "--save-dir",
        required=True,
        type=str,
        default="data2017",
        help="path to the save the organized dataset",
    )
    parser.add_argument(
        "--cropped",
        required=False,
        action='store_true',
        default=False,
        help="Crop images",
    )
    parser.add_argument(
        "--test",
        required=False,
        action='store_true',
        default=False,
        help="Organize test images",
    )
    return parser.parse_args()


args = parse_args()
print("Called with args:")
print(args)

h, w = 1024, 1280
h_start, w_start = 28, 320

data_folders = ["instrument_dataset_" + str(i) for i in range(1, 9)]
"""
TernausNet 4-fold cross-val splits
folds = {0: [1, 3],
         1: [2, 5],
         2: [4, 8],
         3: [6, 7]}
"""
fold_seq_dict = {'1':'0', '3':'0',
                 '2':'1', '5':'1',
                 '4':'2', '8':'2',
                 '6':'3', '7':'3'}
if args.test:
    data_folders = ["instrument_dataset_" + str(i) for i in range(1, 11)]

names_dict = {
    1: "Bipolar Forceps",
    2: "Prograsp Forceps",
    3: "Large Needle Driver",
    4: "Vessel Sealer",
    5: "Grasping Retractor",
    6: "Monopolar Curved Scissors",
    7: "Other",
}


def get_cat_id(folder_name, names_dict):
    coincidence = [
        folder_name.find(name.replace(" ", "_"))
        for name in names_dict.values()
    ]
    coincidence_idx = np.where(np.array(coincidence) > -1)[0][0]
    cat_id = coincidence_idx + 1
    return cat_id


def get_binary_mask(mask_name):
    mask = io.imread(mask_name)
    if len(mask.shape) > 2:
        mask = mask[:, :, 0]
    bw_mask = mask > 0
    return bw_mask


def crop_image(image, h_start, w_start, h, w):
    image = image[h_start : h_start + h, w_start : w_start + w]
    return image


if __name__ == "__main__":
    if args.test:
        if not osp.exists(osp.join(args.save_dir, "test")):
            os.makedirs(osp.join(args.save_dir, "test", "images"))

    for data_f in tqdm(data_folders):
        if args.test:
            save_path = osp.join(args.save_dir, "test")
        else:
            fold_name = "fold{}".format(fold_seq_dict[data_f[-1]])
            save_path = osp.join(args.save_dir, fold_name)
            if not osp.exists(save_path):
                os.makedirs(osp.join(save_path, "images"))
                os.makedirs(osp.join(save_path, "annotations"))
                os.makedirs(osp.join(save_path, "binary_annotations"))
        image_names = glob.glob(
            osp.join(args.data_dir, data_f, "left_frames", "*.png")
        )
        for im_name in tqdm(image_names):
            num_inst = {n: 0 for n in range(1, len(names_dict) + 1)}
            base_name = osp.join(
                save_path,
                "folder",
                "seq" + data_f[-1] + "_" + osp.basename(im_name)[:-4],
            )
            # create empty mask w/ labels
            im = io.imread(im_name)
            im = crop_image(im, h_start, w_start, h, w)
            if not args.test:
                h, w, _ = np.shape(im)
                mask = np.zeros((h, w), dtype="uint8")
                # index mask with binary class masks
                inner_folders = os.listdir(
                    osp.join(args.data_dir, data_f, "ground_truth")
                )
                inner_folders.sort()
                for in_f in inner_folders:
                    # get folder cat_id and name
                    cat_id = get_cat_id(in_f, names_dict)
                    # get binary mask
                    mask_name = im_name.replace(
                        "left_frames", osp.join("ground_truth", in_f)
                    )
                    bw_mask = get_binary_mask(mask_name)
                    bw_mask = crop_image(bw_mask, h_start, w_start, h, w)
                    mask[bw_mask] = cat_id
                    # save binary_mask
                    if bw_mask.sum() > 0:
                        this_inst = num_inst[cat_id]
                        num_inst[cat_id] += 1
                        bw_filename = base_name.replace(
                            "folder", "binary_annotations"
                        ) + "_class{}_inst{}.png".format(cat_id, this_inst)
                        io.imsave(bw_filename, bw_mask.astype(np.uint8) * 255)
                # save mask
                destination = base_name.replace("folder", "images") + ".png"
                io.imsave(destination.replace("images", "annotations"), mask)
            # save image
            destination = base_name.replace("folder", "images") + ".png"
            io.imsave(destination, im)

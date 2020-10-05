# !/usr/bin/env python3
# Modified from https://github.com/waspinator/pycococreator/

import argparse
import datetime
import json
import os
import re
import fnmatch
from PIL import Image
import numpy as np
from pycococreatortools import pycococreatortools
from tqdm import tqdm



def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(
        description="Convert robotic segmentation dataset into COCO format"
    )

    parser.add_argument(
        "--image-dir",
        dest="image_dir",
        required=True,
        help="Complete path to images",
    )
    parser.add_argument(
        "--anns-dir",
        dest="anns_dir",
        required=True,
        help="Complete path to binary annotations",
    )
    parser.add_argument(
        "--save-dir",
        dest="save_dir",
        required=True,
        help="Complete path to save coco format annotations",
    )
    parser.add_argument(
        "--dataset",
        dest="dataset",
        required=True,
        help="Dataset name",
    )
    parser.add_argument(
        "--group-name",
        dest="group_name",
        required=True,
        help="Dataset split group name",
    )
    parser.add_argument(
        "--json-prefix",
        dest="json_prefix",
        required=True,
        help="Prefix for json filename",
    )
    return parser.parse_args()


args = parse_args()
print("Called with args:")
print(args)

# setup paths to data and annotations
IMAGE_DIR = args.image_dir
ANNOTATION_DIR = args.anns_dir
SAVE_DIR = args.save_dir
DATASET = args.dataset
GROUP_NAME = args.group_name

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

INFO = {
    "description": "Robotic Instrument Type Segmentation",
    "url": "",
    "version": GROUP_NAME,
    "year": DATASET,
    "contributor": "C.Gonzalez, L. Bravo-Sanchez, P. Arbelaez",
    "date_created": datetime.datetime.utcnow().isoformat(" "),
}

LICENSES = [{"id": 1, "name": "", "url": ""}]

CATEGORIES = [
    {"id": 1, "name": "Bipolar Forceps", "supercategory": "Instrument"},
    {"id": 2, "name": "Prograsp Forceps", "supercategory": "Instrument"},
    {"id": 3, "name": "Large Needle Driver", "supercategory": "Instrument"},
    {"id": 4, "name": "Vessel Sealer", "supercategory": "Instrument"},
    {"id": 5, "name": "Grasping Retractor", "supercategory": "Instrument"},
    {
        "id": 6,
        "name": "Monopolar Curved Scissors",
        "supercategory": "Instrument",
    },
    {"id": 7, "name": "Ultrasound Probe", "supercategory": "Instrument"},
    {"id": 8, "name": "Suction Instrument", "supercategory": "Instrument"},
    {"id": 9, "name": "Clip Applier", "supercategory": "Instrument"},
]

if DATASET == "2017":
    CATEGORIES = CATEGORIES[:7]
elif DATASET == "2018":
    CATEGORIES = CATEGORIES[:3] + CATEGORIES[5:]
    for i, c in enumerate(CATEGORIES):
        c['id'] = i + 1

def filter_for_jpeg(root, files):
    file_types = ["*.jpeg", "*.jpg"]
    file_types = r"|".join([fnmatch.translate(x) for x in file_types])
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]

    return files


def filter_for_png(root, files):
    file_types = ["*.png"]
    file_types = r"|".join([fnmatch.translate(x) for x in file_types])
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]

    return files


def filter_for_annotations(root, files, image_filename):
    file_types = ["*.png"]
    file_types = r"|".join([fnmatch.translate(x) for x in file_types])
    basename_no_extension = os.path.splitext(os.path.basename(image_filename))[
        0
    ]
    file_name_prefix = basename_no_extension + ".*"
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    files = [
        f
        for f in files
        if re.match(file_name_prefix, os.path.splitext(os.path.basename(f))[0])
    ]

    return files


def main():

    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": [],
    }

    image_id = 1
    segmentation_id = 1
    
    # filter for jpeg images
    for root, _, files in os.walk(IMAGE_DIR):
        image_files = filter_for_png(root, files)
        image_files.sort()  # ensure order

        # go through each image
        for image_filename in tqdm(image_files):
            image = Image.open(image_filename)
            image_info = pycococreatortools.create_image_info(
                image_id, os.path.basename(image_filename), image.size
            )
            coco_output["images"].append(image_info)

            # filter for associated png annotations
            for root, _, files in os.walk(ANNOTATION_DIR):
                annotation_files = filter_for_annotations(
                    root, files, image_filename
                )
                # go through each associated annotation
                for annotation_filename in annotation_files:
                    class_id = int(re.search(r'\d(?=_inst\d.png)', annotation_filename).group())

                    category_info = {
                        "id": class_id,
                        "is_crowd": "crowd" in image_filename,
                    }
                    binary_mask = np.asarray(
                        Image.open(annotation_filename)
                    ).astype(np.uint8)

                    annotation_info = pycococreatortools.create_annotation_info(
                        segmentation_id,
                        image_id,
                        category_info,
                        binary_mask,
                        image.size,
                        tolerance=2,
                    )

                    if annotation_info is not None:
                        coco_output["annotations"].append(annotation_info)

                    segmentation_id = segmentation_id + 1

            image_id = image_id + 1

    with open(
        "{0}/{1}_{2}.json".format(SAVE_DIR, args.json_prefix, GROUP_NAME), "w"
    ) as output_json_file:
        json.dump(coco_output, output_json_file, indent=4)


if __name__ == "__main__":
    main()

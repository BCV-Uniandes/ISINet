# matplotlib inline
import matplotlib
matplotlib.use("Agg")
import argparse
from pycocotools.coco import COCO
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import os as os

# turn off plotting visual
plt.ioff()


def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(
        description="Separate annotations into binaries"
    )
    parser.add_argument("--image_dir", help="Complete path to images")
    parser.add_argument(
        "--anns_file", help="Complete path to coco annotation file"
    )
    parser.add_argument(
        "--save_dir",
        help="Complete path to save visualizations of coco format annotations",
    )
    parser.add_argument(
        "--save_freq",
        dest="save_freq",
        required=False,
        default=1,
        type=int,
        help="Complete path to save coco format annotations",
    )
    return parser.parse_args()


# Show all images in set with annotations of all categories

args = parse_args()
print("Called with args:")
print(args)
image_directory = args.image_dir
annotation_file = args.anns_file
save_directory = args.save_dir

if not os.path.exists(save_directory):
    os.makedirs(save_directory)

example_coco = COCO(annotation_file)

categories = example_coco.loadCats(example_coco.getCatIds())
category_names = [category["name"] for category in categories]
category_names = set([category["supercategory"] for category in categories])

category_ids = example_coco.getCatIds(catNms=["clover"])
image_ids = example_coco.getImgIds(catIds=category_ids)

for current_im in image_ids:
    if current_im % args.save_freq == 0:
        image_data = example_coco.loadImgs(current_im)[0]
        # load and display instance annotations
        image = io.imread(
            os.path.join(image_directory, image_data["file_name"])
        )
        plt.imshow(image)
        plt.axis("off")
        pylab.rcParams["figure.figsize"] = (8.0, 10.0)
        annotation_ids = example_coco.getAnnIds(
            imgIds=image_data["id"], catIds=category_ids, iscrowd=None
        )
        annotations = example_coco.loadAnns(annotation_ids)
        example_coco.showAnns(annotations)
        plt.savefig(
            os.path.join(save_directory, image_data["file_name"]),
            bbox_inches="tight",
        )

        print("Done saving image: " + str(current_im))
        plt.clf()
        plt.cla()
        plt.close()

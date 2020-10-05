# Datasets
Here we describe the steps for using the Endoscopic Vision 2017 [1] and 2018 [2] for instrument-type segmentation.

## Downloading the data
1. Download the 2017 dataset from the challenge webpage [here](https://endovissub2017-roboticinstrumentsegmentation.grand-challenge.org).
2. Download the 2018 dataset with our instrument-type annotations [here](https://biomedicalcomputervision.uniandes.edu.co/index.php/research?id=44). 

## Organizing the data
Organize the data of both datasets into the appropriate splits. Note for training on all the data, use the 2018 labels compatible with the 2017 dataset (final_labels_2017.json).
```
python organize2017.py --data-dir /path/to/original/data \
--save-dir /path/to/save/organized/data --cropped
python organize2018.py --data-dir /path/to/data/with/our/annotations \
--save-dir /path/to/save/organized/data --label-path /path/to/labels.json
```

Afterwards, convert the dataset to the MS-COCO format.  Required for the temporal consistency module and using our pre-trained weights that employ the Mask R-CNN Detectron v.1 code.
```
python robotseg_to_coco.py --image-dir /path/to/images \
    --anns-dir /path/to/binary_annotations --save-dir /path/to/save/coco/annotations \
    --dataset <dataset_name> --group-name <dataset_split> \
    --json-prefix <prefix_for_json_file>
```

## References
[1] Allan, M., Shvets, A., Kurmann, T., Zhang, Z., Duggal, R., Su, Y.H., , et al.: 2017 robotic instrument segmentation challenge. arXiv preprint arXiv:1902.06426 (2019)
[2] Allan, M., Kondo, S., Bodenstedt, S., Leger, S., Kadkhodamohammadi, R., Luengo, I., Fuentes, F., Flouty, E., Mohammed, A., Pedersen, M., et al.: 2018 robotic scene segmentation challenge. arXiv preprint arXiv:2001.11190 (2020)




# ISINet

This is the Pytorch implementation of [ISINet: An Instance-Based Approach for Surgical Instrument Segmentation](https://arxiv.org/abs/2007.05533) published at [MICCAI2020](https://www.miccai2020.org/en/).

## Installation
Requirements:
- Python >= 3.6
- Pytorch == 1.4
- numpy 
- scikit-image 
- tqdm 
- scipy == 1.1
- [flownet2](https://github.com/NVIDIA/flownet2-pytorch)
- [Detectron v.1](https://github.com/facebookresearch/maskrcnn-benchmark) (for using our pre-trained weights)

## Pre-trained Weights
Pre-trained weights are publicly available on the [project page](https://biomedicalcomputervision.uniandes.edu.co/index.php/research?id=44).

## Additional Annotations EndoVis 2018 Dataset
Additional annotations for the EndoVis 2018 Dataset are publicly available on the [project page](https://biomedicalcomputervision.uniandes.edu.co/index.php/research?id=44).

## Data Preparation
Check the instructions detailed in [data/README.md](data/README.md)
## Perform Inference
````
python -W ignore main.py --inference --model FlowNet2  --batch_size batch_size --number_workers num_workers \
--inference_dataset RobotsegTrackerDataset \ --inference_dataset_img_dir /path/to/images \ --inference_batch_size batch_size \
  --inference_dataset_coco_ann_path /path/to/coco/annotations/file.json \
  --inference_dataset_segm_path /path/to/mask-rcnn/inference/segm.json \
  --inference_dataset_ann_dir /path/to/annotations \
  --inference_dataset_cand_dir /path/to/save/candidates \ --inference_dataset_nms 'True' \
  --save /path/to/save/predictions \
  --inference_dataset_dataset '2017' or '2018' \
  --inference_dataset_maskrcnn_inference 'True' \
  --assignment_strategy 'weighted_mode' \ --inference_dataset_prev_frames 7 \
  --threshold 0.0 for 2017 and 0.5 for 2018 \
  --resume /path/to/flownet/checkpoint --num-classes number_of_classes
````

## Reference
If you found our work useful in your research, please use the following BibTeX entry for citation:

````
@article{ISINet2020,
  title={ISINet: An Instance-Based Approach for Surgical Instrument Segmentation},
  author={Cristina Gonz{\'a}lez and Laura Bravo-S{\'a}nchez and Pablo Arbelaez},
  journal={arXiv preprint arXiv:2007.05533},
  year={2020}
}
````

## Acknowledgements
Our code is build upon [FlowNet2](https://github.com/NVIDIA/flownet2-pytorch), we thank the authors for their contributions to the community.


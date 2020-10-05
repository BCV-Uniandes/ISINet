# coco simple G2
#CUDA_VISIBLE_DEVICES=0 python tools/train_net.py --config-file "configs/robotseg_e2e_mask_rcnn_R_50_FPN_1x_simple_g2.yaml" OUTPUT_DIR /media/SSD1/candidate-experiments/R-50-FPN/inst-simple-coco/group2

#coco class G1
#CUDA_VISIBLE_DEVICES=0 python tools/train_net.py --config-file "configs/robotseg_e2e_mask_rcnn_R_50_FPN_1x_class_g1.yaml" OUTPUT_DIR /media/SSD1/candidate-experiments/R-50-FPN/inst-class-coco-cropped/group1

#coco class 50_50
#CUDA_VISIBLE_DEVICES=3 python tools/train_net.py --config-file "configs/robotseg_e2e_mask_rcnn_R_50_FPN_1x_class_g1_50_50.yaml" OUTPUT_DIR /media/SSD1/candidate-experiments/R-50-FPN/inst-class-coco-cropped/50_50/group1_50_50


#coco test split
#CUDA_VISIBLE_DEVICES=3 python tools/train_net.py --config-file "configs/robotseg_e2e_mask_rcnn_R_50_FPN_1x_class_test_split.yaml" OUTPUT_DIR /media/SSD1/candidate-experiments/R-50-FPN/inst-class-coco-cropped/test_split/

#CUDA_VISIBLE_DEVICES=2 python tools/train_net.py --config-file "configs/robotseg_e2e_mask_rcnn_R_50_FPN_1x_class_g2_50_50.yaml" OUTPUT_DIR /media/SSD1/candidate-experiments/R-50-FPN/inst-class-coco-cropped/50_50/group2_50_50

# coco class fold 0
#CUDA_VISIBLE_DEVICES=0 python -W ignore tools/train_net.py --config-file "configs/robotseg_e2e_mask_rcnn_R_50_FPN_1x_class_fold0.yaml" OUTPUT_DIR /media/SSD2/candidate-experiments/R-50-FPN/inst-class-coco-cropped/4_folds/fold0

# coco class fold 1
# CUDA_VISIBLE_DEVICES=1 python -W ignore tools/train_net.py --config-file "configs/robotseg_e2e_mask_rcnn_R_50_FPN_1x_class_fold1.yaml" OUTPUT_DIR /media/SSD2/candidate-experiments/R-50-FPN/inst-class-coco-cropped/4_folds/fold1

# coco class fold 2
# CUDA_VISIBLE_DEVICES=1 python -W ignore tools/train_net.py --config-file "configs/robotseg_e2e_mask_rcnn_R_50_FPN_1x_class_fold2.yaml" OUTPUT_DIR /media/SSD2/candidate-experiments/R-50-FPN/inst-class-coco-cropped/4_folds/fold2

# coco class fold 3
#CUDA_VISIBLE_DEVICES=3 python -W ignore tools/train_net.py --config-file "configs/robotseg_e2e_mask_rcnn_R_50_FPN_1x_class_fold3.yaml" OUTPUT_DIR /media/SSD2/candidate-experiments/R-50-FPN/inst-class-coco-cropped/4_folds/fold3

# coco class comp split
# CUDA_VISIBLE_DEVICES=0 python -W ignore tools/train_net.py --config-file "configs/robotseg_e2e_mask_rcnn_R_50_FPN_1x_class_comp_split.yaml" OUTPUT_DIR /media/SSD1/comp-fold

# coco class comp split
# CUDA_VISIBLE_DEVICES=0 python -W ignore tools/train_net.py --config-file "configs/robotseg_e2e_mask_rcnn_R_50_FPN_1x_class_all.yaml" OUTPUT_DIR /media/SSD1/candidate-experiments/R-50-FPN/inst-class-coco-cropped/all

# coco class 10 percent split
#CUDA_VISIBLE_DEVICES=1 python -W ignore tools/train_net.py --config-file "configs/robotseg_e2e_mask_rcnn_R_50_FPN_1x_class_10percent.yaml" OUTPUT_DIR /media/SSD1/candidate-experiments/R-50-FPN/inst-class-coco-cropped/10percent

# coco class 80-20 split
#CUDA_VISIBLE_DEVICES=1 python -W ignore tools/train_net.py --config-file "configs/robotseg_e2e_mask_rcnn_R_50_FPN_1x_class_80_20.yaml" OUTPUT_DIR /media/SSD1/candidate-experiments/R-50-FPN/inst-class-coco-cropped/80_20

# coco two train split 10_?_?
#CUDA_VISIBLE_DEVICES=0 python -W ignore tools/train_net.py --config-file "configs/robotseg_e2e_mask_rcnn_R_50_FPN_1x_class_10_45_45.yaml" OUTPUT_DIR /media/SSD2/candidate-experiments/R-50-FPN/inst-class-coco-cropped/10_45_45/train2

# coco class 2018+fold 0
CUDA_VISIBLE_DEVICES=2 python -W ignore tools/train_net.py --config-file "configs/robotseg_e2e_mask_rcnn_R_50_FPN_1x_class_fold0_2018.yaml" OUTPUT_DIR /media/SSD1/candidate-experiments/R-50-FPN/inst-class-coco/4_folds/2018+fold3

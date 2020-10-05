# 2017 4-fold inference
# SAVE_DIR="/media/SSD7/prueba_ISINet/ISINet/atemporal_inference2/2017/fold3"
# CONFIG_DIR="configs/robotseg_e2e_mask_rcnn_R_50_FPN_1x_class_fold3.yaml"
# WEIGHT_DIR="/home/labravo/ISINet/ISINet/best_weights/2017/fold3/model_0008112.pth"
# CUDA_VISIBLE_DEVICES=3 python tools/test_net.py --config-file $CONFIG_DIR OUTPUT_DIR $SAVE_DIR MODEL.WEIGHT $WEIGHT_DIR

# 2017 aug 4-fold inference
# SAVE_DIR="/media/SSD7/prueba_ISINet/ISINet/atemporal_inference2/2017_aug/fold3"
# CONFIG_DIR="configs/robotseg_e2e_mask_rcnn_R_50_FPN_1x_class_aug_fold3.yaml"
# WEIGHT_DIR="/home/labravo/ISINet/ISINet/best_weights/2017_aug/fold3/model_0011648.pth"
# CUDA_VISIBLE_DEVICES=3 python tools/test_net.py --config-file $CONFIG_DIR OUTPUT_DIR $SAVE_DIR MODEL.WEIGHT $WEIGHT_DIR

# 2018 inference
SAVE_DIR="/media/SSD7/prueba_ISINet/ISINet/atemporal_inference2/2018"
CONFIG_DIR="configs/robotseg_e2e_mask_rcnn_R_50_FPN_1x_class_2018.yaml"
WEIGHT_DIR="/home/labravo/ISINet/ISINet/best_weights/2018/model_0011890.pth"
CUDA_VISIBLE_DEVICES=3 python tools/test_net.py --config-file $CONFIG_DIR OUTPUT_DIR $SAVE_DIR MODEL.WEIGHT $WEIGHT_DIR

# 2018 aug inference
# SAVE_DIR="/media/SSD7/prueba_ISINet/ISINet/atemporal_inference2/2018_aug"
# CONFIG_DIR="configs/robotseg_e2e_mask_rcnn_R_50_FPN_1x_class_2018_aug.yaml"
# WEIGHT_DIR="/home/labravo/ISINet/ISINet/best_weights/2018_aug/model_0003096.pth"
# CUDA_VISIBLE_DEVICES=3 python tools/test_net.py --config-file $CONFIG_DIR OUTPUT_DIR $SAVE_DIR MODEL.WEIGHT $WEIGHT_DIR
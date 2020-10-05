#!/bin/bash
SPLIT='train'
DATA_PATH='/media/SSD2/'$SPLIT
BATCH_SIZE=1 #3
WORKERS=0
export CUDA_VISIBLE_DEVICES=0

# declare -a group_names=('instrument_dataset_1' 'instrument_dataset_2' 'instrument_dataset_3' 'instrument_dataset_4' 'instrument_dataset_5' 'instrument_dataset_6' 'instrument_dataset_7' 'instrument_dataset_8')
declare -a group_names=('instrument_dataset_1')

for group in "${group_names[@]}"
do
   LEFT_PATH=$DATA_PATH'/'$group'/left_frames'
   RIGHT_PATH=$DATA_PATH'/'$group'/right_frames'
   BW_PATH=$DATA_PATH'/'$group'/ground_truth'
   WEIGHTS_PATH='/media/SSD2/candidate-experiments/pretrained_weights/flownet2/'
   WEIGHTS_NAME='FlowNet2_checkpoint.pth.tar'
   WARPED_ANN_DIR=$DATA_PATH'/'$group'/right_ground_truth'
   COCO_DIR=$DATA_PATH'/'$group'/coco_anns.json'
   SEGM_DIR=$DATA_PATH'/'$group'/segm.json'

   python -W ignore main.py --inference --model FlowNet2 \
   --inference_dataset EndoVis2017Dataset --inference_dataset_left_images $LEFT_PATH \
   --inference_dataset_right_images $RIGHT_PATH --inference_batch_size $BATCH_SIZE \
   --inference_dataset_coco_ann_dir $COCO_DIR --inference_dataset_segm_dir $SEGM_DIR \
   --resume $WEIGHTS_PATH$WEIGHTS_NAME --skip_training --skip_validation \
   --save_dir $WARPED_ANN_DIR --ann_dir $BW_PATH --num_workers $WORKERS
done
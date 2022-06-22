DISP_NET=checkpoints/UpdatedModels/resnet50_depth_256/dispnet_model_best.pth.tar
# DISP_NET=checkpoints/resnet50_depth_256/dispnet_model_best.pth.tar

DATA_ROOT=/home/tuannghust/Data/Inference
RESULTS_DIR=/home/tuannghust/Data/Inference/results/test

# test
python test_disp.py --resnet-layers 50 --img-height 352 --img-width 640 \
--pretrained-dispnet $DISP_NET --dataset-dir $DATA_ROOT/color \
--output-dir $RESULTS_DIR

# evaluate
python eval_depth.py \
--dataset kitti \
--pred_depth=$RESULTS_DIR/predictions.npy \
--gt_depth=$DATA_ROOT/depth
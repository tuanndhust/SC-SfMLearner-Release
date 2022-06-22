DATASET_DIR=/home/tuannghust/Duong_proc_test/sequences/
OUTPUT_DIR=hex_square/

POSE_NET=/home/tuannghust/SC-SfMLearner-Release/checkpoints/resnet50_pose_123/06-05-16:20/exp_pose_model_best.pth.tar

python test_vo.py \
--img-height 352 --img-width 640 \
--sequence 09 \
--pretrained-posenet $POSE_NET --dataset-dir $DATASET_DIR --output-dir $OUTPUT_DIR

python test_vo.py \
--img-height 352 --img-width 640 \
--sequence 10 \
--pretrained-posenet $POSE_NET --dataset-dir $DATASET_DIR --output-dir $OUTPUT_DIR

python ./kitti_eval/eval_odom.py --result=$OUTPUT_DIR --align='7dof'
DATA_ROOT=/home/tuannghust/Data
TRAIN_SET=$DATA_ROOT/kitti_vo_256
python train.py $TRAIN_SET \
--resnet-layers 50 \
--num-scales 1 \
-b2 -s0.1 -c0.5 --epoch-size 1000 --sequence-length 5 \
--with-ssim 1 \
--with-mask 1 \
--with-auto-mask 1 \
--with-pretrain 1 \
--log-output \
--name resnet50_pose_256
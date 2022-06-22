TRAIN_SET=/home/tuannghust/new_square
python train.py $TRAIN_SET \
--resnet-layers 50 \
--num-scales 1 \
-b2 -s0.1 -c0.5 --epoch-size 0 --epochs 100 --sequence-length 5 \
--with-ssim 1 \
--with-mask 1 \
--with-auto-mask 1 \
--with-pretrain 1 \
--log-output \
--name resnet50_pose_123
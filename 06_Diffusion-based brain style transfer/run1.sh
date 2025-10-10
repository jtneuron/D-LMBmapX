# sobel ksize =3
MODEL_FLAGS="--image_size 128 --num_channels 128 --num_res_blocks 3 --in_channels 2 --resolution 128 128"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear"
TRAIN_FLAGS="--lr 1e-5 --batch_size 4 --gpu 2  --log_interval 100 --max_save_num 3 --lr_anneal_steps 500000
 --logdir output/diffusion/task2/BraTS/rm/T2/train3"
DATA_FLAGS="--data_dir datasets/data_config/BraTs_t2_train.json"
VAL_FLAGS="--val_data_dir datasets/data_config/BraTs_t2_test.json
--val_batch_size 4
--val_data_num 16
--val_interval 10000"

mkdir -p ${TRAIN_FLAGS##*logdir} ; cp $(pwd)"/"$0 ${TRAIN_FLAGS##*logdir}
python scripts/image_train_sr3_target.py $DATA_FLAGS  $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS $VAL_FLAGS\
  --save_interval 10000
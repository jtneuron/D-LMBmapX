# sobel ksize =3
MODEL_FLAGS="
--image_size 128
--num_channels 128
--num_res_blocks 3
--in_channels 2
--resolution 224 160
"

DIFFUSION_FLAGS="
--diffusion_steps 1000
--noise_schedule linear
"

TRAIN_FLAGS="
--lr 1e-5
--batch_size 2
--gpu 0
--log_interval 100
--max_save_num 2
--logdir output/diffusion/ex02_sdc/allen/train05
"

DATA_FLAGS="
--data_dir datasets/data_config/allen_s_n_gan_train.json
"

VAL_FLAGS="
--val_data_dir datasets/data_config/allen_s_n_gan_test.json
--val_batch_size 16
--val_data_num 16
--val_interval 10000
"

mkdir -p ${TRAIN_FLAGS##*logdir}
cp $(pwd)"/"$0 ${TRAIN_FLAGS##*logdir}
python scripts/image_train_sr3_sdc.py $DATA_FLAGS $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS $VAL_FLAGS --save_interval 10000

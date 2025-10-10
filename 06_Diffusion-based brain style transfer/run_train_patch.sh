# sobel ksize =3
MODEL_FLAGS="
  --image_size 128
  --num_channels 128
  --num_res_blocks 3
  --in_channels 2
  --resolution 448 320
  "

DIFFUSION_FLAGS="
  --diffusion_steps 1500
  --noise_schedule linear
  "

TRAIN_FLAGS="
  --lr 1e-5
  --batch_size 1
  --gpu 1
  --log_interval 100
  --max_save_num 3
  --logdir output/diffusion/ex08_segbranch_patch/lsfm_512_448_320/train01_sdc
  "

DATA_FLAGS="
  --data_dir datasets/data_config/lsfm_n_gan_train_6.json
  --patch_size 224 160
  --stride 160
  "

VAL_FLAGS="
  --val_data_dir datasets/test_data_config/lsfm-allen.json
  --val_batch_size 1
  --val_data_num 8
  --val_interval 10000
  "

mkdir -p ${TRAIN_FLAGS##*logdir}
cp $(pwd)"/"$0 ${TRAIN_FLAGS##*logdir}
python scripts/image_train_sr3_segbranch_patch.py $DATA_FLAGS $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS $VAL_FLAGS --save_interval 10000 > train_1.log 2>&1
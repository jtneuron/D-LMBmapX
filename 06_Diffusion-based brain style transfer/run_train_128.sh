# sobel ksize =3
MODEL_FLAGS="
  --image_size 256
  --num_channels 128
  --num_res_blocks 3
  --in_channels 2
  --attention_resolutions 16,8
  --resolution 224 160
"

DIFFUSION_FLAGS="
  --diffusion_steps 1000
  --noise_schedule linear
  "

TRAIN_FLAGS="
  --lr 1e-5
  --batch_size 6
  --gpu 1
  --log_interval 1000
  --max_save_num 3
  --logdir ex_output/diffusion/ex07_segbranch_a/lsfm_224_160/train01_image_size_256
  "

DATA_FLAGS="
  --data_dir datasets/data_config_a/lsfm_6_train.json
  --type x
  "

VAL_FLAGS="
  --val_data_dir datasets/data_config_a_test/lsfm_3_test.json
  --val_type x
  --val_batch_size 64
  --val_data_num 256
  --val_interval 10000
  --save_interval 10000
  "

mkdir -p ${TRAIN_FLAGS##*logdir}
cp $(pwd)"/"$0 ${TRAIN_FLAGS##*logdir}
python scripts/image_train_sr3_segbranch.py $DATA_FLAGS $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS $VAL_FLAGS > train.log 2>&1
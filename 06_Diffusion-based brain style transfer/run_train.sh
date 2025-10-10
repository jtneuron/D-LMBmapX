# conda install mpi4py
# pip install opencv-contrib-python
# pip install PyWavelets
# pip install einops
# pip install scikit-image
# pip install pytorch_msssim

# sobel ksize =3
MODEL_FLAGS="
  --image_size 256
  --num_channels 256
  --num_res_blocks 2
  --in_channels 2
  --attention_resolutions 32,16,8
  --resolution 256 256
  "

DIFFUSION_FLAGS="
  --diffusion_steps 1000
  --noise_schedule linear
  "

TRAIN_FLAGS="
  --lr 1e-5
  --batch_size 1
  --gpu 0
  --log_interval 1000
  --max_save_num 1
  --logdir output/diffusion/ex07_segbranch_s/P28_avg_256_256/train00_image_size_256
  "

DATA_FLAGS="
  --data_dir datasets/data_config_s/P28_avg_train.json
  --type avg
  "

VAL_FLAGS="
  --val_data_dir datasets/data_config_s_test/P28_avg_test.json
  --val_type avg
  --val_batch_size 8
  --val_data_num 32
  --val_interval 10000
  --save_interval 10000
  "

mkdir -p ${TRAIN_FLAGS##*logdir}
cp $(pwd)"/"$0 ${TRAIN_FLAGS##*logdir}
python scripts/image_train_sr3_segbranch.py $DATA_FLAGS $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS $VAL_FLAGS > train.log 2>&1

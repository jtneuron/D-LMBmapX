# sobel ksize =3
MODEL_FLAGS="
   --image_size 128
   --num_channels 128
   --num_res_blocks 3
   --in_channels 2
   --resolution 896 560
   --use_checkpoint True
"
DIFFUSION_FLAGS="
   --diffusion_steps 1000
   --noise_schedule linear
"
TRAIN_FLAGS="
   --lr 1e-5
   --batch_size 1
   --gpu 0
   --log_interval 100
   --max_save_num 3
   --logdir output/diffusion/ex02_segbranch_s/P28_avg/train00_896_560_continue
   --model_path output/diffusion/ex02_segbranch_s/P28_avg/train00_896_560/ema_0.9999_240000.pt
"
DATA_FLAGS="
  --data_dir datasets/data_config_s/P28_avg_all.json
"
VAL_FLAGS="
  --val_data_dir datasets/data_config_s/P28_dev_40.json
  --val_batch_size 8
  --val_data_num 8
  --val_interval 10000
"

mkdir -p ${TRAIN_FLAGS##*logdir}
cp $(pwd)"/"$0 ${TRAIN_FLAGS##*logdir}
python scripts/image_train_sr3_sdc.py $DATA_FLAGS $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS $VAL_FLAGS --save_interval 10000

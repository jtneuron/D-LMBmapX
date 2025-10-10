MODEL_FLAGS="--image_size 128 --num_channels 128 --num_res_blocks 3 --in_channels 2 --resolution 240 240"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear"
SAMPLE_FLAGS="--gpu 2
--model_path output/diffusion/task2/BraTS/T2/train2/ema_0.9999_400000.pt
--data_dir datasets/data_config/BraTs_t2_test.json
--continous False
--logdir output/diffusion/task2/BraTS/T2/train2/test3"
mkdir -p ${SAMPLE_FLAGS##*logdir} ; cp $(pwd)"/"$0 ${SAMPLE_FLAGS##*logdir}
python scripts/image_sample_sr3_target.py $MODEL_FLAGS $DIFFUSION_FLAGS $SAMPLE_FLAGS  \
 --num_samples -1 --batch_size 64
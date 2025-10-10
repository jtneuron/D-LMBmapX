MODEL_FLAGS="
--image_size 128
--num_channels 128
--num_res_blocks 3
--in_channels 3
--resolution 224 160"

DIFFUSION_FLAGS="
--diffusion_steps 1000
--noise_schedule linear"

SAMPLE_FLAGS="
--gpu 0
--model_path output/diffusion/ex03_edge/P28/train01/ema_0.9999_620000.pt
--data_dir datasets/data_config/allen_s_n_gan_all.json
--continous False
--logdir output/translation/ex03_edge/allen_to_P28/sample01"

mkdir -p ${SAMPLE_FLAGS##*logdir}
cp $(pwd)"/"$0 ${SAMPLE_FLAGS##*logdir}

python scripts/image_sample_sr3_edge.py $MODEL_FLAGS $DIFFUSION_FLAGS $SAMPLE_FLAGS \
  --num_samples -1 --batch_size 1

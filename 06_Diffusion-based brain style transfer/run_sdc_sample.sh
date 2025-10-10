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

SAMPLE_FLAGS="
--gpu 0
--model_path output/diffusion/ex02_sdc/lsfm/train06/ema_0.9999_670000.pt
--data_dir datasets/data_config/fvbex_brain_m_re_test_all.json
--continous False
--logdir output/translation/ex02_sdc/allen_to_lsfm/sample06
"

mkdir -p ${SAMPLE_FLAGS##*logdir}
cp $(pwd)"/"$0 ${SAMPLE_FLAGS##*logdir}

python scripts/image_sample_sr3_sdc.py $MODEL_FLAGS $DIFFUSION_FLAGS $SAMPLE_FLAGS \
  --num_samples -1 --batch_size 1

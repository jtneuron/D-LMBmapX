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
--gpu 1
--model_path output/diffusion/ex07_segbranch/allen_224_160/train00/ema_0.9999_700000.pt
--data_dir datasets/data_config/lsfm_n_gan_test_all.json
--continous False
--logdir output/translation/test/lsfm_to_allen/sample00
"
mkdir -p ${SAMPLE_FLAGS##*logdir}
cp $(pwd)"/"$0 ${SAMPLE_FLAGS##*logdir}
python scripts/image_sample_sr3_segbranch.py $MODEL_FLAGS $DIFFUSION_FLAGS $SAMPLE_FLAGS --num_samples -1 --batch_size 1

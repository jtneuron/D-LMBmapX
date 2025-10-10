# --timestep_respacing ddim250
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
--batch_size 64
--source_path ex_output/diffusion/ex01_s/lsfm_224_160/train00_image_size_128/best_0.9999_180000_0.910923957824707.pt
--target_path ex_output/diffusion/ex01_s/allen_224_160/train00_image_size_128/best_0.9999_080000_0.9655824899673462.pt
--data_dir datasets/data_config_s_test/lsfm_all_test.json
--val_type_1 lsfm
--val_type_2 allen
--continous False
--logdir ex_output/translation/ex01_s/lsfm_to_allen_to_lsfm/sample00_image_size_224
"

mkdir -p ${SAMPLE_FLAGS##*logdir}
cp $(pwd)"/"$0 ${SAMPLE_FLAGS##*logdir}
python scripts/image_sample_sr3_cycle.py $MODEL_FLAGS $DIFFUSION_FLAGS $SAMPLE_FLAGS --num_samples -1

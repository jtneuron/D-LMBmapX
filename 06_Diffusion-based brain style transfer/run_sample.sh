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

SAMPLE_FLAGS="
--gpu 0
--batch_size 64
--model_path ex_output/diffusion/ex07_segbranch_c/allen_224_160/train00_image_size_256/best_0.9999_580000_0.9832178354263306.pt
--data_dir datasets/data_config_c_test/mri_all_test.json
--val_type mri
--continous False
--logdir ex_output/translation/ex07_segbranch_c/mri_to_allen/sample00_image_size_224
"

mkdir -p ${SAMPLE_FLAGS##*logdir}
cp $(pwd)"/"$0 ${SAMPLE_FLAGS##*logdir}
python scripts/image_sample_sr3_segbranch.py $MODEL_FLAGS $DIFFUSION_FLAGS $SAMPLE_FLAGS --num_samples -1


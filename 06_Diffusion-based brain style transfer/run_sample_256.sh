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

SAMPLE_FLAGS="
--gpu 1
--batch_size 1
--model_path ex_output/diffusion/ex07_segbranch_c/P28_avg_256_256/train00_image_size_256/best_0.9999_630000_0.964510589838028.pt
--data_dir datasets/data_config_c_test/allen_s_m_all.json
--val_type allen
--continous False
--logdir ex_output/translation/ex07_segbranch_c/allen_to_P28_avg/sample00_image_size_256
"

mkdir -p ${SAMPLE_FLAGS##*logdir}
cp $(pwd)"/"$0 ${SAMPLE_FLAGS##*logdir}
python scripts/image_sample_sr3_segbranch.py $MODEL_FLAGS $DIFFUSION_FLAGS $SAMPLE_FLAGS --num_samples -1


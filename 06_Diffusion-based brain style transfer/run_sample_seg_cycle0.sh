# --timestep_respacing ddim250
MODEL_FLAGS="--image_size 128 --num_channels 128 --num_res_blocks 3 --in_channels 2 --resolution 228 160"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear"
SAMPLE_FLAGS="--gpu 1
--source_path output/diffusion/task2/allen/train2/ema_0.9999_210000.pt
--target_path output/diffusion/task2/soma_nuclei/train2/ema_0.9999_210000.pt
--source_seg_model_path output/seg/task4/allen/train1/model000100.pt
--target_seg_model_path output/seg/task3/soma_nuclei9/train1/model000100.pt
--data_dir datasets/data_config/allen_test.json
--continous False
--logdir output/cycle_translation/task2_2/2d/allen_to_somanuclei_to_allen/test5"
mkdir -p ${SAMPLE_FLAGS##*logdir} ; cp $(pwd)"/"$0 ${SAMPLE_FLAGS##*logdir}
python scripts/sample_seg/seg_sample_sr3_1_cycle.py  $MODEL_FLAGS $DIFFUSION_FLAGS $SAMPLE_FLAGS  \
 --num_samples -1 --batch_size 8
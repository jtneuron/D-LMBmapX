python create_data.py --base /media/root/e8449930-91ce-40a4-a5c5-87d4d2cd1568/lpq/nnUNet/nnUNetData/base/Task2641_P28 --source /media/root/e8449930-91ce-40a4-a5c5-87d4d2cd1568/lpq/nnUNet/nnUNetData/base/Task2641_P28 --task_id 26488 --task_name TH_P28_hog

nnUNet_plan_and_preprocess -t 26488

CUDA_VISIBLE_DEVICES=1 nnUNet_train 3d_fullres MyTrainerVitResMLP 26488 0
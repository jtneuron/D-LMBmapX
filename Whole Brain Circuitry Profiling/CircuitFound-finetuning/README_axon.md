## 0 Preparation
First of all, prepare a nnU-Net environment. You can refer to https://github.com/MIC-DKFZ/nnUNet/readme.md.

## 1 Cube selection and data augmentation
Run `create_data.py`, in which base and source directories should be prepared ahead as below. *Note that the number of 
the skeletonized labels, the labels and the volumes need to be exactly the same.*
```
└── base(original train data)
　　 └── train
　 　 　　├── volumes
　　  　　│　　└──volume-001.tiff
　　  　　├── labels
　　 　 　│　　└──label-001.tiff
　　  　　├── labels_sk(skeletonized labels)
　　 　 　│　　└──label-001.tiff
　　 　 　└── artifacts
　　 　　　　　└──volume-200.tiff
└── source(data used for histogram matching)
　　 └── train
　 　 　　├── volumes
　　  　　│　　└──volume-001.tiff
　　  　　├── labels
　　 　 　│　　└──label-001.tiff
　　  　　├── labels_sk(skeletonized labels)
　　 　 　│　　└──label-001.tiff
　　 　 　└── artifacts
　　 　　　　　└──volume-200.tiff
```
Change the parameters of function `histogram_match_data` in `create_data.py` to choose using histogram matching/cutmix or 
not. If you want to use histogram matching, it is better to set both **match_flag** and **join_flag** True so that both 
original cubes and matched cubes can be used for training.
```
cutmix=True  # use cutmix, mix up axon cubes and artifact cubes
match_flag=True, join_flag=True  # use histogram matching, join matched and original cubes
match_flag=True, join_flag=False  # use histogram matching, use only matched cubes
```
## 2 Experiment planning and preprocessing for nnU-Net
After step 1 the raw training dataset will be in the folder prepared in step 0 (`nnUNet_raw_data_base/nnUNet_raw_data/TaskXXX_MYTASK`, 
also see [here](https://github.com/MIC-DKFZ/nnUNet/documentation/dataset_conversion.md)), where task id `XXX` and task name 
`MYTASK` are set in `create_data.py`. 

For nnU-Net model training a preprocess is needed:
```bash
nnUNet_plan_and_preprocess -t XXX
```
Running `nnUNet_plan_and_preprocess` will populate your folder with preprocessed data. You will find the output in 
`nnUNet_preprocessed/TaskXXX_MYTASK`. `nnUNet_plan_and_preprocess` creates subfolders with preprocessed data for the 2D 
U-Net as well as all applicable 3D U-Nets. It will also create 'plans' files (with the ending.pkl) for the 2D and 
3D configurations. These files contain the generated segmentation pipeline configuration and will be read by the 
nnUNetTrainer (see below). Note that the preprocessed data folder only contains the training cases. 

Note that `nnUNet_plan_and_preprocess` accepts several additional input arguments. Running `-h` will list all of them 
along with a description. If you run out of RAM during preprocessing, you may want to adapt the number of processes 
used with the `-tl` and `-tf` options. The default configuration make use of a GPU with 8 GB memory. Larger memory size 
can be used with options like `-pl3d ExperimentPlanner3D_v21_16GB`.

## 3 Model training
nnU-Net trains all U-Net configurations in a 5-fold cross-validation. This enables nnU-Net to determine the 
postprocessing and ensembling (see next step) on the training dataset. 
Training models is done with the `nnUNet_train` command. The general structure of the command is:
```bash
nnUNet_train CONFIGURATION TRAINER_CLASS_NAME TASK_NAME_OR_ID FOLD --npz (additional options)
```
CONFIGURATION is a string that identifies the requested U-Net configuration. TASK_NAME_OR_ID specifies what dataset should 
be trained on and FOLD specifies which fold of the 5-fold-cross-validaton is trained. 

TRAINER_CLASS_NAME is the name of the model trainer. To be specific, a normal U-Net will be trained with TRAINER_CLASS_NAME 
`nnUNetTrainerV2`. You can use TRAINER_CLASS_NAME `MyTranerAxial` to train a U-Net with attention modules. 

nnU-Net stores a checkpoint every 50 epochs. If you need to continue a previous training, just add a `-c` to the 
training command.

`--npz` makes the models save the softmax outputs during the final validation. It should only be used for trainings 
where you plan to run `nnUNet_find_best_configuration` afterwards. If you are developing new trainer classes you may not 
need the softmax predictions and should therefore omit the `--npz` flag. Exported softmax predictions are very large and 
therefore can take up a lot of disk space. 

See `nnUNet_train -h` for additional options. 

##### 3D full resolution U-Net
For FOLD in [0, 1, 2, 3, 4], a sample command is (if `-pl3d ExperimentPlanner3D_v21_16GB` used in step 2): 
```
nnUNet_train 3d_fullres MyTrainerAxial TaskXXX_MYTASK FOLD -p nnUNetPlansv2.1_16GB
```
The trained models will be written to the RESULTS_FOLDER/nnUNet folder. Each training obtains an automatically generated 
output folder name `nnUNet_preprocessed/CONFIGURATION/TaskXXX_MYTASKNAME/TRAINER_CLASS_NAME__PLANS_FILE_NAME/FOLD`.

**Multi GPU training is experimental and NOT RECOMMENDED!**

## 4 Inference
Once all 5-fold models are trained, use the following 
command to automatically determine what U-Net configuration(s) to use for test set prediction:
```bash
nnUNet_find_best_configuration -m 3d_fullres -t XXX --strict
```

`nnUNet_find_best_configuration` will print a string to the terminal with the inference commands you need to use. 
The easiest way to run inference is to simply use these commands. 

For each of the desired configurations(e.g. 3d_fullres), run:
```
nnUNet_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -t TASK_NAME_OR_ID -m CONFIGURATION --save_npz
```
Only specify `--save_npz` if you intend to use ensembling. `--save_npz` will make the command save the softmax 
probabilities alongside of the predicted segmentation masks requiring a lot of disk space. A separate `OUTPUT_FOLDER` 
should be selected for each configuration. You can also use `-f` to specify folder id(s) if not all 5-folds has been trained. 
`--tr` option can be used to specify TRAINER_CLASS_NAME, which should be consistent with the class used in model training. 

A sample command using an U-Net with attention module to generate predictions is: 
`
nnUNet_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -t XXX  --tr nnUNetTrainerV2 -ctr nnUNetTrainerV2CascadeFullRes -m 3d_fullres -p nnUNetPlansv2.1_16GB
`

## 5 Whole brain segmentation

import SimpleITK as sitk
from skimage.io import imread
import os

def load_tiff_convert_to_nifti(img_file, img_out):
    img = imread(img_file)

    img_itk = sitk.GetImageFromArray(img)

    sitk.WriteImage(img_itk, os.path.join(img_out))



if __name__ == "__main__":
    base_dir = ''
    out_dir = ''
    for v in os.listdir(base_dir):
        v_name = v.split('.')[0]
        v_path = os.path.join(base_dir, v)
        out_path = os.path.join(out_dir, v_name + '".nii.gz"')
        load_tiff_convert_to_nifti(v_path, out_path)

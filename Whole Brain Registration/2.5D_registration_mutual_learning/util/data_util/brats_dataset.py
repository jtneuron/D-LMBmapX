import os

import numpy as np
import torch
from torch.utils.data import Dataset

from util.data_util.io_util import read_3d_data_normalize


class BraTSTrainPairDataset(Dataset):
    def __init__(self, dataset_config, dataset_type, transform=None):
        super(BraTSTrainPairDataset, self).__init__()
        self.transform = transform
        self.data_paths = dataset_config[dataset_type]

    def __getitem__(self, index):
        img1_path = self.data_paths[index][1]
        img2_index = np.random.randint(0,len(self.data_paths))
        img2_path = self.data_paths[img2_index][0]

        img = dict()
        img['volume1'] = read_3d_data_normalize(img1_path, max_value=1.)
        img['volume2'] = read_3d_data_normalize(img2_path, max_value=1.)

        img['id1'] = os.path.basename(img1_path).split(".")[0]
        img['id2'] = os.path.basename(img2_path).split(".")[0]

        img = self.as_type_to_tensor(img)

        img['volume1'] = img['volume1'].float()
        img['volume2'] = img['volume2'].float()

        if self.transform is None:
            return img

        img = self.transform(img)

        return img

    def __len__(self):
        return len(self.data_paths)

    @staticmethod
    def as_type_to_tensor(img):
        for key, value in img.items():
            if isinstance(value, np.ndarray):
                img[key] = torch.from_numpy(img[key])
        return img


class BraTSValPairDataset(Dataset):
    def __init__(self, dataset_config, dataset_type, transform=None):
        super(BraTSValPairDataset, self).__init__()
        self.transform = transform
        self.data_paths = dataset_config[dataset_type]
        N = len(self.data_paths)
        self.data_paths = [[path1[1], path2[0], path1[2], path2[2]] for path1 in self.data_paths for path2 in self.data_paths if
                           path1[0] != path2[0]]
        np.random.shuffle(self.data_paths)
        self.data_paths = self.data_paths[:N]

    def __getitem__(self, index):
        img1_path = self.data_paths[index][0]
        img2_path = self.data_paths[index][1]
        img3_path = self.data_paths[index][2]
        img4_path = self.data_paths[index][3]

        img = dict()
        img['volume1'] = read_3d_data_normalize(img1_path, max_value=1.)
        img['volume2'] = read_3d_data_normalize(img2_path, max_value=1.)
        img['volume3'] = read_3d_data_normalize(img3_path, max_value=1.)
        img['volume4'] = read_3d_data_normalize(img4_path, max_value=1.)

        img['id1'] = os.path.basename(img1_path).split(".")[0]
        img['id2'] = os.path.basename(img2_path).split(".")[0]
        img['id3'] = os.path.basename(img3_path).split(".")[0]
        img['id4'] = os.path.basename(img4_path).split(".")[0]

        img = self.as_type_to_tensor(img)

        img['volume1'] = img['volume1'].float()
        img['volume2'] = img['volume2'].float()
        img['volume3'] = img['volume3'].float()
        img['volume4'] = img['volume4'].float()

        if self.transform is None:
            return img

        img = self.transform(img)

        return img

    def __len__(self):
        return len(self.data_paths)

    @staticmethod
    def as_type_to_tensor(img):
        for key, value in img.items():
            if isinstance(value, np.ndarray):
                img[key] = torch.from_numpy(img[key])
        return img


class BraTSInferPairDataset(Dataset):
    def __init__(self, dataset_config, dataset_type, transform=None):
        super(BraTSInferPairDataset, self).__init__()
        self.transform = transform
        self.data_paths = dataset_config[dataset_type]
        self.data_paths = [[path1[1], path2[0], path1[2], path2[2]] for path1 in self.data_paths for path2 in self.data_paths if
                           path1[0] != path2[0]]

    def __getitem__(self, index):
        img1_path = self.data_paths[index][0]
        img2_path = self.data_paths[index][1]
        img3_path = self.data_paths[index][2]
        img4_path = self.data_paths[index][3]

        img = dict()
        img['volume1'] = read_3d_data_normalize(img1_path, max_value=1.)
        img['volume2'] = read_3d_data_normalize(img2_path, max_value=1.)
        img['volume3'] = read_3d_data_normalize(img3_path, max_value=1.)
        img['volume4'] = read_3d_data_normalize(img4_path, max_value=1.)

        img['id1'] = os.path.basename(img1_path).split(".")[0]
        img['id2'] = os.path.basename(img2_path).split(".")[0]
        img['id3'] = os.path.basename(img3_path).split(".")[0]
        img['id4'] = os.path.basename(img4_path).split(".")[0]

        img = self.as_type_to_tensor(img)

        img['volume1'] = img['volume1'].float()
        img['volume2'] = img['volume2'].float()
        img['volume3'] = img['volume3'].float()
        img['volume4'] = img['volume4'].float()

        if self.transform is None:
            return img

        img = self.transform(img)

        return img

    def __len__(self):
        return len(self.data_paths)

    @staticmethod
    def as_type_to_tensor(img):
        for key, value in img.items():
            if isinstance(value, np.ndarray):
                img[key] = torch.from_numpy(img[key])
        return img


if __name__ == '__main__':
    pass

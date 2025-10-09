import os.path

import numpy as np
import torch
from torch.utils.data import Dataset

from util.data_util.io_util import read_3d_data, read_3d_mask


class TrainPairDataset(Dataset):
    def __init__(self, dataset_config, dataset_type, registration_type, transform=None):
        super(TrainPairDataset, self).__init__()
        self.registration_type = registration_type
        self.transform = transform
        self.data_paths = dataset_config[dataset_type]
        self.atlas = dataset_config.get('atlas', 'the config has not atlas')

    def __getitem__(self, index):
        if self.registration_type == 0:
            img1_path = self.data_paths[index]
            img2_path = np.random.choice(self.data_paths)
        elif self.registration_type == 1:
            img1_path = self.data_paths[index]
            img2_path = self.atlas
        elif self.registration_type == 2:
            img1_path = self.atlas
            img2_path = self.data_paths[index]
        elif self.registration_type == 4:
            img1_path = self.data_paths['moving'][index]
            img2_path = self.data_paths['fixed'][index]
        else:
            raise Exception("registration type error!")
        img = dict()
        img['volume1'] = read_3d_data(img1_path)
        img['volume2'] = read_3d_data(img2_path)

        img['label1'] = read_3d_mask(img1_path, "label_cast")
        img['label_test'] = read_3d_mask(img1_path)
        img['label2'] = read_3d_mask(img2_path)

        img['id1'] = os.path.basename(img1_path).split(".")[0]
        img['id2'] = os.path.basename(img2_path).split(".")[0]

        img['volume1'] = img['volume1'].astype(np.float32)
        img['volume2'] = img['volume2'].astype(np.float32)
        img['volume1'] = img['volume1'] / 255.
        img['volume2'] = img['volume2'] / 255.
        
        img = self.as_type_to_tensor(img)

        if self.transform is None:
            return img

        img = self.process_transform(img)

        return img

    def __len__(self):
        if self.registration_type == 4:
            assert len(self.data_paths['moving']) == len(self.data_paths['fixed'])
            return len(self.data_paths['moving'])
            
        return len(self.data_paths)

    def process_transform(self, img):
        if 'label1' in img.keys() and img['label1'] == []:
            img.pop('label1')
        if 'label2' in img.keys() and img['label2'] == []:
            img.pop('label2')

        img = self.transform(img)

        if 'label1' not in img.keys():
            img['label1'] = []
        if 'label2' not in img.keys():
            img['label2'] = []
        return img

    @staticmethod
    def as_type_to_tensor(img):
        for key, value in img.items():
            if isinstance(value, np.ndarray):
                # TEST
                if key == 'label1' or key == 'label2':
                    img[key] = img[key].astype(np.uint8)
                img[key] = torch.from_numpy(img[key])
        return img


class ValPairDataset(Dataset):
    def __init__(self, dataset_config, dataset_type, registration_type, transform=None):
        super(ValPairDataset, self).__init__()
        self.registration_type = registration_type
        self.transform = transform
        data_paths = dataset_config[dataset_type]
        self.atlas = dataset_config.get('atlas', 'the config has not atlas')
        N = len(data_paths)
        if self.registration_type == 0:
            self.data_paths = [[path1, path2] for path1 in data_paths for path2 in data_paths if
                               path1 != path2]
            np.random.shuffle(self.data_paths)
            self.data_paths = self.data_paths[:N]
        elif self.registration_type == 1:
            self.data_paths = [[path1, self.atlas] for path1 in data_paths]
        elif self.registration_type == 2:
            self.data_paths = [[self.atlas, path2] for path2 in data_paths]
        else:
            raise Exception("registration type error!")

    def __getitem__(self, index):
        img1_path, img2_path = self.data_paths[index]
        img = dict()
        img['volume1'] = read_3d_data(img1_path)
        img['volume2'] = read_3d_data(img2_path)

        img['label1'] = read_3d_mask(img1_path)
        img['label2'] = read_3d_mask(img2_path)

        img['id1'] = os.path.basename(img1_path).split(".")[0]
        img['id2'] = os.path.basename(img2_path).split(".")[0]


        img['volume1'] = img['volume1'].astype(np.float32)
        img['volume2'] = img['volume2'].astype(np.float32)
        img['volume1'] = img['volume1'] / 255.
        img['volume2'] = img['volume2'] / 255.
        
        img = self.as_type_to_tensor(img)
        
        if self.transform is None:
            return img

        img = self.process_transform(img)

        return img

    def __len__(self):
        return len(self.data_paths)

    def process_transform(self, img):
        if 'label1' in img.keys() and img['label1'] == []:
            img.pop('label1')
        if 'label2' in img.keys() and img['label2'] == []:
            img.pop('label2')

        img = self.transform(img)

        if 'label1' not in img.keys():
            img['label1'] = []
        if 'label2' not in img.keys():
            img['label2'] = []
        return img

    @staticmethod
    def as_type_to_tensor(img):
        for key, value in img.items():
            if isinstance(value, np.ndarray):
                img[key] = torch.from_numpy(img[key])
        return img


class InferPairDataset(Dataset):
    def __init__(self, dataset_config, dataset_type, registration_type, transform=None):
        self.registration_type = registration_type
        self.transform = transform
        self.data_paths = dataset_config[dataset_type]
        self.atlas = dataset_config.get('atlas', 'the config has not atlas')
        data_paths = dataset_config[dataset_type]
        if registration_type == 0:
            self.data_paths = [[path1, path2] for path1 in data_paths for path2 in data_paths if path1 != path2]
        elif registration_type == 1:
            self.data_paths = [[path1, self.atlas] for path1 in data_paths]
        elif registration_type == 2:
            self.data_paths = [[self.atlas, path1] for path1 in data_paths]
        else:
            raise Exception("registration type error!")

    def __getitem__(self, index):
        img1_path, img2_path = self.data_paths[index]

        img = dict()
        img['volume1'] = read_3d_data(img1_path)
        img['volume2'] = read_3d_data(img2_path)

        img['label1'] = read_3d_mask(img1_path)
        img['label2'] = read_3d_mask(img2_path)

        img['id1'] = os.path.basename(img1_path).split(".")[0]
        img['id2'] = os.path.basename(img2_path).split(".")[0]

 

        img['volume1'] = img['volume1'].astype(np.float32)
        img['volume2'] = img['volume2'].astype(np.float32)
        img['volume1'] = img['volume1'] / 255.
        img['volume2'] = img['volume2'] / 255.
        
        img = self.as_type_to_tensor(img)

        if self.transform is None:
            return img

        img = self.process_transform(img)

        return img

    def __len__(self):
        return len(self.data_paths)

    def process_transform(self, img):
        if 'label1' in img.keys() and img['label1'] == []:
            img.pop('label1')
        if 'label2' in img.keys() and img['label2'] == []:
            img.pop('label2')

        img = self.transform(img)

        if 'label1' not in img.keys():
            img['label1'] = []
        if 'label2' not in img.keys():
            img['label2'] = []
        return img

    @staticmethod
    def as_type_to_tensor(img):
        for key, value in img.items():
            if isinstance(value, np.ndarray):
                img[key] = torch.from_numpy(img[key])
        return img

class TrainPairDataset2D(Dataset):
    def __init__(self, dataset_config, dataset_type):
        super(TrainPairDataset2D, self).__init__()
        self.dataset_config = dataset_config
        self.dataset_type = dataset_type
        self.data_paths = dataset_config[dataset_type]
    
    def __len__(self):
        return len(self.data_paths)
    
    def __getitem__(self, index):
        '''
        train:
            [
                ["......",
                "......"]
            ],
            [
                ["......",
                "......"]
            ],
            ...
        '''
        img1_path, img2_path, img3_path = self.data_paths[index]
        

    
        img = dict()
        img['volume1'] = read_3d_data(img1_path)
        img['volume2'] = read_3d_data(img2_path)
        img['volume3'] = read_3d_data(img3_path)
        img['id1'] = os.path.basename(img1_path).split('.')[0]
        img['id2'] = os.path.basename(img2_path).split('.')[0]
        img['id3'] = os.path.basename(img3_path).split('.')[0]

        img = self.as_type_to_tensor(img)
        

        img['volume1'] = img['volume1'].float()
        img['volume2'] = img['volume2'].float()
        img['volume3'] = img['volume3'].float()

        # img['volume1'] = (1 - (img['volume1'] < 60).float()) * img['volume1']
        
        img['volume1'] = img['volume1'] / 255.
        img['volume2'] = img['volume2'] / 255.
        img['volume3'] = img['volume3'] / 255.

        return img
    
    @staticmethod
    def as_type_to_tensor(img):
        for key, value in img.items():
            if isinstance(value, np.ndarray):
                img[key] = torch.from_numpy(img[key])
        return img

class ValPairDataset2D(Dataset):
    def __init__(self, dataset_config, dataset_type):
        super(ValPairDataset2D, self).__init__()
        self.dataset_config = dataset_config
        self.dataset_type = dataset_type
        self.data_paths = dataset_config[dataset_type]
    
    def __len__(self):
        return len(self.data_paths)
    
    def __getitem__(self, index):
        '''
        train:
            [
                ["......",
                "......"]
            ],
            [
                ["......",
                "......"]
            ],
            ...
        '''
        img1_path, img2_path = self.data_paths[index]
        

    
        img = dict()
        img['volume1'] = read_3d_data(img1_path)
        img['volume2'] = read_3d_data(img2_path)
        img['id1'] = os.path.basename(img1_path).split('.')[0]
        img['id2'] = os.path.basename(img2_path).split('.')[0]
        
        img = self.as_type_to_tensor(img)

        img['volume1'] = img['volume1'].float()
        img['volume2'] = img['volume2'].float()
        img['volume1'] = img['volume1'] / 255.
        img['volume2'] = img['volume2'] / 255.


        
        return img
    
    @staticmethod
    def as_type_to_tensor(img):
        for key, value in img.items():
            if isinstance(value, np.ndarray):
                img[key] = torch.from_numpy(img[key])
        return img

class TrainPairDataset_with_anno(Dataset):
    def __init__(self, dataset_config, dataset_type, registration_type, transform=None):
        super(TrainPairDataset_with_anno, self).__init__()
        self.registration_type = registration_type
        self.transform = transform
        self.data_paths = dataset_config[dataset_type]
        self.atlas = dataset_config.get('atlas', 'the config has not atlas')

    def __getitem__(self, index):
        if self.registration_type == 0:
            img1_path = self.data_paths[index]
            img2_path = np.random.choice(self.data_paths)
        elif self.registration_type == 1:
            img1_path = self.data_paths[index]
            img2_path = self.atlas
        elif self.registration_type == 2:
            img1_path = self.atlas
            img2_path = self.data_paths[index]
        else:
            raise Exception("registration type error!")
        img = dict()
        img['volume1'] = read_3d_data(img1_path)
        img['volume2'] = read_3d_data(img2_path)

        img['label1'] = read_3d_mask(img1_path)
        img['label2'] = read_3d_mask(img2_path)
        
        # test 
        img["anno1"] = read_3d_mask(os.path.join(os.path.dirname(img1_path), "test", os.path.basename(img1_path)))
        img["anno1"] = img["anno1"].astype(np.int32)

        img['id1'] = os.path.basename(img1_path).split(".")[0]
        img['id2'] = os.path.basename(img2_path).split(".")[0]

        img['volume1'] = img['volume1'].astype(np.float32)
        img['volume2'] = img['volume2'].astype(np.float32)
        img['volume1'] = img['volume1'] / 255.
        img['volume2'] = img['volume2'] / 255.
        
        img = self.as_type_to_tensor(img)

        if self.transform is None:
            return img

        img = self.process_transform(img)

        return img

    def __len__(self):
        return len(self.data_paths)

    def process_transform(self, img):
        if 'label1' in img.keys() and img['label1'] == []:
            img.pop('label1')
        if 'label2' in img.keys() and img['label2'] == []:
            img.pop('label2')

        img = self.transform(img)

        if 'label1' not in img.keys():
            img['label1'] = []
        if 'label2' not in img.keys():
            img['label2'] = []
        return img

    @staticmethod
    def as_type_to_tensor(img):
        for key, value in img.items():
            if isinstance(value, np.ndarray):
                img[key] = torch.from_numpy(img[key])
        return img



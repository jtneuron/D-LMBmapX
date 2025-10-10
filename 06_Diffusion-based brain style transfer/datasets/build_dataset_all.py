import blobfile as bf
import os
from tqdm import tqdm
import json
import random


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        name = entry.split(".")[0]
        if name.startswith("mask"):
            continue
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


def _get_paths(A_path):
    A_files = _list_image_files_recursively(A_path)
    pair_data = []
    for a_file in tqdm(A_files):
        pair_data.append([a_file])
    return pair_data


def split_data(file_name, train_rate):
    data_name_list = _get_paths(file_name)
    random.shuffle(data_name_list)
    data_num = len(data_name_list)
    print(file_name, data_num)

    test_name_list = data_name_list[0:int(data_num * train_rate)]
    train_name_list = data_name_list[int(data_num * train_rate):]

    # train_name_list = []
    # test_name_list = []
    # train_data_list = [
    #     'data_180724_O2_n_gan',
    #     'data_180725_C2_n_gan',
    #     'data_s_190312_17_26_21_n_gan',
    #     'data_s_190313_01_04_17_n_gan',
    #     'data_soma_181207_10_39_06_n_gan',
    #     'data_soma_181207_18_26_44_n_gan',
    # ]
    # test_data_list = [
    #     # 'data_180724_O2_n_gan',
    #     # 'data_180725_C2_n_gan',
    #     # 'data_s_190312_17_26_21_n_gan',
    #     # 'data_s_190313_01_04_17_n_gan',
    #     # 'data_soma_181207_10_39_06_n_gan',
    #     # 'data_soma_181207_18_26_44_n_gan',
    #     'data_180921_O11_n_gan',
    #     'data_s_190524_18_43_49_n_gan',
    #     'data_soma_181208_15_21_38_n_gan',
    # ]
    # for data_name in data_name_list:
    #     if data_name[0].split('/')[-2] in train_data_list:
    #         train_name_list.append(data_name)
    #     elif data_name[0].split('/')[-2] in test_data_list:
    #         test_name_list.append(data_name)

    return data_name_list, test_name_list, train_name_list


if __name__ == '__main__':
    data_path = r'/media/user/hdd1/liuhe/brain_data/i2i_data/s/mri'

    _, test_paths, train_paths = split_data(data_path, 1) # c:0.125 a: 0.2 s:0.143
    print('len(train_paths): ', len(train_paths), 'len(test_paths): ', len(test_paths))
    train_data, test_data = dict(), dict()
    train_data['data'] = train_paths
    test_data['data'] = test_paths

    output_path_test = r"./data_config_s_test"
    os.makedirs(output_path_test, exist_ok=True)
    output_path_train = r"./data_config_s"
    os.makedirs(output_path_train, exist_ok=True)

    with open(os.path.join(output_path_test, 'mri_all_test.json'), 'w') as f:
        json.dump(test_data, f, indent=4)
    # with open(os.path.join(output_path_train, 'mri_train.json'), 'w') as f:
    #     json.dump(train_data, f, indent=4)


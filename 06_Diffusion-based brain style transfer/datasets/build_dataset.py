import blobfile as bf
import os
from tqdm import tqdm
import json
import random


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        name = entry.split(".")[0]
        if name.startswith("mask_"):
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
    # random.shuffle(data_name_list)  # 打乱
    data_num = len(data_name_list)
    print(file_name, data_num)
    train_name_list = data_name_list[0:int(data_num * train_rate)]
    test_name_list = data_name_list[int(data_num * train_rate):]
    return data_name_list, train_name_list, test_name_list


if __name__ == '__main__':
    data_path = r'/media/user/hdd1/liuhe/brain_data/i2i_data/c/mri'
    _, train_path_1, _ = split_data(os.path.join(data_path, 'data_fvbex_brain1_m_re'), 1)
    _, train_path_2, _ = split_data(os.path.join(data_path, 'data_fvbex_brain5_m_re'), 1)
    _, train_path_3, _ = split_data(os.path.join(data_path, 'data_fvbex_brain6_m_re'), 1)
    _, train_path_4, _ = split_data(os.path.join(data_path, 'data_fvbex_brain2_m_re'), 1)
    _, train_path_5, _ = split_data(os.path.join(data_path, 'data_fvbex_brain3_m_re'), 1)
    _, train_path_6, _ = split_data(os.path.join(data_path, 'data_fvbex_brain4_m_re'), 1)
    _, train_path_7, _ = split_data(os.path.join(data_path, 'data_fvbex_brain7_m_re'), 1)
    _, train_path_8, _ = split_data(os.path.join(data_path, 'data_fvbex_brain8_m_re'), 1)
    # _, train_path_9, _ = split_data(os.path.join(data_path, 'data_soma_181208_15_21_38_n_gan'), 1)

    train_paths = []
    train_paths.extend(train_path_1)
    train_paths.extend(train_path_2)
    train_paths.extend(train_path_3)
    train_paths.extend(train_path_4)
    train_paths.extend(train_path_5)
    train_paths.extend(train_path_6)
    train_paths.extend(train_path_7)
    train_paths.extend(train_path_8)
    # train_paths.extend(train_path_9)

    # random.shuffle(train_paths)

    print('len(train_paths): ',len(train_paths))
    train_data, _ = dict(), dict()
    train_data['data'] = train_paths
    # test_data['data'] = test_paths
    output_path = r"data_config"
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, 'fvbex_brain_m_re_test_all.json'), 'w') as f:
        json.dump(train_data, f, indent=4)
    # with open(os.path.join(output_path, 'fvbex_brain_m_re_test.json'), 'w') as f:
    #     json.dump(test_data, f, indent=4)

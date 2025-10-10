import os

import torch


class ModelSaver:
    def __init__(self, max_save_num):
        """
        :param max_save_num: max checkpoint number to save
        """
        # self.model_save_path_list = []
        # self.optimizer_save_path_list = []
        # self.max_save_num = max_save_num

        self.save_path_list = []
        self.max_save_num = max_save_num

    def save(self, dir, state_dict, step):
        path = os.path.join(dir, f"state{step:06d}.pt")
        if path not in self.save_path_list:
            self.save_path_list.append(path)
        if len(self.save_path_list) > self.max_save_num:
            top = self.save_path_list.pop(0)
            os.remove(top)
        torch.save(state_dict, path)

    # def save(self, dir, model_state_dict, optimizer_state_dict, step):
    #     filename = f"model{step:06d}.pt"
    #     model_file_path = os.path.join(dir, filename)
    #
    #     filename = f"opt{step:06d}.pt"
    #     opt_file_path = os.path.join(dir, filename)
    #
    #     if model_file_path not in self.model_save_path_list:
    #         self.model_save_path_list.append(model_file_path)
    #         self.optimizer_save_path_list.append(opt_file_path)
    #
    #     if len(self.model_save_path_list) > self.max_save_num:
    #         top = self.model_save_path_list.pop(0)
    #         os.remove(top)
    #         top = self.optimizer_save_path_list.pop(0)
    #         os.remove(top)
    #
    #     torch.save(model_state_dict, model_file_path)
    #     torch.save(optimizer_state_dict, opt_file_path)

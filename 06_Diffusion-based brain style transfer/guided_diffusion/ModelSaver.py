import os


class ModelSaver:
    def __init__(self, max_save_num=3):
        """
        :param max_save_num: max checkpoint number to save
        """
        # self.save_path_list = []
        self.eam_path_list = []
        self.mode_path_list = []
        self.opt_path_list = []
        self.best_path_list = []
        self.max_save_num = max_save_num

    def post_handle(self, path):
        if "ema" in os.path.basename(path):
            if path not in self.eam_path_list:
                self.eam_path_list.append(path)
            if len(self.eam_path_list) > self.max_save_num:
                top = self.eam_path_list.pop(0)
                print(f"remove {top}")
                os.remove(top)
        elif "model" in os.path.basename(path):
            if path not in self.mode_path_list:
                self.mode_path_list.append(path)
            if len(self.mode_path_list) > self.max_save_num:
                top = self.mode_path_list.pop(0)
                print(f"remove {top}")
                os.remove(top)
        elif "opt" in os.path.basename(path):
            if path not in self.opt_path_list:
                self.opt_path_list.append(path)
            if len(self.opt_path_list) > self.max_save_num:
                top = self.opt_path_list.pop(0)
                print(f"remove {top}")
                os.remove(top)
        elif "best" in os.path.basename(path):
            if path not in self.opt_path_list:
                self.best_path_list.append(path)
            if len(self.best_path_list) > self.max_save_num:
                top = self.best_path_list.pop(0)
                print(f"remove {top}")
                os.remove(top)

import torch
import pdb
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.utilities.nd_softmax import softmax_helper

from nnunet.network_architecture.vit_res_mlp import ViT_Res_MLP
from nnunet.network_architecture.custom_modules.vit_res import VitResnet
from nnunet.network_architecture.custom_modules.MLP_decoder import SegFormerHead_3

class MyTrainerVitResMLP(nnUNetTrainerV2):

    def load_my_checkpoint(self):
        # pretrain = "/media/root/18TB_HDD/lpq/SelfMedMAE/pretrain/checkpoint_1000.pth.tar"
        # pretrain = "/mnt/18TB_HDD2/lpq/MAE_run_data/MAE_axons_1121_vit_base_TH_mixed/ckpts/checkpoint_1000.pth.tar"

        pretrain = "/media/root/18TB_HDD/lpq/SelfMedMAE/pretrain/10w_checkpoint_1000.pth.tar"
        # pretrain = "/media/root/18TB_HDD/lpq/SelfMedMAE/pretrain/HOG_10w_checkpoint_1000.pth.tar"

        self.print_to_log_file(f"=> Start loading encoder pretrained weights from {pretrain}")
        checkpoint = torch.load(pretrain, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        # 遍历模型的参数和名称
        for key in list(state_dict.keys()):
            if key.startswith('encoder.'):
                state_dict[key[len('encoder.'):]] = state_dict[key]
                del state_dict[key]
            if key.startswith('decoder.')  or key.startswith('decoder_pixel.') or key.startswith('decoder_hog.'):
                del state_dict[key]
            if key == 'encoder_pos_embed':
                pe = torch.zeros([1, 1, state_dict[key].size(-1)])
                state_dict['pos_embed'] = torch.cat([pe, state_dict[key]], dim=1)
                del state_dict[key]
            if key == 'patch_embed.proj.weight' and \
                    state_dict[
                        'patch_embed.proj.weight'].shape != self.network.encoder.patch_embed.proj.weight.shape:
                del state_dict['patch_embed.proj.weight']
                del state_dict['patch_embed.proj.bias']
            if key == 'pos_embed' and \
                    state_dict['pos_embed'].shape != self.network.encoder.pos_embed.shape:
                del state_dict[key]
        msg = self.network.encoder.vit.load_state_dict(state_dict, strict=False)
        self.print_to_log_file(f'Loading messages: \n {msg}')
        self.print_to_log_file(f"=> Finish loading pretrained weights from {pretrain}")

    def count_params(self, model):
        """
        计算模型的参数量
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.print_to_log_file(f"Total parameters: {total_params:,}")
        self.print_to_log_file(f"Trainable parameters: {trainable_params:,}")
        return total_params, trainable_params

    def initialize_network(self):
        # change network architecture
        self.network = ViT_Res_MLP(VitResnet, SegFormerHead_3)
        self.load_my_checkpoint()

        self.count_params(self.network)
        pdb.set_trace()

        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper

        self.print_to_log_file("=============Total eopchs: {}".format(self.max_num_epochs))

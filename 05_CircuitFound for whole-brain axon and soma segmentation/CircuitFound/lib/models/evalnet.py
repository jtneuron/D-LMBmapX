import os
import pdb
import torch
import shutil
import tifffile
import multiprocessing
import configparser
from PIL import Image
from tqdm import tqdm
import myutils
from lib.losses import *
from torch.autograd import Variable
from torch.utils.data import DataLoader
from monai.losses import DiceCELoss
import torch.nn as nn
import numpy as np
import lib.models as models
import lib.networks as networks
from lib.data.med_transforms import get_post_transforms

def gamma_correction(image, gamma):
    # 首先确保图像数据在0到1之间
    image = image / 65535
    image_corrected = np.power(image, gamma)
    return image_corrected

def equal(v, mean, std):
    v = v / 65535
    # return (v - v.min()) / (v.max() - v.min())
    # return v / 65535
    return (v - v.mean()) / (v.std() + 1e-8)
    # return (v - mean) / (std + 1e-8)

def read_tiff_stack(path):
    if os.path.isdir(path):
        images = [np.array(Image.open(os.path.join(path, p))) for p in sorted(os.listdir(path))]
        return np.array(images)
    else:
        img = Image.open(path)
        images = []
        for i in range(img.n_frames):
            img.seek(i)
            slice = np.array(img)
            images.append(slice)
        return np.array(images)


class EvalNet:
    def __init__(self, args):
        self.args = args
        self.model_name = self.args.model_name
        self.model = None
        self.input_dim = self.args.input_dim
        self.device = torch.device('cuda:{}'.format(self.args.gpu)) if self.args.gpu is not None else torch.device(
            'cpu')

    def eval_two_volumes_maxpool(self):
        pre = read_tiff_stack(self.args.dataroot)
        label = read_tiff_stack(self.args.data_target)
        k = self.args.pool_kernel
        kernel = (k, k, k)
        pre[pre < self.args.threshold] = 0
        pre[pre >= self.args.threshold] = 1
        label[label > 0] = 1
        pre = torch.Tensor(pre).view((1, 1, *pre.shape)).to(self.device)
        # pre = torch_dilation(pre, 5)
        label = torch.Tensor(label).view((1, 1, *label.shape)).to(self.device)
        # label = torch_dilation(label, 5)
        pre = torch.nn.functional.max_pool3d(pre, kernel, 1, 0)
        label = torch.nn.functional.max_pool3d(label, kernel, 1, 0)

        dice_score = dice_error(pre, label)

        total_loss_iou = iou(pre, label).cpu()
        total_loss_tiou = t_iou(pre, label).cpu()
        recall, acc = soft_cldice_f1(pre, label)
        cldice = (2. * recall * acc) / (recall + acc)

        print('\n Validation IOU: {}\n T-IOU: {}'
              '\n ClDice: {} \n ClAcc: {} \n ClRecall: {} \n Dice-score: {}'
              .format(total_loss_iou, total_loss_tiou, cldice, acc, recall, dice_score, '.8f'))

    def get_model(self):
        if self.model_name != 'Unknown' and self.model is None:
            args = self.args
            print(f"=> creating model {self.model_name}")

            # self.post_pred, self.post_label = get_post_transforms(args)

            # setup mixup and loss functions
            if args.mixup > 0:
                raise NotImplemented("Mixup for segmentation has not been implemented.")
            else:
                self.mixup_fn = None

            self.model = getattr(models, self.model_name)(encoder=getattr(networks, args.enc_arch),
                                                          decoder=getattr(networks, args.dec_arch),
                                                          args=args)
            # self.model.to(self.device)

            # 输出模型参数
            # for name, param in self.model.named_parameters():
            #     print(name, param.size())
            # load pretrained weights
            if args.pretrain is not None and args.pretrain_type == 'enc_dec' and os.path.exists(args.pretrain):
                print(f"=> Start loading the model weights from {args.pretrain} for test")
                checkpoint = torch.load(args.pretrain, map_location='cpu')
                state_dict = checkpoint['state_dict']
                msg = self.model.load_state_dict(state_dict, strict=False)
                print(f'Loading messages: \n {msg}')
                print(f"=> Finish loading pretrained weights from {args.pretrain}")
            elif args.pretrain is not None and args.pretrain_type == 'enc' and os.path.exists(args.pretrain):
                print(f"=> Start loading encoder pretrained weights from {args.pretrain}")
                checkpoint = torch.load(args.pretrain, map_location='cpu')
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
                # pdb.set_trace()
                if self.model_name == 'UNETR3D':
                    for key in list(state_dict.keys()):
                        if key.startswith('encoder.'):
                            state_dict[key[len('encoder.'):]] = state_dict[key]
                            del state_dict[key]
                        # need to concat and load pos embed. too
                        # TODO: unify the learning of pos embed of pretraining and finetuning
                        if key == 'encoder_pos_embed':
                            pe = torch.zeros([1, 1, state_dict[key].size(-1)])
                            state_dict['pos_embed'] = torch.cat([pe, state_dict[key]], dim=1)
                            del state_dict[key]
                        if key == 'patch_embed.proj.weight' and \
                                state_dict[
                                    'patch_embed.proj.weight'].shape != self.model.encoder.patch_embed.proj.weight.shape:
                            del state_dict['patch_embed.proj.weight']
                            del state_dict['patch_embed.proj.bias']
                        if key == 'pos_embed' and \
                                state_dict['pos_embed'].shape != self.model.encoder.pos_embed.shape:
                            del state_dict[key]
                    msg = self.model.encoder.load_state_dict(state_dict, strict=False)
                elif self.model_name == 'DynSeg3d':
                    if args.pretrain_load == 'enc+dec':
                        for key in list(state_dict.keys()):
                            if key.startswith('decoder.head.') or (
                                    key.startswith('decoder.blocks.') and int(key[15]) > 7):
                                del state_dict[key]
                    elif args.pretrain_load == 'enc':
                        for key in list(state_dict.keys()):
                            if key.startswith('decoder.'):
                                del state_dict[key]
                    msg = self.model.load_state_dict(state_dict, strict=False)
                # self.model.load(state_dict)
                print(f'Loading messages: \n {msg}')
                print(f"=> Finish loading pretrained weights from {args.pretrain}")

        elif self.model_name == 'Unknown':
            raise ValueError("=> Model name is still unknown")
        else:
            raise ValueError("=> Model has been created. Do not create twice")

        self.model.eval()
        self.wrap_model()

    # def eval_volumes_batch(self):
    #     self.get_model()
    #     testLoader = create_dataset(self.args)
    #     n_val = len(testLoader)
    #     loss_dir = self.eval_net(self.model, testLoader, self.device, n_val)
    #     iou, t_iou = loss_dir['iou'], loss_dir['tiou']
    #     cldice, clacc, clrecall = loss_dir['cldice'], loss_dir['cl_acc'], loss_dir['cl_recall']
    #     junk_rat = loss_dir['junk_ratio']
    #     print('\n Validation IOU: {}\n T-IOU: {}'
    #           '\n ClDice: {} \n ClAcc: {} \n ClRecall: {}'
    #           '\n Junk-ratio: {}'
    #           .format(iou, t_iou, cldice, clacc, clrecall, junk_rat, '.8f'))

    def wrap_model(self):
        """
        1. Distribute model or not
        2. Rewriting batch size and workers
        """
        args = self.args
        model = self.model
        assert model is not None, "Please build model before wrapping model"

        if args.distributed:
            ngpus_per_node = args.ngpus_per_node
            # Apply SyncBN
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                model.cuda(args.gpu)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs we have
                self.batch_size = args.batch_size // ngpus_per_node
                self.workers = (args.workers + ngpus_per_node - 1) // ngpus_per_node
                print("=> Finish adapting batch size and workers according to gpu number")
                model = nn.parallel.DistributedDataParallel(model,
                                                            device_ids=[args.gpu],
                                                            find_unused_parameters=True)
            else:
                model.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                model = nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
        elif args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model = model.cuda(args.gpu)
        else:
            # AllGather/rank implementation in this code only supports DistributedDataParallel
            raise NotImplementedError("Must Specify GPU or use DistributeDataParallel.")
        # pdb.set_trace()
        self.model = model

    def split_data(self, data, block_size):
        """
        将数据切分成指定大小的块
        """
        blocks = []
        for i in range(0, data.shape[0], block_size[0]):
            for j in range(0, data.shape[1], block_size[1]):
                for k in range(0, data.shape[2], block_size[2]):
                    block = data[i:i + block_size[0], j:j + block_size[1], k:k + block_size[2]]
                    blocks.append(block[np.newaxis, ...])
        return blocks

    def merge_data(self, blocks, original_shape):
        """
        将切分的块拼接回原始大小
        """
        merged_data = np.zeros(original_shape)
        idx = 0
        block_size = blocks[0].shape
        for i in range(0, original_shape[0], block_size[0]):
            for j in range(0, original_shape[1], block_size[1]):
                for k in range(0, original_shape[2], block_size[2]):
                    if idx < len(blocks):  # Check if idx is within the bounds of the blocks list
                        merged_data[i:i + block_size[0], j:j + block_size[1], k:k + block_size[2]] = blocks[idx]
                        idx += 1
        return merged_data

    def eval_my_net(self):
        self.get_model()
        multiprocessing.set_start_method('spawn')
        # config = configparser.ConfigParser()
        # config.read(self.args.data_path, encoding="utf-8")
        # volume_section = self.args.section

        self.target = os.path.join(self.args.output_dir, self.args.exp_name)

        original_data = read_tiff_stack(self.args.data_root)
        # original_data = original_data / 65535
        original_data = original_data.astype(np.int64)

        # pdb.set_trace()

        # original_data = ((original_data - original_data.min()) / (original_data.max() - original_data.min()))

        original_data_shape = original_data.shape
        # pdb.set_trace()

        # 将数据切分成150*150*150的块
        block_size = (150, 150, 150)
        data_blocks = self.split_data(original_data, block_size)

        seg_sets = DataLoader(data_blocks, batch_size=self.args.batch_size, shuffle=False)

        predicted_blocks = []

        for datas in seg_sets:
            normalized_images = []

            for img in datas:
                img = img / 65535  # 假设图像数据是uint16格式的，先转换为 float 类型
                img_min = img.min()
                img_max = img.max()
                normalized_img = (img - img_min) / (img_max - img_min)
                normalized_images.append(normalized_img)

            images = torch.stack(normalized_images)

            pred = self.model(images.to(self.device))
            # pdb.set_trace()

            if self.args.num_classes == 1:
                pred = torch.sigmoid(pred)
            else:
                pred = torch.softmax(pred, dim=1)[:, 1, ...]

            # pred = pred.reshape(-1, self.input_dim, self.input_dim, self.input_dim).detach().cpu().numpy()
            pred = pred.detach().cpu().numpy()

            pred = pred * 255

            if len(predicted_blocks) == 0:
                predicted_blocks = pred
            else:
                predicted_blocks = np.concatenate([predicted_blocks, pred])

            # pdb.set_trace()

        print("=====> Prediction list: {}".format(len(predicted_blocks)))
        # # 将预测的块拼接回原始大小
        # predicted_data = self.merge_data(predicted_blocks, original_data_shape)

        for idx, block in enumerate(predicted_blocks):
            tifffile.imwrite(self.target, str(idx) + ".tiff", np.squeeze(block))

        print("=====> Finish Prediction")

    def test_3D_volume(self):
        self.get_model()
        multiprocessing.set_start_method('spawn')
        config = configparser.ConfigParser()
        config.read(self.args.data_path, encoding="utf-8")
        volume_section = self.args.section
        dataroot = config.get(volume_section, "dataroot")
        self.target = os.path.join(self.args.output_dir, self.args.exp_name)
        if os.path.exists(self.target):
            shutil.rmtree(self.target)
        os.mkdir(self.target)
        self.overlap = self.args.overlap
        self.cube = self.input_dim - self.args.overlap * 2

        self.files = [os.path.join(dataroot, pth) for pth in sorted(os.listdir(dataroot))]
        # mapping = {}
        # for i in os.listdir(dataroot):
        #     prefix = i.split('-')[0]
        #     mapping[prefix] = i[len(prefix):]
        # self.files = [os.path.join(dataroot, "z:" + str(i) + mapping["z:" + str(i)]) for i in range(1, 1524)]

        shape_y, shape_x = np.array(Image.open(self.files[0])).shape
        self.s_x, self.s_y, self.s_z = [int(i) for i in config.get(volume_section, "start_point").split(',')]
        self.e_x, self.e_y, self.e_z = [int(i) for i in config.get(volume_section, "end_point").split(',')]

        assert 0 <= self.s_x and self.e_x <= shape_x and 0 <= self.s_y and self.e_y <= shape_y
        assert self.s_x < self.e_x and self.s_y < self.e_y and self.s_z < self.e_z

        self.begin_x, self.begin_y = max(0, self.s_x - self.overlap), max(0, self.s_y - self.overlap)
        self.end_x, self.end_y = min(shape_x, self.e_x + self.overlap), min(shape_y, self.e_y + self.overlap)
        self.pad_s_x = self.begin_x - self.s_x + self.overlap
        self.pad_s_y = self.begin_y - self.s_y + self.overlap
        self.pad_e_x = self.e_x + self.overlap - self.end_x
        self.pad_e_y = self.e_y + self.overlap - self.end_y

        assert self.s_x - self.e_x < self.cube and self.s_y - self.e_y < self.cube and self.s_z - self.e_z < self.cube

        return [z for z in range(self.s_z, self.e_z - self.cube, self.cube)] + [int(self.e_z) - self.cube]

    def segment_brain_batch(self, z):
        volume = []
        for i in range(z - self.overlap, z + self.cube + self.overlap):
            if 0 <= i < len(self.files):
                im = np.array(Image.open(self.files[i])).astype(np.float32)
                img = im[self.begin_y: self.end_y, self.begin_x: self.end_x]
                img = np.pad(img, ((self.pad_s_y, self.pad_e_y), (self.pad_s_x, self.pad_e_x)), 'edge')
                volume.append(img)
            else:
                blank = np.zeros((self.end_y - self.begin_y + self.pad_s_y + self.pad_e_y,
                                  self.end_x - self.begin_x + self.pad_s_x + self.pad_e_x))
                volume.append(blank)
        volume = np.array(volume).astype(np.float32)

        # volume = gamma_correction(volume, 1/4)

        seg_res = np.zeros_like(volume)
        shape_y, shape_x = volume.shape[1:]
        seg = []
        overlap = self.overlap
        cube = self.cube
        mean = im.mean()
        std = im.std()
        print(mean, std)
        for y in range(overlap, shape_y - cube - overlap + 1, cube):
            for x in range(overlap, shape_x - cube - overlap + 1, cube):
                v = volume[:, y - overlap: y - overlap + self.input_dim, x - overlap: x - overlap + self.input_dim]
                seg.append(equal(v[np.newaxis, ...], mean, std))
                # seg.append(equal(v)[np.newaxis, ...])
                if x + 2 * cube + overlap >= shape_x and y + 2 * cube + overlap >= shape_y:
                    v = volume[:, shape_y - self.input_dim: shape_y, shape_x - self.input_dim: shape_x][np.newaxis, ...]
                    seg.append(equal(v, mean, std))
                    # seg.append(equal(v))
                if x + 2 * cube + overlap >= shape_x:
                    v = volume[:, y - overlap: y + overlap + cube, shape_x - self.input_dim: shape_x][np.newaxis, ...]
                    seg.append(equal(v, mean, std))
                    # seg.append(equal(v))
                if y + 2 * cube + overlap >= shape_y:
                    v = volume[:, shape_y - self.input_dim: shape_y, x - overlap: x + overlap + cube][np.newaxis, ...]
                    seg.append(equal(v, mean, std))
                    # seg.append(equal(v))
        print('crop finished.')
        seg_sets = DataLoader(seg, batch_size=self.args.batch_size, shuffle=False)

        segments = []
        for datas in seg_sets:
            pred = self.model(datas.to(self.device))
            if self.args.num_classes == 1:
                pred = torch.sigmoid(pred)
            else:
                pred = torch.softmax(pred, dim=1)[:, 1, ...]

            pred = pred.reshape(-1, self.input_dim, self.input_dim, self.input_dim).detach().cpu().numpy()
            pred = pred[:, overlap: overlap + cube, overlap: overlap + cube, overlap: overlap + cube] * 255

            # pred[pred < self.args.threshold] = 0
            # pred[pred >= self.args.threshold] = 255

            if len(segments) == 0:
                segments = pred
            else:
                segments = np.concatenate((segments, pred), axis=0)
        i = 0
        for y in range(overlap, shape_y - cube - overlap + 1, cube):
            for x in range(overlap, shape_x - cube - overlap + 1, cube):
                seg_res[overlap: self.input_dim - overlap, y: y + cube, x: x + cube] = segments[i]
                i += 1
                if x + 2 * cube + overlap >= shape_x and y + 2 * cube + overlap >= shape_y:
                    seg_res[overlap: self.input_dim - overlap, shape_y - overlap - cube: shape_y - overlap,
                    shape_x - cube - overlap: shape_x - overlap] = segments[i]
                    i += 1
                if x + 2 * cube + overlap >= shape_x:
                    seg_res[overlap: self.input_dim - overlap, y: y + cube,
                    shape_x - cube - overlap: shape_x - overlap] = segments[i]
                    i += 1
                if y + 2 * cube + overlap >= shape_y:
                    seg_res[overlap: self.input_dim - overlap, shape_y - overlap - cube: shape_y - overlap,
                    x: x + cube] = segments[i]
                    i += 1
        i = z
        for img in seg_res[overlap: self.input_dim - overlap, overlap: shape_y - overlap, overlap: shape_x - overlap]:
            tifffile.imsave(os.path.join(self.target, str(i).zfill(4) + '.tiff'), img.astype(np.uint8))
            i += 1

        print(z)

    @staticmethod
    def eval_net(model, testloader, device, n_val):
        total_loss_iou = 0
        total_loss_tiou = 0
        junk_rat = 0
        cl_recall, cl_acc = 0, 0
        global_steps = 0
        model.eval()
        with tqdm(total=n_val, desc='Validation round', unit='batch') as pbar:
            for batch, (data, label) in enumerate(testloader):
                data = Variable(data.to(device))
                label = Variable(label.clone().to(device))
                with torch.no_grad():
                    pre = model(data)
                    if len(label.shape) == 4:
                        pre = torch.argmax(torch.softmax(pre, dim=1), dim=1).unsqueeze(1).float()
                        label = label[:, np.newaxis, ...].float()
                    else:
                        pre = torch.sigmoid(pre)
                pre[pre > 0.5] = 1
                pre[pre <= 0.5] = 0
                # label[label == 255] = 0
                # label[label > 0.5] = 1
                # label[label <= 0.5] = 0
                # tifffile.imsave('./predict/' + str(batch) + '.tiff', pre.cpu().numpy()[0][0])
                # tifffile.imsave('./predict/' + str(batch) + '_label.tiff', label.cpu().numpy()[0][0])
                # tifffile.imsave('./predict/' + str(batch) + '_data.tiff', data.cpu().numpy()[0][0])

                total_loss_iou += iou(pre, label).cpu()
                total_loss_tiou += t_iou(pre, label).cpu()
                junk_rat += junk_ratio(pre, label).cpu()
                recall, acc = soft_cldice_f1(pre, label)
                cl_recall += recall.cpu()
                cl_acc += acc.cpu()
                global_steps += 1
                pbar.update(data.shape[0])
        model.train()
        cl_recall_mean = cl_recall / global_steps
        cl_acc_mean = cl_acc / global_steps
        f = open("../dcn_trans_cur_cldice.txt", "a", newline='')
        f.write(str(((2. * cl_recall_mean * cl_acc_mean) / (cl_recall_mean + cl_acc_mean)).item()) + '\n')
        f.close()
        return {'iou': total_loss_iou / global_steps,
                'cldice': (2. * cl_recall_mean * cl_acc_mean) / (cl_recall_mean + cl_acc_mean),
                'cl_acc': cl_acc_mean,
                'cl_recall': cl_recall_mean,
                'tiou': total_loss_tiou / global_steps,
                'junk_ratio': junk_rat / global_steps}

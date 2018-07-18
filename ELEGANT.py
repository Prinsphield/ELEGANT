# -*- coding:utf-8 -*-
# Created Time: 2018/03/12 10:48:38
# Author: Taihong Xiao <xiaotaihong@126.com>

from dataset import config, MultiCelebADataset
from nets import Encoder, Decoder, Discriminator

import os
import argparse
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from tensorboardX import SummaryWriter
from itertools import chain


class ELEGANT(object):
    def __init__(self, args,
                 config=config, dataset=MultiCelebADataset, \
                 encoder=Encoder, decoder=Decoder, discriminator=Discriminator):

        self.args = args
        self.attributes = args.attributes
        self.n_attributes = len(self.attributes)
        self.gpu = args.gpu
        self.mode = args.mode
        self.restore = args.restore

        # init dataset and networks
        self.config = config
        self.dataset = dataset(self.attributes)
        self.Enc = encoder()
        self.Dec = decoder()
        self.D1  = discriminator(self.n_attributes, self.config.nchw[-1])
        self.D2  = discriminator(self.n_attributes, self.config.nchw[-1]//2)

        self.adv_criterion = torch.nn.BCELoss()
        self.recon_criterion = torch.nn.MSELoss()

        self.set_mode_and_gpu()
        self.restore_from_file()


    def set_mode_and_gpu(self):
        if self.mode == 'train':
            self.Enc.train()
            self.Dec.train()
            self.D1.train()
            self.D2.train()

            if self.gpu:
                with torch.cuda.device(self.gpu[0]):
                    self.Enc.cuda()
                    self.Dec.cuda()
                    self.D1.cuda()
                    self.D2.cuda()
                    self.adv_criterion.cuda()
                    self.recon_criterion.cuda()

            if len(self.gpu) > 1:
                self.Enc = torch.nn.DataParallel(self.Enc, device_ids=self.gpu)
                self.Dec = torch.nn.DataParallel(self.Dec, device_ids=self.gpu)
                self.D1  = torch.nn.DataParallel(self.D1, device_ids=self.gpu)
                self.D2  = torch.nn.DataParallel(self.D2, device_ids=self.gpu)

        elif self.mode == 'test':
            self.Enc.eval()
            self.Dec.eval()

            if self.gpu:
                with torch.cuda.device(self.gpu[0]):
                    self.Enc.cuda()
                    self.Dec.cuda()

            if len(self.gpu) > 1:
                self.Enc = torch.nn.DataParallel(self.Enc, device_ids=self.gpu)
                self.Dec = torch.nn.DataParallel(self.Dec, device_ids=self.gpu)

        else:
            raise NotImplementationError()

    def restore_from_file(self):
        if self.restore is not None:
            ckpt_file_enc = os.path.join(self.config.model_dir, 'Enc_iter_{:06d}.pth'.format(self.restore))
            assert os.path.exists(ckpt_file_enc)
            self.Enc.load_state_dict(torch.load(ckpt_file_enc))

            ckpt_file_dec = os.path.join(self.config.model_dir, 'Dec_iter_{:06d}.pth'.format(self.restore))
            assert os.path.exists(ckpt_file_dec)
            self.Dec.load_state_dict(torch.load(ckpt_file_dec))

            if self.mode == 'train':
                ckpt_file_d1  = os.path.join(self.config.model_dir, 'D1_iter_{:06d}.pth'.format(self.restore))
                assert os.path.exists(ckpt_file_d1)
                self.D1.load_state_dict(torch.load(ckpt_file_d1))

                ckpt_file_d2  = os.path.join(self.config.model_dir, 'D2_iter_{:06d}.pth'.format(self.restore))
                assert os.path.exists(ckpt_file_d2)
                self.D2.load_state_dict(torch.load(ckpt_file_d2))

            self.start_step = self.restore + 1
        else:
            self.start_step = 1

    def tensor2var(self, tensors, volatile=False):
        if not hasattr(tensors, '__iter__'): tensors = [tensors]
        out = []
        for tensor in tensors:
            if len(self.gpu):
                tensor = tensor.cuda(self.gpu[0])
            var = torch.autograd.Variable(tensor, volatile=volatile)
            out.append(var)
        if len(out) == 1:
            return out[0]
        else:
            return out

    def get_attr_chs(self, encodings, attribute_id):
        num_chs = encodings.size(1)
        per_chs = float(num_chs) / self.n_attributes
        start = int(np.rint(per_chs * attribute_id))
        end = int(np.rint(per_chs * (attribute_id + 1)))
        return encodings[:, start:end]


    def forward_G(self):
        self.z_A, self.A_skip = self.Enc(self.A, return_skip=True)
        self.z_B, self.B_skip = self.Enc(self.B, return_skip=True)

        self.z_C = torch.cat([self.get_attr_chs(self.z_A, i) if i != self.attribute_id \
                              else self.get_attr_chs(self.z_B, i)  for i in range(self.n_attributes)], 1)
        self.z_D = torch.cat([self.get_attr_chs(self.z_B, i) if i != self.attribute_id \
                              else self.get_attr_chs(self.z_A, i)  for i in range(self.n_attributes)], 1)

        self.R_A = self.Dec(self.z_A, self.z_A, skip=self.A_skip)
        self.R_B = self.Dec(self.z_B, self.z_B, skip=self.B_skip)
        self.R_C = self.Dec(self.z_C, self.z_A, skip=self.A_skip)
        self.R_D = self.Dec(self.z_D, self.z_B, skip=self.B_skip)

        self.A1 = torch.clamp(self.A + self.R_A, -1, 1)
        self.B1 = torch.clamp(self.B + self.R_B, -1, 1)
        self.C  = torch.clamp(self.A + self.R_C, -1, 1)
        self.D  = torch.clamp(self.B + self.R_D, -1, 1)

    def forward_D_real_sample(self):
        self.d1_A = self.D1(self.A, self.y_A)
        self.d1_B = self.D1(self.B, self.y_B)
        self.d2_A = self.D2(self.A, self.y_A)
        self.d2_B = self.D2(self.B, self.y_B)

    def forward_D_fake_sample(self, detach):
        self.y_C, self.y_D = self.y_A.clone(), self.y_B.clone()
        self.y_C.data[:, self.attribute_id] = self.y_B.data[:, self.attribute_id]
        self.y_D.data[:, self.attribute_id] = self.y_A.data[:, self.attribute_id]

        if detach:
            self.d1_C = self.D1(self.C.detach(), self.y_C)
            self.d1_D = self.D1(self.D.detach(), self.y_D)
            self.d2_C = self.D2(self.C.detach(), self.y_C)
            self.d2_D = self.D2(self.D.detach(), self.y_D)
        else:
            self.d1_C = self.D1(self.C, self.y_C)
            self.d1_D = self.D1(self.D, self.y_D)
            self.d2_C = self.D2(self.C, self.y_C)
            self.d2_D = self.D2(self.D, self.y_D)

    def compute_loss_D(self):
        self.D_loss = {
            'D1':   self.adv_criterion(self.d1_A, torch.ones_like(self.d1_A))  + \
                    self.adv_criterion(self.d1_B, torch.ones_like(self.d1_B))  + \
                    self.adv_criterion(self.d1_C, torch.zeros_like(self.d1_C)) + \
                    self.adv_criterion(self.d1_D, torch.zeros_like(self.d1_D)),

            'D2':   self.adv_criterion(self.d2_A, torch.ones_like(self.d2_A))  + \
                    self.adv_criterion(self.d2_B, torch.ones_like(self.d2_B))  + \
                    self.adv_criterion(self.d2_C, torch.zeros_like(self.d2_C)) + \
                    self.adv_criterion(self.d2_D, torch.zeros_like(self.d2_D)),
        }
        self.loss_D = (self.D_loss['D1'] + 0.5 * self.D_loss['D2']) / 4

    def compute_loss_G(self):
        self.G_loss = {
            'reconstruction': self.recon_criterion(self.A1, self.A) + self.recon_criterion(self.B1, self.B),
            'adv1': self.adv_criterion(self.d1_C, torch.ones_like(self.d1_C)) + \
                    self.adv_criterion(self.d1_D, torch.ones_like(self.d1_D)),
            'adv2': self.adv_criterion(self.d2_C, torch.ones_like(self.d2_C))  + \
                    self.adv_criterion(self.d2_D, torch.ones_like(self.d2_D)),
        }
        self.loss_G = 5 * self.G_loss['reconstruction'] + self.G_loss['adv1'] + 0.5 * self.G_loss['adv2']

    def backward_D(self):
        self.loss_D.backward()
        self.optimizer_D.step()

    def backward_G(self):
        self.loss_G.backward()
        self.optimizer_G.step()

    def img_denorm(self, img, scale=255):
        return (img + 1) * scale / 2.

    def save_image_log(self, save_num=20):
        image_info = {
            'A/img'   : self.img_denorm(self.A.data.cpu(), 1)[:save_num],
            'B/img'   : self.img_denorm(self.B.data.cpu(), 1)[:save_num],
            'C/img'   : self.img_denorm(self.C.data.cpu(), 1)[:save_num],
            'D/img'   : self.img_denorm(self.D.data.cpu(), 1)[:save_num],
            'A1/img'  : self.img_denorm(self.A1.data.cpu(), 1)[:save_num],
            'B1/img'  : self.img_denorm(self.B1.data.cpu(), 1)[:save_num],
            'R_A/img' : self.img_denorm(self.R_A.data.cpu(), 1)[:save_num],
            'R_B/img' : self.img_denorm(self.R_B.data.cpu(), 1)[:save_num],
            'R_C/img' : self.img_denorm(self.R_C.data.cpu(), 1)[:save_num],
            'R_D/img' : self.img_denorm(self.R_D.data.cpu(), 1)[:save_num],
        }
        for tag, images in image_info.items():
            for idx, image in enumerate(images):
                self.writer.add_image(tag+'/{}_{:02d}'.format(self.attribute_id, idx), image, self.step)

    def save_sample_images(self, save_num=5):
        canvas = torch.cat((self.A, self.B, self.C, self.D, self.A1, self.B1), -1)
        img_array = np.transpose(self.img_denorm(canvas.data.cpu().numpy()), (0,2,3,1)).astype(np.uint8)
        for i in range(save_num):
            Image.fromarray(img_array[i]).save(os.path.join(self.config.img_dir, 'step_{:06d}_attr_{}_{:02d}.jpg'.format(self.step, self.attribute_id, i)))

    def save_scalar_log(self):
        scalar_info = {
            'loss_D': self.loss_D.data.cpu().numpy(),
            'loss_G': self.loss_G.data.cpu().numpy(),
            'G_lr'  : self.G_lr_scheduler.get_lr()[0],
            'D_lr'  : self.D_lr_scheduler.get_lr()[0],
        }

        for key, value in self.G_loss.items():
            scalar_info['G_loss/' + key] = value.data[0]

        for key, value in self.D_loss.items():
            scalar_info['D_loss/' + key] = value.data[0]

        for tag, value in scalar_info.items():
            self.writer.add_scalar(tag, value, self.step)

    def save_model(self):
        torch.save({key: val.cpu() for key, val in self.Enc.state_dict().items()}, os.path.join(self.config.model_dir, 'Enc_iter_{:06d}.pth'.format(self.step)))
        torch.save({key: val.cpu() for key, val in self.Dec.state_dict().items()}, os.path.join(self.config.model_dir, 'Dec_iter_{:06d}.pth'.format(self.step)))
        torch.save({key: val.cpu() for key, val in self.D1.state_dict().items()},  os.path.join(self.config.model_dir, 'D1_iter_{:06d}.pth'.format(self.step)))
        torch.save({key: val.cpu() for key, val in self.D2.state_dict().items()},  os.path.join(self.config.model_dir, 'D2_iter_{:06d}.pth'.format(self.step)))

    def train(self):
        self.writer = SummaryWriter(self.config.log_dir)

        self.optimizer_G = torch.optim.Adam(chain(self.Enc.parameters(), self.Dec.parameters()),
                                       lr=self.config.G_lr, betas=(0.5, 0.999),
                                       weight_decay=self.config.weight_decay)

        self.optimizer_D = torch.optim.Adam(chain(self.D1.parameters(), self.D2.parameters()),
                                       lr=self.config.D_lr, betas=(0.5, 0.999),
                                       weight_decay=self.config.weight_decay)

        self.G_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer_G, step_size=self.config.step_size, gamma=self.config.gamma)
        self.D_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer_D, step_size=self.config.step_size, gamma=self.config.gamma)

        # start training
        for step in range(self.start_step, 1 + self.config.max_iter):
            self.step = step
            self.G_lr_scheduler.step()
            self.D_lr_scheduler.step()

            for attribute_id in range(self.n_attributes):
                self.attribute_id = attribute_id
                A, y_A = next(self.dataset.gen(attribute_id, True))
                B, y_B = next(self.dataset.gen(attribute_id, False))
                self.A, self.y_A, self.B, self.y_B = self.tensor2var([A, y_A, B, y_B])

                # forward
                self.forward_G()

                # update D
                self.forward_D_real_sample()
                self.forward_D_fake_sample(detach=True)
                self.compute_loss_D()
                self.optimizer_D.zero_grad()
                self.backward_D()

                # update G
                self.forward_D_fake_sample(detach=False)
                self.compute_loss_G()
                self.optimizer_G.zero_grad()
                self.backward_G()

                if self.step % 100 == 0:
                    self.save_image_log()

                if self.step % 2000 == 0:
                    self.save_sample_images()

            print('step: %06d, loss D: %.6f, loss G: %.6f' % (self.step, self.loss_D.data.cpu().numpy(), self.loss_G.data.cpu().numpy()))

            if self.step % 100 == 0:
                self.save_scalar_log()

            if self.step % 2000 == 0:
                self.save_model()

        print('Finished Training!')
        self.writer.close()

    def transform(self, *images):
        transform1 = transforms.Compose([
            transforms.Resize(self.config.nchw[-2:]),
            transforms.ToTensor(),
        ])
        transform2 = lambda x: x.view(1, *x.size()) * 2 - 1
        out = [transform2(transform1(image)) for image in images]
        return out

    def swap(self):
        '''
        swap attributes of two images.
        '''
        self.attribute_id = self.args.swap_list[0]
        self.B, self.A = self.tensor2var(self.transform(Image.open(self.args.input), Image.open(self.args.target[0])), volatile=True)

        self.forward_G()
        img = torch.cat((self.B, self.A, self.D, self.C), -1)
        img = np.transpose(self.img_denorm(img.data.cpu().numpy()), (0,2,3,1)).astype(np.uint8)[0]
        Image.fromarray(img).save('swap.jpg')

    def linear(self):
        '''
        linear interpolation of two images.
        '''
        self.attribute_id = self.args.swap_list[0]
        self.B, self.A = self.tensor2var(self.transform(Image.open(self.args.input), Image.open(self.args.target[0])), volatile=True)

        self.z_A = self.Enc(self.A, return_skip=False)
        self.z_B, self.B_skip = self.Enc(self.B, return_skip=True)

        self.z_D = torch.cat([self.get_attr_chs(self.z_B, i) if i != self.attribute_id \
                              else self.get_attr_chs(self.z_A, i)  for i in range(self.n_attributes)], 1)

        m = self.args.size[0]
        out = [self.B]
        for i in range(1, 1+m):
            z_i = float(i) / m * (self.z_D - self.z_B) + self.z_B
            R_i = self.Dec(z_i, self.z_B, skip=self.B_skip)
            D_i = torch.clamp(self.B + R_i, -1, 1)
            out.append(D_i)
        out.append(self.A)
        out = torch.cat(out, -1)
        img = np.transpose(self.img_denorm(out.data.cpu().numpy()), (0,2,3,1)).astype(np.uint8)[0]
        Image.fromarray(img).save('linear_interpolation.jpg')

    def matrix1(self):
        '''
        matrix interpolation with respect to one attribute.
        '''
        self.attribute_id = self.args.swap_list[0]
        self.B = self.tensor2var(self.transform(Image.open(self.args.input)), volatile=True)
        self.As = [self.tensor2var(self.transform(Image.open(self.args.target[i])), volatile=True) for i in range(3)]

        self.z_B, self.B_skip = self.Enc(self.B, return_skip=True)
        self.z_As = [self.Enc(self.As[i], return_skip=False) for i in range(3)]

        self.z_Ds = [torch.cat([self.get_attr_chs(self.z_B, i) if i != self.attribute_id \
                              else self.get_attr_chs(self.z_As[j], i)  for i in range(self.n_attributes)], 1)
                     for j in range(3)]

        m, n = self.args.size
        h, w = self.config.nchw[-2:]

        out = torch.ones(1, 3, m * h, n * w)
        for i in range(m):
            for j in range(n):
                a = i / float(m - 1)
                b = j / float(n - 1)
                four = [(1-a) * (1-b), (1-a) * b, a * (1-b), a * b]
                z_ij = four[0] * self.z_B + four[1] * self.z_Ds[0] + four[2] * self.z_Ds[1] + four[3] * self.z_Ds[2]
                R_ij = self.Dec(z_ij, self.z_B, skip=self.B_skip)
                D_ij = torch.clamp(self.B + R_ij, -1, 1)
                out[:,:, i*h:(i+1)*h, j*w:(j+1)*w] = D_ij.data.cpu()

        first_col = torch.cat((self.B.data.cpu(), torch.ones(1,3,(m-2)*h,w), self.As[1].data.cpu()), -2)
        last_col = torch.cat((self.As[0].data.cpu(), torch.ones(1,3,(m-2)*h,w), self.As[2].data.cpu()), -2)
        canvas = torch.cat((first_col, out, last_col), -1)
        img = np.transpose(self.img_denorm(canvas.numpy()), (0,2,3,1)).astype(np.uint8)[0]
        Image.fromarray(img).save('matrix_interpolation1.jpg')

    def matrix2(self):
        '''
        matrix interpolation with respect to two attributes simultaneously.
        '''
        self.attribute_ids = self.args.swap_list
        self.B, self.A1, self.A2 = self.tensor2var(self.transform(Image.open(self.args.input), Image.open(self.args.target[0]), Image.open(self.args.target[1])), volatile=True)

        self.z_B, self.B_skip = self.Enc(self.B, return_skip=True)
        self.z_A1, self.z_A2 = self.Enc(self.A1, return_skip=False), self.Enc(self.A2, return_skip=False)

        self.z_D1 = torch.cat([self.get_attr_chs(self.z_B, i) if i != self.attribute_ids[0]
                              else self.get_attr_chs(self.z_A1, i)  for i in range(self.n_attributes)], 1)

        self.z_D2 = torch.cat([self.get_attr_chs(self.z_B, i) if i != self.attribute_ids[1]
                              else self.get_attr_chs(self.z_A2, i)  for i in range(self.n_attributes)], 1)

        m, n = self.args.size
        h, w = self.config.nchw[-2:]

        out = torch.ones(1, 3, m * h, n * w)
        for i in range(m):
            for j in range(n):
                a = i / float(m - 1)
                b = j / float(n - 1)
                z_ij = a * self.z_D1 + b * self.z_D2 + (1 - a - b) * self.z_B
                R_ij = self.Dec(z_ij, self.z_B, skip=self.B_skip)
                D_ij = torch.clamp(self.B + R_ij, -1, 1)
                out[:,:, i*h:(i+1)*h, j*w:(j+1)*w] = D_ij.data.cpu()

        first_col = torch.cat((self.B.data.cpu(), torch.ones(1,3,(m-2)*h,w), self.A1.data.cpu()), -2)
        last_col = torch.cat((self.A2.data.cpu(), torch.ones(1,3,(m-1)*h,w)), -2)
        canvas = torch.cat((first_col, out, last_col), -1)
        img = np.transpose(self.img_denorm(canvas.numpy()), (0,2,3,1)).astype(np.uint8)[0]
        Image.fromarray(img).save('matrix_interpolation2.jpg')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--attributes', nargs='+', type=str, help='Specify attribute names.')
    parser.add_argument('-g', '--gpu', default=[], nargs='+', type=int, help='Specify GPU ids.')
    parser.add_argument('-m', '--mode', default='train', type=str, choices=['train', 'test'])
    parser.add_argument('-r', '--restore', default=None, action='store', type=int, help='Specify checkpoint id to restore')

    # test parameters
    parser.add_argument('--swap', action='store_true', help='Swap attributes.')
    parser.add_argument('--linear', action='store_true', help='Linear interpolation.')
    parser.add_argument('--matrix', action='store_true', help='Matraix interpolation with respect to one attribute.')
    parser.add_argument('--swap_list', default=[], nargs='+', type=int, help='Specify the attributes ids for swapping.')
    parser.add_argument('-i', '--input', type=str, help='Specify the input image.')
    parser.add_argument('-t', '--target', nargs='+', type=str, help='Specify target images.')
    parser.add_argument('-s', '--size', nargs='+', type=int, help='Specify the interpolation size.')

    args = parser.parse_args()
    print(args)

    if args.mode == 'test':
        assert args.swap + args.linear + args.matrix == 1
        assert args.restore is not None

    model = ELEGANT(args)
    if args.mode == 'train':
        model.train()
    elif args.mode == 'test' and args.swap:
        assert len(args.swap_list) == 1 and args.input and len(args.target) == 1
        model.swap()
    elif args.mode == 'test' and args.linear:
        assert len(args.swap_list) == 1 and len(args.size) == 1
        model.linear()
    elif args.mode == 'test' and args.matrix:
        assert len(args.swap_list) in [1,2]
        if len(args.swap_list) == 1:
            assert len(args.target) == 3 and len(args.size) == 2
            model.matrix1()
        elif len(args.swap_list) == 2:
            assert len(args.target) == 2 and len(args.size) == 2
            model.matrix2()
    else:
        raise NotImplementationError()


if __name__ == "__main__":
    main()

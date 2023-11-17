import numpy as np

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.nn.functional as nnf
from scipy.ndimage.filters import gaussian_filter
import random
import math

def random_noise(data, ranges=[-1, 1], min_data=-0.98, rand_point=[3, 8]):
    back_ground_coord = torch.where(data < min_data)
    back_ground_valu = data[back_ground_coord]
    control_point = random.randint(rand_point[0], rand_point[1])
    control_point = 8
    distribu = torch.rand(control_point) * (ranges[1] - ranges[0]) + ranges[0]
    distribu, _ = torch.sort(distribu)
    ### --> 0 point1 ... pointN, 1
    distribu = torch.cat([torch.tensor([ranges[0]]), distribu])
    distribu = torch.cat([distribu, torch.tensor([ranges[1]])]).cuda()
    shuffle_part = torch.randperm(control_point + 1)

    new_image = torch.zeros_like(data)
    for i in range(control_point + 1):
        target_part = shuffle_part[i]
        min1, max1 = distribu[i], distribu[i + 1]
        min2, max2 = distribu[target_part], distribu[target_part + 1]
        # print (min1,max1)
        coord = torch.where((min1 <= data) & (data < max1))
        new_image[coord] = ((data[coord] - min1) / (max1 - min1)) * (max2 - min2) + min2

    # if torch.rand(1) < 0.2:
    #     new_image = -new_image
    #
    # if torch.rand(1) < 0.2:
    #     new_image = torch.from_numpy(histgram_shift(new_image)).to(torch.float32)
    # if torch.rand(1) < 0.2:
    #     new_image = aug_func(new_image)
    # if torch.rand(1) < 0.2:
    new_image[back_ground_coord] = back_ground_valu
    new_image = torch.clamp(new_image, -1, 1).to(torch.float32)

    return new_image
def small_histgram_shift(data, pointranges=[4, 10]):



    data = data.squeeze()
    data = data.cpu().detach().numpy()
    num_control_point = random.randint(pointranges[0], pointranges[1])
    reference_control_points = torch.linspace(-1, 1, num_control_point)
    floating_control_points = reference_control_points.clone()

    for i in range(1, num_control_point - 1):
        floating_control_points[i] = floating_control_points[i - 1] + torch.rand(
            1) * (floating_control_points[i + 1] - floating_control_points[i - 1])
    img_min, img_max = data.min(), data.max()
    reference_control_points_scaled = (reference_control_points *
                                       (img_max - img_min) + img_min).numpy()
    floating_control_points_scaled = (floating_control_points *
                                      (img_max - img_min) + img_min).numpy()
    data_shifted = np.interp(data, reference_control_points_scaled,
                             floating_control_points_scaled)

    data_shifted = torch.from_numpy(data_shifted).to(torch.float32).cuda()
    data_shifted = torch.clamp(data_shifted, -1, 1).unsqueeze(0).unsqueeze(0)



    return data_shifted



class Transformer2D(nn.Module):
    def __init__(self):
        super(Transformer2D, self).__init__()

    def forward(self, src, flow, padding_mode="border"):
        b = flow.shape[0]
        size = flow.shape[2:]
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = grid.to(torch.float32)
        grid = grid.repeat(b, 1, 1, 1).to(flow.device)
        new_locs = grid + flow
        shape = flow.shape[2:]
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)
        new_locs = new_locs.permute(0, 2, 3, 1)
        new_locs = new_locs[..., [1, 0]].cuda()
        warped = F.grid_sample(src, new_locs, align_corners=True, mode = "nearest",padding_mode=padding_mode,)
        return warped
def create_affine_transformation_matrix(n_dims, scaling=None, rotation=None, shearing=None, translation=None):
    """
        create a 4x4 affine transformation matrix from specified values
    :param n_dims: integer
    :param scaling: list of 3 scaling values
    :param rotation: list of 3 angles (degrees) for rotations around 1st, 2nd, 3rd axis
    :param shearing: list of 6 shearing values
    :param translation: list of 3 values
    :return: 4x4 numpy matrix
    """

    trans_scaling = np.eye(n_dims + 1)
    trans_shearing = np.eye(n_dims + 1)
    trans_translation = np.eye(n_dims + 1)

    if scaling is not None:
        trans_scaling[np.arange(n_dims + 1), np.arange(n_dims + 1)] = np.append(scaling, 1)

    if shearing is not None:
        shearing_index = np.ones((n_dims + 1, n_dims + 1), dtype='bool')
        shearing_index[np.eye(n_dims + 1, dtype='bool')] = False
        shearing_index[-1, :] = np.zeros((n_dims + 1))
        shearing_index[:, -1] = np.zeros((n_dims + 1))
        trans_shearing[shearing_index] = shearing

    if translation is not None:
        trans_translation[np.arange(n_dims), n_dims *
                          np.ones(n_dims, dtype='int')] = translation

    if n_dims == 2:
        if rotation is None:
            rotation = np.zeros(1)
        else:
            rotation = np.asarray(rotation) * (math.pi / 180)
        trans_rot = np.eye(n_dims + 1)
        trans_rot[np.array([0, 1, 0, 1]), np.array([0, 0, 1, 1])] = [np.cos(rotation), np.sin(rotation),
                                                                     np.sin(rotation) * -1, np.cos(rotation)]
        return trans_translation @ trans_rot @ trans_shearing @ trans_scaling

    else:
        if rotation is None:
            rotation = np.zeros(n_dims)
        else:
            rotation = np.asarray(rotation) * (math.pi / 180)
        trans_rot1 = np.eye(n_dims + 1)
        trans_rot1[np.array([1, 2, 1, 2]), np.array([1, 1, 2, 2])] = [np.cos(rotation[0]),
                                                                      np.sin(
                                                                          rotation[0]),
                                                                      np.sin(
                                                                          rotation[0]) * -1,
                                                                      np.cos(rotation[0])]
        trans_rot2 = np.eye(n_dims + 1)
        trans_rot2[np.array([0, 2, 0, 2]), np.array([0, 0, 2, 2])] = [np.cos(rotation[1]),
                                                                      np.sin(
                                                                          rotation[1]) * -1,
                                                                      np.sin(
                                                                          rotation[1]),
                                                                      np.cos(rotation[1])]
        trans_rot3 = np.eye(n_dims + 1)
        trans_rot3[np.array([0, 1, 0, 1]), np.array([0, 0, 1, 1])] = [np.cos(rotation[2]),
                                                                      np.sin(
                                                                          rotation[2]),
                                                                      np.sin(
                                                                          rotation[2]) * -1,
                                                                      np.cos(rotation[2])]
        return trans_translation @ trans_rot3 @ trans_rot2 @ trans_rot1 @ trans_shearing @ trans_scaling
def affine(random_numbers, imgs, padding_modes, opt):
    if not isinstance(imgs, list) and not isinstance(imgs, tuple):
        imgs = [imgs]
    if not isinstance(padding_modes, list) and not isinstance(padding_modes, tuple):
        padding_modes = [padding_modes]


    scaling = random_numbers[0:2] * opt['scaling'] + 1
    rotation = random_numbers[2] * opt['rotation']*2
    translation = random_numbers[3] * opt['translation']

    theta = create_affine_transformation_matrix(
        n_dims=2, scaling=scaling, rotation=rotation, shearing=None, translation=translation)
    theta = theta[:-1, :]
    theta = torch.from_numpy(theta).to(torch.float32)

    # print (imgs[0].shape)
    size = imgs[0].size()
    # print (size)
    grid = F.affine_grid(theta.unsqueeze(0), size, align_corners=True).cuda()

    res_img = []
    for img, mode in zip(imgs, padding_modes):
        res_img.append(F.grid_sample(img, grid, align_corners=True,mode="nearest", padding_mode=mode).squeeze(0))

    return res_img[0] if len(res_img) == 1 else res_img
def non_affine_2d(imgs, padding_modes, opt, elastic_random=None):
    if not isinstance(imgs, list) and not isinstance(imgs, tuple):
        imgs = [imgs]
    if not isinstance(padding_modes, list) and not isinstance(padding_modes, tuple):
        padding_modes = [padding_modes]

    w, h = imgs[0].shape[-2:]
    if elastic_random is None:
        elastic_random = torch.rand([2, w, h]).numpy() * 2 - 1  # .numpy()

    sigma = 12  # 需要根据图像大小调整
    alpha = opt['non_affine_alpha']  # 需要根据图像大小调整

    dx = gaussian_filter(elastic_random[0], sigma) * alpha
    dy = gaussian_filter(elastic_random[1], sigma) * alpha
    dx = np.expand_dims(dx, 0)
    dy = np.expand_dims(dy, 0)
    flow = np.concatenate((dx, dy), 0)
    flow = np.expand_dims(flow, 0)
    flow = torch.from_numpy(flow).to(torch.float32)

    results = []
    for img, mode in zip(imgs, padding_modes):
        img = Transformer2D()(img.unsqueeze(0), flow, padding_mode=mode)
        results.append(img.squeeze(0))

    return results[0] if len(results) == 1 else results


class ResnetAdaILNBlock(nn.Module):
    def __init__(self, dim, use_bias):
        super(ResnetAdaILNBlock, self).__init__()
        self.pad1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias)
        self.norm1 = adaILN(dim)
        self.relu1 = nn.ReLU(True)

        self.pad2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias)
        self.norm2 = adaILN(dim)

    def forward(self, x, gamma, beta):
        out = self.pad1(x)
        out = self.conv1(out)
        out = self.norm1(out, gamma, beta)
        out = self.relu1(out)
        out = self.pad2(out)
        out = self.conv2(out)
        out = self.norm2(out, gamma, beta)

        return out + x


class adaILN(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.9, using_moving_average=True, using_bn=False):
        super(adaILN, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.using_bn = using_bn
        self.num_features = num_features

        if self.using_bn:
            self.rho = Parameter(torch.Tensor(1, num_features, 3))
            self.rho[:, :, 0].data.fill_(3)
            self.rho[:, :, 1].data.fill_(1)
            self.rho[:, :, 2].data.fill_(1)
            self.register_buffer('running_mean', torch.zeros(1, num_features, 1, 1))
            self.register_buffer('running_var', torch.zeros(1, num_features, 1, 1))
            self.running_mean.zero_()
            self.running_var.zero_()
        else:
            self.rho = Parameter(torch.Tensor(1, num_features, 2))
            self.rho[:, :, 0].data.fill_(3.2)
            self.rho[:, :, 1].data.fill_(1)

    def forward(self, input, gamma, beta):
        in_mean, in_var = torch.mean(input, dim=[2, 3], keepdim=True), torch.var(input, dim=[2, 3], keepdim=True)
        out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
        ln_mean, ln_var = torch.mean(input, dim=[1, 2, 3], keepdim=True), torch.var(input, dim=[1, 2, 3], keepdim=True)
        out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)
        softmax = nn.Softmax(2)
        rho = softmax(self.rho)

        if self.using_bn:
            if self.training:
                bn_mean, bn_var = torch.mean(input, dim=[0, 2, 3], keepdim=True), torch.var(input, dim=[0, 2, 3],
                                                                                            keepdim=True)
                if self.using_moving_average:
                    self.running_mean.mul_(self.momentum)
                    self.running_mean.add_((1 - self.momentum) * bn_mean.data)
                    self.running_var.mul_(self.momentum)
                    self.running_var.add_((1 - self.momentum) * bn_var.data)
                else:
                    self.running_mean.add_(bn_mean.data)
                    self.running_var.add_(bn_mean.data ** 2 + bn_var.data)
            else:
                bn_mean = torch.autograd.Variable(self.running_mean)
                bn_var = torch.autograd.Variable(self.running_var)
            out_bn = (input - bn_mean) / torch.sqrt(bn_var + self.eps)
            rho_0 = rho[:, :, 0]
            rho_1 = rho[:, :, 1]
            rho_2 = rho[:, :, 2]

            rho_0 = rho_0.view(1, self.num_features, 1, 1)
            rho_1 = rho_1.view(1, self.num_features, 1, 1)
            rho_2 = rho_2.view(1, self.num_features, 1, 1)
            rho_0 = rho_0.expand(input.shape[0], -1, -1, -1)
            rho_1 = rho_1.expand(input.shape[0], -1, -1, -1)
            rho_2 = rho_2.expand(input.shape[0], -1, -1, -1)
            out = rho_0 * out_in + rho_1 * out_ln + rho_2 * out_bn
        else:
            rho_0 = rho[:, :, 0]
            rho_1 = rho[:, :, 1]
            rho_0 = rho_0.view(1, self.num_features, 1, 1)
            rho_1 = rho_1.view(1, self.num_features, 1, 1)
            rho_0 = rho_0.expand(input.shape[0], -1, -1, -1)
            rho_1 = rho_1.expand(input.shape[0], -1, -1, -1)
            out = rho_0 * out_in + rho_1 * out_ln

        out = out * gamma.unsqueeze(2).unsqueeze(3) + beta.unsqueeze(2).unsqueeze(3)
        return out


class ILN(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.9, using_moving_average=True, using_bn=False):
        super(ILN, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.using_bn = using_bn
        self.num_features = num_features

        if self.using_bn:
            self.rho = Parameter(torch.Tensor(1, num_features, 3))
            self.rho[:, :, 0].data.fill_(1)
            self.rho[:, :, 1].data.fill_(3)
            self.rho[:, :, 2].data.fill_(3)
            self.register_buffer('running_mean', torch.zeros(1, num_features, 1, 1))
            self.register_buffer('running_var', torch.zeros(1, num_features, 1, 1))
            self.running_mean.zero_()
            self.running_var.zero_()
        else:
            self.rho = Parameter(torch.Tensor(1, num_features, 2))
            self.rho[:, :, 0].data.fill_(1)
            self.rho[:, :, 1].data.fill_(3.2)

        self.gamma = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.beta = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.gamma.data.fill_(1.0)
        self.beta.data.fill_(0.0)

    def forward(self, input):
        in_mean, in_var = torch.mean(input, dim=[2, 3], keepdim=True), torch.var(input, dim=[2, 3], keepdim=True)
        out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
        ln_mean, ln_var = torch.mean(input, dim=[1, 2, 3], keepdim=True), torch.var(input, dim=[1, 2, 3], keepdim=True)
        out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)

        softmax = nn.Softmax(2)
        rho = softmax(self.rho)

        if self.using_bn:
            if self.training:
                bn_mean, bn_var = torch.mean(input, dim=[0, 2, 3], keepdim=True), torch.var(input, dim=[0, 2, 3],
                                                                                            keepdim=True)
                if self.using_moving_average:
                    self.running_mean.mul_(self.momentum)
                    self.running_mean.add_((1 - self.momentum) * bn_mean.data)
                    self.running_var.mul_(self.momentum)
                    self.running_var.add_((1 - self.momentum) * bn_var.data)
                else:
                    self.running_mean.add_(bn_mean.data)
                    self.running_var.add_(bn_mean.data ** 2 + bn_var.data)
            else:
                bn_mean = torch.autograd.Variable(self.running_mean)
                bn_var = torch.autograd.Variable(self.running_var)
            out_bn = (input - bn_mean) / torch.sqrt(bn_var + self.eps)
            rho_0 = rho[:, :, 0]
            rho_1 = rho[:, :, 1]
            rho_2 = rho[:, :, 2]

            rho_0 = rho_0.view(1, self.num_features, 1, 1)
            rho_1 = rho_1.view(1, self.num_features, 1, 1)
            rho_2 = rho_2.view(1, self.num_features, 1, 1)
            rho_0 = rho_0.expand(input.shape[0], -1, -1, -1)
            rho_1 = rho_1.expand(input.shape[0], -1, -1, -1)
            rho_2 = rho_2.expand(input.shape[0], -1, -1, -1)
            out = rho_0 * out_in + rho_1 * out_ln + rho_2 * out_bn
        else:
            rho_0 = rho[:, :, 0]
            rho_1 = rho[:, :, 1]
            rho_0 = rho_0.view(1, self.num_features, 1, 1)
            rho_1 = rho_1.view(1, self.num_features, 1, 1)
            rho_0 = rho_0.expand(input.shape[0], -1, -1, -1)
            rho_1 = rho_1.expand(input.shape[0], -1, -1, -1)
            out = rho_0 * out_in + rho_1 * out_ln

        out = out * self.gamma.expand(input.shape[0], -1, -1, -1) + self.beta.expand(input.shape[0], -1, -1, -1)
        return out


class Resize():
    def __init__(self, size_tuple, use_cv=True):
        self.size_tuple = size_tuple
        self.use_cv = use_cv

    def __call__(self, tensor):
        """
            Resized the tensor to the specific size

            Arg:    tensor  - The torch.Tensor obj whose rank is 4
            Ret:    Resized tensor
        """
        tensor = tensor.unsqueeze(0)
        tensor = F.interpolate(tensor, size=[self.size_tuple[0], self.size_tuple[1]])
        tensor = tensor.squeeze(0)
        return tensor  # F.interpolate(tensor, size = [self.size_tuple[0]])


class ToTensor():
    def __call__(self, tensor):
        tensor = np.expand_dims(tensor, 0)
        return torch.from_numpy(tensor)


class Normalize():
    def __call__(self, tensor):
        """
            Normalized the tensor into the range [-1, 1]

            Arg:    tensor  - The torch.Tensor obj whose rank is 4
            Ret:    Normalized tensor
        """
        tensor = (tensor - 127.5) / 127.5
        assert (torch.min(tensor) >= -1) and (torch.max(tensor) <= 1)
        return tensor


def image_for_torch(array_fix):
    tensor_fix = torch.from_numpy(array_fix).unsqueeze(0).unsqueeze(0).cuda()
    resize_fix = F.interpolate(tensor_fix, size=[64, 160, 160],
                               align_corners=True, mode='trilinear')  # .squeeze(0)
    return resize_fix


class ResnetBlock(nn.Module):
    def __init__(self, dim, use_bias):
        super(ResnetBlock, self).__init__()
        conv_block = []
        conv_block += [nn.ReflectionPad2d(1),
                       nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias),
                       nn.InstanceNorm2d(dim),
                       nn.ReLU(True)]

        conv_block += [nn.ReflectionPad2d(1),
                       nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias),
                       nn.InstanceNorm2d(dim)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class ResnetAdaILNBlock(nn.Module):
    def __init__(self, dim, use_bias):
        super(ResnetAdaILNBlock, self).__init__()
        self.pad1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias)
        self.norm1 = adaILN(dim)
        self.relu1 = nn.ReLU(True)

        self.pad2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias)
        self.norm2 = adaILN(dim)

    def forward(self, x, gamma, beta):
        out = self.pad1(x)
        out = self.conv1(out)
        out = self.norm1(out, gamma, beta)
        out = self.relu1(out)
        out = self.pad2(out)
        out = self.conv2(out)
        out = self.norm2(out, gamma, beta)

        return out + x


class adaILN(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.9, using_moving_average=True, using_bn=False):
        super(adaILN, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.using_bn = using_bn
        self.num_features = num_features

        if self.using_bn:
            self.rho = Parameter(torch.Tensor(1, num_features, 3))
            self.rho[:, :, 0].data.fill_(3)
            self.rho[:, :, 1].data.fill_(1)
            self.rho[:, :, 2].data.fill_(1)
            self.register_buffer('running_mean', torch.zeros(1, num_features, 1, 1))
            self.register_buffer('running_var', torch.zeros(1, num_features, 1, 1))
            self.running_mean.zero_()
            self.running_var.zero_()
        else:
            self.rho = Parameter(torch.Tensor(1, num_features, 2))
            self.rho[:, :, 0].data.fill_(3.2)
            self.rho[:, :, 1].data.fill_(1)

    def forward(self, input, gamma, beta):
        in_mean, in_var = torch.mean(input, dim=[2, 3], keepdim=True), torch.var(input, dim=[2, 3], keepdim=True)
        out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
        ln_mean, ln_var = torch.mean(input, dim=[1, 2, 3], keepdim=True), torch.var(input, dim=[1, 2, 3], keepdim=True)
        out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)
        softmax = nn.Softmax(2)
        rho = softmax(self.rho)

        if self.using_bn:
            if self.training:
                bn_mean, bn_var = torch.mean(input, dim=[0, 2, 3], keepdim=True), torch.var(input, dim=[0, 2, 3],
                                                                                            keepdim=True)
                if self.using_moving_average:
                    self.running_mean.mul_(self.momentum)
                    self.running_mean.add_((1 - self.momentum) * bn_mean.data)
                    self.running_var.mul_(self.momentum)
                    self.running_var.add_((1 - self.momentum) * bn_var.data)
                else:
                    self.running_mean.add_(bn_mean.data)
                    self.running_var.add_(bn_mean.data ** 2 + bn_var.data)
            else:
                bn_mean = torch.autograd.Variable(self.running_mean)
                bn_var = torch.autograd.Variable(self.running_var)
            out_bn = (input - bn_mean) / torch.sqrt(bn_var + self.eps)
            rho_0 = rho[:, :, 0]
            rho_1 = rho[:, :, 1]
            rho_2 = rho[:, :, 2]

            rho_0 = rho_0.view(1, self.num_features, 1, 1)
            rho_1 = rho_1.view(1, self.num_features, 1, 1)
            rho_2 = rho_2.view(1, self.num_features, 1, 1)
            rho_0 = rho_0.expand(input.shape[0], -1, -1, -1)
            rho_1 = rho_1.expand(input.shape[0], -1, -1, -1)
            rho_2 = rho_2.expand(input.shape[0], -1, -1, -1)
            out = rho_0 * out_in + rho_1 * out_ln + rho_2 * out_bn
        else:
            rho_0 = rho[:, :, 0]
            rho_1 = rho[:, :, 1]
            rho_0 = rho_0.view(1, self.num_features, 1, 1)
            rho_1 = rho_1.view(1, self.num_features, 1, 1)
            rho_0 = rho_0.expand(input.shape[0], -1, -1, -1)
            rho_1 = rho_1.expand(input.shape[0], -1, -1, -1)
            out = rho_0 * out_in + rho_1 * out_ln

        out = out * gamma.unsqueeze(2).unsqueeze(3) + beta.unsqueeze(2).unsqueeze(3)
        return out


class ILN(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.9, using_moving_average=True, using_bn=False):
        super(ILN, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.using_bn = using_bn
        self.num_features = num_features

        if self.using_bn:
            self.rho = Parameter(torch.Tensor(1, num_features, 3))
            self.rho[:, :, 0].data.fill_(1)
            self.rho[:, :, 1].data.fill_(3)
            self.rho[:, :, 2].data.fill_(3)
            self.register_buffer('running_mean', torch.zeros(1, num_features, 1, 1))
            self.register_buffer('running_var', torch.zeros(1, num_features, 1, 1))
            self.running_mean.zero_()
            self.running_var.zero_()
        else:
            self.rho = Parameter(torch.Tensor(1, num_features, 2))
            self.rho[:, :, 0].data.fill_(1)
            self.rho[:, :, 1].data.fill_(3.2)

        self.gamma = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.beta = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.gamma.data.fill_(1.0)
        self.beta.data.fill_(0.0)

    def forward(self, input):
        in_mean, in_var = torch.mean(input, dim=[2, 3], keepdim=True), torch.var(input, dim=[2, 3], keepdim=True)
        out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
        ln_mean, ln_var = torch.mean(input, dim=[1, 2, 3], keepdim=True), torch.var(input, dim=[1, 2, 3], keepdim=True)
        out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)

        softmax = nn.Softmax(2)
        rho = softmax(self.rho)

        if self.using_bn:
            if self.training:
                bn_mean, bn_var = torch.mean(input, dim=[0, 2, 3], keepdim=True), torch.var(input, dim=[0, 2, 3],
                                                                                            keepdim=True)
                if self.using_moving_average:
                    self.running_mean.mul_(self.momentum)
                    self.running_mean.add_((1 - self.momentum) * bn_mean.data)
                    self.running_var.mul_(self.momentum)
                    self.running_var.add_((1 - self.momentum) * bn_var.data)
                else:
                    self.running_mean.add_(bn_mean.data)
                    self.running_var.add_(bn_mean.data ** 2 + bn_var.data)
            else:
                bn_mean = torch.autograd.Variable(self.running_mean)
                bn_var = torch.autograd.Variable(self.running_var)
            out_bn = (input - bn_mean) / torch.sqrt(bn_var + self.eps)
            rho_0 = rho[:, :, 0]
            rho_1 = rho[:, :, 1]
            rho_2 = rho[:, :, 2]

            rho_0 = rho_0.view(1, self.num_features, 1, 1)
            rho_1 = rho_1.view(1, self.num_features, 1, 1)
            rho_2 = rho_2.view(1, self.num_features, 1, 1)
            rho_0 = rho_0.expand(input.shape[0], -1, -1, -1)
            rho_1 = rho_1.expand(input.shape[0], -1, -1, -1)
            rho_2 = rho_2.expand(input.shape[0], -1, -1, -1)
            out = rho_0 * out_in + rho_1 * out_ln + rho_2 * out_bn
        else:
            rho_0 = rho[:, :, 0]
            rho_1 = rho[:, :, 1]
            rho_0 = rho_0.view(1, self.num_features, 1, 1)
            rho_1 = rho_1.view(1, self.num_features, 1, 1)
            rho_0 = rho_0.expand(input.shape[0], -1, -1, -1)
            rho_1 = rho_1.expand(input.shape[0], -1, -1, -1)
            out = rho_0 * out_in + rho_1 * out_ln

        out = out * self.gamma.expand(input.shape[0], -1, -1, -1) + self.beta.expand(input.shape[0], -1, -1, -1)
        return out


class SpatialTransformation(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        self.grid = self.grid.to('cuda')

        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, mode=self.mode)


def smooothing_loss(y_pred):
    dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
    dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])

    dx = dx * dx
    dy = dy * dy
    d = torch.mean(dx) + torch.mean(dy)
    grad = d
    return d





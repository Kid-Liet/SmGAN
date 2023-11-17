#-*- coding : utf-8-*-
# coding:unicode_escape


from  src.ITKFuncs import *
import SimpleITK as sitk
import torch
#import cv2
import glob
from src.dicom_util import ExportDicom
from src.models import Generator
import torchvision.transforms as transforms
import numpy as np
from src.utils import ToTensor
import sys,os
from src.aescipher import AESCipher
import torch.nn.functional as F
from src.get_incomplete_layer_index import get_incomplete_layer_index
import pickle
from src.DicomIO import PyWriteDicom
cipher = AESCipher()
transforms_ = [ ToTensor(), ]
transform = transforms.Compose(transforms_)
max, min = 2000., -1000.


def _Crop(img, crop_size):
    W, H = img.shape[0], img.shape[1]

    if W < crop_size:
        dist_W = crop_size - W
        if dist_W % 2 == 0:
            img = F.pad(img, (0, 0, int((dist_W) / 2), int((dist_W) / 2)), 'constant', value=img.min())
        else:
            img = F.pad(img, (0, 0, int((dist_W - 1) / 2) + 1, int((dist_W - 1) / 2)), 'constant', value=img.min())
    if H < crop_size:
        dist_H = crop_size - H
        if dist_H % 2 == 0:
            img = F.pad(img, (int((dist_H) / 2), int((dist_H) / 2), 0, 0), 'constant', value=img.min())
        else:
            img = F.pad(img, (int((dist_H - 1) / 2) + 1, int((dist_H - 1) / 2), 0, 0), 'constant', value=img.min())


    crop_img = img[int(img.shape[0] / 2) - int(crop_size / 2): int(img.shape[0] / 2) + int(crop_size / 2),
               int(img.shape[1] / 2) - int(crop_size / 2): int(img.shape[1] / 2) + int(crop_size / 2)]


    return crop_img.unsqueeze(0)

def Crop_(array_input,fake_B_nii,crop_size):
    W,H  = array_input.shape[1],array_input.shape[2]
    if W <= crop_size:
        dist_W = crop_size - W
        if dist_W % 2 == 0:
            fake_B_nii = fake_B_nii[:, int(crop_size / 2) - int(array_input.shape[1] / 2): int(crop_size / 2) + int(array_input.shape[1] / 2), : ]
        else:
            fake_B_nii = fake_B_nii[:, int(crop_size / 2) - int(array_input.shape[1] / 2): int(crop_size / 2) + int(array_input.shape[1] / 2)-1, : ]

    else:
        dist_W = crop_size - W
        if dist_W % 2 == 0:
            pad_w = int((array_input.shape[1] - crop_size) / 2)
            fake_B_nii = np.pad(fake_B_nii, ((0, 0), (pad_w, pad_w), (0, 0)), 'constant',
                                 constant_values=fake_B_nii.min())
        else:
            pad_w = int((array_input.shape[1] - crop_size) / 2)
            fake_B_nii = np.pad(fake_B_nii, ((0, 0), (pad_w, pad_w+1), (0, 0)), 'constant',
                                constant_values=fake_B_nii.min())


    if H <= crop_size:
        dist_H = crop_size - H
        if dist_H % 2 == 0:
            fake_B_nii = fake_B_nii[:, : , int(crop_size / 2) - int(array_input.shape[2] / 2): int(crop_size / 2) + int(array_input.shape[2] / 2)]
        else:
            fake_B_nii = fake_B_nii[:,:  , int(crop_size / 2) - int(array_input.shape[2] / 2): int(crop_size / 2) + int(array_input.shape[2] / 2)-1]

    else:
        dist_H = crop_size - H
        if dist_H % 2 == 0:
            pad_h = int((array_input.shape[2]-crop_size)/2)
            fake_B_nii = np.pad(fake_B_nii,((0,0),(0,0),(pad_h,pad_h)),'constant',constant_values=fake_B_nii.min())
        else:
            pad_h = int((array_input.shape[2] - crop_size) / 2)
            fake_B_nii = np.pad(fake_B_nii, ((0, 0), (0, 0), (pad_h, pad_h+1)), 'constant',
                                constant_values=fake_B_nii.min())
    return fake_B_nii

def process_img(itk_image_f_cutted, itk_image_m_cutted):
    idxtop, idxbottom = get_incomplete_layer_index(itk.GetArrayFromImage(itk_image_f_cutted))
    incompleteidx = idxtop + idxbottom

    origin_spacing = np.array(itk_image_f_cutted.GetSpacing())
    origin_origin = np.array(itk_image_f_cutted.GetOrigin())
    #origin_direct = np.array(itk_image_f_cutted.GetDirection())
    origin_size = itk_image_f_cutted.GetLargestPossibleRegion().GetSize()
    origin_size = np.array([origin_size[0], origin_size[1], origin_size[2]])
    resample_spacing = [1, 1, origin_spacing[2]]
    itk_image_f_resample, itk_image_m_resample = resample_image_to_11(itk_image_f_cutted,itk_image_m_cutted,resample_spacing)
    array_f, array_m = itk.GetArrayFromImage(itk_image_f_resample) ,itk.GetArrayFromImage(itk_image_m_resample)
    array_f = np.clip(array_f, np.percentile(array_f, 1), np.percentile(array_f, 99))
    array_f = (array_f - array_f.min()) / (array_f.max() - array_f.min()) * 2 - 1  # (-1,1)
    array_m = np.clip(array_m, min, max)
    array_m = (array_m - min) / (max - min) * 2 - 1  # (-1,1)


    return array_f,array_m,origin_spacing,origin_origin,origin_size,incompleteidx#,max_m,min_m


def read(fix_path,mov_path,rots):

    if fix_path[-4:] == ".nii" or  fix_path[-3:] == ".gz":
        itk_image_f = itk_image_nifti_reader(fix_path)
    else:
        itk_image_f = itk_image_series_reader(fix_path)
    if mov_path[-4:] == ".nii"or  mov_path[-3:] == ".gz":
        itk_image_m = itk_image_nifti_reader(mov_path)
    else:
        itk_image_m = itk_image_series_reader(mov_path)


    return itk_image_f, itk_image_m

def IMSE_img2img(cbct_path,ct_path,output_path,rot,weight_path,log="./"):
    sys.stdout.write('#{:03d}#\\r'.format(0))
    sys.stdout.flush()
    itk_image_f_cutted, itk_image_m_cutted = read(cbct_path,ct_path,rot)
    array_f,array_m,origin_spacing,origin_origin,origin_size,incompleteidx = process_img(itk_image_f_cutted,itk_image_m_cutted)

    net_G = Generator(2, 1)
    crop_size = 256

    net_G.load_state_dict(torch.load(weight_path))
    net_G.cuda()
    net_G.eval()
    sys.stdout.write('#{:03d}#\\r'.format(10))
    sys.stdout.flush()


    sum_iters = array_f.shape[0]

    iters = sum_iters / 60
    for s in range(sum_iters):
        f_slice, m_slice = transform(array_f[s]).float().cuda(), transform(array_m[s]).float().cuda()
        resize_f = _Crop(f_slice.squeeze(), crop_size).unsqueeze(0)
        resize_m = _Crop(m_slice.squeeze(), crop_size).unsqueeze(0)
        AB = torch.cat([resize_f, resize_m], 1)


        fake_B = (net_G(AB) + 1) / 2  # 0-1
        fake_B = ((fake_B.squeeze().detach().cpu().numpy()) * (max - min) + min).astype('int') # -1000 2000

        if s == 0:
            fake_B_nii = np.expand_dims(fake_B, axis=0)
        else:
            fake_B = np.expand_dims(fake_B, axis=0)
            fake_B_nii = np.concatenate((fake_B_nii, fake_B), axis=0)
        sys.stdout.write('#{:03d}#\\r'.format(10 + int(s / iters)))
        sys.stdout.flush()

    fake_B_nii = Crop_(array_f, fake_B_nii, crop_size)
    fake_B_nii = sitk.GetImageFromArray(fake_B_nii)
    fake_B_nii.SetSpacing([1, 1, origin_spacing[2]])
    fake_B_nii.SetOrigin(origin_origin)
    sys.stdout.write('#{:03d}#\\r'.format(95))
    sys.stdout.flush()
    fake_B_nii = rest2(fake_B_nii, origin_size)
    if output_path[-4:] == ".nii" or  output_path[-3:] == ".gz":
        sitk.WriteImage(fake_B_nii, os.path.join(output_path))
    else:
        fake_B_npy = sitk.GetArrayFromImage(fake_B_nii)
        input_dcm = glob.glob(os.path.join('%s' % ct_path, '*'))[0]
        export = ExportDicom(origin_spacing, origin_origin, input_dcm)
        export.write_image(fake_B_npy, output_path,incompleteidx)

    sys.stdout.write('#{:03d}#\\r'.format(100))
    sys.stdout.flush()
if __name__ == '__main__':
    rots = "1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"
    IMSE_img2img(cbct_path = "../test_data/CBCT.nii.gz", ct_path = "../test_data/plan_CT.nii.gz",output_path = "../test_data/sCT.nii.gz",
                   rot =rots,weight_path= "../weight/netG_A2B.pth",log="./debug.log")

import datetime
import os
import random
from glob import glob

import cv2
import numpy as np
import pydicom as dicom
import SimpleITK as sitk
#import skimage.draw

# from jellyfish import levenshtein_distance
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence
from pydicom.uid import generate_uid


class ExportDicom(object):
    def __init__(self, spacings,origin,reference_root ,meta=dict(), modality='CT'):
        self.dcmData = dicom.read_file(reference_root,force=True)
        self.pixel_spacing = [
            spacing if spacing is not None else 1.0 for spacing in spacings]
        self.modality = modality
        time = datetime.datetime.now()
        self.DATE = time.strftime('%Y%m%d')
        self.TIME = time.strftime('%H%M%S')
        meta = get_meta(reference_root)
        self.meta = {
            'PatientID': meta.get('PatientID'),#'{}-ID-{}'.format(dset_name, patient_id)
            'PatientName': meta.get('PatientName',),#'{}-Name-{}'.format(dset_name, patient_id)
            'SOPClassUID': meta.get('SOPClassUID','1.2.840.10008.5.1.4.1.1.2' if modality == 'CT' else '1.2.840.10008.5.1.4.1.1.4'),
            'SOPInstanceUID': meta.get('SOPInstanceUID', []), # shengcheng
            'StudyInstanceUID': meta.get('StudyInstanceUID',),
            'SeriesInstanceUID': [generate_uid()],
            'FrameOfReferenceUID': [generate_uid()],#meta.get('FrameOfReferenceUID',generate_uid()),
            'ImagePositionPatient': list(origin),#meta.get('ImagePositionPatient', ['0.0', '0.0', '0.0'])
        }
        #print (self.meta)
        #print (meta.get('KVP'))
        #qwer
        # print (self.meta['PatientID'])
        # print (meta.get('SOPClassUID'))
        # print (self.meta['SOPClassUID'])


    def write_image(self, image_array, save_dir, incompleteidx):
        """
        Param:
            image_array : N * H * W numpy array
            save_dir    : directory to save dicom files
        """
        image_array = image_array.astype(np.int16)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        # if len(self.meta['SOPInstanceUID']) == 0:
        #     self.meta['SOPInstanceUID'] = [generate_uid()
        #                                    for _ in range(image_array.shape[0])]

        for i in range(image_array.shape[0]):
            if i in incompleteidx:
                continue
            save_path = os.path.join(save_dir, '{}.dcm'.format(i))
            current_position = [float(t)
                                for t in self.meta['ImagePositionPatient']]
            current_position[2] += i * self.pixel_spacing[2]

            ds = Dataset()
            SOPInstanceUID = generate_uid()
            ds.SOPInstanceUID = SOPInstanceUID[:-6] + '.' + str.zfill(str(i + 1), 5)
            ds.PatientID = self.meta['PatientID']
            ds.PatientName = self.meta['PatientName']
            ds.SOPClassUID = self.meta['SOPClassUID']

            #ds.SOPInstanceUID = self.meta['SOPInstanceUID'][i]
            ds.StudyInstanceUID = self.meta['StudyInstanceUID']
            ds.SeriesInstanceUID = self.meta['SeriesInstanceUID']
            ds.FrameOfReferenceUID = self.meta['FrameOfReferenceUID']
            ds.Modality = self.modality
            ds.ContentDate = self.DATE
            ds.ContentTime = self.TIME
            ds.SeriesNumber = '1'

            ds.Columns = image_array.shape[2]
            ds.Rows = image_array.shape[1]

            rescale_intercept = image_array.min()
            pixel_array = image_array[i] - rescale_intercept
            ds.PixelData = pixel_array.tobytes()

            # ds.ImagePositionPatient = '\\'.join(map(str, current_position))
            # ds.PixelSpacing = '\\'.join(map(str, self.pixel_spacing[:2]))
            # ds.ImageOrientationPatient = '\\'.join(
            #     map(str, self.pixel_direct[:6]))  # 1.0, 0.0, 0.0, 0.0, 1.0, 0.0
            # ds.SliceLocation = '{}'.format(i * self.pixel_spacing[2])
            ds.ImagePositionPatient = list(current_position)
            ds.PixelSpacing = self.pixel_spacing[:2]
            ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
            ds.SliceLocation = i * self.pixel_spacing[2]
            ds.RescaleIntercept = rescale_intercept
            ds.SliceThickness = self.pixel_spacing[2]
            #
            current_tag = ds.dir()
            for key in self.dcmData.dir():
                if key == "PixelData":
                    continue
                if key not in current_tag:
                    exec('ds.' + key + '= getattr(self.dcmData, key, '') ')
            #
            ds.SeriesDescription = "Manteia sCT"
            ds.RescaleSlope = 1
            ds.PatientPosition = 'HFS'
            ds.SamplesPerPixel = 1
            ds.BitsAllocated = 16
            ds.BitsStored = 16
            ds.HighBit = 15
            ds.PixelRepresentation = 0
            ds.PhotometricInterpretation = 'MONOCHROME2'
            #
            ds.file_meta = Dataset()
            ds.is_implicit_VR = True
            ds.is_little_endian = True
            ds.fix_meta_info()


            #print (ds)
            ds.save_as(save_path, write_like_original=False)




    def write_rt(self, label_array, roi_names, save_path, image_array=None, slices=None):
        """
        Param:
            label_array : N * H * W * C mask array
            roi_names   : names of label set
            image_array : N * H * W numpy array
            save_path   : path to save rt file
        """
        assert not (image_array is None and slices is None)
        assert label_array.shape[-1] == len(roi_names)

        # extract contours from label data, assume label shape is N*H*W*C
        label_contours = [[] for _ in range(label_array.shape[-1])]
        for slice_index in range(label_array.shape[0]):
            for label_index in range(label_array.shape[-1]):
                contours, _ = cv2.findContours(label_array[slice_index, :, :, label_index].copy(),
                                               mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_NONE)
                for contour in contours:
                    # skip the contour that has points less than 5
                    if len(contour) < 5:
                        continue
                    xys = []
                    contour = np.squeeze(contour).astype(np.float64)
                    for x, y in contour:
                        xys.append((x, y))
                    label_contours[label_index].append((slice_index, xys))

        # write to a dicom structure file
        ds = Dataset()
        ds.PatientID = self.meta['PatientID']
        ds.PatientName = self.meta['PatientName']
        ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.481.3'
        ds.SOPInstanceUID = generate_uid()
        ds.StudyInstanceUID = self.meta['StudyInstanceUID']
        ds.SeriesInstanceUID = self.meta['SeriesInstanceUID']
        ds.StudyDate = self.DATE
        ds.StudyTime = self.TIME
        ds.SeriesDate = self.DATE
        ds.SeriesTime = self.TIME
        ds.Modality = 'RTSTRUCT'
        ds.SeriesNumber = '1'
        ds.StructureSetLabel = 'RTStruct'
        ds.StructureSetDate = self.DATE
        ds.StructureSetTime = self.TIME

        ReferencedFrameOfReferenceSequence = Sequence()
        ReferencedFrameOfReference = Dataset()
        ReferencedFrameOfReference.FrameOfReferenceUID = self.meta['FrameOfReferenceUID']
        RTReferencedStudySequence = Sequence()
        RTReferencedStudy = Dataset()
        RTReferencedStudy.ReferencedSOPClassUID = '1.2.840.10008.5.1.4.1.1.481.3'
        RTReferencedStudy.ReferencedSOPInstanceUID = ds.SOPInstanceUID
        RTReferencedSeriesSequence = Sequence()
        RTReferencedSeries = Dataset()
        RTReferencedSeries.SeriesInstanceUID = self.meta['SeriesInstanceUID']
        ContourImageSequence = Sequence()
        for slice_index in range(label_array.shape[0]):
            ContourImage = Dataset()
            ContourImage.ReferencedSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
            ContourImage.ReferencedSOPInstanceUID = self.meta['SOPInstanceUID'][slice_index]
        RTReferencedSeries.ContourImageSequence = ContourImageSequence
        RTReferencedSeriesSequence.append(RTReferencedSeries)
        RTReferencedStudy.RTReferencedSeriesSequence = RTReferencedSeriesSequence
        RTReferencedStudySequence.append(RTReferencedStudy)
        ReferencedFrameOfReference.RTReferencedStudySequence = RTReferencedStudySequence
        ReferencedFrameOfReferenceSequence.append(ReferencedFrameOfReference)
        ds.ReferencedFrameOfReferenceSequence = ReferencedFrameOfReferenceSequence

        StructureSetROISequence = Sequence()
        for label_index in range(label_array.shape[-1]):
            StructureSetROI = Dataset()
            StructureSetROI.ROINumber = label_index
            StructureSetROI.ReferencedFrameOfReferenceUID = self.meta['FrameOfReferenceUID']
            StructureSetROI.ROIName = roi_names[label_index]
            StructureSetROI.ROIVolume = '0.0'
            StructureSetROI.ROIGenerationAlgorithm = 'MANUAL'
            StructureSetROISequence.append(StructureSetROI)
        ds.StructureSetROISequence = StructureSetROISequence

        RTROIObservationsSequence = Sequence()
        for label_index in range(label_array.shape[-1]):
            RTROIObservations = Dataset()
            RTROIObservations.ObservationNumber = label_index
            RTROIObservations.ReferencedROINumber = label_index
            RTROIObservations.RTROIInterpretedType = 'ORGAN'
            RTROIObservations.ROIInterpreter = ''
            RTROIObservationsSequence.append(RTROIObservations)
        ds.RTROIObservationsSequence = RTROIObservationsSequence

        if slices is None:
            slices = sitk.GetImageFromArray(image_array)
            slices.SetSpacing(self.pixel_spacing)
            slices.SetOrigin(self.meta['ImagePositionPatient'])
        ROIContourSequence = Sequence()
        for label_index in range(label_array.shape[-1]):
            ROIContour = Dataset()
            ROIContour.ROIDisplayColor = '\\'.join(
                [str(random.randint(0, 256)) for _ in range(3)])
            ROIContour.ReferencedROINumber = label_index
            ContourSequence = Sequence()
            for contour in label_contours[label_index]:
                slice_index, xys = contour
                Contour = Dataset()
                ContourImageSequence = Sequence()
                ContourImage = Dataset()
                ContourImage.ReferencedSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
                ContourImage.ReferencedSOPInstanceUID = self.meta['SOPInstanceUID'][slice_index]
                ContourImageSequence.append(ContourImage)
                Contour.ContourImageSequence = ContourImageSequence
                Contour.ContourGeometricType = 'CLOSED_PLANAR'
                Contour.NumberOfContourPoints = len(xys)
                ContourData = ''
                for x, y in xys:
                    point = slices.TransformContinuousIndexToPhysicalPoint(
                        (float(x), float(y), float(slice_index)))
                    ContourData += '{}\\{}\\{}\\'.format(
                        point[0], point[1], point[2])
                if len(ContourData) == 0:
                    continue
                Contour.ContourData = ContourData[:-2]
                ContourSequence.append(Contour)
            ROIContour.ContourSequence = ContourSequence
            ROIContourSequence.append(ROIContour)
        ds.ROIContourSequence = ROIContourSequence
        ds.file_meta = Dataset()
        ds.is_implicit_VR = True
        ds.is_little_endian = True
        ds.fix_meta_info()
        if not os.path.exists(os.path.dirname(save_path)):
            os.mkdir(os.path.dirname(save_path))
        ds.save_as(save_path, write_like_original=False)


def dcm_is_valid(dcm):
    if hasattr(dcm, 'SOPClassUID'):
        return True
    else:
        return False


def try_save_dcm(dcm, fn):
    try:
        dicom.dcmwrite(fn, dcm)
    except:
        print('file: {} is cannot be write'.format(os.path.basename(fn)))


def find_rt_file(patient_dir, begin_prefix=''):
    rt_count = 0
    out = None
    out_path = None
    for path in glob(patient_dir+'/*'):
        if os.path.isdir(path):
            continue
        if not os.path.basename(path).startswith(begin_prefix):
            continue
        dcm = dicom.dcmread(path, force=True)
        if not dcm_is_valid(dcm):
            print('file: {} is corrupted'.format(os.path.basename(path)))
            continue
        if dcm.SOPClassUID == '1.2.840.10008.5.1.4.1.1.481.3':
            rt_count += 1
            out = dcm
            out_path = path
    if rt_count != 1:
        print('patient [{}] has {} rt file'.format(
            os.path.basename(patient_dir), rt_count))
    return out, out_path


def get_meta(dcm_files):
    dcm = dicom.dcmread(dcm_files,force=True)

    def add_key_to_meta(meta, dcm, key):
        try:
            value = getattr(dcm, key)
            if value == '':
                return
            meta[key] = value
        except:
            return
    meta = {}
    key_list = ['PatientID', 'PatientName', 'ImagePositionPatient', 'SOPClassUID',
                'StudyInstanceUID', 'SeriesInstanceUID', 'FrameOfReferenceUID','']
    for k in key_list:
        add_key_to_meta(meta, dcm, k)



    return meta


def sitk_read_dicom_dir(series_dir):
    reader = sitk.ImageSeriesReader()
    dcm_files = reader.GetGDCMSeriesFileNames(series_dir)
    reader.SetFileNames(dcm_files)
    reader.SetLoadPrivateTags(True)
    slices = reader.Execute()

    return slices


def get_roi_contour(rt_dcm, roi_names):
    index_to_name = {}
    if isinstance(roi_names, str):
        one_contour_to_go = True
        roi_names = [roi_names]
    else:
        one_contour_to_go = False
    for ss_roi in rt_dcm.StructureSetROISequence:
        roi_index = int(ss_roi.ROINumber)
        roi_name = str(ss_roi.ROIName).lower()
        if roi_name not in roi_names:
            # print('\tSkip roi [{}]'.format(roi_name))
            continue
        # print('got roi name: [{}]'.format(roi_name))
        index_to_name[roi_index] = roi_name
        if one_contour_to_go:
            break

    contours = []
    for roi_contour in rt_dcm.ROIContourSequence:
        ref_roi_number = int(roi_contour.ReferencedROINumber)
        if ref_roi_number not in index_to_name:
            continue

        if hasattr(roi_contour, 'ContourSequence'):
            contours.append(roi_contour.ContourSequence)
        else:
            raise RuntimeError('invalid roi: {}'.format(
                index_to_name[ref_roi_number]))

    return contours


# def get_patient_label(slices, rt_dcm, roi_names, fill_contour=True):
#     index_to_name = {}
#     if isinstance(roi_names, str):
#         one_contour_to_go = True
#         roi_names = [roi_names]
#     else:
#         one_contour_to_go = False
#     for ss_roi in rt_dcm.StructureSetROISequence:
#         roi_index = int(ss_roi.ROINumber)
#         roi_name = str(ss_roi.ROIName).lower()
#         if roi_name not in roi_names:
#             # print('\tSkip roi [{}]'.format(roi_name))
#             continue
#         print('got roi name: [{}]'.format(roi_name))
#         index_to_name[roi_index] = roi_name
#         if one_contour_to_go:
#             break
#
#     W, H, Z = slices.GetSize()
#     label = np.zeros((Z, H, W, len(roi_names)), dtype=np.float32)
#
#     for roi_contour in rt_dcm.ROIContourSequence:
#         ref_roi_number = int(roi_contour.ReferencedROINumber)
#         if ref_roi_number not in index_to_name:
#             continue
#
#         contours_each_slice = [(i, []) for i in range(Z)]
#
#         if hasattr(roi_contour, 'ContourSequence'):
#             for contour in roi_contour.ContourSequence:
#                 contour_data = contour.ContourData
#                 assert len(contour_data) % 3 == 0
#                 contour_data = np.array(contour_data).reshape((-1, 3))
#                 contour_data = [slices.TransformPhysicalPointToContinuousIndex(
#                     t) for t in contour_data]
#                 contour_data = np.array(contour_data)
#                 if len(contour_data) < 3:
#                     print('\t\tless than 3 points in contour',
#                           contour_data.shape)
#                     continue
#                 slice_index = round(contour_data[0, 2], 0)
#                 # print(slice_index, round(slice_index, 0))
#                 assert abs(slice_index - int(slice_index)) < 1e-5
#                 if int(slice_index) >= Z:
#                     print('out of slice')
#                     continue
#                 contours_each_slice[int(slice_index)][1].append(
#                     contour_data)
#
#             for slice_index, contours in contours_each_slice:
#                 # shapely need points more than 4
#                 contours = list(filter(lambda t: len(t) >= 4, contours))
#                 masks = []
#                 if len(contours) == 0:
#                     continue
#                 for contour in contours:
#                     mask = np.zeros((H, W), dtype=np.float32)
#                     cor_xy = contour[:, :2].tolist()
#                     cv2.fillPoly(mask, np.int32([cor_xy]), 1)
#                     mask = mask.astype(np.uint8)
#                     masks.append(mask)
#                 inner_polys = []
#                 for idx, mask in enumerate(masks):
#                     for jdx in range(idx+1, len(masks)):
#                         union_area = (masks[jdx] | mask).sum()
#                         if union_area == masks[jdx].sum():
#                             inner_polys.append(idx)
#                         elif union_area == mask.sum():
#                             inner_polys.append(jdx)
#                 inner_polys = list(set(inner_polys))
#
#                 if fill_contour:
#                     func = skimage.draw.polygon
#                 else:
#                     func = skimage.draw.polygon_perimeter
#
#                 for contour in contours:
#                     rr, cc = func(contour[:, 1], contour[:, 0])
#                     label[slice_index, rr, cc,
#                           roi_names.index(index_to_name[ref_roi_number])] = 1
#                 for cont_index in inner_polys:
#                     rr, cc = func(contours[cont_index]
#                                   [:, 1], contours[cont_index][:, 0])
#                     label[slice_index, rr, cc,
#                           roi_names.index(index_to_name[ref_roi_number])] = 0
#     return label


def get_roi_names(dcm):
    roi_names = set()
    for StructureSetROI in dcm.StructureSetROISequence:
        roi_names.add(StructureSetROI.ROIName)
    return list(roi_names)


def get_roi_numbers(dcm):
    roi_numbers = []
    for RTROIObservations in dcm.RTROIObservationsSequence:
        roi_numbers.append(RTROIObservations.ReferencedROINumber)
    return roi_numbers


def rename_roi(dcm, old_name, new_name):
    StructureSetROISequence = Sequence()
    for StructureSetROI in dcm.StructureSetROISequence:
        if StructureSetROI.ROIName == old_name:
            StructureSetROI.ROIName = new_name
        StructureSetROISequence.append(StructureSetROI)
    dcm.StructureSetROISequence = StructureSetROISequence
    return dcm


def remove_roi(dcm, name):
    StructureSetROISequence = Sequence()
    for StructureSetROI in dcm.StructureSetROISequence:
        if StructureSetROI.ROIName == name:
            continue
        StructureSetROISequence.append(StructureSetROI)
    dcm.StructureSetROISequence = StructureSetROISequence

    return dcm


# def nearest_names(standard_names, name):
#     dists = [levenshtein_distance(name, t) for t in standard_names]
#     inds = np.argsort(dists)
#     return list(np.array(standard_names)[inds])


def add_roi(dcm, name, contours):
    roi_number = str(len(get_roi_numbers(dcm)) + 1)
    StructureSetROISequence = dcm.StructureSetROISequence
    StructureSetROI = Dataset()
    # StructureSetROI.ReferencedFrameOfReferenceUID = dcm.FrameOfReferenceUID
    StructureSetROI.ROINumber = roi_number
    StructureSetROI.ROIName = name
    StructureSetROI.ROIVolume = '0.0'
    StructureSetROI.ROIGenerationAlgorithm = 'MANUAL'
    StructureSetROISequence.append(StructureSetROI)
    ROIContourSequence = dcm.ROIContourSequence
    ROIContour = Dataset()
    ROIContour.ROIDisplayColor = '{}\\{}\\{}'.format(random.randint(0, 256), random.randint(0, 256),
                                                     random.randint(0, 256))
    ROIContour.ReferencedROINumber = roi_number
    ROIContour.ContourSequence = contours
    ROIContourSequence.append(ROIContour)

    RTROIObservationsSequence = dcm.RTROIObservationsSequence
    RTROIObservations = Dataset()
    RTROIObservations.ObservationNumber = roi_number
    RTROIObservations.ReferencedROINumber = roi_number
    RTROIObservations.RTROIInterpretedType = 'ORGAN'
    RTROIObservations.ROIInterpreter = ''
    RTROIObservationsSequence.append(RTROIObservations)

    dcm.StructureSetROISequence = StructureSetROISequence
    dcm.ROIContourSequence = ROIContourSequence
    dcm.RTROIObservationsSequence = RTROIObservationsSequence
    return dcm


def fix_roi_number(dcm):
    ref_roi_numbers = []
    StructureSetROISequence = Sequence()
    for StructureSetROI in dcm.StructureSetROISequence:
        StructureSetROISequence.append(StructureSetROI)
        ref_roi_numbers.append(StructureSetROI.ROINumber)

    ROIContourSequence = Sequence()
    for ROIContour in dcm.ROIContourSequence:
        if not ROIContour.ReferencedROINumber in ref_roi_numbers:
            continue
        ROIContourSequence.append(ROIContour)
    dcm.ROIContourSequence = ROIContourSequence

    RTROIObservationsSequence = Sequence()
    for RTROIObservations in dcm.RTROIObservationsSequence:
        if not RTROIObservations.ObservationNumber in ref_roi_numbers:
            continue
        if not RTROIObservations.ReferencedROINumber in ref_roi_numbers:
            continue
        RTROIObservationsSequence.append(RTROIObservations)
    dcm.RTROIObservationsSequence = RTROIObservationsSequence

    return dcm


def reset_roi_number(dcm):
    roi_number_old_to_new = {}
    roi_id = 0
    for StructureSetROI in sorted(dcm.StructureSetROISequence, key=lambda t: t.ROIName):
        roi_number_old_to_new[StructureSetROI.ROINumber] = str(roi_id)
        roi_id += 1
    StructureSetROISequence = Sequence()
    for StructureSetROI in dcm.StructureSetROISequence:
        StructureSetROI.ROINumber = roi_number_old_to_new[StructureSetROI.ROINumber]
        StructureSetROISequence.append(StructureSetROI)
    ROIContourSequence = Sequence()
    for ROIContour in dcm.ROIContourSequence:
        ROIContour.ReferencedROINumber = roi_number_old_to_new[ROIContour.ReferencedROINumber]
        ROIContourSequence.append(ROIContour)
    RTROIObservationsSequence = Sequence()
    for RTROIObservations in dcm.RTROIObservationsSequence:
        RTROIObservations.ObservationNumber = roi_number_old_to_new[
            RTROIObservations.ObservationNumber]
        RTROIObservations.ReferencedROINumber = roi_number_old_to_new[
            RTROIObservations.ReferencedROINumber]
        RTROIObservationsSequence.append(RTROIObservations)

    dcm.StructureSetROISequence = StructureSetROISequence
    dcm.ROIContourSequence = ROIContourSequence
    dcm.RTROIObservationsSequence = RTROIObservationsSequence

    return dcm


def append_left_or_right_to_roi(patient_fn, dcm, roi_name):
    slices = sitk_read_dicom_dir(patient_fn)
    label = get_patient_label(slices, dcm, roi_name).squeeze()

    W = np.shape(label)[2]
    left, right = label[..., :round(W/2)].sum(), label[..., round(W/2):].sum()
    ratio = (left-right)/(left+right+1e-6)
    if ratio > 0:
        rename_roi(dcm, roi_name, roi_name+'_l')
    else:
        rename_roi(dcm, roi_name, roi_name+'_r')

    if abs(ratio) < 0.5:
        print('ratio {} is too low, need to check'.format(ratio))
        for label_slice in label:
            cv2.imshow('label_slice', label_slice)
            cv2.waitKey()


if __name__ == "__main__":
    dcm = dicom.dcmread(
        'F:/zwspace/data/RT_MAC/all/RTMAC-TRAIN-004/MR1.3.6.1.4.1.14519.5.2.1.1706.6003.104481923742345531146674543340.dcm')
    print(dcm.PatientName)

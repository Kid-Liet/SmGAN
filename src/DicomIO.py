import itk
import datetime
import os
import numpy as np
import sys,glob
import pydicom as dicom
#from pydicom.dataset import Dataset
#from pydicom.sequence import Sequence
from src.dicom_util import get_meta
from pydicom.uid import generate_uid

def ItkReadDicom(fileDir):
    '''
    Dicom读取
    输入目标Dicom路径,
    返回ImageType image3d, ReaderType reader
    '''
    PixelType = itk.F
    Dimension = 3
    ImageType = itk.Image[PixelType, Dimension]
    ReaderType = itk.ImageSeriesReader[ImageType]
    reader = ReaderType.New()
    ImageIOType = itk.GDCMImageIO
    dicomIO = ImageIOType.New()
    reader.SetImageIO(dicomIO)
    NamesGeneratorType = itk.GDCMSeriesFileNames
    namesGenerator = NamesGeneratorType.New()
    namesGenerator.SetUseSeriesDetails(True)
    namesGenerator.SetDirectory(fileDir)
    seriesUID = namesGenerator.GetSeriesUIDs()
    fileNames = namesGenerator.GetFileNames(seriesUID[0])
    reader.SetFileNames(fileNames)
    reader.Update()
    image3d = reader.GetOutput()
    return image3d, reader



def ItkWriteDicom(input, reader, inFileDir, outFileDir):
    '''
    输入Dicom数据, ReaderType reader, 输入路径, 输出路径,
    输出Dicom
    '''
    PixelType = itk.F
    Dimension = 3
    OutputDimension = 2
    ImageType = itk.Image[PixelType, Dimension]
    Image2DType = itk.Image[PixelType, OutputDimension]
    ImageIOType = itk.GDCMImageIO
    dicomIO = ImageIOType.New()
    SeriesWriterType = itk.ImageSeriesWriter[ImageType, Image2DType]
    seriesWriter = SeriesWriterType.New()
    seriesWriter.SetInput(input)
    seriesWriter.SetImageIO(dicomIO)
    NamesGeneratorType = itk.GDCMSeriesFileNames
    namesGenerator = NamesGeneratorType.New()
    namesGenerator.SetUseSeriesDetails(True)
    namesGenerator.SetInputDirectory(inFileDir)
    namesGenerator.SetOutputDirectory(outFileDir)
    seriesWriter.SetFileNames(namesGenerator.GetOutputFileNames())
    seriesWriter.SetMetaDataDictionaryArray(reader.GetMetaDataDictionaryArray())
    seriesWriter.Update()


def itk_image_nifti_writer(itk_image, nii_path):
    """
    Write NIFTI data Image
    :param itk_image: itk Image
    :param nii_path: path of the writing path
    :return: 1
    """
    # typedef image type
    image_type = itk.Image[itk.F, 3]
    # read dicom"
    itk.NiftiImageIOFactory.RegisterOneFactory
    writer = itk.ImageFileWriter[image_type].New()
    writer.SetInput(itk_image)
    writer.SetFileName(nii_path)
    writer.Update()
    # print(image)
    return 1


def PyWriteDicom(dicomPath,savePath,imgArray,filename='',incompleteidx=[]):
    meta = get_meta(glob.glob(os.path.join('%s' % dicomPath, '*'))[0])
    NamesGeneratorType = itk.GDCMSeriesFileNames
    namesGenerator = NamesGeneratorType.New()
    namesGenerator.SetUseSeriesDetails(True)
    namesGenerator.SetDirectory(dicomPath)
    seriesUID = namesGenerator.GetSeriesUIDs()
    reference_root = namesGenerator.GetFileNames(seriesUID[0])

    time = datetime.datetime.now()
    DATE = time.strftime('%Y%m%d')
    TIME = time.strftime('%H%M%S.%f')

    SeriesUID = generate_uid()


    if not os.path.exists(savePath):
        os.mkdir(savePath)

    # get corner median value, so background will be -1000
    imgArray = imgArray.astype(np.int16)
    rescale_intercept = imgArray.min()
    #print (rescale_intercept)
    #imgArray = imgArray - rescale_intercept
    #print (imgArray.min())
    #imgArray[imgArray<0]=0


    idx = 1
    for i in range(imgArray.shape[0]):
        if i in incompleteidx:
            continue

        ds = dicom.read_file(reference_root[i],force=True)

        ds.SeriesDescription = meta.get('SeriesDescription','')

        # GeneratingRegistrationSerise.cpp
        ds.SeriesInstanceUID = SeriesUID
        SOPInstanceUID = generate_uid()
        ds.SOPInstanceUID = SOPInstanceUID[:-6] + '.' + str.zfill(str(idx+1), 5)
        ds.Columns = imgArray.shape[2]
        ds.Rows = imgArray.shape[1]
        ds.SeriesDate = DATE  # Series_Date
        ds.SeriesTime = TIME  # Series_Time
        ds.RescaleIntercept = rescale_intercept
        ds.Modality = "CT"
        ds.PixelData = imgArray[i].tobytes()

        ds.RescaleSlope = 1
        ds.PatientPosition = 'HFS'
        ds.SamplesPerPixel = 1
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.PhotometricInterpretation = 'MONOCHROME2'
        # ds.is_implicit_VR = True
        # ds.is_little_endian = True
        # ds.fix_meta_info()


        if filename:
            tmpfilename = '{}{}.dcm'.format(filename,idx)
        else:
            tmpfilename =  '{}.dcm'.format(idx)
        save_path = os.path.join(savePath, tmpfilename)
        ds.save_as(save_path, write_like_original=False)
        idx = idx+1
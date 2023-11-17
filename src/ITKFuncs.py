import itk
import numpy as np
import SimpleITK as sitk


"""
    Re-sample for rigid or deformable transform
"""
def rest(image,new_spacing):
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkBSpline)
    origin = image.GetOrigin()
    origin = np.array([origin[0], origin[1], origin[2]])
    resampler.SetOutputOrigin(origin)
    direction = image.GetDirection()#[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

    resampler.SetOutputDirection(direction)
    spacing = image.GetSpacing()
    spacing = np.array([spacing[0], spacing[1], spacing[2]])
    size = image.GetSize()
    size = np.array([size[0], size[1], size[2]])

    new_size_x = (size[0]* spacing[0])/new_spacing[0]
    new_size_y = (size[1]* spacing[1])/new_spacing[1]
    new_size_z = size[2]
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize([round(new_size_x), round(new_size_y), round(new_size_z)])
    resampled_image = resampler.Execute(image)

    return resampled_image
def rest2(image,new_size):
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkBSpline)
    origin = image.GetOrigin()
    resampler.SetOutputOrigin(origin)
    direction = image.GetDirection()
    resampler.SetOutputDirection(direction)
    spacing = image.GetSpacing()
    #spacing = np.array([spacing[0], spacing[1], spacing[2]])
    size = image.GetSize()
    #size = np.array([size[0], size[1], size[2]])
    new_spacing_x = (size[0]* spacing[0])/new_size[0]
    new_spacing_y = (size[1]* spacing[1])/new_size[1]
    new_spacing_z = (size[2]* spacing[2])/new_size[2]
    resampler.SetOutputSpacing([new_spacing_x,new_spacing_y,new_spacing_z])

    resampler.SetSize(new_size.tolist())

    # print (origin,spacing,size)
    # print(origin, new_spacing, [int(new_size_x), int(new_size_y), int(new_size_z)])
    resampled_image = resampler.Execute(image)

    return resampled_image
def itk_read_image(imagepath):


    if imagepath[-4:] == ".nii":
        itk_img = sitk.ReadImage(imagepath)
    else:
        reader = sitk.ImageSeriesReader()
        names = reader.GetGDCMSeriesFileNames(imagepath)
        reader.SetFileNames(names)
        itk_img = reader.Execute()

    return itk_img#image
def angle_trans_to_vtk_transform(fixed_center, moving_center, angle, trans, inverse_flag):
    """
    Convert euler angle and translation to 4x4 rotation matrix
    :param fixed_center: fixed image center torch tensor
    :param moving_center: moving image center torch tensor
    :param angle: euler angle
    :param trans: translation
    :param inverse_flag: fixed and moving inversed?
    :return: vtk_transform
    """
    # gpu or cpu -> cpu
    if angle.is_cuda:
        angle = angle.cpu()
    if trans.is_cuda:
        trans = trans.cpu()
    if fixed_center.is_cuda:
        fixed_center = fixed_center.cpu()
    if moving_center.is_cuda:
        moving_center = moving_center.cpu()

    # transform
    cos_value = np.cos(np.array([angle[0], angle[1], angle[2]]))
    sin_value = np.sin(np.array([angle[0], angle[1], angle[2]]))
    # rot X Y Z
    rot_x = np.array([[1., 0, 0],
                      [0, cos_value[0], -sin_value[0]],
                      [0, sin_value[0], cos_value[0]]])
    rot_y = np.array([[cos_value[1], 0, sin_value[1]],
                      [0, 1., 0],
                      [-sin_value[1], 0, cos_value[1]]])
    rot_z = np.array([[cos_value[2], -sin_value[2], 0],
                      [sin_value[2], cos_value[2], 0],
                      [0, 0, 1.]])
    rot_matrix = rot_z.dot(rot_y.dot(rot_x))

    if not inverse_flag:
        offset = rot_matrix.dot(-np.array([fixed_center[0],
                                           fixed_center[1],
                                           fixed_center[2]])) + \
                 np.array([moving_center[0],
                           moving_center[1],
                           moving_center[2]]) + \
                 np.array([trans[0],
                           trans[1],
                           trans[2]])
    else:
        offset = rot_matrix.dot(-np.array([fixed_center[0],
                                           fixed_center[1],
                                           fixed_center[2]]) + \
                                np.array([trans[0],
                                          trans[1],
                                          trans[2]]))+ \
                 np.array([moving_center[0],
                           moving_center[1],
                           moving_center[2]])

    vtk_transform = itk.AffineTransform[itk.D, 3].New()
    # rot = itk.Matrix[itk.D, 3, 3]()
    off = itk.Vector[itk.D, 3]()
    off.SetElement(0, offset[0])
    off.SetElement(1, offset[1])
    off.SetElement(2, offset[2])
    # for i in range(3):
    #     for j in range(3):
    #         rot.GetVnlMatrix().put(i, j, rot_matrix[i, j])
    rot = itk.GetMatrixFromArray(rot_matrix)
    vtk_transform.SetMatrix(rot)
    vtk_transform.SetOffset(off)
    return vtk_transform


def resample_full_moving_image_after_rigid_transform(image_fixed, image_moving, vtk_transform):
    """
    Resample the moving image based on the fixed image's grid when given a rotation and translation
    This function give a full size as the same of fixed image
    :param image_fixed: fixed image (using its grids)
    :param image_moving: moving image
    :param vtk_transform: vtk rigid transform
    :return: resampled itk image
    """
    # image type
    ImageType = itk.Image[itk.F, 3]

    # back grounf value
    minCalculator = itk.MinimumMaximumImageCalculator[ImageType].New()
    minCalculator.SetImage(image_moving)
    minCalculator.Compute()
    edgePaddingValue = minCalculator.GetMinimum()

    # property of fixed image
    region_fixed = image_fixed.GetLargestPossibleRegion()
    start_fixed = region_fixed.GetIndex()
    size_fixed = region_fixed.GetSize()
    origin_fixed = image_fixed.GetOrigin()
    spacing_fixed = image_fixed.GetSpacing()

    # resample image
    resampleFilter = itk.ResampleImageFilter[ImageType, ImageType].New()
    interpolatorType = itk.LinearInterpolateImageFunction[ImageType, itk.D]
    interpolator = interpolatorType.New()
    resampleFilter.SetInterpolator(interpolator)
    resampleFilter.SetTransform(vtk_transform)
    resampleFilter.SetSize(size_fixed)
    resampleFilter.SetOutputOrigin(origin_fixed)
    resampleFilter.SetOutputSpacing(spacing_fixed)
    resampleFilter.SetOutputStartIndex(start_fixed)
    resampleFilter.SetDefaultPixelValue(edgePaddingValue)
    resampleFilter.SetInput(image_moving)
    resampleFilter.Update()
    image_resampled_rigid = resampleFilter.GetOutput()
    return image_resampled_rigid


def resample_moving_image_after_rigid_transform(image_fixed, image_moving, vtk_transform):
    """
    Resample the moving image based on the fixed image's grid when given a rotation and translation
    This function only give a overlap size of fixed image and rotated moving image
    :param image_fixed: fixed image (using its grids)
    :param image_moving: moving image
    :param vtk_transform: vtk rigid transform
    :return: resampled itk image of fixed and moving
    """
    # image type
    ImageType = itk.Image[itk.F, 3]

    # back grounf value fixed
    minCalculatorF = itk.MinimumMaximumImageCalculator[ImageType].New()
    minCalculatorF.SetImage(image_fixed)
    minCalculatorF.Compute()
    edgePaddingValueF = minCalculatorF.GetMinimum()

    # back grounf value moving
    minCalculatorM = itk.MinimumMaximumImageCalculator[ImageType].New()
    minCalculatorM.SetImage(image_moving)
    minCalculatorM.Compute()
    edgePaddingValueM = minCalculatorM.GetMinimum()

    # property of fixed image
    region_fixed = image_fixed.GetLargestPossibleRegion()
    start_fixed = region_fixed.GetIndex()
    size_fixed = region_fixed.GetSize()
    origin_fixed = image_fixed.GetOrigin()
    spacing_fixed = image_fixed.GetSpacing()
    # property of moving image
    region_moving = image_moving.GetLargestPossibleRegion()
    start_moving = region_moving.GetIndex()
    size_moving = region_moving.GetSize()
    origin_moving = image_moving.GetOrigin()
    spacing_moving = image_moving.GetSpacing()
    # center
    origin_fixed_np = np.array([origin_fixed[0], origin_fixed[1], origin_fixed[2]])
    start_fixed_np = np.array([start_fixed[0], start_fixed[1], start_fixed[2]])
    size_fixed_np = np.array([size_fixed[0], size_fixed[1], size_fixed[2]])
    spacing_fixed_np = np.array([spacing_fixed[0], spacing_fixed[1], spacing_fixed[2]])

    origin_moving_np = np.array([origin_moving[0], origin_moving[1], origin_moving[2]])
    start_moving_np = np.array([start_moving[0], start_moving[1], start_moving[2]])
    size_moving_np = np.array([size_moving[0], size_moving[1], size_moving[2]])
    spacing_moving_np = np.array([spacing_moving[0], spacing_moving[1], spacing_moving[2]])

    rot_matrix = itk.GetArrayFromVnlMatrix(vtk_transform.GetMatrix().GetVnlMatrix().as_matrix())
    offset = np.array([vtk_transform.GetOffset()[0], vtk_transform.GetOffset()[1], vtk_transform.GetOffset()[2]])

    # get the overlap area
    # define the bound box
    minX = 1e10
    maxX = -1e10
    minY = 1e10
    maxY = -1e10
    minZ = 1e10
    maxZ = -1e10

    # corner
    indexI = np.array([0, round(size_moving_np[0]) - 1], 'int')
    indexJ = np.array([0, round(size_moving_np[1]) - 1], 'int')
    indexK = np.array([0, round(size_moving_np[2]) - 1], 'int')

    # loop for 2^3 corner
    for I in range(2):
        for J in range(2):
            for K in range(2):
                corner = rot_matrix.transpose().dot(np.array([(indexI[I] + start_moving_np[0]) * spacing_moving_np[0] +
                                                              origin_moving_np[0],
                                                              (indexJ[J] + start_moving_np[1]) * spacing_moving_np[1] +
                                                              origin_moving_np[1],
                                                              (indexK[K] + start_moving_np[2]) * spacing_moving_np[2] +
                                                              origin_moving_np[2]]) - offset)
                minX = np.minimum(minX, corner[0])
                maxX = np.maximum(maxX, corner[0])
                minY = np.minimum(minY, corner[1])
                maxY = np.maximum(maxY, corner[1])
                minZ = np.minimum(minZ, corner[2])
                maxZ = np.maximum(maxZ, corner[2])
    minXf = origin_fixed[0] + start_fixed_np[0] * size_fixed_np[0]
    maxXf = origin_fixed[0] + (size_fixed_np[0] + start_fixed_np[0]) * spacing_fixed_np[0]
    minYf = origin_fixed[1] + start_fixed_np[1] * size_fixed_np[1]
    maxYf = origin_fixed[1] + (size_fixed_np[1] + start_fixed_np[1]) * spacing_fixed_np[1]
    minZf = origin_fixed[2] + start_fixed_np[2] * size_fixed_np[2]
    maxZf = origin_fixed[2] + (size_fixed_np[2] + start_fixed_np[2]) * spacing_fixed_np[2]

    # compare for get the overlap region
    minXr = np.maximum(minX, minXf)
    maxXr = np.minimum(maxX, maxXf)
    minYr = np.maximum(minY, minYf)
    maxYr = np.minimum(maxY, maxYf)
    minZr = np.maximum(minZ, minZf)
    maxZr = np.minimum(maxZ, maxZf)

    # stick to the fixed grid
    minXr = np.round((minXr - minXf) / spacing_fixed_np[0]) * spacing_fixed_np[0] + minXf
    maxXr = np.round((maxXr - minXf) / spacing_fixed_np[0]) * spacing_fixed_np[0] + minXf
    minYr = np.round((minYr - minYf) / spacing_fixed_np[1]) * spacing_fixed_np[1] + minYf
    maxYr = np.round((maxYr - minYf) / spacing_fixed_np[1]) * spacing_fixed_np[1] + minYf
    minZr = np.round((minZr - minZf) / spacing_fixed_np[2]) * spacing_fixed_np[2] + minZf
    maxZr = np.round((maxZr - minZf) / spacing_fixed_np[2]) * spacing_fixed_np[2] + minZf

    # resample ruler
    size_re = itk.Size[3]()
    size_re[0] = int(np.round((maxXr - minXr) / spacing_fixed_np[0]))
    size_re[1] = int(np.round((maxYr - minYr) / spacing_fixed_np[1]))
    size_re[2] = int(np.round((maxZr - minZr) / spacing_fixed_np[2]))
    origin_re = [0, 0, 0]
    origin_re[0] = minXr
    origin_re[1] = minYr
    origin_re[2] = minZr
    start_re = itk.Index[3]()
    start_re[0] = 0
    start_re[1] = 0
    start_re[2] = 0
    spacing_re = [0, 0, 0]
    spacing_re[0] = spacing_fixed[0]
    spacing_re[1] = spacing_fixed[1]
    spacing_re[2] = spacing_fixed[2]

    # resample image for fixed image
    resampleFilterF = itk.ResampleImageFilter[ImageType, ImageType].New()
    interpolatorF = itk.LinearInterpolateImageFunction[ImageType, itk.D].New()
    resampleFilterF.SetInterpolator(interpolatorF)
    resampleFilterF.SetSize(size_re)
    resampleFilterF.SetOutputOrigin(origin_re)
    resampleFilterF.SetOutputSpacing(spacing_re)
    resampleFilterF.SetOutputStartIndex(start_re)
    resampleFilterF.SetDefaultPixelValue(edgePaddingValueF)
    resampleFilterF.SetInput(image_fixed)
    resampleFilterF.Update()
    image_resampled_rigid_f = resampleFilterF.GetOutput()

    # resample image for moving image
    resampleFilterM = itk.ResampleImageFilter[ImageType, ImageType].New()
    interpolatorM = itk.LinearInterpolateImageFunction[ImageType, itk.D].New()
    resampleFilterM.SetInterpolator(interpolatorM)
    resampleFilterM.SetTransform(vtk_transform)
    resampleFilterM.SetSize(size_re)
    resampleFilterM.SetOutputOrigin(origin_re)
    resampleFilterM.SetOutputSpacing(spacing_re)
    resampleFilterM.SetOutputStartIndex(start_re)
    resampleFilterM.SetDefaultPixelValue(edgePaddingValueM)
    resampleFilterM.SetInput(image_moving)
    resampleFilterM.Update()
    image_resampled_rigid_m = resampleFilterM.GetOutput()

    # return
    return image_resampled_rigid_f, image_resampled_rigid_m


def resample_moving_image_after_def_transform(image_fixed, image_moving, displacement_field_itk):
    """
    resample the moving image after rigid and deformable transform
    :param image_fixed: itk fixed image
    :param image_moving: itk moving image
    :param vtk_transform: rigid registration result fixed->moving Affine transform
    :param displacement_field_itk: deformable registration result fixed->moved affine transformed
    :return: the resampled itk image
    """
    # image type
    ImageType = itk.Image[itk.F, 3]

    # back grounf value
    minCalculator = itk.MinimumMaximumImageCalculator[ImageType].New()
    minCalculator.SetImage(image_moving)
    minCalculator.Compute()
    edgePaddingValue = minCalculator.GetMinimum()
    # deformable
    # typedef
    VectorComponentType = itk.F
    VectorPixelType = itk.Vector[VectorComponentType, 3]
    DisplacementFieldType = itk.Image[VectorPixelType, 3]

    # resample
    filterType = itk.ResampleImageFilter[DisplacementFieldType, DisplacementFieldType]
    resampleImageFilter = filterType.New()
    interpolatorType = itk.LinearInterpolateImageFunction[DisplacementFieldType, itk.D]
    interpolator = interpolatorType.New()
    resampleImageFilter.SetInterpolator(interpolator)
    resampleImageFilter.SetDefaultPixelValue([0, 0, 0])
    resampleImageFilter.SetOutputSpacing(image_fixed.GetSpacing())
    resampleImageFilter.SetOutputOrigin(image_fixed.GetOrigin())
    resampleImageFilter.SetOutputStartIndex(image_fixed.GetLargestPossibleRegion().GetIndex())
    resampleImageFilter.SetSize(image_fixed.GetLargestPossibleRegion().GetSize())
    resampleImageFilter.SetInput(displacement_field_itk)
    resampleImageFilter.Update()
    displacement_resampled_itk = resampleImageFilter.GetOutput()
    # wrap displacement field
    warpFilter = itk.WarpImageFilter[ImageType, ImageType, DisplacementFieldType].New()
    interpolator = itk.LinearInterpolateImageFunction[ImageType, itk.D].New()
    warpFilter.SetInterpolator(interpolator)
    warpFilter.SetOutputSpacing(image_fixed.GetSpacing())
    warpFilter.SetOutputOrigin(image_fixed.GetOrigin())
    warpFilter.SetOutputDirection(image_fixed.GetDirection())
    warpFilter.SetDisplacementField(displacement_resampled_itk)
    warpFilter.SetInput(image_moving)
    warpFilter.SetEdgePaddingValue(edgePaddingValue)
    warpFilter.Update()
    image_resampled_total = warpFilter.GetOutput()
    return image_resampled_total




def resample_moving_image_after_total_transform(image_fixed, image_moving, vtk_transform, displacement_field_itk):
    """
    resample the moving image after rigid and deformable transform
    :param image_fixed: itk fixed image
    :param image_moving: itk moving image
    :param vtk_transform: rigid registration result fixed->moving Affine transform
    :param displacement_field_itk: deformable registration result fixed->moved affine transformed
    :return: the resampled itk image
    """
    # image type
    ImageType = itk.Image[itk.F, 3]

    # back grounf value
    minCalculator = itk.MinimumMaximumImageCalculator[ImageType].New()
    minCalculator.SetImage(image_moving)
    minCalculator.Compute()
    edgePaddingValue = minCalculator.GetMinimum()

    # property of fixed image
    region_fixed = image_fixed.GetLargestPossibleRegion()
    start_fixed = region_fixed.GetIndex()
    size_fixed = region_fixed.GetSize()
    origin_fixed = image_fixed.GetOrigin()
    spacing_fixed = image_fixed.GetSpacing()

    # resample image
    resampleFilter = itk.ResampleImageFilter[ImageType, ImageType].New()
    interpolatorType = itk.LinearInterpolateImageFunction[ImageType, itk.D]
    interpolator = interpolatorType.New()
    resampleFilter.SetInterpolator(interpolator)
    resampleFilter.SetTransform(vtk_transform)
    resampleFilter.SetSize(size_fixed)
    resampleFilter.SetOutputOrigin(origin_fixed)
    resampleFilter.SetOutputSpacing(spacing_fixed)
    resampleFilter.SetOutputStartIndex(start_fixed)
    resampleFilter.SetDefaultPixelValue(edgePaddingValue)
    resampleFilter.SetInput(image_moving)
    resampleFilter.Update()
    image_after_rigid = resampleFilter.GetOutput()

    # deformable
    # typedef
    VectorComponentType = itk.F
    VectorPixelType = itk.Vector[VectorComponentType, 3]
    DisplacementFieldType = itk.Image[VectorPixelType, 3]

    # resample
    filterType = itk.ResampleImageFilter[DisplacementFieldType, DisplacementFieldType]
    resampleImageFilter = filterType.New()
    interpolatorType = itk.LinearInterpolateImageFunction[DisplacementFieldType, itk.D]
    interpolator = interpolatorType.New()
    resampleImageFilter.SetInterpolator(interpolator)
    resampleImageFilter.SetDefaultPixelValue([0, 0, 0])
    resampleImageFilter.SetOutputSpacing(image_fixed.GetSpacing())
    resampleImageFilter.SetOutputOrigin(image_fixed.GetOrigin())
    resampleImageFilter.SetOutputStartIndex(image_fixed.GetLargestPossibleRegion().GetIndex())
    resampleImageFilter.SetSize(image_fixed.GetLargestPossibleRegion().GetSize())
    resampleImageFilter.SetInput(displacement_field_itk)
    resampleImageFilter.Update()
    displacement_resampled_itk = resampleImageFilter.GetOutput()
    # wrap displacement field
    warpFilter = itk.WarpImageFilter[ImageType, ImageType, DisplacementFieldType].New()
    interpolator = itk.LinearInterpolateImageFunction[ImageType, itk.D].New()
    warpFilter.SetInterpolator(interpolator)
    warpFilter.SetOutputSpacing(image_fixed.GetSpacing())
    warpFilter.SetOutputOrigin(image_fixed.GetOrigin())
    warpFilter.SetOutputDirection(image_fixed.GetDirection())
    warpFilter.SetDisplacementField(displacement_resampled_itk)
    warpFilter.SetInput(image_after_rigid)
    warpFilter.SetEdgePaddingValue(edgePaddingValue)
    warpFilter.Update()
    image_resampled_total = warpFilter.GetOutput()
    return image_resampled_total



def itk_image_regularization(itk_image):
    """
    Regularization of itk Image when direction is nor identity
    :param itk_image: origin itk Image
    :return: image regularized
    """
    # typedef
    ImageType3dF = itk.Image[itk.F, 3]
    # property of 3d image
    region3d = itk_image.GetLargestPossibleRegion()
    start3d = region3d.GetIndex()
    size3d = region3d.GetSize()
    direction3d = itk_image.GetDirection()
    spacing3d = itk_image.GetSpacing()

    # direction matrix
    R = itk.GetArrayFromVnlMatrix(direction3d.GetVnlMatrix().as_matrix())
    # spacing matrix
    S = np.diag(spacing3d)

    # if not a identity matrix we need to re-sample
    if np.abs(R[0,0]-1)>1e-6 or np.abs(R[1,1]-1)>1e-6 or np.abs(R[1,1]-1)>1e-6:
        # min or max
        minMaxCaiculator = itk.MinimumMaximumImageCalculator[ImageType3dF].New()
        minMaxCaiculator.SetImage(itk_image)
        minMaxCaiculator.Compute()
        default_value = minMaxCaiculator.GetMinimum ()

        # new index start with 0 this is good
        startIndex = itk.Index[3]()
        startIndex[0] = 0
        startIndex[1] = 0
        startIndex[2] = 0

        # get the origin
        origin = itk_image.GetOrigin()

        # define the bound box
        minX = 1e10
        maxX = -1e10
        minY = 1e10
        maxY = -1e10
        minZ = 1e10
        maxZ = -1e10

        # corner
        indexI = np.array([0,round(size3d[0])-1],'int')
        indexJ = np.array([0,round(size3d[1])-1],'int')
        indexK = np.array([0,round(size3d[2])-1],'int')

        #loop for 2^3 corner
        for I in range(2):
            for J in range(2):
                for K in range(2):
                    corner = origin + R.dot(S.dot( np.array([indexI[I]+start3d[0],
                                                             indexJ[J]+start3d[1],
                                                             indexK[K]+start3d[2]])))
                    minX = np.minimum(minX, corner[0])
                    maxX = np.maximum(maxX, corner[0])
                    minY = np.minimum(minY, corner[1])
                    maxY = np.maximum(maxY, corner[1])
                    minZ = np.minimum(minZ, corner[2])
                    maxZ = np.maximum(maxZ, corner[2])

        # new origion
        originResample = [0,0,0]
        originResample[0] = minX
        originResample[1] = minY
        originResample[2] = minZ

        #new size
        sizeResample = itk.Size[3]()
        sizeResample[0] = int(round((maxX-minX)/spacing3d[0]))+1
        sizeResample[1] = int(round((maxY-minY)/spacing3d[1]))+1
        sizeResample[2] = int(round((maxZ-minZ)/spacing3d[2]))+1

        desiredDirection = itk.GetMatrixFromArray(np.array([[1., 0, 0], [0, 1., 0], [0, 0, 1.]]))

        # res-ample identity
        resampleFilter = itk.ResampleImageFilter[ImageType3dF,ImageType3dF].New()
        interpolatorType = itk.LinearInterpolateImageFunction[ImageType3dF, itk.D]
        interpolator = interpolatorType.New()
        resampleFilter.SetInterpolator(interpolator)
        resampleFilter.SetOutputDirection(desiredDirection)
        resampleFilter.SetSize(sizeResample)
        resampleFilter.SetOutputOrigin(originResample)
        resampleFilter.SetOutputSpacing(spacing3d)
        resampleFilter.SetOutputStartIndex(startIndex)
        resampleFilter.SetDefaultPixelValue(default_value)
        resampleFilter.SetInput(itk_image)

        resampleFilter.Update()
        itk_image_resample = resampleFilter.GetOutput()
    else:
        itk_image_resample = itk_image
    return itk_image_resample

def itk_image_series_reader(dicom_path):
    """
        Read DICOM Series
        :param dicom_path: path of dicom folder
        :param default_value: background -1024 for ct 0 for PT&MR
        :return: itk_image
    """
    # typedef image type
    image_type = itk.Image[itk.F, 3]
    # read dicom"
    reader = itk.ImageSeriesReader[image_type].New()
    dicom_io = itk.GDCMImageIO.New()
    names_generator = itk.GDCMSeriesFileNames.New()
    names_generator.SetGlobalWarningDisplay(False)
    asciiStr = dicom_path.encode('utf-8')
    asciiStr2 = asciiStr.decode('Latin')
    names_generator.SetDirectory(asciiStr2)
    series_uid = names_generator.GetSeriesUIDs()
    series_identifier = None

    for uid in series_uid:
        series_identifier = uid
        break
    assert series_identifier is not None

    file_names = names_generator.GetFileNames(series_identifier)
    reader.SetImageIO(dicom_io)
    reader.SetFileNames(file_names)
    reader.Update()
    itk_image = reader.GetOutput()
    if itk_image is None:
        return

    itk_image = itk_image_regularization(itk_image)
    return itk_image

def itk_image_nifti_reader(nii_path):
    """
    Read NIFTI data Image
    :param nii_path: path of *.nii file
    :return: itk_image
    """
    # typedef image type
    image_type = itk.Image[itk.F, 3]
    # read dicom"
    itk.NiftiImageIOFactory.RegisterOneFactory
    reader = itk.ImageFileReader[image_type].New()
    reader.SetFileName(nii_path)
    reader.Update()
    itk_image = reader.GetOutput()
    itk_image = itk_image_regularization(itk_image)
    return itk_image

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

def itk_displacement_nifti_writer(itk_displacement, nii_path):
    """
    Write NIFTI data displacemnet field
    :param itk_displacement: itk_displacement 4d image
    :param nii_path: path of the writing path
    :return: 1
    """
    # typedef image type
    VectorComponentType = itk.F
    VectorPixelType = itk.Vector[VectorComponentType, 3]
    DisplacementFieldType = itk.Image[VectorPixelType, 3]
    # read dicom"
    itk.NiftiImageIOFactory.RegisterOneFactory
    writer = itk.ImageFileWriter[DisplacementFieldType].New()
    writer.SetInput(itk_displacement)
    writer.SetFileName(nii_path)
    writer.Update()
    # print(image)
    return 1

def displacement_to_itk(displacement,fix_image):
    """
    Convert class Image to itk image
    :param displacement: Class Displacement
    :return: itk displacement
    """
    # displacement field to itk

    displacement_origin = fix_image.GetOrigin()
    displacement_spacing = fix_image.GetSpacing()
    displacement_size = fix_image.GetLargestPossibleRegion().GetSize()
    # itk type
    np_displacement = displacement.copy()

    VectorComponentType = itk.F
    VectorPixelType = itk.Vector[VectorComponentType, 3]
    DisplacementFieldType = itk.Image[VectorPixelType, 3]
    ImageType = itk.Image[VectorComponentType, 3]
    # ComposeImageFilter
    composeImageFilter = itk.ComposeImageFilter[ImageType, DisplacementFieldType].New()
    itk_displacement_0 = itk.GetImageViewFromArray(np_displacement[0, :, :, :])
    itk_displacement_1 = itk.GetImageViewFromArray(np_displacement[1, :, :, :])
    itk_displacement_2 = itk.GetImageViewFromArray(np_displacement[2, :, :, :])
    composeImageFilter.SetInput(0, itk_displacement_0)
    composeImageFilter.SetInput(1, itk_displacement_1)
    composeImageFilter.SetInput(2, itk_displacement_2)
    composeImageFilter.Update()
    itk_displacement = composeImageFilter.GetOutput()
    # index
    displacement_start = itk.Index[3]()
    displacement_start.Fill(0)
    # region
    displacement_region = itk.ImageRegion[3]()
    displacement_region.SetSize(displacement_size)
    displacement_region.SetIndex(displacement_start)
    # set region
    itk_displacement.SetRegions(displacement_region)
    # set origin
    itk_displacement.SetOrigin(displacement_origin)
    # set spacing
    itk_displacement.SetSpacing(displacement_spacing)
    return itk_displacement










def resample_image(image_fixed,spacing_fixed):
    """
    Resample the moving image based on the fixed image's grid when given a rotation and translation
    This function only give a overlap size of fixed image and rotated moving image
    :param image_fixed: fixed image (using its grids)
    :param image_moving: moving image
    :param vtk_transform: vtk rigid transform
    :return: resampled itk image of fixed and moving
    """
    # image type
    ImageType = itk.Image[itk.F, 3]

    # back grounf value fixed
    minCalculatorF = itk.MinimumMaximumImageCalculator[ImageType].New()
    minCalculatorF.SetImage(image_fixed)
    minCalculatorF.Compute()
    edgePaddingValueF = minCalculatorF.GetMinimum()



    # property of fixed image
    region_fixed = image_fixed.GetLargestPossibleRegion()

    start_fixed = region_fixed.GetIndex()
    size_fixed = region_fixed.GetSize()
    origin_fixed = image_fixed.GetOrigin()
    #spacing_fixed = image_fixed.GetSpacing()
    # center
    origin_fixed_np = np.array([origin_fixed[0], origin_fixed[1], origin_fixed[2]])
    start_fixed_np = np.array([start_fixed[0], start_fixed[1], start_fixed[2]])
    size_fixed_np = np.array([size_fixed[0], size_fixed[1], size_fixed[2]])
    spacing_fixed_np = np.array([spacing_fixed[0], spacing_fixed[1], spacing_fixed[2]])

    minX = 1e10
    maxX = -1e10
    minY = 1e10
    maxY = -1e10
    minZ = 1e10
    maxZ = -1e10

    # corner
    indexI = np.array([0, round(size_fixed_np[0]) - 1], 'int')
    indexJ = np.array([0, round(size_fixed_np[1]) - 1], 'int')
    indexK = np.array([0, round(size_fixed_np[2]) - 1], 'int')

    # loop for 2^3 corner
    for I in range(2):
        for J in range(2):
            for K in range(2):
                corner = np.array([(indexI[I] + start_fixed_np[0]) * spacing_fixed_np[0] +
                                                              origin_fixed_np[0],
                                                              (indexJ[J] + start_fixed_np[1]) * spacing_fixed_np[1] +
                                                              origin_fixed_np[1],
                                                              (indexK[K] + start_fixed_np[2]) * spacing_fixed_np[2] +
                                                              origin_fixed_np[2]])
                minX = np.minimum(minX, corner[0])
                maxX = np.maximum(maxX, corner[0])
                minY = np.minimum(minY, corner[1])
                maxY = np.maximum(maxY, corner[1])
                minZ = np.minimum(minZ, corner[2])
                maxZ = np.maximum(maxZ, corner[2])

    minXf = origin_fixed[0] + start_fixed_np[0] * size_fixed_np[0]
    maxXf = origin_fixed[0] + (size_fixed_np[0] + start_fixed_np[0]) * spacing_fixed_np[0]
    minYf = origin_fixed[1] + start_fixed_np[1] * size_fixed_np[1]
    maxYf = origin_fixed[1] + (size_fixed_np[1] + start_fixed_np[1]) * spacing_fixed_np[1]
    minZf = origin_fixed[2] + start_fixed_np[2] * size_fixed_np[2]
    maxZf = origin_fixed[2] + (size_fixed_np[2] + start_fixed_np[2]) * spacing_fixed_np[2]

    minXr = np.maximum(minX, minXf)
    maxXr = np.minimum(maxX, maxXf)
    minYr = np.maximum(minY, minYf)
    maxYr = np.minimum(maxY, maxYf)
    minZr = np.maximum(minZ, minZf)
    maxZr = np.minimum(maxZ, maxZf)

    # stick to the fixed grid
    minXr = np.round((minXr - minXf) / spacing_fixed_np[0]) * spacing_fixed_np[0] + minXf
    maxXr = np.round((maxXr - minXf) / spacing_fixed_np[0]) * spacing_fixed_np[0] + minXf
    minYr = np.round((minYr - minYf) / spacing_fixed_np[1]) * spacing_fixed_np[1] + minYf
    maxYr = np.round((maxYr - minYf) / spacing_fixed_np[1]) * spacing_fixed_np[1] + minYf
    minZr = np.round((minZr - minZf) / spacing_fixed_np[2]) * spacing_fixed_np[2] + minZf
    maxZr = np.round((maxZr - minZf) / spacing_fixed_np[2]) * spacing_fixed_np[2] + minZf

    # resample ruler
    size_re = itk.Size[3]()
    size_re[0] = int(np.round((maxXr - minXr) / spacing_fixed_np[0]))
    size_re[1] = int(np.round((maxYr - minYr) / spacing_fixed_np[1]))
    size_re[2] = int(np.round((maxZr - minZr) / spacing_fixed_np[2]))
    origin_re = [0, 0, 0]
    origin_re[0] = minXr
    origin_re[1] = minYr
    origin_re[2] = minZr
    start_re = itk.Index[3]()
    start_re[0] = 0
    start_re[1] = 0
    start_re[2] = 0
    spacing_re = [0, 0, 0]
    spacing_re[0] = spacing_fixed[0]
    spacing_re[1] = spacing_fixed[1]
    spacing_re[2] = spacing_fixed[2]

    # resample image for fixed image
    resampleFilterF = itk.ResampleImageFilter[ImageType, ImageType].New()
    interpolatorF = itk.LinearInterpolateImageFunction[ImageType, itk.D].New()
    resampleFilterF.SetInterpolator(interpolatorF)
    resampleFilterF.SetSize(size_re)
    resampleFilterF.SetOutputOrigin(origin_re)
    resampleFilterF.SetOutputSpacing(spacing_re)
    resampleFilterF.SetOutputStartIndex(start_re)
    resampleFilterF.SetDefaultPixelValue(edgePaddingValueF)
    resampleFilterF.SetInput(image_fixed)
    resampleFilterF.Update()
    image_resampled_rigid_f = resampleFilterF.GetOutput()

    return image_resampled_rigid_f






def resample_image_to_11(image_fixed, image_moving,resample_spacing):
    ImageType = itk.Image[itk.F, 3]
    # back grounf value fixed
    minCalculatorF = itk.MinimumMaximumImageCalculator[ImageType].New()
    minCalculatorF.SetImage(image_fixed)
    minCalculatorF.Compute()
    edgePaddingValueF = minCalculatorF.GetMinimum()

    # back grounf value moving
    minCalculatorM = itk.MinimumMaximumImageCalculator[ImageType].New()
    minCalculatorM.SetImage(image_moving)
    minCalculatorM.Compute()
    edgePaddingValueM = minCalculatorM.GetMinimum()

    region_fixed = image_fixed.GetLargestPossibleRegion()
    old_size = region_fixed.GetSize()
    old_size = np.array([old_size[0], old_size[1], old_size[2]])
    origin = image_fixed.GetOrigin()
    origin = np.array([origin[0], origin[1], origin[2]])
    old_spacing = image_fixed.GetSpacing()
    old_spacing = np.array([old_spacing[0], old_spacing[1], old_spacing[2]])

    size_re = itk.Size[3]()
    size_re[0] = int(np.round(old_size[0]* old_spacing[0]))
    size_re[1] = int(np.round(old_size[1]* old_spacing[1]))
    size_re[2] = int(old_size[2])

    spacing_re = np.array([resample_spacing[0], resample_spacing[1], resample_spacing[2]])



    # resample image for fixed image
    resampleFilterF = itk.ResampleImageFilter[ImageType, ImageType].New()
    interpolatorF = itk.LinearInterpolateImageFunction[ImageType, itk.D].New()
    resampleFilterF.SetInterpolator(interpolatorF)
    resampleFilterF.SetSize(size_re)
    resampleFilterF.SetOutputOrigin(origin)
    resampleFilterF.SetOutputSpacing(spacing_re)
   # resampleFilterF.SetOutputStartIndex(start_re)
    resampleFilterF.SetDefaultPixelValue(edgePaddingValueF)
    resampleFilterF.SetInput(image_fixed)
    resampleFilterF.Update()
    image_resampled_rigid_f = resampleFilterF.GetOutput()

    # resample image for moving image
    resampleFilterM = itk.ResampleImageFilter[ImageType, ImageType].New()
    interpolatorM = itk.LinearInterpolateImageFunction[ImageType, itk.D].New()
    resampleFilterM.SetInterpolator(interpolatorM)
    resampleFilterM.SetSize(size_re)
    resampleFilterM.SetOutputOrigin(origin)
    resampleFilterM.SetOutputSpacing(spacing_re)
   # resampleFilterM.SetOutputStartIndex(start_re)
    resampleFilterM.SetDefaultPixelValue(edgePaddingValueM)
    resampleFilterM.SetInput(image_moving)
    resampleFilterM.Update()
    image_resampled_rigid_m = resampleFilterM.GetOutput()
    return image_resampled_rigid_f, image_resampled_rigid_m
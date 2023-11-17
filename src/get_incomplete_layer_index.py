import numpy as np
import skimage.morphology as sm


def max_distance_to_center(bw):
    x, y = np.meshgrid( np.linspace(0, bw.shape[1]-1, bw.shape[1]),
                      np.linspace(0, bw.shape[0]-1, bw.shape[0]))
    x0 = np.floor(bw.shape[0]/2)
    y0 = np.floor(bw.shape[1]/2)
    distance = np.sqrt((x-x0)**2 + (y-y0)**2)
    bwdis = distance * bw
    maxdis = np.max(bwdis)
    return maxdis


def is_incomplete_image(img, airHU=-1000, bodyHU=0, maxradiu=-1):
    '''
    judge the 2d image is incomplete.
    '''
    airbw = img > airHU
    bodybw = img > bodyHU
    airbw = sm.opening(airbw, sm.square(3))
    bodybw = sm.opening(bodybw, sm.square(3))

    airdis = max_distance_to_center(airbw)
    bodydis = max_distance_to_center(bodybw)

    eps = 6
    incomplete = False
    if maxradiu < 0:
        maxradiu = min(img.shape)/2
    # if air area is incomplete
    if (airdis + eps) < maxradiu:
        incomplete = True
    # if find no body (no matter air area is complete or not),it's incomplete
    if bodydis == 0:
        incomplete = True

    return incomplete


def get_max_radius_of_air(img3d, airHU=-1000):
    '''
    get radius of air by coronal and sagittal
    '''
    shape1 = (int)(img3d.shape[1]/2)
    coronal = img3d[:, shape1, :]
    coronalbw = coronal > airHU
    coronalbw = np.sum(coronalbw, axis=0)
    coronalIdx = np.nonzero(coronalbw)
    coronaldis = np.max(np.abs(coronalIdx[0]-shape1))

    shape2 = (int)(img3d.shape[2]/2)
    sagittal = img3d[:, :, shape2]
    sagittalbw = sagittal > airHU
    sagittalbw = np.sum(sagittalbw, axis=0)
    sagittalIdx = np.nonzero(sagittalbw)
    sagittaldis = np.max(np.abs(sagittalIdx[0]-shape2))
    return max(coronaldis, sagittaldis)


# def get_incomplete_layer_index(img3d):
#     '''
#     remove cbct incomplete layer in Z direction
#     '''
#     airHU = np.min(img3d)
#     bodyHU = -100
#     maxradiu = get_max_radius_of_air(img3d, airHU)
#     # isincomplete = is_incomplete_image(img3d[2], airHU=airHU, bodyHU=bodyHU, maxradiu=maxradiu)
#     incompleteidxtop = []
#     incompleteidxbottom = []
#     for i in range(len(img3d)):
#         isincomplete = is_incomplete_image(
#             img3d[i], airHU=airHU, bodyHU=bodyHU, maxradiu=maxradiu)
#         if not isincomplete:
#             break
#         incompleteidxtop.append(i)
#         continue
#
#     for i in range(len(img3d)-1, 0, -1):
#         isincomplete = is_incomplete_image(
#             img3d[i], airHU=airHU, bodyHU=bodyHU, maxradiu=maxradiu)
#         if not isincomplete:
#             break
#         incompleteidxbottom.append(i)
#         continue
#
#     return incompleteidxtop, incompleteidxbottom

def get_incomplete_layer_index(img3d):
    '''
    remove cbct incomplete layer in Z direction
    '''
    airHU = np.min(img3d)
    incompleteidxtop = []
    incompleteidxbottom = []
    for i in range(img3d.shape[0]):
        if img3d[i].max() == airHU:
            incompleteidxtop.append(i)
        else:
             break
    for i in range(len(img3d) - 1, 0, -1):
        if img3d[i].max() == airHU:
            incompleteidxbottom.append(i)
        else:
             break

    return incompleteidxtop, incompleteidxbottom
if __name__ == '__main__':
    import DicomIO
    import itk
    import matplotlib.pyplot as plt

    files = {
        R"F:\1.MyTest\2.CBCT.Process\Export_0001149462\WKY-0001149462\CBCT1\image":
        [0, 1, 2, 3, 4, 87, 86, 85, 84, 83],
        R"F:\1.MyTest\0.CBCTDetectIncompleteLayers\1.2.246.352.61.2.4804164383210528000.10841528546293039793":
        [0, 69, 68, 67, 66, 65, 64, 63, 62, 61, 60,
            59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49],
        R"F:\1.MyTest\0.CBCTDetectIncompleteLayers\CBCT":
        [0, 1, 63, 62],
        R"F:\1.MyTest\0.CBCTDetectIncompleteLayers\head-cbct":
        [0, 1, 2, 76, 75]
    }

    inputDirCBCT = R"F:\1.MyTest\0.CBCTDetectIncompleteLayers\CBCT"
    img3dCBCT, reader = DicomIO.ItkReadDicom(inputDirCBCT)
    imgArrayCBCT = itk.GetArrayFromImage(img3dCBCT)
    idxtop, idxbottom = get_incomplete_layer_index(imgArrayCBCT)
    print(idxtop)
    print(idxbottom)
    output = np.delete(imgArrayCBCT, idxtop+idxbottom, 0)
    plt.imshow(output[0])
    plt.imshow(output[len(output)-1])

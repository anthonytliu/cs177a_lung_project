import settings
import helpers
import glob
import os
import cv2  # conda install -c https://conda.anaconda.org/menpo opencv3
import scipy.misc
import dicom  # pip install pydicom
import numpy
from multiprocessing import Pool
import math


def load_patient(src_dir):
    slices = [dicom.read_file(src_dir + '/' + s) for s in os.listdir(src_dir)]
    slices.sort(key=lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = numpy.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = numpy.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness
    return slices


def get_pixels_hu(slices):
    image = numpy.stack([s.pixel_array for s in slices])
    image = image.astype(numpy.int16)
    image[image == -2000] = 0
    for slice_number in range(len(slices)):
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(numpy.float64)
            image[slice_number] = image[slice_number].astype(numpy.int16)
        image[slice_number] += numpy.int16(intercept)

    return numpy.array(image, dtype=numpy.int16)

#not used?
def resample(image, scan, new_spacing=[1, 1, 1]):
    spacing = map(float, ([scan[0].SliceThickness] + scan[0].PixelSpacing))
    spacing = numpy.array(list(spacing))
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = numpy.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)
    return image, new_spacing

def cv_flip(img,cols,rows,degree):
    M = cv2.getRotationMatrix2D((cols / 2, rows /2), degree, 1.0)
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst

def extract_dicom_images_patient(src_dir):
    #directory where the extracted images will go
    target_dir = settings.NDSB3_EXTRACTED_IMAGE_DIR
    print("Dir: ", src_dir)
    #directory of the patient (folder name = patientID)
    dir_path = settings.NDSB3_RAW_SRC_DIR + src_dir + "/"
    patient_id = src_dir
    #sorts slices by acquisition, obtains thickness, applies to all slices, then returns array of slices
    slices = load_patient(dir_path)
    print(len(slices), "\t", slices[0].SliceThickness, "\t", slices[0].PixelSpacing)
    print("Orientation: ", slices[0].ImageOrientationPatient)
    #assert slices[0].ImageOrientationPatient == [1.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000]
    cos_value = (slices[0].ImageOrientationPatient[0])
    cos_degree = round(math.degrees(math.acos(cos_value)),2)
    #convert to image
    pixels = get_pixels_hu(slices)
    image = pixels
    print(image.shape)
    #check if order is inverted
    invert_order = slices[1].ImagePositionPatient[2] > slices[0].ImagePositionPatient[2]
    print("Invert order: ", invert_order, " - ", slices[1].ImagePositionPatient[2], ",", slices[0].ImagePositionPatient[2])
    #rescale images based on pixel spacing
    pixel_spacing = slices[0].PixelSpacing
    pixel_spacing.append(slices[0].SliceThickness)
    image = helpers.rescale_patient_images(image, pixel_spacing, settings.TARGET_VOXEL_MM)
    if not invert_order:
        image = numpy.flipud(image)
    #go through the height of the image (rows = [0])
    for i in range(image.shape[0]):
        patient_dir = target_dir + patient_id + "/"
        if not os.path.exists(patient_dir):
            os.mkdir(patient_dir)
        #write image out as .png file
        img_path = patient_dir + "img_" + str(i).rjust(4, '0') + "_i.png"
        #original image is the one just written to folder
        org_img = image[i]
        # if there exists slope,rotation image with corresponding degree
        if cos_degree>0.0:
            org_img = cv_flip(org_img,org_img.shape[1],org_img.shape[0],cos_degree)
        img, mask = helpers.get_segmented_lungs(org_img.copy())
        #normalize hounsfield units
        org_img = helpers.normalize_hu(org_img)
        cv2.imwrite(img_path, org_img * 255)
        cv2.imwrite(img_path.replace("_i.png", "_m.png"), mask * 255)


def extract_dicom_images(clean_targetdir_first=False, only_patient_id=None):
    print("Extracting images")
    #the directory where the NDSB3 extracted images will go
    target_dir = settings.NDSB3_EXTRACTED_IMAGE_DIR
    #make output dir
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    #check if preexisting files are there, if so, remove them
    if clean_targetdir_first and only_patient_id is not None:
        print("Cleaning target dir")
        for file_path in glob.glob(target_dir + "*.*"):
            os.remove(file_path)
    #patientID will always be NONE here
    if only_patient_id is None:
        #the directory where the raw NDSB3 files are stored
        dirs = os.listdir(settings.NDSB3_RAW_SRC_DIR)
        dirs = [s for s in dirs if not '.' in s]
        #this is always True
        if True:
            #create pools = # of patients, or 8
            pool = Pool(settings.WORKER_POOL_SIZE)
            #each pool (thread) is going to run extract_dicom_images_patient on each patient folder in dirs
            pool.map(extract_dicom_images_patient, dirs)
        else:
            for dir_entry in dirs:
                extract_dicom_images_patient(dir_entry)
    else:
        extract_dicom_images_patient(only_patient_id)


if __name__ == '__main__':
    extract_dicom_images(clean_targetdir_first=False, only_patient_id=None)
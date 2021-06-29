"""
Resize the input images for quick iteration 
"""
import pydicom
from PIL import Image
def read_xray(path, voi_lut = False, fix_monochrome = True):
    dicom = pydicom.read_file(path)
    data = dicom.pixel_array
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data
    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
    return data
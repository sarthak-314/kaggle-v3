from multiprocessing import Pool
from tqdm import tqdm
import pandas as pd
import argparse
import pydicom
import os



args={}
args['input'] = '../input/rsna-miccai-brain-tumor-radiogenomic-classification'
args['output'] = './'
args['dataset'] = 'train'
args['n_jobs'] = 20
args['debug'] = 0


DICOM_FIELDS = [
    'AccessionNumber',
    'AcquisitionMatrix',
    # 'B1rms',  # Empty
    # 'BitsAllocated',  # = 16
    # 'BitsStored',  # = 16
    'Columns',
    'ConversionType',
    # 'DiffusionBValue',  # 0 or empty
    # 'DiffusionGradientOrientation',  # [0.0, 0.0, 0.0] or empty
    'EchoNumbers',
    # 'EchoTime',  # empty
    'EchoTrainLength',
    'FlipAngle',
    # 'HighBit',  # = 15
    # 'HighRRValue',  #  0 or empty
    'ImageDimensions',  # 2 or epty
    'ImageFormat',
    'ImageGeometryType',
    'ImageLocation',
    'ImageOrientation',
    'ImageOrientationPatient',
    'ImagePosition',
    'ImagePositionPatient',
    # 'ImageType',  # ['DERIVED', 'SECONDARY']
    'ImagedNucleus',
    'ImagingFrequency',
    'InPlanePhaseEncodingDirection',
    'InStackPositionNumber',
    'InstanceNumber',
    # 'InversionTime',   # empty
    # 'Laterality',  # empty
    # 'LowRRValue',  # empty
    'MRAcquisitionType',
    'MagneticFieldStrength',
    # 'Modality',  # MR
    'NumberOfAverages',
    'NumberOfPhaseEncodingSteps',
    'PatientID',
    'PatientName',
    # 'PatientPosition',  # HFS
    'PercentPhaseFieldOfView',
    'PercentSampling',
    # 'PhotometricInterpretation',  # MONOCHROME2
    'PixelBandwidth',
    # 'PixelPaddingValue',  # empty or 0
    'PixelRepresentation',
    'PixelSpacing',
    # 'PlanarConfiguration',  # 0 or empty
    # 'PositionReferenceIndicator',  # 'NA' or empty
    'PresentationLUTShape',
    'ReconstructionDiameter',
    # 'RescaleIntercept',  # = 0
    # 'RescaleSlope',  # = 1
    # 'RescaleType',  # = US
    'Rows',
    'SAR',
    'SOPClassUID',
    'SOPInstanceUID',
    # 'SamplesPerPixel',  # = 1
    'SeriesDescription',
    'SeriesInstanceUID',
    'SeriesNumber',
    'SliceLocation',
    'SliceThickness',
    'SpacingBetweenSlices',
    'SpatialResolution',
    'SpecificCharacterSet',
    'StudyInstanceUID',
    # 'TemporalResolution',  # 0 or empty
    # 'TransferSyntaxUID',  # = 1.2.840.10008.1.2
    # 'TriggerWindow',  # = 0
    'WindowCenter',
    'WindowWidth'
]

# All of the FM fields are empty
FM_FIELDS = [
    'FileMetaInformationGroupLength',
    'FileMetaInformationVersion',
    'ImplementationClassUID',
    'ImplementationVersionName',
    'MediaStorageSOPClassUID',
    'MediaStorageSOPInstanceUID',
    'SourceApplicationEntityTitle',
    'TransferSyntaxUID',
]

final = []


def get_meta_info(dicom):
    row = {f: dicom.get(f) for f in DICOM_FIELDS}
    row_fm = {f: dicom.file_meta.get(f) for f in FM_FIELDS}
    row_other = {
    #  'is_original_encoding': dicom.is_original_encoding,  # = True
    #  'is_implicit_VR': dicom.is_implicit_VR,  # = True
    #  'is_little_endian': dicom.is_little_endian, # = True
        'timestamp': dicom.timestamp,
    }
    return {**row,
            #**row_fm,  # All are emtpy
            **row_other}


def get_dicom_files(input_dir, ds='train'):
    dicoms = []

    for subdir, dirs, files in os.walk(f"{input_dir}/{ds}"):
        for filename in files:
            filepath = subdir + os.sep + filename

            if filepath.endswith(".dcm"):
                dicoms.append(filepath)

    return dicoms


def process_dicom(dicom_src, _x):
    dicom = pydicom.dcmread(dicom_src)
    file_data = dicom_src.split("/")
    file_src = "/".join(file_data[-4:])

    tmp = {"BraTS21ID": file_data[-3], "dataset": file_data[-4], "type": file_data[-2], "dicom_src": f"./{file_src}"}
    tmp.update(get_meta_info(dicom))

    return tmp


def update(res):
    if res is not None:
        final.append(res)

    pbar.update()


def error(e):
    print(e)



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="./input")
    ap.add_argument("--output", type=str, default="./")
    ap.add_argument("--dataset", type=str, default="train")
    ap.add_argument("--n_jobs", type=int, default=20)
    ap.add_argument("--debug", type=int, default=0)

    args = vars(ap.parse_args())

    dicom_files = get_dicom_files(args["input"], args["dataset"])

    if args["debug"]:
        dicom_files = dicom_files[:1000]

    pool = Pool(processes=args["n_jobs"])
    pbar = tqdm(total=len(dicom_files))

    for dicom_file in dicom_files:
        pool.apply_async(
            process_dicom,
            args=(dicom_file, ''),
            callback=update,
            error_callback=error,
        )

    pool.close()
    pool.join()
    pbar.close()

    final = pd.DataFrame(final)
    final.to_csv(f"{args['output']}/dicom_meta_{args['dataset']}.csv", index=False)
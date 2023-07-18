import logging
import math
import subprocess
import time
from pathlib import Path

import boto3
import h5py
import s3fs


# curl -OL https://downloaduav.jpl.nasa.gov/Release2z/Haywrd_14501_21043_012_210602_L090_CX_02/Haywrd_14501_21043_012_210602_L090_CX_129_02.h5
# h5stat Haywrd_14501_21043_012_210602_L090_CX_129_02.h5 | grep 'File space page size'
# h5stat Haywrd_14501_21043_012_210602_L090_CX_129_02.h5 | grep 'File metadata'
# h5repack -S PAGE -G 689984 Haywrd_14501_21043_012_210602_L090_CX_129_02.h5 nisar_reformatted.h5
# h5repack -S PAGE -G 689984 -l /science/LSAR/SLC/swaths/frequencyA/VV:CHUNK=1024x1024 Haywrd_14501_21043_012_210602_L090_CX_129_02.h5 nisar_repaged_rechunked.h5
# h5repack -S PAGE -G 689984  Haywrd_14501_21043_012_210602_L090_CX_129_02.h5 nisar_repaged.h5
# h5repack -l /science/LSAR/SLC/swaths/frequencyA/VV:CHUNK=1024x1024 Haywrd_14501_21043_012_210602_L090_CX_129_02.h5 nisar_rechunked.h5

KB = 1024
MB = KB * KB

S3_FILESYSTEM = s3fs.S3FileSystem(anon=True, default_block_size=4 * MB)
S3 = boto3.client('s3')
bucket_name = 'ffwilliams2-shenanigans'
prefix = 'h5repack'
base_s3_url = f's3://{bucket_name}/{prefix}/'
base_s3_url = f'https://{bucket_name}.s3.us-west-2.amazonaws.com/{prefix}/'


def optimize_hdf5(filepath, dataset_path, output_path, chunk_size=8 * MB, page_size=2 * MB):
    # TODO find way to get file metadata_size
    metadata_size = 689984
    assert page_size >= metadata_size
    with h5py.File(filepath, mode='r') as file:
        datatype = file[dataset_path].dtype
        datatype_size = datatype.itemsize

    n_pixels = chunk_size / datatype_size
    dim_2d = int(math.sqrt(n_pixels))

    cmd = f'h5repack -S PAGE -G {page_size} -l {dataset_path}:CHUNK={dim_2d}x{dim_2d} {filepath} {output_path}'
    print(f'Running the command: {cmd}')
    subprocess.run(cmd.split(' '))


def upload_test_datasets(dataset_names, bucket, prefix_path):
    for name in dataset_names:
        S3.upload_file(name, bucket, f'{prefix_path}/name')


class GetRequestCounter(logging.Handler):
    def __init__(self):
        super().__init__()
        self.count = 0

    def emit(self, record):
        if record.getMessage().startswith('CALL: get_object'):
            self.count += 1


def print_hdf5_information(file, dataset):
    datatype = dataset.dtype
    print('Datatype:')
    print('  Name:', datatype.name)
    print('  Bytes per Item:', datatype.itemsize)
    chunks = dataset.chunks
    print('Chunk Layout:')
    print('  Chunk Shape:', chunks)
    print('  Chunk Size (MB):', (chunks[0] * chunks[1] * datatype.itemsize) / MB)
    _, rdcc_nslots, rdcc_nbytes, rdcc_w0 = file.id.get_access_plist().get_cache()
    print('Chunk Cache Size (MB):', rdcc_nbytes / MB)
    mdc_nbytes = file.id.get_mdc_config().initial_size
    mdc_min_nbytes = file.id.get_mdc_config().min_size
    print('Metadata Cache Size (MB):', mdc_nbytes / MB)
    print('Metadata Cache Min Size (MB):', mdc_min_nbytes / MB)


def open_test_dataset():
    with open('Haywrd_14501_21043_012_210602_L090_CX_129_02.h5', mode='rb') as fobj:
        with h5py.File(fobj, mode='r') as hdf_file:
            vv = hdf_file['science']['LSAR']['SLC']['swaths']['frequencyA']['VV']
            print_hdf5_information(hdf_file, vv)


def test_routine(filepath, extra_args):
    with h5py.File(filepath, mode='r', **extra_args) as ds:
        vv = ds['science']['LSAR']['SLC']['swaths']['frequencyA']['VV']
        data = vv[0, 0]
        data = vv[3000:8000, 500:2000]
        # data = vv[:, :]


def benchmark(filepath, name, local=True):
    logger = logging.getLogger('s3fs')
    counter_handler = GetRequestCounter()
    logger.addHandler(counter_handler)
    logger.setLevel(logging.DEBUG)
    extra_args = {}
    if 'repaged' in filepath:
        extra_args['page_buf_size'] = 8 * MB

    if 'rechunkeed' in filepath:
        extra_args['rdcc_bytes'] = 64 * MB

    start_time = time.time()
    if local:
        with open(filepath, 'rb') as fobj:
            test_routine(fobj, extra_args)
    else:
        with S3_FILESYSTEM.open(filepath, mode='rb') as fobj:
            test_routine(fobj, extra_args)
    stop_time = time.time()
    print(f'{name} time: {stop_time-start_time:.4f}')
    print(f'Number of GET requests: {counter_handler.count}')


def benchmark2(filepath, name, page_buf_size=None, rdcc_nbytes=None):
    extra_args = {}
    if filepath.startswith('s3') or filepath.startswith('https'):
        extra_args['driver'] = 'ros3'

    if page_buf_size:
        extra_args['page_buf_size'] = page_buf_size

    if rdcc_nbytes:
        extra_args['rdcc_nbytes'] = rdcc_nbytes

    start_time = time.time()
    test_routine(filepath, extra_args)
    stop_time = time.time()
    print(f'{name} time: {stop_time-start_time:.4f}')


def run_benchmarks():
    benchmark('Haywrd_14501_21043_012_210602_L090_CX_129_02.h5', 'Unaltered Local', local=True)
    benchmark(base_s3_url + 'Haywrd_14501_21043_012_210602_L090_CX_129_02.h5', 'unaltered s3', local=False)
    benchmark(base_s3_url + 'nisar_repaged_.h5', 'repacked metadata s3', local=False)
    benchmark(base_s3_url + 'nisar_rechunked.h5', 'rechunked data s3', local=false)
    benchmark(base_s3_url + 'nisar_repaged_rechunked.h5', 'repacked metadata and rechunked data', local=False)


if __name__ == '__main__':
    file = 'Haywrd_14501_21043_012_210602_L090_CX_129_02.h5'
    benchmark2(file, 'local')
    benchmark2(base_s3_url + file, 's3 base')
    benchmark2(base_s3_url + 'nisar_repaged.h5', 's3 modified', page_buf_size=4 * MB, rdcc_nbytes=64*MB)
    benchmark2(
        base_s3_url + 'nisar_repaged_rechunked.h5', 's3 modified chunk', page_buf_size=8 * MB, rdcc_nbytes=128 * MB
    )

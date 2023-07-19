import logging
import math
import os
import subprocess
import time
from itertools import product
from pathlib import Path

import boto3
import h5py
import numpy as np
import pandas as pd
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

    options = ''
    if page_size:
        options += f' -S PAGE -G {page_size}'

    if chunk_size:
        n_pixels = chunk_size / datatype_size
        dim_2d = int(math.sqrt(n_pixels))
        options += f' -l {dataset_path}:CHUNK={dim_2d}x{dim_2d}'

    options += f' {filepath} {output_path}'
    cmd = 'h5repack' + options
    print(f'Command: {cmd}')
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
        # data = vv[0, 0]
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


def benchmark2(filepath, name=None, page_buf_size=None, rdcc_nbytes=None, repeat=1):
    run_times = []
    for i in range(repeat):
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
        total_time = stop_time - start_time
        run_times.append(total_time)
        if name:
            print(f'{name} time: {total_time:.4f}')

    return np.min(run_times)


def prep_test_dataset(filepath, dataset_path, output_s3_bucket, output_s3_path, chunk_size=8 * MB, page_size=2 * MB):
    optimize_hdf5(filepath, dataset_path, 'tmp.h5', chunk_size, page_size)
    S3.upload_file('tmp.h5', output_s3_bucket, output_s3_path)
    os.remove('tmp.h5')


def prep_random_bytes(num_bytes, output_s3_bucket, output_s3_path):
    with open('tmp.bin', 'wb') as file:
        random_bytes = os.urandom(num_bytes)
        file.write(random_bytes)
    S3.upload_file('tmp.bin', output_s3_bucket, output_s3_path)
    os.remove('tmp.bin')


def prep_dataset_set():
    file = 'Haywrd_14501_21043_012_210602_L090_CX_129_02.h5'
    dataset = '/science/LSAR/SLC/swaths/frequencyA/VV'
    bucket = 'ffwilliams2-shenanigans'
    prefix = 'h5repack'
    # page_sizes = [1, 2, 4, 8, 16]
    # for page_size in page_sizes:
    #     prep_test_dataset(
    #         file,
    #         dataset,
    #         bucket,
    #         f'{prefix}/reoptimize_page{page_size}MB.h5',
    #         chunk_size=None,
    #         page_size=page_size * MB,
    #     )
    chunk_sizes = [2, 4, 8, 16, 24, 48]
    for chunk_size in chunk_sizes:
        prep_test_dataset(
            file,
            dataset,
            bucket,
            f'{prefix}/reoptimize_page2MB_chunk{chunk_size}.h5',
            chunk_size=chunk_size * MB,
            page_size=2 * MB,
        )

    # a quarter of the example dataset size
    n_bytes = 7072*1320*8
    prep_random_bytes(n_bytes, bucket, f'{prefix}/random_bytes_{int(n_bytes / MB)}MB.bin')


def run_page_benchmarks():
    page_sizes = [1, 2, 4, 8, 16]
    page_buffer_sizes = [2, 8, 32, 128, 512]

    run_page_size, run_page_buffer_size, run_time = [[], [], []]
    rows = []
    for page_size, page_buffer_size in product(page_sizes, page_buffer_sizes):
        if page_size > page_buffer_size:
            continue
        time = benchmark2(
            base_s3_url + f'reoptimize_page{page_size}MB.h5',
            page_buf_size=page_buffer_size * MB,
            rdcc_nbytes=None,
            repeat=5,
        )
        rows.append([page_size, page_buffer_size, time])

    df = pd.DataFrame(rows)
    df.columns = ['page_size', 'page_buffer_size', 'time']
    df.to_csv('page_results.csv', index=False)


def run_benchmarks():
    benchmark('Haywrd_14501_21043_012_210602_L090_CX_129_02.h5', 'Unaltered Local', local=True)
    benchmark(base_s3_url + 'Haywrd_14501_21043_012_210602_L090_CX_129_02.h5', 'unaltered s3', local=False)
    benchmark(base_s3_url + 'nisar_repaged_.h5', 'repacked metadata s3', local=False)
    benchmark(base_s3_url + 'nisar_rechunked.h5', 'rechunked data s3', local=false)
    benchmark(base_s3_url + 'nisar_repaged_rechunked.h5', 'repacked metadata and rechunked data', local=False)


if __name__ == '__main__':
    prep_dataset_set()
    # run_page_benchmarks()

    # file = 'Haywrd_14501_21043_012_210602_L090_CX_129_02.h5'
    # optimize_hdf5(file, '/science/LSAR/SLC/swaths/frequencyA/VV', 'tmp.h5')
    # benchmark2(file, 'local')
    # benchmark2(base_s3_url + file, 's3 base')
    # benchmark2(base_s3_url + 'nisar_repaged.h5', 's3 modified', page_buf_size=4 * MB, rdcc_nbytes=64*MB)
    # benchmark2(
    #     base_s3_url + 'nisar_repaged_rechunked.h5', 's3 modified chunk', page_buf_size=8 * MB, rdcc_nbytes=128 * MB
    # )

import io
import logging
import math
import os
import subprocess
import time
from itertools import product

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


def get_metadata_size(filepath):
    cmd = f'h5stat {filepath}'
    output = subprocess.run(cmd.split(' '), stdout=subprocess.PIPE, text=True).stdout
    cleaned_lines = [line.strip() for line in output.replace('\t', '').strip().split('\n')]
    metadata_size_line = [line for line in cleaned_lines if 'File metadata' in line][0]
    _, _, metadata_size_text, _ = metadata_size_line.split(' ')
    metadata_size_bytes = int(metadata_size_text)
    return metadata_size_bytes


def optimize_hdf5(filepath, output_path, dataset_path=None, chunk_size=8 * MB, page_size=2 * MB):
    options = ''
    if page_size:
        metadata_size_bytes = get_metadata_size(filepath)
        print(f'Metadata size / Page size = {metadata_size_bytes / page_size:.2f}')
        options += f' -S PAGE -G {page_size}'

    if chunk_size and dataset_path:
        with h5py.File(filepath, mode='r') as file:
            datatype = file[dataset_path].dtype
            datatype_size = datatype.itemsize

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


def open_test_dataset():
    with open('Haywrd_14501_21043_012_210602_L090_CX_129_02.h5', mode='rb') as fobj:
        with h5py.File(fobj, mode='r') as hdf_file:
            vv = hdf_file['science']['LSAR']['SLC']['swaths']['frequencyA']['VV']
            print_hdf5_information(hdf_file, vv)


def test_routine(filepath, extra_args, index_exp=np.index_exp[0, 0]):
    with h5py.File(filepath, mode='r', **extra_args) as ds:
        vv = ds['science']['LSAR']['SLC']['swaths']['frequencyA']['VV']
        vv[index_exp]


def benchmark2(filepath, name=None, page_buf_size=None, rdcc_nbytes=None, index_exp=np.index_exp[0, 0], repeat=1):
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
        test_routine(filepath, extra_args, index_exp)
        stop_time = time.time()
        total_time = stop_time - start_time
        run_times.append(total_time)
        if name:
            print(f'{name} time: {total_time:.4f}')

    return np.min(run_times)


def prep_test_dataset(filepath, dataset_path, output_s3_bucket, output_s3_path, chunk_size=8 * MB, page_size=2 * MB):
    optimize_hdf5(filepath, 'tmp.h5', dataset_path, chunk_size, page_size)
    S3.upload_file('tmp.h5', output_s3_bucket, output_s3_path)
    os.remove('tmp.h5')


def prep_random_array(shape, output_s3_bucket, output_s3_path):
    array = np.random.rand(*shape).astype(np.csingle)
    np.save('tmp.npy', array, allow_pickle=False)
    S3.upload_file('tmp.npy', output_s3_bucket, output_s3_path)
    os.remove('tmp.npy')


def prep_dataset_set():
    file = 'Haywrd_14501_21043_012_210602_L090_CX_129_02.h5'
    dataset = '/science/LSAR/SLC/swaths/frequencyA/VV'
    bucket = 'ffwilliams2-shenanigans'
    prefix = 'h5repack'
    page_sizes = [1, 2, 4, 8, 16]
    for page_size in page_sizes:
        prep_test_dataset(
            file,
            dataset,
            bucket,
            f'{prefix}/reoptimize_page{page_size}MB.h5',
            chunk_size=None,
            page_size=page_size * MB,
        )
    chunk_sizes = [1, 2, 4, 8, 16, 24]
    for chunk_size in chunk_sizes:
        prep_test_dataset(
            file,
            dataset,
            bucket,
            f'{prefix}/reoptimize_page2MB_chunk{chunk_size}MB.h5',
            chunk_size=chunk_size * MB,
            page_size=2 * MB,
        )

    # a quarter of the example dataset size
    prep_random_array([7072, 1320], bucket, f'{prefix}/random_array_7072x1320.npy')


def run_page_benchmarks():
    page_sizes = [1, 2, 4, 8, 16]
    page_buffer_sizes = [2, 4, 8, 32, 128, 512]

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
    df.columns = ['chunk_size', 'page_buffer_size', 'time']
    df.to_csv('page_results.csv', index=False)


def run_chunk_benchmarks():
    chunk_sizes = [1, 2, 4, 8, 16]
    chunk_buffer_sizes = [4, 8, 16, 32, 64]
    index_exp = np.index_exp[1000:8072, 1000:2320]

    rows = []
    for chunk_size, chunk_buffer_size in product(chunk_sizes, chunk_buffer_sizes):
        time = benchmark2(
            base_s3_url + f'reoptimize_page2MB_chunk{chunk_size}MB.h5',
            page_buf_size=4 * MB,
            rdcc_nbytes=chunk_buffer_size * MB,
            index_exp=index_exp,
            repeat=5,
        )
        rows.append([chunk_size, chunk_buffer_size, time])

    df = pd.DataFrame(rows)
    df.columns = ['chunk_size', 'chunk_buffer_size', 'time']
    df.to_csv('chunk_results.csv', index=False)


def run_control_benchmarks():
    origional_filename = 'Haywrd_14501_21043_012_210602_L090_CX_129_02.h5'
    local_metadata_time = benchmark2(origional_filename, repeat=5)
    s3_metadata_time = benchmark2(base_s3_url + origional_filename, repeat=5)

    index_exp = np.index_exp[1000:8072, 1000:2320]
    s3_time = benchmark2(base_s3_url + origional_filename, index_exp=index_exp, repeat=5)
    paged_only_time = benchmark2(
        base_s3_url + 'reoptimize_page2MB.h5', page_buf_size=4 * MB, index_exp=index_exp, repeat=5
    )

    raw_bytes_times = []
    for _ in range(5):
        start_time = time.time()
        tmp_s3_client = boto3.client('s3')
        response = tmp_s3_client.get_object(Bucket='ffwilliams2-shenanigans', Key='h5repack/random_array_7072x1320.npy')
        byte_data = response['Body'].read()
        np.load(io.BytesIO(byte_data))
        stop_time = time.time()
        raw_bytes_times.append(stop_time - start_time)
    raw_bytes_time = np.min(raw_bytes_times)

    names = ['local (metadata only)', 's3 (metadata only)', 's3', 'paged only', 'raw bytes']
    times = [local_metadata_time, s3_metadata_time, s3_time, paged_only_time, raw_bytes_time]
    df = pd.DataFrame({'name': names, 'time': times})
    df.to_csv('control_results.csv', index=False)


if __name__ == '__main__':
    with h5py.File('Haywrd_14501_21043_012_210602_L090_CX_129_02.h5') as file:
        vv = file['science']['LSAR']['SLC']['swaths']['frequencyA']['VV']
        print_hdf5_information(file, vv)

    prep_dataset_set()
    run_page_benchmarks()
    run_chunk_benchmarks()
    run_control_benchmarks()

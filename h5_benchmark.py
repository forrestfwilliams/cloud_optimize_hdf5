import time
import logging
from pathlib import Path

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

S3_FILESYSTEM = s3fs.S3FileSystem(anon=True, default_block_size=24*MB)
bucket_name = 'ffwilliams2-shenanigans'
prefix = 'h5repack'
# filename = '/Users/ffwilliams2/data/hdf5/Haywrd_14501_21043_012_210602_L090_CX_129_02.h5'
# filename = '/Users/ffwilliams2/data/hdf5/nisar_repaged.h5'
# filename = '/Users/ffwilliams2/data/hdf5/nisar_repaged.h5'
base_s3_url = f's3://{bucket_name}/{prefix}/'


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


def test_routine(fobj, extra_args):
    ds = h5py.File(fobj, mode='r', **extra_args)
    vv = ds['science']['LSAR']['SLC']['swaths']['frequencyA']['VV']
    # data = vv[0:20, 0:20]
    data = vv[3000:8000, 500:2000]
    # data = vv[:, :]


def benchmark(filepath, name, local=True):
    logger = logging.getLogger('s3fs')
    counter_handler = GetRequestCounter()
    logger.addHandler(counter_handler)
    logger.setLevel(logging.DEBUG)
    extra_args = {}
    if 'repaged' in filepath:
        extra_args['page_buf_size'] = 8*MB

    if 'rechunkeed' in filepath:
        extra_args['rdcc_bytes'] = 64*MB

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


def run_benchmarks():
    benchmark('Haywrd_14501_21043_012_210602_L090_CX_129_02.h5', 'Unaltered Local', local=True)
    benchmark(base_s3_url + 'Haywrd_14501_21043_012_210602_L090_CX_129_02.h5', 'unaltered s3', local=False)
    # benchmark(base_s3_url + 'nisar_repaged_.h5', 'repacked metadata s3', local=False)
    # benchmark(base_s3_url + 'nisar_rechunked.h5', 'rechunked data s3', local=false)
    benchmark(base_s3_url + 'nisar_repaged_rechunked.h5', 'repacked metadata and rechunked data', local=False)


def run_benchmark():
    start_time = time.time()
    with open(filename, 'rb') as fobj:
        test_routine(fobj)
    stop_time = time.time()
    print(f'Base time: {stop_time-start_time:.4f}')

    start_time = time.time()
    with s3_filesystem.open(base_s3_url + filename, mode='rb') as fobj:
        test_routine(fobj)
    stop_time = time.time()
    print(f'Unaltered S3 time: {stop_time-start_time:.4f}')

    start_time = time.time()
    with s3_filesystem.open(base_s3_url + 'nisar_repaged.h5', mode='rb') as fobj:
        test_routine(fobj)
    stop_time = time.time()
    print(f'Repacked metadata S3 time: {stop_time-start_time:.4f}')

    start_time = time.time()
    with s3_filesystem.open(base_s3_url + 'nisar_rechunked.h5', mode='rb') as fobj:
        test_routine(fobj)
    stop_time = time.time()
    print(f'Rechunked data S3 time: {stop_time-start_time:.4f}')

    start_time = time.time()
    with s3_filesystem.open(base_s3_url + 'nisar_repaged_rechunked.h5', mode='rb') as fobj:
        test_routine(fobj)
    stop_time = time.time()
    print(f'Repacked metadata AND rechunked data S3 time: {stop_time-start_time:.4f}')


if __name__ == '__main__':
    run_benchmarks()
    # with open(filename, 'rb') as fobj:
    #     file = h5py.File(fobj, mode='r', rdcc_nbytes=2**27, page_buf_size=2**21, min_meta_keep=20, min_raw_keep=20)
    #     vv = file['science']['LSAR']['SLC']['swaths']['frequencyA']['VV']
    #     print_hdf5_information(file, vv)

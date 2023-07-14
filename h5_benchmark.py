import time

import h5py
import s3fs


# curl -OL https://downloaduav.jpl.nasa.gov/Release2z/Haywrd_14501_21043_012_210602_L090_CX_02/Haywrd_14501_21043_012_210602_L090_CX_129_02.h5
# h5stat Haywrd_14501_21043_012_210602_L090_CX_129_02.h5 | grep "File space page size"
# h5stat Haywrd_14501_21043_012_210602_L090_CX_129_02.h5 | grep "File metadata"
# h5repack -S PAGE -G 689984 Haywrd_14501_21043_012_210602_L090_CX_129_02.h5 nisar_reformatted.h5
# h5repack -S PAGE -G 689984 -l /science/LSAR/SLC/swaths/frequencyA/VV:CHUNK=1024x1024 Haywrd_14501_21043_012_210602_L090_CX_129_02.h5 nisar_repaged_rechunked.h5
# h5repack -S PAGE -G 689984  Haywrd_14501_21043_012_210602_L090_CX_129_02.h5 nisar_repaged.h5
# h5repack -l /science/LSAR/SLC/swaths/frequencyA/VV:CHUNK=1024x1024 Haywrd_14501_21043_012_210602_L090_CX_129_02.h5 nisar_rechunked.h5

KB = 1024
MB = KB * KB

s3_filesystem = s3fs.S3FileSystem(anon=True)# , default_block_size=8*MB
bucket_name = 'ffwilliams2-shenanigans'
prefix = 'h5repack'
filename = 'Haywrd_14501_21043_012_210602_L090_CX_129_02.h5'
base_s3_url = f's3://{bucket_name}/{prefix}/'


def print_chunk_layout(dataset):
    datatype = dataset.dtype
    print("Datatype:")
    print("  Name:", datatype.name)
    print("  Bytes per Item:", datatype.itemsize)
    chunks = dataset.chunks
    print("Chunk Layout:")
    print("  Chunk Shape:", chunks)
    print("  Chunk Size (MB):", (chunks[0] * chunks[1] * datatype.itemsize) / MB)


def test_routine(fobj):
    ds = h5py.File(fobj, 'r')
    vv = ds['science']['LSAR']['SLC']['swaths']['frequencyA']['VV']
    # data = vv[3000:8000, 500:2000]
    data = vv[:,:]


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

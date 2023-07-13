import time

import s3fs
import h5py

# curl -OL https://downloaduav.jpl.nasa.gov/Release2z/Haywrd_14501_21043_012_210602_L090_CX_02/Haywrd_14501_21043_012_210602_L090_CX_129_02.h5
# h5stat Haywrd_14501_21043_012_210602_L090_CX_129_02.h5 | grep "File space page size"
# h5stat Haywrd_14501_21043_012_210602_L090_CX_129_02.h5 | grep "File metadata"
# h5repack -S PAGE -G 689984 Haywrd_14501_21043_012_210602_L090_CX_129_02.h5 nisar_reformatted.h5

bucket_name = 'ffwilliams2-shenanigans'
prefix = 'h5repack'
filename = 'Haywrd_14501_21043_012_210602_L090_CX_129_02.h5'
modified_filename = 'nisar_example_reformatted.h5'
base_s3_url = f's3://{bucket_name}/{prefix}/{filename}'
modified_s3_url = f's3://{bucket_name}/{prefix}/{modified_filename}'


s3_filesystem = s3fs.S3FileSystem(anon=True)


def test_routine(fobj):
    ds = h5py.File(fobj, 'r')
    vv = ds['science']['LSAR']['SLC']['swaths']['frequencyA']['VV']
    data = vv[:, :]

start_time = time.time()
with open(filename, 'rb') as fobj:
    test_routine(fobj)
stop_time = time.time()
print(f'Base time: {stop_time-start_time:.4f}')


start_time = time.time()
with s3_filesystem.open(base_s3_url, mode='rb') as fobj:
    test_routine(fobj)
stop_time = time.time()
print(f'Unaltered S3 time: {stop_time-start_time:.4f}')


start_time = time.time()
with s3_filesystem.open(modified_s3_url, mode='rb') as fobj:
    test_routine(fobj)
stop_time = time.time()
print(f'Repacked metadata S3 time: {stop_time-start_time:.4f}')

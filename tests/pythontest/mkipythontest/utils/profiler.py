import glob
import os

import pandas
import torch_npu

PROFILE_TEMP_PATH = os.path.dirname(os.path.abspath(__file__))+'/tmp/'

DEVICE_PROFILE_PATH = '/ASCEND_PROFILER_OUTPUT/kernel_details.csv'

def get_profiler_time(profiling_file_path:str)->float:
    task_time_sum = 0
    profile_path = os.listdir(PROFILE_TEMP_PATH)
    task_time_path = os.path.join(PROFILE_TEMP_PATH, profile_path[0],DEVICE_PROFILE_PATH)
    file_list = glob.glob(task_time_path)
    task_time= pandas.read_csv(file_list[0]).loc[:,"Duration(us)"]
    # remove the first and the second
    task_time = task_time[2:]
    task_time.sort_values()
    for i in range(10):
        task_time_sum += task_time[i+5]
    task_time_avg = task_time_sum/10
    return task_time_avg

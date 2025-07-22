import numpy as np
import pandas as pd
import gnss_lib_py as glp
import os
import urllib.request

# 1. Make sure the directory exists
data_dir = "data/TutData"
os.makedirs(data_dir, exist_ok=True)

# 2. Download the file only if it doesn't exist
data_path = os.path.join(data_dir, "myreceiver.csv")
url = "https://raw.githubusercontent.com/Stanford-NavLab/gnss_lib_py/main/notebooks/tutorials/data/myreceiver.csv"

if not os.path.isfile(data_path):
    print(f"Downloading {url} to {data_path}...")
    urllib.request.urlretrieve(url, data_path)
    print("Download complete.")
else:
    print("File already exists, skipping download.")

navdata = glp.NavData(csv_path=data_path)
print(navdata)

# for timestamp, delta_t, navdata_subset in glp.loop_time(navdata,'myTimestamp'):
#     print('Current timestamp: ', timestamp)
#     print('Difference between current and future time step', delta_t)
#     print('Current group of data')
#     print(navdata_subset)
   
   
double_navdata = glp.concat(navdata, navdata, axis=1)
print(double_navdata) 
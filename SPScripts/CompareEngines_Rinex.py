# This script compare the performance of the GNSS algorithms using the rinex navigation and observation files
import os
import requests
import gnss_lib_py as glp

# Define the target directory and file path
data_dir = "data/TutData"
# file_name = "rinex_obs_mixed_types.20o"
file_name = "aspa3240.24o"
# file_path = os.path.join(data_dir, "brdc1370.20n")
file_path = os.path.join(data_dir, file_name)

# url = "https://raw.githubusercontent.com/Stanford-NavLab/gnss_lib_py/main/data/unit_test/rinex/nav/brdc1370.20n"

# url = "https://raw.githubusercontent.com/Stanford-NavLab/gnss_lib_py/main/data/unit_test/rinex/obs/rinex_obs_mixed_types.20o"

# # Create the directory if it does not exist
# # This replaces `glp.make_dir("../data")`
# os.makedirs(data_dir, exist_ok=True)

# # Download the file only if it doesn't already exist
# if not os.path.exists(file_path):
#     print(f"Downloading {os.path.basename(file_path)}...")
#     response = requests.get(url)
#     response.raise_for_status()  # Ensure the download was successful
    
#     with open(file_path, "wb") as f:
#         f.write(response.content)
#     print("Download complete.")
# else:
#     print(f"'{file_path}' already exists. Skipping download.")
    
# rinex_nav = glp.RinexNav("data/TutData/sfdm3240.24n")   
# rinex_nav = glp.RinexNav("data/TutData/WROC00POL_R_20243240000_01D_GN.rnx")

# print(rinex_nav)

rinex_obs_3 = glp.RinexObs(file_path)

print("\nLoaded Rinex Obs 3 data for the first time instant:\n",
      rinex_obs_3.where('gps_millis', rinex_obs_3['gps_millis', 0], 'eq'))
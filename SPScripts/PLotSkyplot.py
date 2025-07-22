import os
import urllib.request
import gnss_lib_py as glp
import matplotlib.pyplot as plt

# Ensure the data directory exists
data_dir = "data/TutData"
glp.make_dir(data_dir)

# Download the file if it doesn't exist
csv_path = os.path.join(data_dir, "Pixel4XL_derived.csv")
url = "https://raw.githubusercontent.com/Stanford-NavLab/gnss_lib_py/main/data/unit_test/google_decimeter_2021/Pixel4XL_derived.csv"

if not os.path.isfile(csv_path):
    print(f"Downloading {url} to {csv_path}...")
    urllib.request.urlretrieve(url, csv_path)
    print("Download complete.")
else:
    print(f"File {csv_path} already exists, skipping download.")

# Load the Android Google Challenge data
derived_data = glp.AndroidDerived2021(csv_path, remove_timing_outliers=False)
state_estimate = glp.solve_wls(derived_data)

fig = glp.plot_skyplot(derived_data, state_estimate)
plt.show()
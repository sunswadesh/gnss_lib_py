import os
import urllib.request
import gnss_lib_py as glp

# Helper function to download a file if not already present
def download_if_needed(url, path):
    if not os.path.isfile(path):
        print(f"Downloading {url} to {path} ...")
        urllib.request.urlretrieve(url, path)
        print("Download complete.")
    else:
        print(f"File {path} already exists, skipping download.")


# Data directory setup
data_dir = "data/TutData"
glp.make_dir(data_dir)


# Download the CSV file if needed
derived_fname = "Pixel4XL_derived.csv"
derived_url = "https://raw.githubusercontent.com/Stanford-NavLab/gnss_lib_py/main/data/unit_test/google_decimeter_2021/Pixel4XL_derived.csv"
derived_path = os.path.join(data_dir, derived_fname)

download_if_needed(derived_url, derived_path)


# Load Android Google Challenge data
derived_data = glp.AndroidDerived2021(derived_path, remove_timing_outliers=False)

# Solve EKF state estimate
state_ekf = glp.solve_gnss_ekf(derived_data)

# Plot and save map visualization
fig = glp.plot_map(state_ekf)
fig.write_html("data/TutData/results/EKF1.html")

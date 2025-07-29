# This script compares the performance of the GNSS algorithms using the Android Google Challenge data
import os
import urllib.request
import gnss_lib_py as glp
import matplotlib.pyplot as plt
import plotly.io as pio

# Set default renderer to 'iframe' (or 'notebook' if in Jupyter)
pio.renderers.default = 'iframe'


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

# Downloaded files if needed
# derived_fname = "2023-09-06-22-49_device_gnss.csv"
# truth_fname   = "2023-09-06-22-49_ground_truth.csv"
derived_fname = "2023-03-08-21-34_Pixel6_device_gnss.csv"
truth_fname   = "2023-03-08-21-34_Pixel6_ground_truth.csv"


# derived_fname = "device_gnss.csv"
# truth_fname   = "ground_truth.csv"

# Load data
# derived_data = glp.AndroidDerived2021(os.path.join(data_dir, derived_fname), remove_timing_outliers=False)
derived_data = glp.AndroidDerived2023(os.path.join(data_dir, derived_fname))
true_data_path = os.path.join(data_dir, truth_fname)

# Weighted Least Squares
state_estimate = glp.solve_wls(derived_data)
# fig = glp.plot_map(state_estimate)

# simple Kalman Filter
state_kf = glp.solve_gnss_kf(derived_data)

# Extended Kalman Filter
state_ekf = glp.solve_gnss_ekf(derived_data)
# fig = glp.plot_map(state_ekf)

# # Forward and backward Kalman Filter
state_fbkf = glp.solve_gnss_ekf_with_smoothing(derived_data)

# Factor Graph filter
# state_fg = glp.solve_gnss_factor_graph(derived_data)

# truth_data_second_trace = glp.AndroidGroundTruth2021(true_data_path)
truth_data_second_trace = glp.AndroidGroundTruth2023(true_data_path)


# fig = glp.plot_map(state_estimate, truth_data_second_trace, state_ekf)

# fig.write_html("data/TutData/results/2023_Pixel7pro_FULL_WLS_EKFvsTruth.html")
# # Now open this file in any browser manually (e.g., by double-clicking or with `xdg-open my_map.html` from a WSL or Linux shell)

# Export to KML
# kml_output_path = "data/TutData/results/2023-09-06-22-49_Pixel7pro_allFilters_FULLtrajectory.kml"
kml_output_path = os.path.join(data_dir + "/results/", derived_fname[:-16] + "_Pixel6pro_allFilters_FULLtrajectory.kml")

glp.export_navdata_to_kml(state_estimate, truth_data_second_trace, state_kf, state_ekf, state_fbkf,
                         filename=kml_output_path)


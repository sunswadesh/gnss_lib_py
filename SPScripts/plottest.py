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

# Download files if needed
derived_fname = "2023-09-07-18-59_device_gnss.csv"
truth_fname   = "2023-09-07-18-59_ground_truth.csv"

download_if_needed(
    "https://raw.githubusercontent.com/Stanford-NavLab/gnss_lib_py/main/data/unit_test/google_decimeter_2023/2023-09-07-18-59-us-ca/pixel7pro/device_gnss.csv",
    os.path.join(data_dir, derived_fname)
)
download_if_needed(
    " https://raw.githubusercontent.com/Stanford-NavLab/gnss_lib_py/main/data/unit_test/google_decimeter_2023/2023-09-07-18-59-us-ca/pixel7pro/ground_truth.csv",
    os.path.join(data_dir, truth_fname)
)

# Load data
# derived_data = glp.AndroidDerived2021(os.path.join(data_dir, derived_fname), remove_timing_outliers=False)
derived_data = glp.AndroidDerived2023(os.path.join(data_dir, derived_fname))
true_data_path = os.path.join(data_dir, truth_fname)

# plot the pseudorange over time of each individual satellite
# glonass_data = derived_data.where("gnss_id","glonass")
# fig1 = glp.plot_metric(glonass_data, "raw_pr_m", linestyle="None")
# # plt.show()

# # plot the iono and tropo delay over time of each individual satellite. Show both plots simulataneously
# fig2 = glp.plot_metric(glonass_data, "iono_delay_m", "tropo_delay_m", groupby="sv_id",
#                   linestyle="None", markeredgecolor="g", markersize=12,
#                   markeredgewidth=1.0)
# plt.show()
state_estimate = glp.solve_wls(derived_data)
# fig = glp.plot_map(state_estimate)

# truth_data_second_trace = glp.AndroidGroundTruth2021(true_data_path)
truth_data_second_trace = glp.AndroidGroundTruth2023(true_data_path)
fig = glp.plot_map(state_estimate, truth_data_second_trace)

fig.write_html("data/TutData/results/2023_Pixel7pro_FULL_DereivedvsTruth.html")
# Now open this file in any browser manually (e.g., by double-clicking or with `xdg-open my_map.html` from a WSL or Linux shell)


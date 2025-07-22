import os
import urllib.request
import gnss_lib_py as glp
import matplotlib.pyplot as plt

# Helper function to download a file if not already present
def download_if_needed(url, path):
    if not os.path.isfile(path):
        print(f"Downloading {url} to {path} ...")
        urllib.request.urlretrieve(url, path)
        print("Download complete.")
    else:
        print(f"File {path} already exists, skipping download.")

# Create data directory if it doesn't exist
data_dir = "data/TutData"
glp.make_dir(data_dir)

# Define the filename and URL
derived_fname = "Pixel4XL_derived.csv"
derived_url = "https://raw.githubusercontent.com/Stanford-NavLab/gnss_lib_py/main/data/unit_test/google_decimeter_2021/Pixel4XL_derived.csv"
derived_path = os.path.join(data_dir, derived_fname)

# Download the CSV file if not already downloaded
download_if_needed(derived_url, derived_path)

# Load the Android Google Challenge data
derived_data = glp.AndroidDerived2021(derived_path, remove_timing_outliers=False)

state_wls = glp.solve_wls(derived_data)
# When assuming that SV positions are given in the ECEF frame when signals are received use
# state_wls = glp.solve_wls(derived_data, sv_rx_time=True)

# Save the plot as an HTML file
fig = glp.plot_map(state_wls)
fig.write_html("data/TutData/results/Snapshots.html")

# Custom Weighting Schemes for WLS
# Morton, Y. Jade, et al., eds. Position, navigation, and timing technologies in
# the 21st century: Integrated satellite navigation, sensor systems, and civil
# applications, volume 1. John Wiley & Sons, 2021. Section 11.3.1.
derived_data["weights"] = 1./derived_data["raw_pr_sigma_m"]**2

state_wls_sigma = glp.solve_wls(derived_data,weight_type="weights")

state_wls_sigma.rename({"lat_rx_wls_deg":"lat_rx_" + "sigma" + "_deg",
                        "lon_rx_wls_deg":"lon_rx_" + "sigma" + "_deg",
                         "alt_rx_wls_m":"alt_rx_" + "sigma" + "_m",
                         }, inplace=True)

fig = glp.plot_map(state_wls, state_wls_sigma)
fig.write_html("data/TutData/results/Snapshots_sigma.html")

# Estimating Only Receiver Clock Bias with WLS

state_with_clock_bias = glp.solve_wls(derived_data, only_bias=True, receiver_state=state_wls)

fig = glp.plot_metric(state_with_clock_bias,"gps_millis","b_rx_wls_m")
plt.show()

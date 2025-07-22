import simplekml
import numpy as np
import pandas as pd
from gnss_lib_py.navdata.navdata import NavData
from gnss_lib_py.navdata.operations import find_wildcard_indexes
from gnss_lib_py.visualizations import style

def export_navdata_to_kml(*args, filename="output.kml"):
    """
    Export one or more NavData trajectory objects to a KML file.

    Parameters
    ----------
    *args : gnss_lib_py.navdata.navdata.NavData
        One or more NavData objects. Must include latitude and longitude rows named
        like 'lat_*_deg' and 'lon_*_deg', excluding sigma rows.
    filename : str
        Output filepath for the .kml file.
    """

    if len(args) == 0:
        raise ValueError("No NavData objects provided for KML export.")

    kml = simplekml.Kml()
    
    # Predefined colors to cycle through (as simplekml colors)
    colors = [simplekml.Color.red, simplekml.Color.blue, simplekml.Color.green,
              simplekml.Color.orange, simplekml.Color.yellow, simplekml.Color.purple]

    for idx, traj_data in enumerate(args):
        if not isinstance(traj_data, NavData):
            raise TypeError(f"Argument {idx} is not a NavData instance")

        # Find lat/lon row names (exclude sigma)
        traj_idxs = find_wildcard_indexes(traj_data,
                                         wildcards=["lat_*_deg", "lon_*_deg"],
                                         max_allow=1,
                                         excludes=[["lat_sigma_*_deg"], ["lon_sigma_*_deg"]])

        if not traj_idxs["lat_*_deg"] or not traj_idxs["lon_*_deg"]:
            raise ValueError(f"NavData argument {idx} missing latitude or longitude rows")

        lat_key = traj_idxs["lat_*_deg"][0]
        lon_key = traj_idxs["lon_*_deg"][0]

        label_name = style.get_label({"": "_".join((lat_key.split("_"))[1:-1])})

        latitudes = traj_data[lat_key]
        longitudes = traj_data[lon_key]

        coords = list(zip(longitudes, latitudes))  # simplekml expects (lon, lat)

        # Create a linestring for the trajectory
        linestring = kml.newlinestring(name=label_name, coords=coords)
        linestring.style.linestyle.width = 3
        # Cycle through the preset colors
        linestring.style.linestyle.color = colors[idx % len(colors)]
        linestring.style.polystyle.color = simplekml.Color.changealphaint(100, colors[idx % len(colors)])

    kml.save(filename)
    print(f"KML saved to {filename}")

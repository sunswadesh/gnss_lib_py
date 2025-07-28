"""Parses Rinex .o files"""

__authors__ = "Ashwin Kanhere, Swadesh S"
__date__ = "24 July 2025"

import os
from datetime import timezone

import numpy as np
import pandas as pd
import georinex as gr

from gnss_lib_py.navdata.navdata import NavData
import gnss_lib_py.utils.constants as consts
from gnss_lib_py.utils.time_conversions import datetime_to_gps_millis
from gnss_lib_py.navdata.operations import sort, concat

class RinexObs(NavData):
    """Class handling Rinex observation files [1]_.

    The Rinex Observation files (of the format .yyo or .rnx) contain measured
    pseudoranges, carrier phase, doppler and signal-to-noise ratio
    measurements for multiple constellations and bands.

    ## NEW: This loader is designed to be version-agnostic. It robustly
    ## handles both RINEX 2.x and RINEX 3.x formats by using a more
    ## efficient pandas-based reshaping method instead of looping.

    This loader converts observation files into a NavData instance where
    measurements from different bands are treated as separate measurement
    rows. Inherits from NavData().

    References
    ----------
    .. [1] https://files.igs.org/pub/data/format/rinex305.pdf

    """
    def __init__(self, input_paths):
        """Loading Rinex observation files into a NavData based class.

        Should input path to a RINEX observation file (e.g., .o or .rnx).

        Parameters
        ----------
        input_paths : string or path-like or list of paths
            Path to rinex observation file(s).

        """
        ## NEW: The main processing logic has been moved to a separate
        ## `preprocess` method for better code structure and readability.
        obs_df = self.preprocess(input_paths)
        super().__init__(pandas_df=obs_df)
        self.postprocess()

    def preprocess(self, input_paths):
        """
        Loads and reshapes RINEX observation data from wide to long format.
        
        ## NEW: This method is the new core of the parser. It uses pandas
        ## melt and pivot operations to efficiently transform the data from
        ## georinex's output format into the long format required by NavData.
        ## This approach is significantly faster and more robust for handling
        ## both RINEX 2 and 3 formats compared to nested loops.

        Parameters
        ----------
        input_paths : string or path-like or list of paths
            Path to rinex observation file(s).

        Returns
        -------
        pd.DataFrame
            A long-format DataFrame with standardized measurement columns.
        """

        if isinstance(input_paths, (str, os.PathLike)):
            input_paths = [input_paths]
        
        all_obs_df = []

        for path in input_paths:
            # georinex.load is version-agnostic and is the first step.
            ############################################
            ds = gr.load(path)
            print(ds)
            print(ds.coords['time'].dtype)
            obs_df = ds.to_dataframe()
            print(obs_df.index.get_level_values('time').dtype)
            ##########################################

            obs = gr.load(path).to_dataframe()
            obs.dropna(how='all', inplace=True)
            obs.reset_index(inplace=True)

            # Melt the dataframe from wide to long format.
            # Each row will now be a single measurement for a sat at a given time.
            id_vars = ['time', 'sv']
            df_long = obs.melt(id_vars=id_vars, var_name='rinex_code', value_name='measurement')

            # Remove any rows where the measurement is NaN
            df_long.dropna(subset=['measurement'], inplace=True)
            if df_long.empty:
                continue

            # Extract measurement type (C, L, D, S) and observation code (1C, 5X, etc.)
            df_long['measurement_type'] = df_long['rinex_code'].str[0]
            df_long['observation_code'] = df_long['rinex_code'].str[1:]
            
            # Pivot the table to create columns for each measurement type
            df_pivot = df_long.pivot_table(index=['time', 'sv', 'observation_code'],
                                           columns='measurement_type',
                                           values='measurement').reset_index()

            # Rename columns to the gnss_lib_py standard
            rename_map = self._measure_type_dict()
            df_pivot.rename(columns=rename_map, inplace=True)
            all_obs_df.append(df_pivot)

        if not all_obs_df:
            raise FileNotFoundError("No valid observation data found in the provided file(s).")
        
        final_df = pd.concat(all_obs_df, ignore_index=True)
        
        # Convert time column to gps_millis
        final_df['gps_millis'] = datetime_to_gps_millis(final_df['time'])
        final_df.drop(columns=['time'], inplace=True)
        final_df.rename(columns={"sv":"sv_id"}, inplace=True)

        return final_df

    def postprocess(self):
        """
        Performs final data structuring after preprocessing.
        
        This includes splitting satellite IDs, adding signal type information,
        and ensuring the data is sorted.
        """
        # NEW: Add a check to see if sv_id has already been converted to a number.
        # If it's numeric, this postprocessing has likely already run, so we can exit.
        if np.issubdtype(self['sv_id'].dtype, np.number):
            # The check is complete, no need to run the rest of the function.
            return

        # If we are here, sv_id is a string, so we proceed with parsing.
        self['gnss_sv_id'] = self['sv_id']
        gnss_chars = [sv_id[0] for sv_id in np.atleast_1d(self['sv_id'])]
        gnss_nums = [sv_id[1:] for sv_id in np.atleast_1d(self['sv_id'])]
        gnss_id = [consts.CONSTELLATION_CHARS[gnss_char] for gnss_char in gnss_chars]
        self['gnss_id'] = np.asarray(gnss_id)
        # This is the line that converts 'sv_id' to an integer type.
        self['sv_id'] = np.asarray(gnss_nums, dtype=int)
        
        # Assign the gnss_lib_py standard names for signal_type
        rx_constellations = np.unique(self['gnss_id'])
        signal_type_dict = self._signal_type_dict()
        signal_types = np.empty(len(self), dtype=object)
        
        for constellation in rx_constellations:
            # Find rows corresponding to the current constellation
            const_idx = self['gnss_id'] == constellation
            # Get the observation codes for these rows
            obs_codes = self['observation_code'][const_idx]
            # Map observation codes to signal types using the dictionary
            # Handle cases where a code might not be in the dictionary
            mapped_signals = [signal_type_dict[constellation].get(code, 'unknown') for code in obs_codes]
            signal_types[const_idx] = mapped_signals
            
        self['signal_type'] = signal_types
        sort(self,'gps_millis', inplace=True)

    @staticmethod
    def _measure_type_dict():
        """Map of Rinex observation measurement types to standard names.

        Returns
        -------
        measure_type_dict : Dict
            Dictionary of the form {rinex_character : measure_name}
        """

        measure_type_dict = {'C': 'raw_pr_m',
                             'L': 'carrier_phase_cyc',
                             'D': 'doppler_hz',
                             'S': 'cn0_dbhz',
                             # For RINEX 2 compatibility
                             'P': 'raw_pr_m',
                             }
        return measure_type_dict

    @staticmethod
    def _signal_type_dict():
        """Dictionary from constellation and signal bands to signal types.

        Transformations from Section 5.1 in [2]_ and 5.2.17 from [3]_.

        Returns
        -------
        signal_type_dict : Dict
            Dictionary of the form {constellation_band : {band : signal_type}}

        References
        ----------
        .. [2] https://files.igs.org/pub/data/format/rinex304.pdf
        .. [3] https://files.igs.org/pub/data/format/rinex305.pdf

        """
        signal_type_dict = {}
        # GPS
        gps_map = {}
        for code in ['1C','1S','1L','1X','1P','1W','1Y','1M','1N']: gps_map[code] = 'l1'
        for code in ['2C','2D','2S','2L','2X','2P','2W','2Y','2M','2N']: gps_map[code] = 'l2'
        for code in ['5I','5Q','5X']: gps_map[code] = 'l5'
        # RINEX 2 codes
        gps_map.update({'1': 'l1', '2': 'l2', '5': 'l5'})
        signal_type_dict['gps'] = gps_map
        
        # GLONASS
        glo_map = {}
        for code in ['1C','1P']: glo_map[code] = 'g1'
        for code in ['4A','4B','4X']: glo_map[code] = 'g1a'
        for code in ['2C','2P']: glo_map[code] = 'g2'
        for code in ['6A','6B','6X']: glo_map[code] = 'g2a'
        for code in ['3I','3Q','3X']: glo_map[code] = 'g3'
        # RINEX 2 codes
        glo_map.update({'1': 'g1', '2': 'g2', '3': 'g3'})
        signal_type_dict['glonass'] = glo_map

        # Galileo
        gal_map = {}
        for code in ['1A','1B','1C','1X','1Z']: gal_map[code] = 'e1'
        for code in ['5I','5Q','5X']: gal_map[code] = 'e5a'
        for code in ['7I','7Q','7X']: gal_map[code] = 'e5b'
        for code in ['8I','8Q','8X']: gal_map[code] = 'e5'
        for code in ['6A','6B','6C','6X','6Z']: gal_map[code] = 'e6'
        # RINEX 2 codes
        gal_map.update({'1': 'e1', '5': 'e5a', '7': 'e5b', '8':'e5', '6':'e6'})
        signal_type_dict['galileo'] = gal_map

        # SBAS
        sbas_map = {}
        for code in ['1C']: sbas_map[code] = 'l1'
        for code in ['5I','5Q','5X']: sbas_map[code] = 'l5'
        # RINEX 2 codes
        sbas_map.update({'1': 'l1', '5': 'l5'})
        signal_type_dict['sbas'] = sbas_map

        # QZSS
        qzss_map = {}
        for code in ['1C','1S','1L','1X','1Z','1B']: qzss_map[code] = 'l1'
        for code in ['2S','2L','2X']: qzss_map[code] = 'l2'
        for code in ['5I','5Q','5X','5D','5P','5Z']: qzss_map[code] = 'l5'
        for code in ['6S','6L','6X','6E','6Z']: qzss_map[code] = 'l6'
        # RINEX 2 codes
        qzss_map.update({'1': 'l1', '2': 'l2', '5': 'l5', '6': 'l6'})
        signal_type_dict['qzss'] = qzss_map

        # BeiDou
        bds_map = {}
        for code in ['2I','2Q','2X']: bds_map[code] = 'b1'
        for code in ['1D','1P','1X','1A','1N']: bds_map[code] = 'b1c'
        for code in ['1S','1L','1Z']: bds_map[code] = 'b1a'
        for code in ['5D','5P','5X']: bds_map[code] = 'b2a'
        for code in ['7I','7Q','7X','7D','7P','7Z']: bds_map[code] = 'b2b'
        for code in ['8D','8P','8X']: bds_map[code] = 'b2'
        for code in ['6I','6Q','6X','6A']: bds_map[code] = 'b3'
        for code in ['6D','6P','6Z']: bds_map[code] = 'b3a'
        # RINEX 2 codes
        bds_map.update({'1': 'b1', '2': 'b1', '7':'b2', '6':'b3'})
        signal_type_dict['beidou'] = bds_map
        
        # IRNSS/NavIC
        irnss_map = {}
        for code in ['5A','5B','5C','5X']: irnss_map[code] = 'l5'
        for code in ['9A','9B','9C','9X']: irnss_map[code] = 's'
        irnss_map.update({'5': 'l5', '9':'s'})
        signal_type_dict['irnss'] = irnss_map

        return signal_type_dict
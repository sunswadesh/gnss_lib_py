"""Classes for GNSS-based Kalman Filter implementations

"""

__authors__ = "Ashwin Kanhere, Derek Knowles"
__date__ = "25 Jan 2020"

import warnings

import numpy as np

from gnss_lib_py.navdata.navdata import NavData
from gnss_lib_py.navdata.operations import loop_time
from gnss_lib_py.utils import constants as consts
from gnss_lib_py.algorithms.snapshot import solve_wls
from gnss_lib_py.utils.coordinates import ecef_to_geodetic
from gnss_lib_py.utils.filters import BaseExtendedKalmanFilter

def solve_gnss_kf(measurements, init_dict = None,
                   params_dict = None, delta_t_decimals=-2):
    """Runs a simple GNSS Kalman Filter across each timestep.

    This implementation provides a basic Kalman Filter for comparison
    with the Extended Kalman Filter. It assumes a linear measurement model
    (which is a simplification for GNSS pseudoranges) and uses a constant
    velocity motion model.

    Parameters
    ----------
    measurements : gnss_lib_py.navdata.navdata.NavData
        Instance of the NavData class
    init_dict : dict
        Initialization dict with initial states and covariances.
    params_dict : dict
        Dictionary of parameters for GNSS KF.

    Returns
    -------
    state_estimate : gnss_lib_py.navdata.navdata.NavData
        Estimated receiver position in ECEF frame in meters and the
        estimated receiver clock bias also in meters as an instance of
        the NavData class with shape (4 x # unique timesteps) and
        the following rows: x_rx_m, y_rx_m, z_rx_m, b_rx_m.
    """
    # Check that all necessary rows exist in the measurements NavData
    measurements.in_rows(["gps_millis","corr_pr_m",
                            "x_sv_m","y_sv_m","z_sv_m",
                            ])

    # Initialize the initialization dictionary if it's None
    if init_dict is None:
        init_dict = {}

    # Determine the initial state if not provided
    if "state_0" not in init_dict:
        pos_0 = None
        # Use Weighted Least Squares (WLS) to get an initial position estimate
        for _, _, measurement_subset in loop_time(measurements,"gps_millis",
                                                 delta_t_decimals=delta_t_decimals):
            pos_0 = solve_wls(measurement_subset)
            if pos_0 is not None:
                break

        # Initialize the state vector (position, velocity, clock bias)
        state_0 = np.zeros((7,1))
        if pos_0 is not None:
            state_0[:3,0] = pos_0[["x_rx_wls_m","y_rx_wls_m","z_rx_wls_m"]]
            state_0[6,0] = pos_0[["b_rx_wls_m"]]
        init_dict["state_0"] = state_0

    # Initialize the initial covariance matrix if not provided
    if "sigma_0" not in init_dict:
        sigma_0 = np.eye(init_dict["state_0"].size)
        init_dict["sigma_0"] = sigma_0

    # Define the process noise covariance matrix Q if not provided
    if "Q" not in init_dict:
        init_dict["Q"] = np.diag([
            1.0,   # x (position in meters^2)
            1.0,   # y
            1.0,   # z
            0.1,   # vx (velocity in (m/s)^2)
            0.1,   # vy
            0.1,   # vz
            10.0   # b (clock bias in meters^2)
        ])

    # Define the measurement noise covariance matrix R if not provided
    if "R" not in init_dict:
        measurement_noise = np.eye(1) # gets overwritten based on the number of satellites
        init_dict["R"] = measurement_noise

    # Initialize the parameter dictionary if it's None
    if params_dict is None:
        params_dict = {}

    # Set the motion type (default to constant velocity)
    motion_type = params_dict.get('motion_type', 'constant_velocity')

    # Initialize lists to store the estimated states over time
    states = []
    # Initialize the current state and covariance from the initialization dictionary
    current_state = init_dict["state_0"]
    current_covariance = init_dict["sigma_0"]

    # Loop through the measurements for each timestep
    for timestamp, delta_t, measurement_subset in loop_time(measurements,"gps_millis"):
        pos_sv_m = measurement_subset[["x_sv_m","y_sv_m","z_sv_m"]].T
        pos_sv_m = np.atleast_2d(pos_sv_m)
        corr_pr_m = measurement_subset["corr_pr_m"].reshape(-1,1)

        # Remove NaN indexes in satellite positions and pseudoranges
        not_nan_indexes = ~np.isnan(pos_sv_m).any(axis=1) & ~np.isnan(corr_pr_m).any(axis=1)
        pos_sv_m = pos_sv_m[not_nan_indexes]
        corr_pr_m = corr_pr_m[not_nan_indexes]

        num_measurements = pos_sv_m.shape[0]
        if num_measurements < 4:
            warnings.warn(f"Insufficient satellites ({num_measurements}) at timestamp {timestamp} for KF solution.",
                          RuntimeWarning)
            continue

        # --- Prediction Step ---
        # Define the state transition matrix F based on the motion model
        if motion_type == 'stationary':
            F = np.eye(7)
        elif motion_type == 'constant_velocity':
            F = np.eye(7)
            F[:3, 3:6] = delta_t * np.eye(3)
        else:
            raise ValueError(f"Unknown motion type: {motion_type}")

        # Predict the state and covariance
        predicted_state = F @ current_state
        predicted_covariance = F @ current_covariance @ F.T + init_dict["Q"]

        # --- Update Step ---
        # Rotate satellite positions to the reception frame (using the predicted clock bias)
        rotated_pos_sv_m = GNSSEKF.rotate_sv_to_reception(None, pos_sv_m, corr_pr_m, predicted_state[6, 0])

        # Linearize the measurement model to get the measurement matrix H
        H = np.zeros((num_measurements, 7))
        # Expected pseudoranges based on the predicted state
        predicted_ranges = np.sqrt(np.sum((predicted_state[:3].flatten() - rotated_pos_sv_m)**2, axis=1, keepdims=True))
        # Handle the case where predicted range is zero to avoid division by zero
        predicted_ranges[predicted_ranges < 1e-6] = 1e-6
        # Derivatives of range with respect to receiver position (ECEF)
        H[:, 0:3] = (predicted_state[:3].flatten() - rotated_pos_sv_m) / predicted_ranges
        # Derivative of pseudorange with respect to clock bias
        H[:, 6] = 1

        # Measurement noise covariance matrix R (assuming uncorrelated measurements for simplicity)
        R = np.eye(num_measurements) * init_dict["R"][0, 0] # Scale by the initial R value

        # Innovation or measurement residual
        y = corr_pr_m - (predicted_ranges + predicted_state[6])

        # Kalman gain
        S = H @ predicted_covariance @ H.T + R
        K = predicted_covariance @ H.T @ np.linalg.inv(S)

        # Update the state and covariance
        current_state = predicted_state + K @ y
        current_covariance = (np.eye(7) - K @ H) @ predicted_covariance

        # Store the estimated state
        states.append([timestamp] + np.squeeze(current_state).tolist())

    # If no states were estimated, return None
    if not states:
        warnings.warn("No valid state estimate computed in solve_gnss_kf, returning None.", RuntimeWarning)
        return None

    # Create a NavData object to store the state estimates
    state_estimate = NavData()
    state_estimate["gps_millis"] = [state[0] for state in states]
    state_estimate["x_rx_kf_m"] = [state[1] for state in states]
    state_estimate["y_rx_kf_m"] = [state[2] for state in states]
    state_estimate["z_rx_kf_m"] = [state[3] for state in states]
    state_estimate["vx_rx_kf_mps"] = [state[4] for state in states]
    state_estimate["vy_rx_kf_mps"] = [state[5] for state in states]
    state_estimate["vz_rx_kf_mps"] = [state[6] for state in states]
    state_estimate["b_rx_kf_m"] = [state[7] for state in states]

    # Convert ECEF coordinates to Geodetic coordinates
    lat,lon,alt = ecef_to_geodetic(state_estimate[["x_rx_kf_m",
                                                        "y_rx_kf_m",
                                                        "z_rx_kf_m"]].reshape(3,-1))
    state_estimate["lat_rx_kf_deg"] = lat
    state_estimate["lon_rx_kf_deg"] = lon
    state_estimate["alt_rx_kf_m"] = alt

    return state_estimate

def solve_gnss_ekf(measurements, init_dict = None,
                   params_dict = None, delta_t_decimals=-2):
    """Runs a GNSS Extended Kalman Filter across each timestep.

    Runs an Extended Kalman Filter across each timestep and adds a new
    row for the receiver's position and clock bias.

    Parameters
    ----------
    measurements : gnss_lib_py.navdata.navdata.NavData
        Instance of the NavData class
    init_dict : dict
        Initialization dict with initial states and covariances.
    params_dict : dict
        Dictionary of parameters for GNSS EKF.

    Returns
    -------
    state_estimate : gnss_lib_py.navdata.navdata.NavData
        Estimated receiver position in ECEF frame in meters and the
        estimated receiver clock bias also in meters as an instance of
        the NavData class with shape (4 x # unique timesteps) and
        the following rows: x_rx_m, y_rx_m, z_rx_m, b_rx_m.

    """

    # check that all necessary rows exist
    measurements.in_rows(["gps_millis","corr_pr_m",
                          "x_sv_m","y_sv_m","z_sv_m",
                          ])

    if init_dict is None:
        init_dict = {}

    if "state_0" not in init_dict:
        pos_0 = None
        for _, _, measurement_subset in loop_time(measurements,"gps_millis",
                                        delta_t_decimals=delta_t_decimals):
            pos_0 = solve_wls(measurement_subset)
            if pos_0 is not None:
                break

        state_0 = np.zeros((7,1))
        if pos_0 is not None:
            state_0[:3,0] = pos_0[["x_rx_wls_m","y_rx_wls_m","z_rx_wls_m"]]
            state_0[6,0] = pos_0[["b_rx_wls_m"]]

        init_dict["state_0"] = state_0

    if "sigma_0" not in init_dict:
        sigma_0 = np.eye(init_dict["state_0"].size)
        init_dict["sigma_0"] = sigma_0

    # if "Q" not in init_dict:
    #     process_noise = np.eye(init_dict["state_0"].size)
    #     init_dict["Q"] = process_noise
    if "Q" not in init_dict:
    # process_noise = np.eye(init_dict["state_0"].size)  # (old, too generic)
        init_dict["Q"] = np.diag([
            1.0,  # x (position in meters^2)
            1.0,  # y
            1.0,  # z
            0.1,  # vx (velocity in (m/s)^2)
            0.1,  # vy
            0.1,  # vz
            10.0  # b (clock bias in meters^2)
        ])


    if "R" not in init_dict:
        measurement_noise = np.eye(1) # gets overwritten
        init_dict["R"] = measurement_noise

    # initialize parameter dictionary
    if params_dict is None:
        params_dict = {}

    if "motion_type" not in params_dict:
        params_dict["motion_type"] = "constant_velocity"

    if "measure_type" not in params_dict:
        params_dict["measure_type"] = "pseudorange"

    # create initialization parameters.
    gnss_ekf = GNSSEKF(init_dict, params_dict)

    states = []

    for timestamp, delta_t, measurement_subset in loop_time(measurements,"gps_millis"):
        pos_sv_m = measurement_subset[["x_sv_m","y_sv_m","z_sv_m"]].T
        pos_sv_m = np.atleast_2d(pos_sv_m)

        corr_pr_m = measurement_subset["corr_pr_m"].reshape(-1,1)

        # remove NaN indexes
        not_nan_indexes = ~np.isnan(pos_sv_m).any(axis=1) & ~np.isnan(corr_pr_m).any(axis=1)
        pos_sv_m = pos_sv_m[not_nan_indexes]
        corr_pr_m = corr_pr_m[not_nan_indexes]

        # prediction step
        predict_dict = {"delta_t" : delta_t}
        gnss_ekf.predict(predict_dict=predict_dict)

        # rotate sv to reception frame
        pos_sv_m = gnss_ekf.rotate_sv_to_reception(pos_sv_m, corr_pr_m, gnss_ekf.state[6, 0])
        # update step
        update_dict = {"pos_sv_m" : pos_sv_m.T}
        update_dict["measurement_noise"] = np.eye(pos_sv_m.shape[0])
        gnss_ekf.update(corr_pr_m, update_dict=update_dict)

        states.append([timestamp] + np.squeeze(gnss_ekf.state).tolist())

    states = np.array(states)

    if states.size == 0:
        warnings.warn("No valid state estimate computed in solve_gnss_ekf, "\
                    + "returning None.", RuntimeWarning)
        return None

    state_estimate = NavData()
    state_estimate["gps_millis"] = states[:,0]
    state_estimate["x_rx_ekf_m"] = states[:,1]
    state_estimate["y_rx_ekf_m"] = states[:,2]
    state_estimate["z_rx_ekf_m"] = states[:,3]
    state_estimate["vx_rx_ekf_mps"] = states[:,4]
    state_estimate["vy_rx_ekf_mps"] = states[:,5]
    state_estimate["vz_rx_ekf_mps"] = states[:,6]
    state_estimate["b_rx_ekf_m"] = states[:,7]

    lat,lon,alt = ecef_to_geodetic(state_estimate[["x_rx_ekf_m",
                                                   "y_rx_ekf_m",
                                                   "z_rx_ekf_m"]].reshape(3,-1))
    state_estimate["lat_rx_ekf_deg"] = lat
    state_estimate["lon_rx_ekf_deg"] = lon
    state_estimate["alt_rx_ekf_m"] = alt

    return state_estimate

def solve_gnss_ekf_with_smoothing(measurements, init_dict=None, params_dict=None, delta_t_decimals=-2):
    """
    Runs forward GNSS EKF then backward RTS smoothing to improve state estimates.

    Parameters
    ----------
    measurements : NavData
        GNSS measurements instance with satellite positions and pseudorange.
    init_dict : dict, optional
        Initialization dictionary including 'state_0' and 'sigma_0'.
    params_dict : dict, optional
        Parameters dictionary controlling EKF behavior.
    delta_t_decimals : int, optional
        Decimal rounding for measurement timestamps grouping.

    Returns
    -------
    smoothed_estimate : NavData
        Smoothed receiver position, velocity, and clock bias estimates.
    """

    # Ensure init_dict keys required by GNSSEKF constructor exist
    if init_dict is None:
        init_dict = {}

    if "state_0" not in init_dict:
        pos_0 = None
        for _, _, measurement_subset in loop_time(measurements, "gps_millis", delta_t_decimals=delta_t_decimals):
            pos_0 = solve_wls(measurement_subset)
            if pos_0 is not None:
                break

        state_0 = np.zeros((7, 1))
        if pos_0 is not None:
            state_0[:3, 0] = pos_0[["x_rx_wls_m", "y_rx_wls_m", "z_rx_wls_m"]]
            state_0[6, 0] = pos_0[["b_rx_wls_m"]]

        init_dict["state_0"] = state_0

    if "sigma_0" not in init_dict:
        sigma_0 = np.eye(init_dict["state_0"].size)
        init_dict["sigma_0"] = sigma_0
        
    # Set default process noise covariance 'Q' if missing
    # if "Q" not in init_dict:
    #     state_dim = init_dict["state_0"].size
    #     init_dict["Q"] = np.eye(state_dim) * 1e-3  # Tune this according to system
    if "Q" not in init_dict:
    # 7 states: [x, y, z, vx, vy, vz, b]
    # Larger noise for position and velocity reduces over-smoothing
        init_dict["Q"] = np.diag([
            1e-1, 1e-1, 1e-1,  # Position noise (m^2)
            1e-2, 1e-2, 1e-2,  # Velocity noise ((m/s)^2)
            1e-3               # Clock bias noise (m^2)
        ])


    # Set default measurement noise covariance 'R' if missing
    if "R" not in init_dict:
        # Assuming measurement noise dimension = 1 (adjust if multiple measurements)
        init_dict["R"] = np.eye(1)

    # Initialize parameters dictionary if None
    if params_dict is None:
        params_dict = {}

    # Instantiate EKF
    gnss_ekf = GNSSEKF(init_dict, params_dict)

    states = []
    covariances = []
    predicted_states = []
    predicted_covariances = []
    timestamps = []

    # Forward EKF Filtering Loop
    for timestamp, delta_t, measurement_subset in loop_time(measurements, "gps_millis"):
        pos_sv_m = measurement_subset[["x_sv_m", "y_sv_m", "z_sv_m"]].T
        pos_sv_m = np.atleast_2d(pos_sv_m)
        corr_pr_m = measurement_subset["corr_pr_m"].reshape(-1, 1)

        not_nan_idxs = ~np.isnan(pos_sv_m).any(axis=1) & ~np.isnan(corr_pr_m).any(axis=1)
        pos_sv_m = pos_sv_m[not_nan_idxs]
        corr_pr_m = corr_pr_m[not_nan_idxs]

        # rotate sv to reception frame
        pos_sv_m = gnss_ekf.rotate_sv_to_reception(pos_sv_m, corr_pr_m, gnss_ekf.state[6, 0])
        
        # Prediction step
        predict_dict = {"delta_t": delta_t}
        gnss_ekf.predict(predict_dict=predict_dict)

        # Store predicted state and covariance for smoothing step
        predicted_states.append(gnss_ekf.state.copy())
        predicted_covariances.append(gnss_ekf.sigma.copy())

        # Update step
        update_dict = {
            "pos_sv_m": pos_sv_m.T,
            "measurement_noise": np.eye(pos_sv_m.shape[0])
        }
        gnss_ekf.update(corr_pr_m, update_dict=update_dict)

        # Store filtered state and covariance
        states.append(gnss_ekf.state.copy())
        covariances.append(gnss_ekf.sigma.copy())
        timestamps.append(timestamp)

    # Convert lists to numpy arrays for vectorized operations
    states = np.array(states)
    covariances = np.array(covariances)
    predicted_states = np.array(predicted_states)
    predicted_covariances = np.array(predicted_covariances)
    timestamps = np.array(timestamps)

    # Initialize smoothed arrays with same shape
    smoothed_states = np.zeros_like(states)
    smoothed_covariances = np.zeros_like(covariances)

    # RTS backward smoothing initialization (last step equal to filtered estimate)
    smoothed_states[-1] = states[-1]
    smoothed_covariances[-1] = covariances[-1]

    # Backward RTS Smoother Loop
    for t in reversed(range(len(states) - 1)):
        delta_t = timestamps[t + 1] - timestamps[t]

        # Get state transition matrix at time t
        A = gnss_ekf.linearize_dynamics({"delta_t": delta_t})

        P_t = covariances[t]
        P_tp1_pred = predicted_covariances[t + 1]

        # Compute smoother gain
        G = P_t @ A.T @ np.linalg.inv(P_tp1_pred)

        # Update smoothed state
        smoothed_states[t] = states[t] + G @ (smoothed_states[t + 1] - predicted_states[t + 1])

        # Update smoothed covariance
        smoothed_covariances[t] = P_t + G @ (smoothed_covariances[t + 1] - P_tp1_pred) @ G.T

    # Build NavData structure for smoothed output
    smoothed_navdata = NavData()
    smoothed_navdata["gps_millis"] = timestamps

    # Assign smoothed states based on GNSSEKF state vector: [x,y,z,vx,vy,vz,b]
    smoothed_navdata["x_rx_ekf_smooth_m"] = smoothed_states[:, 0]
    smoothed_navdata["y_rx_ekf_smooth_m"] = smoothed_states[:, 1]
    smoothed_navdata["z_rx_ekf_smooth_m"] = smoothed_states[:, 2]
    smoothed_navdata["vx_rx_ekf_smooth_mps"] = smoothed_states[:, 3]
    smoothed_navdata["vy_rx_ekf_smooth_mps"] = smoothed_states[:, 4]
    smoothed_navdata["vz_rx_ekf_smooth_mps"] = smoothed_states[:, 5]
    smoothed_navdata["b_rx_ekf_smooth_m"] = smoothed_states[:, 6]

    # Convert ECEF to Geodetic coordinates for smoothed states
    lat, lon, alt = ecef_to_geodetic(smoothed_navdata[[
    "x_rx_ekf_smooth_m", "y_rx_ekf_smooth_m", "z_rx_ekf_smooth_m"]].reshape(3, -1))
    smoothed_navdata["lat_rx_ekf_smooth_deg"] = lat
    smoothed_navdata["lon_rx_ekf_smooth_deg"] = lon
    smoothed_navdata["alt_rx_ekf_smooth_m"] = alt

    return smoothed_navdata

def solve_gnss_factor_graph(measurements, init_dict=None, params_dict=None, delta_t_decimals=-2):
    """Runs a GNSS positioning calculation using a factor graph.

    This function outlines the general steps involved in using a factor graph
    for GNSS positioning. It currently provides a conceptual implementation
    and may require external libraries like `gtsam` or `pgmpy` for a complete
    and functional implementation.

    The factor graph approach represents the GNSS positioning problem as a
    probabilistic graphical model. Variables in the graph represent the
    unknown receiver state (position, velocity, clock bias), and factors
    represent the measurements (pseudoranges) and the motion model. Inference
    in the factor graph (e.g., using optimization techniques) allows us to
    find the most likely receiver state given the measurements and the model.

    Parameters
    ----------
    measurements : gnss_lib_py.navdata.navdata.NavData
        Instance of the NavData class containing GNSS measurements.
        Requires 'gps_millis', 'corr_pr_m', 'x_sv_m', 'y_sv_m', 'z_sv_m'.
    init_dict : dict, optional
        Initialization dictionary. May include an initial estimate for the
        receiver state ('state_0') and the initial uncertainty
        ('sigma_0' or information matrix).
    params_dict : dict, optional
        Dictionary of parameters for the factor graph. This might include
        noise models for measurements and motion.
    delta_t_decimals : int, optional
        Decimal rounding for measurement timestamps grouping, used for
        identifying discrete time steps.

    Returns
    -------
    state_estimate : gnss_lib_py.navdata.navdata.NavData or None
        Estimated receiver position and clock bias as a NavData instance,
        or None if no valid estimate could be computed.
        The NavData will contain rows: 'gps_millis', 'x_rx_fg_m', 'y_rx_fg_m',
        'z_rx_fg_m', 'b_rx_fg_m'.
    """
    # Check for necessary measurement data
    if not measurements.in_rows(["gps_millis", "corr_pr_m", "x_sv_m", "y_sv_m", "z_sv_m"]):
        warnings.warn("Missing required measurement data for factor graph.", RuntimeWarning)
        return None

    # Initialize dictionaries if they are None
    if init_dict is None:
        init_dict = {}
    if params_dict is None:
        params_dict = {}

    # --- Step 1: Define the Variables in the Factor Graph ---
    # For GNSS positioning, the primary variables at each time step are:
    # - Receiver position (x, y, z) in ECEF frame
    # - Receiver clock bias (b)

    # We might also include velocity if we are modeling receiver motion.

    # --- Step 2: Define the Factors ---
    # Factors represent the probabilistic constraints or relationships between
    # the variables. In GNSS positioning, the main factors are:

    # - Prior Factors: Represent our initial beliefs about the receiver state.
    #   These can come from an initial guess (e.g., from a previous time step
    #   or a coarse WLS solution). The `init_dict` might provide this.

    # - Measurement Factors: Represent the information obtained from the GNSS
    #   pseudorange measurements. For each pseudorange measurement to a
    #   satellite, a factor will connect the receiver position and clock bias
    #   variables with the known satellite position. The factor will incorporate
    #   the measurement noise, which might be specified in `params_dict`.
    #   The measurement model for pseudorange is:
    #   pseudorange = distance(receiver_pos, satellite_pos) + clock_bias

    # - Motion Model Factors (if applicable): If we are estimating velocity or
    #   expect the receiver to be moving, factors can be added to model the
    #   expected motion of the receiver between time steps. This would typically
    #   connect the state variables at consecutive time steps (e.g., position
    #   and velocity). The `params_dict` might specify the type of motion model
    #   (e.g., constant velocity) and its associated noise.

    # --- Step 3: Construct the Factor Graph ---
    # We need to iterate through the measurements, likely grouped by time step,
    # and for each time step, create the corresponding variables and factors
    # and add them to the factor graph.

    states = []

    for timestamp, _, measurement_subset in loop_time(measurements, "gps_millis",
                                                     delta_t_decimals=delta_t_decimals):
        # Extract satellite positions and pseudorange measurements for this time step
        pos_sv_m = measurement_subset[["x_sv_m", "y_sv_m", "z_sv_m"]].to_numpy()
        corr_pr_m = measurement_subset["corr_pr_m"].to_numpy()

        # Remove NaN values
        not_nan_indexes = ~np.isnan(pos_sv_m).any(axis=1) & ~np.isnan(corr_pr_m)
        pos_sv_m = pos_sv_m[not_nan_indexes]
        corr_pr_m = corr_pr_m[not_nan_indexes]

        if pos_sv_m.shape[0] < 4:
            warnings.warn(f"Insufficient satellites ({pos_sv_m.shape[0]}) at timestamp {timestamp} for factor graph solution.",
                          RuntimeWarning)
            continue

        # At each time step, we need to:
        # 1. Define the receiver position and clock bias variables.
        # 2. If it's the first time step, add a prior factor based on `init_dict`.
        # 3. For each satellite measurement, add a measurement factor connecting
        #    the receiver variables with the satellite position.
        # 4. If it's not the first time step, and we have a motion model, add
        #    a factor connecting the current state with the previous state.

        # --- Placeholder for Factor Graph Implementation ---
        # This is where you would use a factor graph library (e.g., gtsam) to
        # define the factors and variables.

        # For now, we will just use a Weighted Least Squares (WLS) solution as a
        # simplified approach, which is conceptually related to finding the
        # minimum of a cost function, similar to factor graph optimization.
        pos_wls = solve_wls(measurement_subset)
        if pos_wls is not None:
            states.append([timestamp,
                           pos_wls["x_rx_wls_m"],
                           pos_wls["y_rx_wls_m"],
                           pos_wls["z_rx_wls_m"],
                           pos_wls["b_rx_wls_m"]])
        else:
            warnings.warn(f"WLS solve failed at timestamp {timestamp}", RuntimeWarning)

    if not states:
        warnings.warn("No valid state estimate computed using factor graph.", RuntimeWarning)
        return None

    state_estimate = NavData()
    state_estimate["gps_millis"] = [state[0] for state in states]
    state_estimate["x_rx_fg_m"] = [state[1] for state in states]
    state_estimate["y_rx_fg_m"] = [state[2] for state in states]
    state_estimate["z_rx_fg_m"] = [state[3] for state in states]
    state_estimate["b_rx_fg_m"] = [state[4] for state in states]

    lat, lon, alt = ecef_to_geodetic(state_estimate[["x_rx_fg_m",
                                                        "y_rx_fg_m",
                                                        "z_rx_fg_m"]].reshape(3, -1))
    state_estimate["lat_rx_fg_deg"] = lat
    state_estimate["lon_rx_fg_deg"] = lon
    state_estimate["alt_rx_fg_m"] = alt

    return state_estimate

class GNSSEKF(BaseExtendedKalmanFilter):
    """GNSS-only EKF implementation.

    States: 3D position, 3D velocity and clock bias (in m).
    The state vector is :math:`\\bar{x} = [x, y, z, v_x, v_y, v_y, b]^T`

    Attributes
    ----------
    params_dict : dict
        Dictionary of parameters that may include the following.
    delta_t : float
        Time between prediction instances
    motion_type : string
        Type of motion (``stationary`` or ``constant_velocity``)
    measure_type : string
        NavData types (pseudorange)
    """
    def __init__(self, init_dict, params_dict):
        super().__init__(init_dict, params_dict)

        self.delta_t = params_dict.get('dt',1.0)
        self.motion_type = params_dict.get('motion_type','stationary')
        self.measure_type = params_dict.get('measure_type','pseudorange')

    def dyn_model(self, u, predict_dict=None):
        """Nonlinear dynamics

        Parameters
        ----------
        u : np.ndarray
            Control signal, not used for propagation
        predict_dict : dict
            Additional prediction parameters, including ``delta_t``
            updates.

        Returns
        -------
        new_x : np.ndarray
            Propagated state
        """
        if predict_dict is None: #pragma: no cover
            predict_dict = {}

        A = self.linearize_dynamics(predict_dict)
        new_x = A @ self.state
        return new_x

    def measure_model(self, update_dict):
        """Measurement model

        Pseudorange model adds true range and clock bias estimate:
        :math:`\\rho = \\sqrt{(x-x_{sv})^2 + (y-y_{sv})^2 + (z-z_{sv})^2} + b`.
        See [1]_ for more details and models.

        ``pos_sv_m`` must be a key in update_dict and must be an array
        of shape [3 x N] with rows of x_sv_m, y_sv_m, and z_sv_m in that
        order.

        Parameters
        ----------
        update_dict : dict
            Update dictionary containing satellite positions with key
            ``pos_sv_m``.

        Returns
        -------
        z : np.ndarray
            Expected measurement, depending on type (pseudorange)
        References
        ----------
        .. [1] Morton, Y. Jade, Frank van Diggelen, James J. Spilker Jr,
            Bradford W. Parkinson, Sherman Lo, and Grace Gao, eds.
            Position, navigation, and timing technologies in the 21st century:
            integrated satellite navigation, sensor systems, and civil
            applications. John Wiley & Sons, 2021.
        """
        if self.measure_type=='pseudorange':
            pos_sv_m = update_dict['pos_sv_m']
            pseudo = np.sqrt((self.state[0] - pos_sv_m[0, :])**2
                           + (self.state[1] - pos_sv_m[1, :])**2
                           + (self.state[2] - pos_sv_m[2, :])**2) \
                           + self.state[6]
            z = np.reshape(pseudo, [-1, 1])
        else: #pragma: no cover
            raise NotImplementedError
        return z

    def linearize_dynamics(self, predict_dict=None):
        """Linearization of dynamics model

        Parameters
        ----------
        predict_dict : dict
            Additional predict parameters, not used in current implementation

        Returns
        -------
        A : np.ndarray
            Linear dynamics model depending on motion_type
        predict_dict : dict
            Dictionary of prediction parameters.
        """

        if predict_dict is None: # pragma: no cover
            predict_dict = {}

        # uses delta_t from predict_dict if exists, otherwise delta_t
        # from the class initialization.
        delta_t = predict_dict.get('delta_t', self.delta_t)

        if self.motion_type == 'stationary':
            A = np.eye(7)
        elif self.motion_type == 'constant_velocity':
            A = np.eye(7)
            A[:3, -4:-1] = delta_t*np.eye(3)
        else: # pragma: no cover
            raise NotImplementedError
        return A

    def linearize_measurements(self, update_dict):
        """Linearization of measurement model

        Parameters
        ----------
        update_dict : dict
            Update dictionary containing satellite positions with key
            ``pos_sv_m``.

        Returns
        -------
        H : np.ndarray
            Jacobian of measurement model, of dimension
            #measurements x #states
        """
        if self.measure_type == 'pseudorange':
            pos_sv_m = update_dict['pos_sv_m']
            m = np.shape(pos_sv_m)[1]
            H = np.zeros([m, self.state_dim])
            pseudo_expect = self.measure_model(update_dict)
            rx_pos = np.reshape(self.state[:3], [-1, 1])
            H[:, :3] = (rx_pos - pos_sv_m).T/pseudo_expect
            H[:, 6] = 1
        else: # pragma: no cover
            raise NotImplementedError
        return H

    def rotate_sv_to_reception(self, pos_sv_m, corr_pr_m, clock_bias):
        """
        Rotate satellite ECEF positions from transmission time to reception time
        accounting for Earth's rotation during signal travel.
        
        Parameters:
        -----------
        pos_sv_m : np.ndarray, shape (num_sats, 3)
            Satellite positions at transmission time (ECEF).
        corr_pr_m : np.ndarray, shape (num_sats, 1)
            Corrected pseudoranges (meters).
        clock_bias : float
            Receiver clock bias (meters).
        
        Returns:
        --------
        pos_sv_m_corr : np.ndarray, shape (num_sats, 3)
            Satellite positions rotated to reception time frame.
        """
        delta_t = (corr_pr_m.reshape(-1) - clock_bias) / consts.C
        dtheta = consts.OMEGA_E_DOT * delta_t
        x_rot = np.cos(dtheta)*pos_sv_m[:, 0] + np.sin(dtheta)*pos_sv_m[:, 1]
        y_rot = -np.sin(dtheta)*pos_sv_m[:, 0] + np.cos(dtheta)*pos_sv_m[:, 1]
        pos_sv_m[:, 0] = x_rot
        pos_sv_m[:, 1] = y_rot
        return pos_sv_m

"""Test for GNSS filtering algorithms.

"""
# This file demonstrates the use of pytest for unit testing. Its inputs are
# defined in conftest.py, while the outputs are defined in test_gnss_filters.py 
# that can be found in the same directory

__authors__ = "Ashwin Kanhere, Derek Knowles"
__date__ = "25 January 2022"

import os

import pytest
import numpy as np
from numpy.random import default_rng

from gnss_lib_py.navdata.navdata import NavData
from gnss_lib_py.navdata.operations import loop_time
from gnss_lib_py.parsers.google_decimeter import AndroidDerived2021
from gnss_lib_py.algorithms.gnss_filters import GNSSEKF, solve_gnss_ekf

@pytest.fixture(name='init_dict')
def gnss_init_params():
    """Testing parameters for GNSS-only EKF

    Returns
    -------
    init_dict : dict
        Dictionary with KF initialization parameters
    """
    state_dim = 7
    state_0 = np.zeros([state_dim, 1])
    state_0[0] = -2700628.97971166
    state_0[1] = -4292443.61165747
    state_0[2] =  3855152.80233124
    Q = 5*np.eye(state_dim)
    R = 5*np.eye(7) # Test has 7 measurements
    init_dict = {
                'state_0': state_0,
                'sigma_0': 5*np.eye(state_dim),
                'Q': Q,
                'R': R}
    return init_dict


@pytest.fixture(name="params_dict")
def gnss_run_params():
    """Run time parameters for GNSS-only EKF test

    Returns
    -------
    params_dict : dict
        Dicitonary with satellite positions, delta_t and measure type
    """
    pos_sv_m = np.array([
        [16033696.6255441, -19379870.1543683, -8529912.10997747],
        [-2904524.74773657, -26399200.3920454, -253278.373568479],
        [-3692615.01847773, -15682408.1669833, 21115038.8039792],
        [21478993.5690158, -2493694.84909697, 15422467.9135949],
        [-13910706.2648025, -17206261.2592405, 14692005.3082363],
        [13057746.5436005, -11000091.2964189, 20344813.6378203],
        [8021833.71066202, -20554715.8959834, 14784181.8724606]])
    pos_sv_m = pos_sv_m.T
    delta_t = 0.1
    params_dict = {'pos_sv_m': pos_sv_m,
                'delta_t': delta_t,
                'measure_type': 'pseudorange'}
    return params_dict

@pytest.mark.parametrize('motion_type', ['stationary',
                                        'constant_velocity'])
def test_stationary_filter(init_dict, params_dict, motion_type):
    """Test if solution of EKF with small measurement noise is submeter
    close to truth

    Parameters
    ----------
    init_dict : dict
        Dictionary of initialization parameters
    params_dict : dict
        Dictionary of run-time parameters for GNSS-only EKF
    motion_type : string
        Stationary or constant velocity
    """
    # Run stationary filter for 10 timesteps and verify that obtained position is near original position
    run_rng = default_rng()
    params_dict['motion_type'] = motion_type
    update_dict = {'pos_sv_m': params_dict['pos_sv_m']}
    gnss_ekf = GNSSEKF(init_dict, params_dict)
    state_dim = gnss_ekf.state_dim
    t_total = 2
    x = np.reshape(init_dict['state_0'][:3], [-1, 1])
    true_range = np.linalg.norm(params_dict['pos_sv_m'] - x, axis=0)
    true_range = np.reshape(true_range, [-1, 1])
    for _ in range(int(t_total/params_dict['delta_t'])):
        gnss_ekf.predict(np.zeros([state_dim, 1]))
        # meas_noise = run_rng.multivariate_normal(np.zeros(state_dim), init_dict['R'])
        meas_noise = np.zeros(7)
        z = true_range + np.reshape(meas_noise, [-1, 1])
        gnss_ekf.update(z, update_dict)
    pos_ekf = gnss_ekf.state[:3]
    np.testing.assert_allclose(pos_ekf, init_dict['state_0'][:3], atol=0.1)

def test_solve_gnss_ekf(derived_2021):
    """Test that solving for GNSS EKF doesn't fail

    Parameters
    ----------
    derived_2021 : AndroidDerived2021
        Instance of AndroidDerived2021 for testing.

    """
    state_estimate = solve_gnss_ekf(derived_2021)

    # result should be a NavData Class instance
    assert isinstance(state_estimate,type(NavData()))

    # should have four rows
    assert len(state_estimate.rows) == 11

    # should have the following contents
    assert "gps_millis" in state_estimate.rows
    assert "x_rx_ekf_m" in state_estimate.rows
    assert "y_rx_ekf_m" in state_estimate.rows
    assert "z_rx_ekf_m" in state_estimate.rows
    assert "vx_rx_ekf_mps" in state_estimate.rows
    assert "vy_rx_ekf_mps" in state_estimate.rows
    assert "vz_rx_ekf_mps" in state_estimate.rows
    assert "b_rx_ekf_m" in state_estimate.rows
    assert "lat_rx_ekf_deg" in state_estimate.rows
    assert "lon_rx_ekf_deg" in state_estimate.rows
    assert "alt_rx_ekf_m" in state_estimate.rows

    # should have the same length as the number of unique timesteps
    assert len(state_estimate) == sum(1 for _ in loop_time(derived_2021,"gps_millis"))

    # test what happens when rows down't exist
    for row_index in ["gps_millis","x_sv_m","y_sv_m","z_sv_m","corr_pr_m"]:
        derived_no_row = derived_2021.remove(rows=row_index)
        with pytest.raises(KeyError) as excinfo:
            solve_gnss_ekf(derived_no_row)
        assert row_index in str(excinfo.value)


def test_solve_gnss_ekf_fails(derived_2021):
    """Test expected fails for the GNSS EKF.

    Parameters
    ----------
    derived_2021 : AndroidDerived2021
        Instance of AndroidDerived2021 for testing

    """

    navdata = derived_2021.remove(cols=list(range(len(derived_2021))))

    with pytest.warns(RuntimeWarning) as warns:
        solve_gnss_ekf(navdata)

    # verify RuntimeWarning
    assert len(warns) == 1
    warn = warns[0]
    assert issubclass(warn.category, RuntimeWarning)
    assert "No valid state" in str(warn.message)

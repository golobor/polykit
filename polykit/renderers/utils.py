import numpy as np

from scipy.interpolate import InterpolatedUnivariateSpline


def interpolate_chain(positions, extra_bonds=[], target_N=90000, fine_grain=10):
    """
    Converts an individual polymer of any length to a smoothed chain with fixed distance between neighboring monomers.
    Proceeds by cubic spline interpolation as follows.

    1. Interpolate the data using cubic spline
    2. Evaluate cubic spline at target_N*fine_grain values
    3. Rescale the evaluated spline such that total distance is target_N
    4. Select target_N points along the path with distance between neighboring points _along the chain_ equal to 1.
    5. Select appropriate bonds along the interpolated path to mimic the original chain topology

    Parameters
    ----------
		positions : Nx3 float array
			Input xyz coordinates
		extra_bonds : Mx2 int array
			Non-contiguous extra bonds to be interpolated (optional)
		target_N : int
			Length of output polymer.
			It is not advised to make it many times less than N

    Returns
    -------
		(about target_N) x 3 float array of interpolated positions
		(about target_N-1 + M) x 2 int array of interpolated bonds
    """

    N = len(positions)
    target_data_size = target_N*fine_grain

    eval_range = np.arange(N)
    target_range = np.arange(0, N-1, N/float(target_data_size))

    splined = np.zeros((len(target_range), 3), float)
    
    for i in range(3):
        spline = InterpolatedUnivariateSpline(eval_range, positions[:, i], k=3)
        splined[:, i] = spline(target_range)

    pos = np.arange(1, target_N)
    dists = np.linalg.norm(np.diff(splined, axis=0), axis=1)
    
    cum_dists = np.cumsum(dists)

    splined /= (cum_dists[-1]  / N)
    cum_dists /= (cum_dists[-1] / target_N)
    
    searched_pos = np.searchsorted(cum_dists, pos)
    searched_bonds = np.searchsorted(target_range[searched_pos], extra_bonds)

    v1 = cum_dists[searched_pos]
    v2 = cum_dists[searched_pos-1]
        
    p1 = (v1-np.floor(v1)) / (v1-v2)
    p2 = 1-p1

    interp_pos = p2[:, None]*splined[searched_pos] + p1[:, None]*splined[searched_pos-1]
    interp_bonds = np.asarray(list(zip(pos[:-1]-1, pos[:-1])) + searched_bonds.tolist())
    
    return interp_pos, interp_bonds

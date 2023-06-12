import numpy as np
import pathlib

import gsd.hoomd

DEFAULT_TRAJ_PATH = "traj.gsd"

def path_to_traj(path):
    if type(path) == gsd.hoomd.HOOMDTrajectory:
        return path
    elif pathlib.Path(path).is_dir():
        return gsd.hoomd.open(name=pathlib.Path(path) / DEFAULT_TRAJ_PATH)
    else:
        return gsd.hoomd.open(name=pathlib.Path(path))


def unwrap_pos(snapshot, get_pos_f=None, max_delta=2):
    """Unwrap polymer coordinates wrapped into periodic boundary conditions

    Args:
        snapshot (_type_): HOOMD snapshot
        get_pos_f (_type_, optional): function to extract particle coordinates from a snapshot. Defaults to None.
        max_delta (int, optional): maximum separation between particles of a chain that is considered for unwrapping. 
        Useful for the case where a multi-particle segment of a chain is stretched across the periodic boundary.
        Defaults to 2.

    Returns:
        _type_: unwrapped coordinates
    """
    if get_pos_f is None:
        get_pos_f = lambda s: s.particles.position
    d = np.copy(get_pos_f(snapshot))

    for delta in range(1, max_delta+1):
        for ax in range(3):
            L = snapshot.configuration.box[ax]

            bond_projection = d[delta:, ax] - d[:-delta, ax]
            img_shifts = (
                np.round(np.abs(bond_projection) / L)
                * 
                np.sign(bond_projection)
            )

            d[delta:,ax] -= np.cumsum(img_shifts) * L

    return d


def _get_last_frame_idx(traj):
    traj = path_to_traj(traj)


    if len(traj) == 0:
        return None

    return len(traj) - 1

def get_abs_frame_idx(traj, idx):
    traj = path_to_traj(traj)

    if idx < 0:
        idx = _get_last_frame_idx(traj) + 1 + idx

    return idx

def fetch_snaphot(
    traj, 
    frame_idx):
    
    traj = path_to_traj(traj)
    
    return traj[get_abs_frame_idx(traj, frame_idx)]


def fetch_frame(
    traj, 
    frame_idx, 
    unwrap=True):
    
    snapshot = fetch_snaphot(traj, frame_idx)
    
    if unwrap:
        d = unwrap_pos(snapshot)
    else:
        d = np.copy(snapshot.particles.position)
        
    return d

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


def _find_chains_sequential(bonds):
    bonds = np.asarray(bonds)
    bonds = bonds[np.argsort(bonds[:,0])]
    bond_starts = bonds[:,0]
    bond_ends = bonds[:,1]

    chain_starts = bond_starts[np.r_[True, (np.diff(bond_starts) != 1)]]
    chain_ends = bond_ends[np.r_[(np.diff(bond_starts) != 1), True]] + 1
    return np.vstack([chain_starts, chain_ends]).T
    

def find_chains_sequential(snap, bond_types=['polymer']):        
    bond_typeids = [i for i, btype in enumerate(snap.bonds.types) if btype in bond_types]
    chain_bonds = snap.bonds.group[np.isin(snap.bonds.typeid, bond_typeids)]

    if ((chain_bonds[:,1] - chain_bonds[:,0]) != 1).sum() != 0:
        raise ValueError('Some input bonds connect non-sequencial particles.')

    chain_spans = _find_chains_sequential(chain_bonds)
    return chain_spans


def _unwrap_chains(d_unwrapped, chains, box):
    d_out = np.copy(d_unwrapped)
    box = np.asarray(box)
    chains = np.asarray(chains)

    for lo, hi in chains:
        d_chain = d_out[lo:hi]
        chain_com = d_chain.mean(axis=0)
        #img_shift_chain = chain_com // box
        img_shift_chain = (
            np.round(np.abs(chain_com) / box)
            * 
            np.sign(chain_com))
        
        chain_shift = - img_shift_chain * box
        
        d_out[lo:hi] += chain_shift
    return d_out


def unwrap_chains(snap, max_delta=2, bond_types=['polymer']):
    d_unwrapped = unwrap_pos(snap, max_delta=max_delta)

    chains = find_chains_sequential(snap, bond_types=bond_types)
    box = snap.configuration.box[:3]
    d_chains = _unwrap_chains(d_unwrapped, chains, box)

    return d_chains, chains
    

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

def get_frame_nsteps(traj, frame_idx):
    snapshot = fetch_snaphot(traj, frame_idx)
    return snapshot.configuration.step


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

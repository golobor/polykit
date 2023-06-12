from ..analysis import polymer_analyses
from . import fetch_hoomd

import collections
import pathlib
import shelve

SCALING_CACHE_FILENAME = "scalings.shlv"


def cached_contact_vs_dist(
    folder,
    fetch_func=fetch_hoomd.fetch_frame,
    get_abs_frame_idx=fetch_hoomd.get_abs_frame_idx,
    frame_idx=-1,
    contact_radius=1.1,
    bins_decade=10,
    bins=None,
    ring=False,
    particle_slice=(0,None),
    random_sigma=None,
    random_reps=10,
    cache_file=SCALING_CACHE_FILENAME,
    
):
    if random_sigma is None:
        random_reps = None

    frame_idx = fetch_hoomd.get_abs_frame_idx(folder, frame_idx)

    path = pathlib.Path(folder) / cache_file
    cache_f = shelve.open(path.as_posix(), "c")

    key_dict = {}
    for k in [
        "frame_idx",
        "bins",
        "bins_decade",
        "contact_radius",
        "ring",
        "random_sigma",
        "random_reps",
        "particle_slice"
    ]:
        key_dict[k] = locals()[k]

    if isinstance(key_dict["bins"], collections.abc.Iterable):
        key_dict["bins"] = tuple(key_dict["bins"])

    # key = '_'.join([i for for kv in sorted(key_dict.items()) for i in kv])
    key = repr(tuple(sorted(key_dict.items())))

    if key in cache_f:
        return cache_f[key]

    coords = fetch_func(folder, frame_idx)
    coords = coords[particle_slice[0]:particle_slice[-1]]
    sc = polymer_analyses.gaussian_contact_vs_dist(
        coords,
        contact_vs_dist_func=polymer_analyses.contact_vs_dist,
        random_sigma=random_sigma,
        random_reps=random_reps,
        bins_decade=bins_decade,
        bins=bins,
        contact_radius=contact_radius,
        ring=ring,
    )
    cache_f[key] = sc
    cache_f.close()

    return sc


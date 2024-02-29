# Code written by: Maksim Imakaev (imakaev@mit.edu)
"""
Analyses of polymer conformations
=================================


This module presents a collection of utils to work with polymer conformations.


Tools for calculating contacts
------------------------------

The main function calculating contacts is: :py:func:`polychrom.polymer_analyses.calculate_contacts`
Right now it is a simple wrapper around scipy.cKDTree.

Another function :py:func:`polychrom.polymer_analyses.smart_contacts` was added recently to help build contact maps
with a large contact radius. It randomly sub-samples the monomers; by default selecting N/cutoff monomers. It then
calculates contacts from sub-sampled monomers only. It is especially helpful when the same code needs to calculate
contacts at large and small contact radii.Because of sub-sampling at large contact radius, it avoids the problem of
having way-too-many-contacts at a large contact radius. For ordinary contacts, the number of contacts scales as
contact_radius^3; however, with smart_contacts it would only scale linearly with contact radius, which leads to
significant speedsups.


Tools to calculate P(s) and R(s)
--------------------------------

We provide functions to calculate P(s), Rg^2(s) and R^2(s) for polymers. By default, they use  log-spaced bins on the
X axis, with about 10 bins per order of magnitude, but aligned such that the last bins ends exactly at (N-1). They
output (bin, scaling) for Rg^2 and R^2, and (bin_mid, scaling) for contacts. In either case, the returned values are
ready to plot. The difference is that Rg and R^2 are evaluated at a given value of s, while contacts are aggregated
for (bins[0].. bins[1]), (bins[1]..bins[2]). Therefore, we have to return bin mids for contacts.

"""


import numpy as np
import pandas as pd

from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter1d


def calculate_contacts(data, cutoff=1.7):
    """Calculates contacts between points give the contact radius (cutoff)

    Parameters
    ----------
    data : Nx3 array
        Coordinates of points
    cutoff : float , optional
        Cutoff distance (contact radius)

    Returns
    -------
    k by 2 array of contacts. Each row corresponds to a contact.
    """
    if data.shape[1] != 3:
        raise ValueError("Incorrect polymer data shape. Must be Nx3.")

    if np.isnan(data).any():
        raise RuntimeError("Data contains NANs")

    tree = cKDTree(data)
    pairs = tree.query_pairs(cutoff, output_type="ndarray")
    return pairs


def calculate_contacts_chunked(data, cutoff=1.7, chunk_size = int(1e8)):
    """
    Calculate contacts between particles, chunked.

    Args:
        data (ndarray): Array of shape Nx3 representing the polymer data.
        cutoff (float, optional): Distance cutoff for identifying contacts. Defaults to 1.7.

    Yields:
        ndarray: Array of shape Mx2 representing the pairs of pairticles that are in contact.

    Raises:
        ValueError: If the shape of the polymer data is not Nx3.
        RuntimeError: If the polymer data contains NaN values.
    """

    if data.shape[1] != 3:
        raise ValueError("Incorrect polymer data shape. Must be Nx3.")

    if np.isnan(data).any():
        raise RuntimeError("Data contains NANs")

    tree = cKDTree(data)
    
    N_RANDOM = 10
    random_sample = tree.query_ball_point(x=data[np.random.choice(len(data), size=N_RANDOM)], r=cutoff, workers = 1)
    n_pairs_avg = np.mean([len(i) for i in random_sample])
    particles_per_chunk = int(chunk_size // n_pairs_avg)

    for i in range(0, len(d_chrom), particles_per_chunk):
        chunk = d_chrom[i:i+particles_per_chunk]
        
        neighbours = tree.query_ball_point(x=chunk, r=cutoff, workers = 1)
        pairs = np.vstack([np.repeat(np.arange(len(neighbours)), [len(i) for i in neighbours]), np.concatenate(neighbours)]).T
        pairs = pairs[pairs[:,0] != pairs[:,1]]

        yield pairs


def smart_contacts(data, cutoff=1.7, min_cutoff=2.1, percent_func=lambda x: 1 / x):
    """Calculates contacts for a polymer, give the contact radius (cutoff)
    This method takes a random fraction of the monomers that is equal to (
    1/cutoff).

    This is done to make contact finding faster, and because if cutoff radius
    is R, and monomer (i,j) are in contact, then monomers (i+a), and (j+b)
    are likely in contact if |a| + |b| <~ R  (the polymer could not run away
    by more than R in R steps)

    This method will have # of contacts grow approximately linearly with
    contact radius, not cubically, which should drastically speed up
    computations of contacts for large (5+) contact radii. This should allow
    using the same code both for small and large contact radius, without the
    need to reduce the # of conformations, subsample the data, or both at
    very large contact radii.


    Parameters
    ----------
    data : Nx3 array
        Polymer coordinates
    cutoff : float , optional
        Cutoff distance that defines contact
    min_cutoff : float, optional
        Apply the "smart" reduction of contacts only when cutoff
        is less than this value
    percent_func : callable, optional
        Function that calculates fraction of monomers to use, as a function of cutoff
        Default is 1/cutoff

    Returns
    -------
    k by 2 array of contacts. Each row corresponds to a contact.
    """
    if data.shape[1] != 3:
        raise ValueError("Incorrect polymer data shape. Must be Nx3.")

    if np.isnan(data).any():
        raise RuntimeError("Data contains NANs")

    if cutoff > min_cutoff:
        frac = percent_func(cutoff)
        inds = np.nonzero(np.random.random(len(data)) < frac)[0]

        conts = calculate_contacts(data[inds], cutoff)
        conts[:, 0] = inds[conts[:, 0]]
        conts[:, 1] = inds[conts[:, 1]]
        return conts

    else:
        return calculate_contacts(data, cutoff)


def generate_bins(end, start=1, bins_decade=10):
    lstart = np.log10(start)
    lend = np.log10(end)
    num = int(np.round(np.ceil((lend - lstart) * bins_decade)))
    bins = np.unique(
        np.round(np.logspace(lstart, lend, num=max(num, 0))).astype(np.int64)
    )
    assert bins[-1] == end
    return bins


def _bin_contacts_df(contacts, N, bins_decade=10, bins=None, ring=False):
    if ring:
        contour_dists = np.abs(contacts[:, 1] - contacts[:, 0])
        mask = contour_dists > N // 2
        contour_dists[mask] = N - contour_dists[mask]
    else:
        contour_dists = np.abs(contacts[:, 1] - contacts[:, 0])

    if bins is None and bins_decade is None:
        bins = np.arange(0, N + 1)
        contacts_per_bin = np.bincount(contour_dists, minlength=N)
    else:
        if bins is None:
            bins = generate_bins(N, bins_decade=bins_decade)
        else:
            bins = np.array(bins)

        contacts_per_bin = np.bincount(
            np.searchsorted(bins, contour_dists, side="right"), minlength=len(bins)
        )
        contacts_per_bin = contacts_per_bin[
            1 : len(bins)
        ]  # ignore contacts outside of distance binds

    if ring:
        pairs_per_bin = np.diff(N * bins)
    else:
        pairs_per_bin = np.diff(N * bins + 0.5 * bins - 0.5 * (bins ** 2))

    contact_freqs = contacts_per_bin / pairs_per_bin

    bin_mids = np.sqrt(bins[:-1] * bins[1:])

    return pd.DataFrame(
        {
            "dist": bin_mids,
            "contact_freq": contact_freqs,
            "n_particle_pairs": pairs_per_bin,
            "n_contacts": contacts_per_bin,
            "min_dist": bins[:-1],
            "max_dist": bins[1:],
        }
    )


def contact_scaling(data, bins0=None, cutoff=1.1, *, ring=False):
    """
    Returns contact probability scaling for a given polymer conformation
    Contact between monomers X and X+1 is counted as s=1


    Parameters
    ----------
    data : Nx3 array of ints/floats
        Input polymer conformation
    bins0 : list or None
        Bins to calculate scaling.
        Bins should probably be log-spaced; log-spaced bins can be quickly
        calculated using mirnylib.numtuis.logbinsnew.
        If None, bins will be calculated automatically
    cutoff : float, optional
        Cutoff to calculate scaling
    ring : bool, optional
        If True, will calculate contacts for the ring

    Returns
    -------
    (mids, contact probabilities) where "mids" contains
    geometric means of bin start/end


    """
    data = np.asarray(data)
    N = data.shape[0]
    assert data.shape[1] == 3

    if bins0 is None:
        bins0 = generate_bins(N)

    bins0 = np.array(bins0)
    bins = [(bins0[i], bins0[i + 1]) for i in range(len(bins0) - 1)]
    contacts = np.array(calculate_contacts(data, cutoff))

    contacts = contacts[:, 1] - contacts[:, 0]  # contact lengthes

    if ring:
        mask = contacts > N // 2
        contacts[mask] = N - contacts[mask]

    scontacts = np.sort(contacts)  # sorted contact lengthes
    # count of contacts
    connumbers = np.diff(np.searchsorted(scontacts, bins0, side="left"))

    if ring:
        possible = np.diff(N * bins0)
    else:
        possible = np.diff(N * bins0 + 0.5 * bins0 - 0.5 * (bins0**2))

    connumbers = connumbers / possible

    a = [np.sqrt(i[0] * (i[1] - 1)) for i in bins]
    return a, connumbers


def slope_contact_scaling(mids, cp, sigma=2):
    def smooth(x):
        return gaussian_filter1d(x, sigma)

    # P(s) has to be smoothed in logspace, and both P and s have to be smoothed.
    # It is discussed in detail here
    # https://gist.github.com/mimakaev/4becf1310ba6ee07f6b91e511c531e73

    # Values sigma=1.5-2 look reasonable for reasonable simulations

    slope = np.diff(smooth(np.log(cp))) / np.diff(smooth(np.log(mids)))

    return mids[1:], slope


def contact_vs_dist_df(
    coords,
    contact_radius=1.1,
    bins_decade=10,
    bins=None,
    ring=False,
):
    """
    Returns the P(s) statistics of a polymer, i.e. the average contact frequency
    vs countour distance. Contact frequencies are averaged across ranges (bins)
    of countour distance.

    Parameters
    ----------
    coords : Nx3 array of ints/floats
        An array of coordinates of the polymer particles.
    bins : array.
        Bins to divide the total span of possible countour distances.
        If neither `bins` or `bins_decade` are provided, distances are not binned.
    bins_decade : int.
        If provided, distance bins are generated automatically
        in the range of [1, N_PARTICLES - 1],
        such that bins edges are approximately equally spaced in
        the log space, with approx. `bins_decade` bins per decade.
        If neither `bins` or `bins_decade` are provided, distances are not binned.

    contact_radius : float
        Particles separated in 3D by less than `contact_radius` are
        considered to be in contact.
    ring : bool, optional
        If True, will calculate contacts for the ring

    Returns
    -------
    (bin_mids, contact_freqs, npairs_per_bin) where "mids" contains
    geometric means of bin start/end


    """
    coords = np.asarray(coords)
    if coords.shape[1] != 3:
        raise ValueError(
            "coords must contain an Nx3 array of particle coordinates in 3D"
        )
    N = coords.shape[0]

    contacts = np.array(
        calculate_contacts(coords, contact_radius)
    )

    assert np.sum(contacts[:, 1] < contacts[:, 0]) == 0

    cvd = _bin_contacts_df(contacts, N, bins_decade=bins_decade, bins=bins, ring=ring)

    return cvd


def gaussian_contact_vs_dist(
    coords, contact_vs_dist_func, random_sigma=3.0, random_reps=10, **kwargs
):
    """Calculates contact_vs_dist_func for a polymer with random normal shifts.

    Args:
        coords (_type_): coordinates of a polymer
        contact_vs_dist_func (_type_): function to calculate contact_vs_dist
        random_sigma (float, optional): the standard deviation of the random gaussian shift along each coordinate. Defaults to 3.0.
        random_reps (int, optional): number of random shifts to average over. Defaults to 10.

    Returns:
        _type_: a pandas dataframe with the average contact frequency at different contour distances
    """
    if random_sigma is None:
        return contact_vs_dist_func(coords, **kwargs)

    contact_freqs = 0
    for _ in range(random_reps):
        shifts = np.random.normal(scale=random_sigma, size=coords.shape)
        res = contact_vs_dist_func(coords + shifts, **kwargs)
        contact_freqs += res["contact_freq"] / random_reps

    res["contact_freq"] = contact_freqs

    return res


def Rg2_scaling(data, bins=None, ring=False):
    """Calculates average gyration radius of subchains a function of s

    Parameters
    ----------

    data: Nx3 array
    bins: subchain lengths at which to calculate Rg
    ring: treat polymer as a ring (default: False)
    """

    data = np.asarray(data, float)
    N = data.shape[0]
    assert data.shape[1] == 3

    data = np.concatenate([[[0, 0, 0]], data])

    if bins is None:
        bins = generate_bins(N)

    coms = np.cumsum(data, 0)  # cumulative sum of locations to calculate COM
    coms2 = np.cumsum(data**2, 0)  # cumulative sum of locations^2 to calculate RG

    def radius_gyration(len2):
        if ring:
            comsadd = coms[1:len2, :].copy()
            coms2add = coms2[1:len2, :].copy()
            comsadd += coms[-1, :][None, :]
            coms2add += coms2[-1, :][None, :]
            comsw = np.concatenate([coms, comsadd], axis=0)
            coms2w = np.concatenate([coms2, coms2add], axis=0)
        else:
            comsw = coms
            coms2w = coms2

        coms2d = (-coms2w[:-len2, :] + coms2w[len2:, :]) / len2
        comsd = ((comsw[:-len2, :] - comsw[len2:, :]) / len2) ** 2
        diffs = coms2d - comsd
        sums = np.sum(diffs, 1)
        return np.mean(sums)

    rads = [0.0 for _ in range(len(bins))]
    for i in range(len(bins)):
        rads[i] = radius_gyration(int(bins[i]))
    return np.array(bins), rads


def R2_scaling(data, bins=None, ring=False):
    """
    Returns end-to-end distance scaling of a given polymer conformation.
    ..warning:: This method averages end-to-end scaling over all possible
     subchains of given length

    Parameters
    ----------

    data: Nx3 array
    bins: the same as in giveCpScaling
    ring: is the polymer a ring?

    """
    data = np.asarray(data, float)
    N = data.shape[0]
    assert data.shape[1] == 3
    data = data.T

    if bins is None:
        bins = generate_bins(N)
    if ring:
        data = np.concatenate([data, data], axis=1)

    rads = [0.0 for _ in range(len(bins))]
    for i in range(len(bins)):
        length = bins[i]
        if ring:
            rads[i] = np.mean((np.sum((data[:, :N] - data[:, length : length + N]) ** 2, 0)))
        else:
            rads[i] = np.mean((np.sum((data[:, :-length] - data[:, length:]) ** 2, 0)))
    return np.array(bins), rads


def Rg2(data):
    """
    Simply calculates gyration radius of a polymer chain.
    """
    data = np.asarray(data)
    assert data.shape[1] == 3
    return np.mean((data - np.mean(data, axis=0)) ** 2) * 3


def Rg2_matrix(data):
    """
    Uses dynamic programming and vectorizing to calculate Rg for each subchain of the polymer.
    Returns a matrix for which an element [i,j] is Rg of a subchain from i to j including  i and j
    """

    data = np.asarray(data, float)
    assert data.shape[1] == 3
    N = data.shape[0]
    data = np.concatenate([[[0, 0, 0]], data])

    coms = np.cumsum(data, 0)  # cumulative sum of locations to calculate COM
    coms2 = np.cumsum(data**2, 0)  # cumulative sum of locations^2 to calculate RG

    dists = np.abs(np.arange(N)[:, None] - np.arange(N)[None, :]) + 1
    coms2d = (-coms2[:-1, None, :] + coms2[None, 1::, :]) / dists[:, :, None]
    comsd = ((coms[:-1, None, :] - coms[None, 1:, :]) / dists[:, :, None]) ** 2
    sums = np.sum(coms2d - comsd, 2)
    np.fill_diagonal(sums, 0)
    mask = np.arange(N)[:, None] > np.arange(N)[None, :]
    sums[mask] = sums.T[mask]
    return sums


def ndarray_groupby_aggregate(
    df,
    ndarray_cols,
    aggregate_cols,
    value_cols=[],
    sample_cols=[],
    preset="sum",
    ndarray_agg=lambda x: np.sum(x, axis=0),
    value_agg=lambda x: x.sum(),
):
    """
    A version of pd.groupby that is aware of numpy arrays as values of columns

    * aggregates columns ndarray_cols using ndarray_agg aggregator,
    * aggregates value_cols using value_agg aggregator,
    * takes the first element in sample_cols,
    * aggregates over aggregate_cols

    It has presets for sum, mean and nanmean.
    """

    if preset == "sum":
        ndarray_agg = lambda x: np.sum(x, axis=0)
        value_agg = lambda x: x.sum()
    elif preset == "mean":
        ndarray_agg = lambda x: np.mean(x, axis=0)
        value_agg = lambda x: x.mean()
    elif preset == "nanmean":
        ndarray_agg = lambda x: np.nanmean(x, axis=0)
        value_agg = lambda x: x.mean()

    def combine_values(in_df):
        """
        splits into ndarrays, 'normal' values, and samples;
        performs aggregation, and returns a Series
        """
        average_arrs = pd.Series(
            index=ndarray_cols,
            data=[ndarray_agg([np.asarray(j) for j in in_df[i].values]) for i in ndarray_cols],
        )
        average_values = value_agg(in_df[value_cols])
        sample_values = in_df[sample_cols].iloc[0]
        agg_series = pd.concat([average_arrs, average_values, sample_values])
        return agg_series

    return df.groupby(aggregate_cols).apply(combine_values)


def streaming_ndarray_agg(
    in_stream,
    ndarray_cols,
    aggregate_cols,
    value_cols=[],
    sample_cols=[],
    chunksize=30000,
    add_count_col=False,
    divide_by_count=False,
):
    """
    Takes in_stream of dataframes

    Applies ndarray-aware groupby-sum or groupby-mean: treats ndarray_cols as numpy arrays,
    value_cols as normal values, for sample_cols takes the first element.

    Does groupby over aggregate_cols

    if add_count_col is True, adds column "count", if it's a string - adds column with add_count_col name

    if divide_by_counts is True, divides result by column "count".
    If it's a string, divides by divide_by_count column

    This function can be used for automatically aggregating P(s), R(s) etc.
    for a set of conformations that is so large that all P(s) won't fit in RAM,
    and when averaging needs to be done over so many parameters
    that for-loops are not an issue. Examples may include simulations in which sweep
    over many parameters has been performed.

    """
    value_cols_orig = [i for i in value_cols]
    ndarray_cols, value_cols = list(ndarray_cols), list(value_cols)
    aggregate_cols, sample_cols = list(aggregate_cols), list(sample_cols)
    if add_count_col is not False:
        if add_count_col is True:
            add_count_col = "count"
        value_cols.append(add_count_col)

    def agg_one(dfs, aggregate):
        """takes a list of DataFrames and old aggregate
        performs groupby and aggregation  and returns new aggregate"""
        if add_count_col is not False:
            for i in dfs:
                i[add_count_col] = 1

        df = pd.concat(dfs + ([aggregate] if aggregate is not None else []), sort=False)
        aggregate = ndarray_groupby_aggregate(
            df,
            ndarray_cols=ndarray_cols,
            aggregate_cols=aggregate_cols,
            value_cols=value_cols,
            sample_cols=sample_cols,
            preset="sum",
        )
        return aggregate.reset_index()

    aggregate = None
    cur = []
    count = 0
    for i in in_stream:
        cur.append(i)
        count += len(i)
        if count > chunksize:
            aggregate = agg_one(cur, aggregate)
            cur = []
            count = 0
    if len(cur) > 0:
        aggregate = agg_one(cur, aggregate)

    if divide_by_count is not False:
        if divide_by_count is True:
            divide_by_count = "count"
        for i in ndarray_cols + value_cols_orig:
            aggregate[i] = aggregate[i] / aggregate[divide_by_count]

    return aggregate


def kabsch_msd(P, Q):
    """
    Calculates MSD between two vectors using Kabash alcorithm
    Borrowed from https://github.com/charnley/rmsd  with some changes

    rmsd is licenced with  a 2-clause BSD licence

    Copyright (c) 2013, Jimmy Charnley Kromann <jimmy@charnley.dk> & Lars Bratholm
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    1. Redistributions of source code must retain the above copyright notice, this
       list of conditions and the following disclaimer.
    2. Redistributions in binary form must reproduce the above copyright notice,
       this list of conditions and the following disclaimer in the documentation
       and/or other materials provided with the distribution.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
    ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
    ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

    """
    P = P - np.mean(P, axis=0)
    Q = Q - np.mean(Q, axis=0)
    C = np.dot(np.transpose(P), Q)

    V, S, W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0
    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]

    # Create Rotation matrix U
    U = np.dot(V, W)
    dist = np.mean((np.dot(P, U) - Q) ** 2) * 3
    return dist
    

def calculate_cistrans(data, chains, chain_id=0, cutoff=5, pbc_box=False, box_size=None):
    """
    Analysis of the territoriality of polymer chains from simulations, using the cis/trans ratio.
    Cis signal is computed for the marked chain ('chain_id') as amount of contacts of the chain with itself
    Trans signal is the total amount of trans contacts for the marked chain with other chains from 'chains'
    (and with all the replicas for 'pbc_box'=True)

    """
    if data.shape[1] != 3:
        raise ValueError("Incorrect polymer data shape. Must be Nx3.")

    if np.isnan(data).any():
        raise RuntimeError("Data contains NANs")

    N = len(data)

    if pbc_box:
        if box_size is None:
            raise ValueError("Box size is not given")
        else:
            data_scaled = np.mod(data, box_size)

    else:
        box_size = None
        data_scaled = np.copy(data)

    if chains is None:
        chains = [[0, N]]
        chain_id = 0

    chain_start = chains[chain_id][0]
    chain_end = chains[chain_id][1]

    # all contact pairs available in the scaled data
    tree = cKDTree(data_scaled, boxsize=box_size)
    pairs = tree.query_pairs(cutoff, output_type="ndarray")

    # total number of contacts of the marked chain:
    # each contact is counted twice if both monomers belong to the marked chain and
    # only once if just one of the monomers in the pair belong to the marked chain
    all_signal = len(pairs[pairs < chain_end]) - len(pairs[pairs < chain_start])

    # contact pairs of the marked chain with itself
    tree = cKDTree(data[chain_start:chain_end], boxsize=None)
    pairs = tree.query_pairs(cutoff, output_type="ndarray")

    # doubled number of contacts of the marked chain with itself (i.e. cis signal)
    cis_signal = 2 * len(pairs)

    assert all_signal >= cis_signal

    trans_signal = all_signal - cis_signal

    return cis_signal, trans_signal

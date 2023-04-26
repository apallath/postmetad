"""
Reconstructs biases and free energy surfaces (FES) from OPES_METAD output files.
"""
import warnings
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from tqdm import tqdm


## PLUMED I/O functions
def read_plumed_colvar_file(fname: str, time_col=0, colvar_cols=[1], bias_col=2) -> pd.DataFrame:
    r"""
    Reads collective variables and OPES bias from PLUMED colvar file.

    Args:
        fname: Path to PLUMED colvar file.
        time_col: Column index to read time from (default=0).
        colvar_cols: List of column indices to read collective variables from (int, default=[1]).
        bias_col: Column index to read bias value from (default=2).

    Returns:
        cv_df: Pandas DataFrame containing colvars and bias at each time, indexed by time. Colvar
            columns are labeled by the field labels specified by the FIELDS line in the PLUMED
            output file. Bias column is labeled as 'bias'.
    """
    times = []
    data = []

    labels = []

    with open(fname) as f:
        # Read data file
        for l in f:
            lstrip = l.strip()

            if len(labels) == 0:
                lsplit = lstrip.split()
                if lsplit[0] == "#!" and lsplit[1] == "FIELDS":
                    for col in colvar_cols:
                        labels.append(lsplit[2 + col])
                    labels.append("bias")

            # Skip zero-length lines
            if len(lstrip) == 0:
                warnings.warn("Skipped blank line in %s" % fname)

            # Parse data
            elif lstrip[0] != "#" and "#" not in lstrip:
                lsplit = lstrip.split()
                # Skip incomplete lines
                if len(lsplit) <= time_col or len(lsplit) <= max(colvar_cols) or len(lsplit) <= bias_col:
                    warnings.warn(
                        "Incomplete data (%s) encountered in %s (or requested column index > actual # of columns). Line skipped."
                        % (lstrip, fname)
                    )
                else:
                    tcur = float(lsplit[time_col])
                    dcur = [float(lsplit[col]) for col in colvar_cols]
                    dcur.append(float(lsplit[bias_col]))

                    times.append(tcur)
                    data.append(dcur)

    times = np.array(times)
    data = np.array(data)

    cv_df = pd.DataFrame(data, index=times, columns=labels)

    return cv_df


def read_generic_data_file(fname: str, commentchar: str = "#", time_col=0, colvar_cols=[1]):
    """
    Reads a generic data file containing timeseries collective variables (such as data files output
    by INDUS or xvg files output by GROMACS).

    Args:
        fname: Path to PLUMED colvar file.
        time_col: Column index to read time from (default=0).
        colvar_cols: List of column indices to read collective variables from (int, default=[1]).

    Returns:
        cv_df: Pandas DataFrame containing colvars at each time, indexed by time. Columns are not
            labeled.
    """
    times = []
    data = []

    with open(fname) as f:
        # Read data file
        for l in f:
            lstrip = l.strip()

            # Skip zero-length lines
            if len(lstrip) == 0:
                warnings.warn("Skipped blank line in %s" % fname)

            # Parse data
            elif lstrip[0] != commentchar and commentchar not in lstrip:
                lsplit = lstrip.split()
                # Skip incomplete lines
                if len(lsplit) <= time_col or len(lsplit) <= max(colvar_cols):
                    warnings.warn(
                        "Incomplete data (%s) encountered in %s (or requested column index > actual # of columns). Line skipped."
                        % (lstrip, fname)
                    )
                else:
                    tcur = float(lsplit[time_col])
                    dcur = [float(lsplit[col]) for col in colvar_cols]

                    times.append(tcur)
                    data.append(dcur)

    times = np.array(times)
    data = np.array(data)

    cv_df = pd.DataFrame(data, index=times)

    return cv_df


def read_plumed_state_file(fname: str, which=-1):
    r"""
    Reads PLUMED state file. At present, this only worked for OPES_METAD_state
    states (and not for OPES_METAD_EXPLORE_state states).

    Args:
        fname: Path to PLUMED colvar file.
        which: Index of stored state to read (default=-1, i.e. read last stored state).

    Returns:
        CV_names: List of CV names.
        params: Dictionary of parameter values.
        kernels: List of kernel parameters.

    """
    state_idxs = []
    print("Reading states...")
    with open(fname) as f:
        for lidx, line in enumerate(f):
            if line.strip()[0] == "#" and line.strip().split()[1] == "FIELDS":
                state_idxs.append(lidx)

    print("Found {} stored states".format(len(state_idxs)))

    start = False
    nCVs = 0
    CV_names = []
    params = {}
    kernels = []

    with open(fname) as f:
        for lidx, line in enumerate(f):
            if lidx == state_idxs[which]:
                start = True
                print("Reading state {}".format(which))

                if line.strip().split()[1] != "FIELDS":
                    raise IOError("Error in reading state file: Expected to find FIELDS at line {}".format(lidx))
                else:
                    vals = line.strip().split()
                    nCVs = int((len(vals) - 4) / 2)
                    params["nCVs"] = nCVs
                    CV_names = vals[3:-1:2]
                    print("Found {} CVs: ".format(nCVs) + ", ".join(CV_names))
                    continue

            if start:
                if line.strip()[0] == "#":
                    vals = line.strip().split()
                    # Are we at the next state?
                    if vals[1] == "FIELDS":
                        break  # we're done, we've reached the next state

                    # Read parameters
                    if vals[2] == "action":
                        if vals[3] != "OPES_METAD_state":
                            raise ValueError("Invalid action: {}".format(vals[3]))
                    else:
                        params[vals[2]] = float(vals[3])

                else:
                    # Read kernels
                    vals = line.strip().split()
                    kernel = [float(v) for v in vals[1:]]
                    kernels.append(kernel)
        print("Done")

    return CV_names, params, kernels


## FES construction from collective variables
def fes_from_colvars(colvar_df: pd.DataFrame, method="reweighting", temp: float = None, beta: float = 1, **kwargs):
    r"""
    Constructs FES from colvars DataFrame.

    Args:
        colvar_df: DataFrame containing colvars and bias.
        method: Method to use to reconstruct FES (options=["reweighting"], default="reweighting")
        temp: Temperature, in K (default=None).
        beta: 1/kT, in units of mol/kJ, calculated from temperature if temp is not None, otherwise set to 1.
        **kwargs: Keyword arguments for FES reconstruction.

    Returns:
        bF: Dimensionless FES, $\beta F$
        edges: Bin edges.
    """
    if temp is not None:
        beta = 1 / (8.314 * temp / 1000)

    cols = colvar_df.columns.values
    cols = np.delete(cols, np.argwhere(cols == "bias"))
    colvars = colvar_df[cols].values
    bias = colvar_df["bias"].values

    print(colvars.shape)
    print(bias.shape)

    if method == "reweighting":
        return fes_from_reweighting(colvars, bias, beta, **kwargs)
    else:
        raise ValueError("Invalid FES construction method")


def fes_from_reweighting(colvars: np.ndarray, bias: np.ndarray, beta: float = 1, bins=20):
    r"""
    Constructs FES from reweighting.

    Args:
        colvars: Array of shape (N, D), consisting of D collective variables for N timesteps.
        bias: Array of shape (N,) containing OPES bias for N timesteps.
        beta: 1/kT, in units of mol/kJ.
        bins: Bins along each FES coordinate (default=20).
            The bin specification must be in one of the following forms:

                - A sequence of arrays describing the bin edges along each dimension.
                - The number of bins for each dimension (nx, ny, ... = bins).
                - The number of bins for all dimensions (nx = ny = ... = bins).

    Returns:
        bF: Dimensionless FES, $\beta F$.
        edges: Bin edges.
    """
    p, edges = np.histogramdd(colvars, weights=np.exp(beta * bias), bins=bins)
    bF = -np.log(p)
    return bF - np.min(bF), edges


## FES construction from PLUMED State.data file
def fes_from_state(params, kernels, temp: float = None, beta: float = 1, bins=20):
    r"""
    Constructs FES using params and kernels extracted from PLUMED State.data file.

    Args:
        params: Dictionary of parameter values extracted from PLUMED State.data file.
        kernels: List of kernel parameters extracted from PLUMED State.data file.
        temp: Temperature, in K (default=None).
        beta: 1/kT, in units of mol/kJ, calculated from temperature if temp is not None, otherwise set to 1.
        bins: Bins along each FES coordinate (default=20).
            The bin specification must be in one of the following forms:

                - A sequence of arrays describing the bin edges along each dimension.
                - The number of bins for each dimension (nx, ny, ... = bins).
                - The number of bins for all dimensions (nx = ny = ... = bins).

        Returns:
            bF: Dimensionless FES, $\beta F$.
            edges: Bin edges.
    """
    if temp is not None:
        beta = 1 / (8.314 * temp / 1000)

    biasfactor = params["biasfactor"]
    epsilon = params["epsilon"]
    cutoff = params["kernel_cutoff"]
    val_at_cutoff = np.exp(-0.5 * cutoff**2)
    zed = params["zed"]


## FES reduction
def integrate_out_fes(bF: np.ndarray, dims: Tuple[int]):
    r"""
    Integrate out multiple dimensions from FES.

    Args:
        bF: Dimensionless FES, $\beta F$.
        dims: int or tuple of ints defining dimensions to integrate.

    Returns:
        bF_red: Reduced FES, after integrating out dims.
    """
    bF_red = -logsumexp(-bF, axis=dims)
    return bF_red - np.min(bF_red)


## FES plotting
def plot_1D_fes(bF: np.ndarray, edges=np.ndarray, dpi=300, **kwargs):
    r"""
    Plots 1D FES $\beta F$.

    Args:
        bF: Dimensionless FES, $\beta F$.
        edges: Bin-edges.
        dpi: DPI (default=300).
    """
    fig, ax = plt.subplots(dpi=dpi)
    bins = (edges[:-1] + edges[1:]) / 2
    ax.plot(bins, bF, **kwargs)
    return fig, ax


def plot_2D_fes(
    bF: np.ndarray,
    edges_x: np.ndarray,
    edges_y: np.ndarray,
    clip_min=None,
    clip_max=None,
    levels=None,
    cmap="RdYlBu",
    dpi=300,
    **kwargs
):
    r"""
    Generates a filled contour plot of 2D FES $\beta F$.

    Args:
        bF: Dimensionless FES, $\beta F$.
        edges_x: Bin-edges along x-coordinate.
        edges_y: Bin-edges along y-coordinate.
        clip_min: Lower value to clip betaF at.
        clip_max: Upper value to clip betaF at.
        levels: Levels to plot contours at (see matplotlib contour/contourf docs for details).
        cmap: Colormap for plot.
        dpi: DPI (default=300).
    """
    fig, ax = plt.subplots(dpi=dpi)

    if clip_min is not None:
        bF = bF.clip(min=clip_min)
    if clip_max is not None:
        bF = bF.clip(max=clip_max)

    bins_x = (edges_x[:-1] + edges_x[1:]) / 2
    bins_y = (edges_y[:-1] + edges_y[1:]) / 2

    cs = ax.contourf(bins_x, bins_y, bF.T, levels=levels, cmap=cmap, **kwargs)
    ax.contour(bins_x, bins_y, bF.T, levels=levels, colors="black", alpha=0.2)
    cbar = fig.colorbar(cs)
    return fig, ax, cbar


## Bias potential from PLUMED State.data file
class OPESBias:
    """
    OPES bias potential.
    """

    def __init__(self, params, kernels, temp: float = None, beta: float = 1):
        kernels = np.array(kernels)

    def bias(self, x: np.ndarray):
        """
        Evaluates bias at a data point.

        Args:
            x: Data point to evaluate bias at.
        """
        pass

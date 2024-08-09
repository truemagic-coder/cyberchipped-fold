import matplotlib.pyplot as plt
import numpy as np
from typing import List

def plot_pae(pae: np.ndarray, dpi: int = 100) -> plt.Figure:
    plt.figure(figsize=(8, 8), dpi=dpi)
    plt.imshow(pae, cmap="bwr", vmin=0, vmax=30)
    plt.colorbar()
    plt.title("Predicted Aligned Error")
    plt.xlabel("Residue")
    plt.ylabel("Residue")
    return plt.gcf()

def plot_plddt(plddt: np.ndarray, dpi: int = 100) -> plt.Figure:
    plt.figure(figsize=(8, 3), dpi=dpi)
    plt.plot(plddt)
    plt.title("Predicted LDDT")
    plt.xlabel("Residue")
    plt.ylabel("pLDDT")
    plt.ylim(0, 100)
    return plt.gcf()

def plot_msa_v2(feature_dict: dict, dpi: int = 100) -> plt.Figure:
    msa = feature_dict['msa']
    seqid = (msa[0] == msa).mean(-1)
    seqid_sort = seqid.argsort()
    non_gaps = (msa != 21).astype(float)
    non_gaps[non_gaps == 0] = np.nan
    plt.figure(figsize=(12, 3), dpi=dpi)
    plt.title("Sequence coverage")
    plt.imshow(non_gaps[seqid_sort] * seqid[seqid_sort, None], interpolation='nearest', aspect='auto', cmap="rainbow")
    plt.ylabel("Sequences")
    plt.xlabel("Positions")
    return plt.gcf()

def plot_paes(paes: List[np.ndarray], Ls: List[int], dpi: int = 100) -> plt.Figure:
    num_models = len(paes)
    fig, axes = plt.subplots(1, num_models, figsize=(4*num_models, 4), dpi=dpi)
    if num_models == 1:
        axes = [axes]
    for n, (pae, ax) in enumerate(zip(paes, axes)):
        im = ax.imshow(pae, cmap="bwr", vmin=0, vmax=30)
        ax.set_title(f"model {n+1}")
        ax.set_xlabel("Positions")
        ax.set_ylabel("Positions")
    fig.colorbar(im, ax=axes, label="Expected position error")
    return fig

def plot_plddts(plddts: List[np.ndarray], Ls: List[int], dpi: int = 100) -> plt.Figure:
    plt.figure(figsize=(8, 5), dpi=dpi)
    for n, plddt in enumerate(plddts):
        plt.plot(plddt, label=f"model {n+1}")
    plt.title("Predicted LDDT per position")
    plt.xlabel("Positions")
    plt.ylabel("Predicted LDDT")
    plt.legend()
    return plt.gcf()
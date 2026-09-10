import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
import pickle

import Utils


def getmaxact(pats, centers, cnbin = False) :
    y_hat = []
    for pat in pats :
        if not cnbin :
            maxidx = np.argmax(pat)
        else :
            maxidx = (np.argmax(pat)//cnbin)
        y_hat.append(centers[maxidx])
    return np.array(y_hat)


def pats_to_dists(pats, nbin, cnbin) :
    ydists = []
    for pat in pats :
        ydists.append(np.sum(pat.reshape(nbin, cnbin), 1))
    return np.array(ydists)


def dodecode(pats, centers, cnbin = False) :
    # A simple decode which just uses the peak x.
    return getmaxact(pats, centers, cnbin = cnbin)


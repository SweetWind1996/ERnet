import linecache
import numpy as np


def txt_saveto_list(path, train_txt_len, namelist):
    for i in np.arange(train_txt_len):
        linedata = linecache.getline(path, i)
        namelist.append(linedata.split())
    namelist[:1] = []
    return namelist

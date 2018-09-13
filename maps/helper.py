import os

import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd

datafile = os.path.expanduser('/home/dave/projects/diploma/data/place.csv')
cols = ['id', 'code', "name"]

df = pd.read_file(datafile, usecols=cols)
df.sam
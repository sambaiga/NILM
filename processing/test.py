import data_dask
from matplotlib import pyplot as plt
import random
import timeit



path = "/home/ibcn079/data/REFIT/"
appliance_name = "Washing Machine"
WINDOW_SIZE= 2000

start = timeit.default_timer()
agg, ind =data_dask.generate_clean_data2(path, appliance=appliance_name, window_size=WINDOW_SIZE, threshold=10, proportion=[2, 1000],test=False, test_on=1)
stop = timeit.default_timer()
print( stop - start )
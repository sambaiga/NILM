import ukdale_data as data
from matplotlib import pyplot as plt
import random
import timeit



path = "/home/ibcn079/data/ukdale/"
WINDOW_SIZE= 591
Threshold =10   
appliance = 'microwave'

#aggregate, iam = data.generate_clean_data(path, appliance,buildings=[1,2])
start = timeit.default_timer()
data.generate_clean_data(path, appliance=appliance, window_size=WINDOW_SIZE, threshold=10, proportion=[2, 1000],test=False, test_on=1)
stop = timeit.default_timer()
print( stop - start )
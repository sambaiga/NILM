import os
import psutil 
import dask
import dask.dataframe as dd
import pandas as pd
from os import walk
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from dask import delayed
import dask.array as da
from dask import compute


def read_channel(filename,channel):
    
    """Method to read home channel data from .csv file into dask dataframe
        Args:
                filename: path to a specific channel_(m) from house(n)
        return:
                [dask.Dataframe]  from house(n)
    """
    channel_to_read = dd.read_csv(filename, skipinitialspace=True, usecols=['Timestamp',channel])
    dd.read_csv(filename)
    
    return channel_to_read




def generate_clean_data2(path, appliance, test=False, test_on='All'):
    
    aggregate_channels = []
    individual_channels = []
    aggregate_channels_test = []
    individual_channels_test = []
   
   
    if appliance == 'Fridge':
        houses = [1]
    elif appliance == 'Washing Machine':
        houses = [1,2]
        
    for house in houses:
        csv_filename = path + str(house) + '.csv'
        iam = read_channel(filename=csv_filename, channel=appliance)
        aggregate = read_channel(filename=csv_filename, channel="Aggregate")
        print("Reading house: {}".format(house))
        aggregate, iam = np.array(aggregate['Aggregate']), np.array(iam[appliance])
        #aggregate, iam = to_dask_array(aggregate['Aggregate']), to_dask_array(iam[appliance]) 
        
        if test:
            split = round(len(aggregate) * 0.8)
            aggregate_test = aggregate[split:]
            iam_test = iam[split:]
            if test_on == 'All':
                aggregate_channels_test.append(aggregate_test)
                individual_channels_test.append(iam_test)
                aggregate = aggregate[:split]
                iam = iam[:split]
            elif test_on == house:
                aggregate_channels_test.append(aggregate_test)
                individual_channels_test.append(iam_test)
                aggregate = aggregate[:split]
                iam = iam[:split]
                
        aggregate_channels.append(aggregate) #appending the aggregate to aggregate list and iam to iam list so that their indices match
        individual_channels.append(iam)
    
    return aggregate_channels, individual_channels


def get_activations(individual_channels, window_size, threshold=10, proportion=[1,1]):
    
    activation_proportion = proportion[0]
    non_activation_proportion = proportion[1]
    
    activations = []
    non_activations = []
    
    for channel in individual_channels: #iterating through appliance power usage in each house
        activations_for_house = [] #buffer list to fill all activations detected in iam
        non_activations_for_house = []
        non_activation_samples = 0
        print("Creating activations for house : {}".format(channel))
        for i in range(len(channel)):
            start = 0
            end = 0
            if channel[i] > threshold: #if power is above threshold power that may possibly be an activation
                if non_activation_samples > window_size:
                    non_activations_for_house.append([i - non_activation_samples, i-1])
                non_activation_samples = 0
                start = i
                while channel[i] > threshold and i < len(channel) - 1:
                    i += 1 #increasing index indicator until it reaches the value below threshold
                end = i
                activation = [start, end]
                activations_for_house.append(activation) #appending activation start and end time to buffer of activations for house
            else:
                non_activation_samples +=1
        activations.append(activations_for_house) #appending whole bufer to list of activations of specific appliance in all houses used for loading activations
        non_activations.append(non_activations_for_house)
        
    return activations, non_activations

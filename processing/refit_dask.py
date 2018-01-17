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


def convert_to_tensor(aggregate, individual, window_size):
    
    """Method for converting aggregate and individual power usage pandas.Dataframes to two torch 2D tensors of the same size so aggregate data is input and
        individual data is desired output.
        Args:
                aggregate: pandas.Dataframe containing concatenated and downsampled aggregate channels data
                individual: pandas.Dataframe containing signal from individual appliance channels
                NOTE: the lengths of aggregate and individual data must be the same.
        return aggregate, individual : Two torch Tensors with data grouped in windows of specified size
    """

    #Converting to numpy array with usage only (dropping Time column completely).
    aggregate = np.array(aggregate)
    individual = np.array(individual)

    #Creating padding so the last row of torch.Tensor can fit to window size of specifiedAppliance
    zeros_padding = window_size - len(aggregate)%window_size

    #Appending zeros padding to the end of numpy arrays of both individual and aggregate
    aggregate = np.append(aggregate, np.zeros(zeros_padding))
    individual = np.append(individual, np.zeros(zeros_padding))

    #Conversion to 1D torch.Tensor from numpy
    aggregate = torch.from_numpy(aggregate)
    individual = torch.from_numpy(individual)

    #Reshaping from 1D to 2d toch.Tensor
    aggregate = aggregate.view(-1, window_size)
    individual = individual.view(-1, window_size)

    return aggregate, individual


def to_dask_array(df):
    """This method is read dask dataframe and convert a its numeric values into an Array.
        Args:
                     df : dask dataframe
                     
        return:
                    [dask] single-chunk dask arrays
                    
        source: https://stackoverflow.com/questions/37444943/dask-array-from-dataframe            
    """
    partitions = df.to_delayed()
    shapes = [part.values.shape for part in partitions]
    dtype = partitions[0].dtype

    results = compute(dtype, *shapes)  # trigger computation to find shape
    dtype, shapes = results[0], results[1:]

    chunks = [da.from_delayed(part.values, shape, dtype) 
              for part, shape in zip(partitions, shapes)]
    return da.concatenate(chunks, axis=0)



def generate_clean_data2(path, appliance, window_size, threshold, proportion=[1,1],test=False, test_on='All'):
    
    activation_proportion = proportion[0]
    non_activation_proportion = proportion[1]
    aggregate_channels = []
    individual_channels = []
    aggregate_channels_test = []
    individual_channels_test = []
    activations = []
    non_activations = []
   
    
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
    
    for channel in individual_channels: #iterating through frigde power usage in each house
        activations_for_house = [] #buffer list to fill all activations detected in iam
        non_activations_for_house = []
        non_activation_samples = 0
        #print("Creating activations for house : {}".format(channel))
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
        
    agg, iam = [], []
    for i in range(len(aggregate_channels)):
        #iterating through aggregate data of each house
        print('Number of activations in this channel: ', len(activations[i]))
        print('Number of non-activations in this channel: ', len(non_activations[i]))
        
        agg_windows, iam_windows = create_overlap_windows(aggregate_channels[i], individual_channels[i], window_size, stride=2)
        agg.extend(agg_windows)
        iam.extend(iam_windows)
        for start, end in activations[i]:
            #then iterating through activation positions in specified house [i]
            for j in range(activation_proportion):
                #randomly generate windows #n times with one activation
                activation_size = end - start
                #randomly positioning activation in window
                start_aggregate = start - random.randint(0, window_size - activation_size)
                
                if start_aggregate + window_size < len(aggregate_channels[i]):
                    agg_buff, iam_buff = aggregate_channels[i][start_aggregate: start_aggregate + window_size], individual_channels[i][start_aggregate: start_aggregate + window_size]
                    agg_buff, iam_buff = np.copy(agg_buff), np.copy(iam_buff)
                    agg.append(agg_buff)
                    iam.append(iam_buff)
               

    print("Finished safely")
    return  agg, iam
        
        
        
def create_windows(agg, iam, window_size):
    
    #Creating padding so the last row of torch.Tensor can fit to window size of specifiedAppliance
    zeros_padding = window_size - len(agg)%window_size

    #Appending zeros padding to the end of numpy arrays of both individual and aggregate
    agg = np.append(agg, np.zeros(zeros_padding))
    iam = np.append(iam, np.zeros(zeros_padding))
    agg = np.reshape(agg, (-1, window_size))
    iam = np.reshape(iam, (-1, window_size))
    agg = agg[:len(agg)-2]
    iam = iam[:len(iam)-2]
    
    return agg, iam


def create_overlap_windows(agg, iam, window_size, stride = 10):
    position = 0
    agg_windows = []
    iam_windows = []
    
    while position < len(agg) - window_size -1:
        agg_buffer = agg[position: position + window_size]
        iam_buffer = iam[position: position + window_size]
        agg_windows.append(agg_buffer)
        iam_windows.append(iam_buffer)
        position += stride
        
    return agg_windows, iam_windows        
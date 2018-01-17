
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



def read_channel(filename):
    """Method to read home channel data from .csv file into dask dataframe
        Args:
                filename: path to a specific channel_(m) from house(n)
        return:
                [dask.Dataframe]  from house(n)
    """
    channel_to_read = dd.read_csv(filename)
    return channel_to_read


def read_all_homes(path, houses): 
    """Returns dictionary of homes with keys "h1", "h2" etc, where each channel is a panda Dataframe
        Args: path - string containing path to where REDD home folders are located.
        Return: dictionary in format:
                                        
    """
    
    mypath = path
    homes = {}
    house_id = 0
    for house in houses:
        house_id += 1
        print("Loading house", house_id, end="... ")
        csv_filename = path + str(house) + '.csv'
        print(csv_filename)
        homes[str(house)]=read_channel(csv_filename)
    return homes


def read_specific_appliance(path, appliance_name, house_id):
    """This method is read aggregate and appliance data from building n for specified appliance.
        Args:
                     path : path to where data is located
                     appliance_name : Name of appliance
                     house_id : house id
        return:
                    [dask.Dataframe] aggregate, individual_appliance - so it can be converted in a torch tensor after that (and possibly some other transformations)
    """
    mypath = path
    column = ["Aggregate", appliance_name ]
    csv_filename = path + str(house_id) + '.csv'
    df = read_channel(csv_filename)
    aggregate = df['Aggregate']
    individual_appliance = df[appliance_name]
    return aggregate, individual_appliance


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
        aggregate, iam = read_specific_appliance(path,appliance,house)
        print("Reading house: {}".format(house))
        aggregate, iam = np.array(aggregate), np.array(iam) 
        
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
               
        for start, end in non_activations[i]:
            for j in range(non_activation_proportion):
                window_start = random.randint(start, end - window_size)
                agg_buff, iam_buff = aggregate_channels[i][window_start: window_start + window_size], individual_channels[i][window_start: window_start + window_size]
                agg_buff, iam_buff = np.copy(agg_buff), np.copy(iam_buff)
                agg.append(agg_buff)
                iam.append(iam_buff)
                
    zipper = list(zip(agg, iam))
    random.shuffle(zipper)
    agg, iam = zip(*zipper)
    agg, iam = np.array(agg), np.array(iam)
    dataset = [agg, iam]

    #Creating test set if test==True
    agg_test = []
    iam_test = []
    isFirst = True
    
    if test:
        for i in range(len(aggregate_channels_test)):
            agg_buff_test, iam_buff_test = create_windows(aggregate_channels_test[i], individual_channels_test[i], window_size=window_size)
            if isFirst:
                agg_test = agg_buff_test
                iam_test = iam_buff_test
                isFirst = False
            else:
                print(agg_test)
                print(agg_buff_test)
                agg_test = np.concatenate((agg_test, agg_buff_test), axis=0)
                iam_test = np.concatenate((iam_test, iam_buff_test), axis=0)
        testset = [agg_test, iam_test]
        
        return dataset, testset

    return dataset
        
     
        
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
        
        
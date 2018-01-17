import os
import pandas as pd
import numpy as np
import os
import psutil 
import time
from datetime import datetime
import random



def read_channel(filename, appliance):
    """Method to read home channel data from .dat file into panda dataframe
        Args:
                filename: path to a specific channel_(m) from house(n)
        return:
                [pandas.Dataframe] of a signle channel_(m) from house(n)
    """
    channel_to_read = pd.read_csv(filename, names=["Time", appliance], delim_whitespace=True)
    channel_to_read['Time'] = pd.to_datetime(channel_to_read['Time'],unit='s')

    return channel_to_read

def load_chan_list(app_name, data_dir, ds_name='UKDALE'):
        """
        Returns corresponding meter numbers given appliance name
        For different channels with same name, it will return a list
        """
        chan_list = []
        if(ds_name=='UKDALE'):
            for line in open(os.path.join(data_dir, 'labels.dat')):
                if(line.strip('\n').split(' ')[1] == app_name or 
                   line.strip('\n').split(' ')[1][:-1] == app_name):
                    chan_list.append(int(line.strip('\n').split(' ')[0]))
        return(chan_list)

    
def load_meter(app_name,  data_dir, ds_name='UKDALE',):
        """
        Take an appliance name, return a list of meters object
        Each meter object is a dictionary with three attributes, appliance name,
        channel number and data which is a pandas series
        """
        data = pd.DataFrame()
        if(ds_name=='UKDALE'):
            chan_list = load_chan_list(app_name,data_dir, ds_name)
            for chan_num in chan_list:
                file_name = 'channel_%d.dat' % chan_num
                df=read_channel(os.path.join(data_dir, file_name),app_name)
                df.dropna(axis=0)
                data = pd.concat([data, df])
        return data    


def generate_clean_data(path, appliance,buildings=[1,2], test=False, test_on='All'):
    
    #activation_proportion = proportion[0]
    #non_activation_proportion = proportion[1]
    #aggregate_channels = []
    #individual_channels = []
    aggregate_channels = pd.DataFrame()
    individual_channels = pd.DataFrame()
    aggregate_channels_test = []
    individual_channels_test = []
    activations = []
    non_activations = []
    
    houses=buildings
    for house in houses:
        data_dir = path + "house" +str(house) + '/'
        print(data_dir)
        iam = load_meter(app_name=appliance,  data_dir=data_dir, ds_name='UKDALE')
        aggregate = load_meter(app_name="aggregate", data_dir=data_dir,ds_name='UKDALE')
        print("Reading house: {}".format(house))
        #aggregate, iam = np.array(aggregate['aggregate']), np.array(iam[appliance])
        
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
    #appending the aggregate to aggregate list and iam to iam list so that their indices match            
    aggregate_channels, individual_channels = pd.concat([aggregate_channels, aggregate]), pd.concat([individual_channels, iam]) 
    print("size of aggregate: {}".format(len(aggregate_channels)))
    print("size of meters: {}".format(len(individual_channels)))
    return aggregate_channels, individual_channels


def resample_data(main, meter):
    '''Convernt time column into index and downsampling aggregate data 
    from 1s data to ~6s data as it was sampled for each individual appliance

    Attributes
    ----------
    main :  Aggregate dataframe
    meter : appliance dataframe

    Return : 
       [pandas.Dataframe] main, meter - so it can be  possibly for some other transformations)

    '''
    main.index = main['Time']
    meter.index = meter['Time']

    del meter['Time']
    del main['Time']

    meter = meter[meter.index.isin(main.index)]
    main = main[main.index.isin(meter.index)]
    
    return main, meter

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
import os
import pandas as pd
import numpy as np
import os
import psutil 
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




def generate_data(path, appliance):
    
    
    aggregate_channels = pd.DataFrame()
    individual_channels = pd.DataFrame()
    
    if appliance =="kettle": 
        houses = [1,2,3]
    elif appliance =="microwave": 
        houses = [1,2,5]    
    elif appliance =="dish_washer": 
        houses = [1,2,5]    
    elif appliance =="fridge": 
         houses = [1,2]   
    elif appliance =="washing_machine": 
        houses = [1,2]  


    for house in houses:
        data_dir = path + "house" +str(house) + '/'
        
        try:
            iam = load_meter(app_name=appliance,  data_dir=data_dir, ds_name='UKDALE')
            aggregate = load_meter(app_name="aggregate", data_dir=data_dir,ds_name='UKDALE')
            print("Reading house: {}".format(house))

        except KeyError:
            print (" Dataframe is empty check wether {} name  is correct".format(appliance))    
        #aggregate, iam = resample_data(aggregate, iam)
        #appending the aggregate to aggregate list and iam to iam list so that their indices match            
        aggregate_channels, individual_channels = pd.concat([aggregate_channels, aggregate]), pd.concat([individual_channels, iam]) 
        
    print("size of aggregate: {}".format(len(aggregate_channels)))
    print("size of meters: {}".format(len(individual_channels)))
    return aggregate_channels, individual_channels


def generate_data2(path, appliance):
    
    
    aggregate_channels = []
    individual_channels = []
    
    if appliance =="kettle": 
        houses = [1,2,3]
    elif appliance =="microwave": 
        houses = [1,2,5]    
    elif appliance =="dish_washer": 
        houses = [1,2,5]    
    elif appliance =="fridge": 
         houses = [1,2]   
    elif appliance =="washing_machine": 
        houses = [1,2]  


    for house in houses:
        data_dir = path + "house" +str(house) + '/'
        print(data_dir)
        iam = load_meter(app_name=appliance,  data_dir=data_dir, ds_name='UKDALE')
        aggregate = load_meter(app_name="aggregate", data_dir=data_dir,ds_name='UKDALE')
        print("Reading house: {}".format(house))
        aggregate, iam = resample_data(aggregate, iam)
        aggregate, iam = np.array(aggregate['aggregate']), np.array(iam[appliance])
        
                
    aggregate_channels.append(aggregate) #appending the aggregate to aggregate list and iam to iam list so that their indices match
    individual_channels.append(iam)

    return aggregate_channels, individual_channels




def generate_clean_data(path, appliance, window_size, threshold, proportion=[1,1],  test=False, test_on='All'):
    
    activation_proportion = proportion[0]
    non_activation_proportion = proportion[1]
    aggregate_channels = []
    individual_channels = []
    aggregate_channels_test = []
    individual_channels_test = []
    activations = []
    non_activations = []
    
    if appliance =="kettle": 
        houses = [1,2,3]
    elif appliance =="microwave": 
        houses = [1,2,5]    
    elif appliance =="dish_washer": 
        houses = [1,2,5]    
    elif appliance =="fridge": 
         houses = [1,2]   
    elif appliance =="washing_machine": 
        houses = [1,2]  


    for house in houses:
        data_dir = path + "house" +str(house) + '/'
        print(data_dir)
        iam = load_meter(app_name=appliance,  data_dir=data_dir, ds_name='UKDALE')
        aggregate = load_meter(app_name="aggregate", data_dir=data_dir,ds_name='UKDALE')
        print("Reading house: {}".format(house))
        aggregate, iam = resample_data(aggregate, iam)
        aggregate, iam = np.array(aggregate['aggregate']), np.array(iam[appliance])
        
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
                if (window_size - activation_size)> 0:
                    start_aggregate = start - np.random.randint(0, (window_size - activation_size))
                
                    #if start_aggregate + window_size < len(aggregate_channels[i]):
                    agg_buff, iam_buff = aggregate_channels[i][start_aggregate: start_aggregate + window_size], individual_channels[i][start_aggregate: start_aggregate + window_size]
                    agg_buff, iam_buff = np.copy(agg_buff), np.copy(iam_buff)
                    agg.append(agg_buff)
                    iam.append(iam_buff)

        for start, end in non_activations[i]:
            for j in range(non_activation_proportion):
                if (end - window_size)>0:
                    window_start = np.random.randint(start, (end - window_size))
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
    print("Finish finding activation with {}".format(len(dataset)))
    return dataset
    

    
    
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


    
           
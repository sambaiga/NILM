import pandas as pd
#import dask.dataframe as pd
from os import walk
import numpy as np
import torch
import random
from sys import stdout
import psutil 
from datetime import datetime
import random




columns_rename = {'1':{'Time': 'Timestamp', 
                        'Appliance1': 'Fridge',
                        'Appliance2':'Freezer(1)',
                        'Appliance3':'Freezer(2)',
                        'Appliance4':'Washer Dryer',
                        'Appliance5':'Washing Machine',
                        'Appliance6':'Dishwasher',
                        'Appliance7':'Computer',
                        'Appliance8':'Television Site',
                        'Appliance9':'Electric Heater' },
                '2':{'Time': 'Timestamp', 
                        'Appliance1': 'Fridge-Freezer',
                        'Appliance2':'Washing Machine',
                        'Appliance3':'Dishwasher',
                        'Appliance4':'Television Site',
                        'Appliance5':'Microwave',
                        'Appliance6':'Toaster',
                        'Appliance7':'Hi-Fi',
                        'Appliance8':'Kettle',
                        'Appliance9':'Overhead Fan' },
                '3':{'Time': 'Timestamp', 
                        'Appliance1': 'Toaster',
                        'Appliance2':'Fridge-Freezer',
                        'Appliance3':'Freezer',
                        'Appliance4':'Tumble Dryer',
                        'Appliance5':'Dishwasher',
                        'Appliance6':'Washing Machine',
                        'Appliance7':'Television Site',
                        'Appliance8':'Microwave',
                        'Appliance9':'Kettle' }, 
                '4':{'Time': 'Timestamp', 
                        'Appliance1': 'Fridge',
                        'Appliance2':'Freezer',
                        'Appliance3':'Fridge-Freezer',
                        'Appliance4':'Washing Machine(1)',
                        'Appliance5':'Washing Machine(2)',
                        'Appliance6':'Desktop Computer',
                        'Appliance7':'Television Site',
                        'Appliance8':'Microwave',
                        'Appliance9':'Kettle' },
                '5':{'Time': 'Timestamp', 
                        'Appliance1': 'Fridge-Freezer',
                        'Appliance2':'Tumble Dryer',
                        'Appliance3':'Washing Machine',
                        'Appliance4':'Dishwasher',
                        'Appliance5':'Desktop Computer',
                        'Appliance6':'Television Site',
                        'Appliance7':'Microwave',
                        'Appliance8':'Kettle',
                        'Appliance9':'Toaster' },
                '6':{'Time': 'Timestamp', 
                        'Appliance1': 'Freezer',
                        'Appliance2':'Washing Machine',
                        'Appliance3':'Dishwasher',
                        'Appliance4':'MJY Computer',
                        'Appliance5':'TV/Satellite',
                        'Appliance6':'Microwave',
                        'Appliance7':'Kettle',
                        'Appliance8':'Toaster',
                        'Appliance9':'PGM Computer' },
                  
                '7':{'Time': 'Timestamp', 
                        'Appliance1': 'Fridge',
                        'Appliance2':'Freezer(1)',
                        'Appliance3':'Freezer(2)',
                        'Appliance4':'Tumble Dryer',
                        'Appliance5':'Washing Machine',
                        'Appliance6':'Dishwasher',
                        'Appliance7':'Television Site',
                        'Appliance8':'Toaster',
                        'Appliance9':'Kettle' },
                  
                 '8':{'Time': 'Timestamp', 
                        'Appliance1': 'Fridge',
                        'Appliance2':'Freezer',
                        'Appliance3':'Washer Dryer',
                        'Appliance4':'Washing Machine',
                        'Appliance5':'Toaster',
                        'Appliance6':'Computer',
                        'Appliance7':'Television Site',
                        'Appliance8':'Microwave',
                        'Appliance9':'Kettle' },
                  
                  '9':{'Time': 'Timestamp', 
                        'Appliance1': 'Fridge-Freezer',
                        'Appliance2':'Washer Dryer',
                        'Appliance3':'Washing Machine',
                        'Appliance4':'Dishwasher',
                        'Appliance5':'Television Site',
                        'Appliance6':'Microwave',
                        'Appliance7':'Kettle',
                        'Appliance8':'Hi-Fi',
                        'Appliance9':'Electric Heater' },
                  
                  '10':{'Time': 'Timestamp', 
                        'Appliance1': 'Magimix(Blender)',
                        'Appliance2':'Toaster',
                        'Appliance3':'Chest Freezer',
                        'Appliance4':'Fridge-Freezer',
                        'Appliance5':'Washing Machine',
                        'Appliance6':'Dishwasher',
                        'Appliance7':'Television Site',
                        'Appliance8':'Microwave',
                        'Appliance9':'K Mix' },
                  
                  '11':{'Time': 'Timestamp', 
                        'Appliance1': 'Fridge',
                        'Appliance2':'Fridge-Freezer',
                        'Appliance3':'Washing Machine',
                        'Appliance4':'Dishwasher',
                        'Appliance5':'Computer Site',
                        'Appliance6':'Microwave',
                        'Appliance7':'Kettle',
                        'Appliance8':'Router',
                        'Appliance9':'Hi-Fi' },
                  
                  '12':{'Time': 'Timestamp', 
                        'Appliance1': 'Fridge-Freezer',
                        'Appliance2':'Unknown1',
                        'Appliance3':'Unknown2',
                        'Appliance4':'Computer Site',
                        'Appliance5':'Microwave',
                        'Appliance6':'Kettle',
                        'Appliance7':'Toaster',
                        'Appliance8':'Television',
                        'Appliance9':'Unknown3' },
                  
                  '13':{'Time': 'Timestamp', 
                        'Appliance1': 'Television Site',
                        'Appliance2':'Freezer',
                        'Appliance3':'Washing Machine',
                        'Appliance4':'Dishwasher',
                        'Appliance5':'Unknown',
                        'Appliance6':'Network Site',
                        'Appliance7':'Microwave',
                        'Appliance8':'Microwave(2)',
                        'Appliance9':'Kettle' },
                  
                  
                  '15':{'Time': 'Timestamp', 
                        'Appliance1': 'Fridge-Freezer',
                        'Appliance2':'Tumble Dryer',
                        'Appliance3':'Washing Machine',
                        'Appliance4':'Dishwasher',
                        'Appliance5':'Computer Site',
                        'Appliance6':'Television Site',
                        'Appliance7':'Microwave',
                        'Appliance8':'Hi-Fi',
                        'Appliance9':'Toaster' },
                  
                  '16':{'Time': 'Timestamp', 
                        'Appliance1': 'Fridge-Freezer(1)',
                        'Appliance2':'Fridge-Freezer(2)',
                        'Appliance3':'Electric Heater(1)',
                        'Appliance4':'Electric Heater(2)',
                        'Appliance5':'Washing Machine',
                        'Appliance6':'Dishwasher',
                        'Appliance7':'Computer Site',
                        'Appliance8':'Television Site',
                        'Appliance9':'Dehumidifier' },
                  
                  '17':{'Time': 'Timestamp', 
                        'Appliance1': 'Freezer',
                        'Appliance2':'Fridge-Freezer',
                        'Appliance3':'Tumble Dryer',
                        'Appliance4':'Washing Machine',
                        'Appliance5':'Computer Site',
                        'Appliance6':'Television Site',
                        'Appliance7':'Microwave',
                        'Appliance8':'Kettle',
                        'Appliance9':'TV Site(Bedroom)' },
                  
                  '18':{'Time': 'Timestamp', 
                        'Appliance1': 'Fridge(garage)',
                        'Appliance2':'Freezer(garage)',
                        'Appliance3':'Fridge-Freezer',
                        'Appliance4':'Washer Dryer(garage)',
                        'Appliance5':'Washing Machine',
                        'Appliance6':'Dishwasher',
                        'Appliance7':'Desktop Computer',
                        'Appliance8':'Television Site',
                        'Appliance9':'Microwave' },
                  
                  '19':{'Time': 'Timestamp', 
                        'Appliance1': 'Fridge Freezer',
                        'Appliance2':'Washing Machine',
                        'Appliance3':'Television Site',
                        'Appliance4':'Microwave',
                        'Appliance5':'Kettle',
                        'Appliance6':'Toaster',
                        'Appliance7':'Bread-maker',
                        'Appliance8':'Games Console',
                        'Appliance9':'Hi-Fi' },
                  
                  '20':{'Time': 'Timestamp', 
                        'Appliance1': 'Fridge',
                        'Appliance2':'Freezer',
                        'Appliance3':'Tumble Dryer',
                        'Appliance4':'Washing Machine',
                        'Appliance5':'Dishwasher',
                        'Appliance6':'Computer Site',
                        'Appliance7':'Television Site',
                        'Appliance8':'Microwave',
                        'Appliance9':'Kettle' },
                  
                  '21':{'Time': 'Timestamp', 
                        'Appliance1': 'Fridge-Freezer',
                        'Appliance2':'Tumble Dryer',
                        'Appliance3':'Washing Machine',
                        'Appliance4':'Dishwasher',
                        'Appliance5':'Food Mixer',
                        'Appliance6':'Television',
                        'Appliance7':'Kettle',
                        'Appliance8':'Vivarium',
                        'Appliance9':'Pond Pump' },}


label = {'1':['Timestamp', 
                     'Fridge',
                     'Freezer(1)',
                     'Freezer(2)',
                     'Washer Dryer',
                     'Washing Machine',
                    'Dishwasher',
                     'Computer',
                    'Television Site',
                    'Electric Heater' ],
                 '2':['Timestamp', 
                    'Fridge-Freezer',
                    'Washing Machine',
                    'Dishwasher',
                    'Television Site',
                     'Microwave',
                    'Toaster',
                    'Hi-Fi',
                    'Kettle',
                    'Overhead Fan'],
                }





def load_csv(filename, columns_rename, house,  tz="Europe/London"):
    """
    Parameters
    ----------
    filename : str
    columns : list of tuples (for hierarchical column index)
    tz : str e.g. 'US/Eastern'
    Returns
    -------
    dataframe
    """
    # Load data
    columns = ['Time',
           'Aggregate',
           'Appliance1',
           'Appliance2',
           'Appliance3',
           'Appliance4',
           'Appliance5',
           'Appliance6',
           'Appliance7',
           'Appliance8',
           'Appliance9']
    
    df = pd.read_csv(filename)
    df = df[columns]
    df = df.rename(columns=columns_rename[str(house)])
    # Convert the integer index column to timezone-aware datetime 
    df['Timestamp'] = pd.to_datetime(df.Timestamp, utc=True)
    #df.set_index('Timestamp', inplace=True)
    #df = df.tz_localize('GMT').tz_convert(tz)
    #df = df.sort_index()

    return df


def read_all_homes(path, houses,outputpath="/home/ibcn079/data/REFIT/house_"): 
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
        csv_filename = path + 'CLEAN_House' + str(house) + '.csv'
        print(csv_filename)
        homes[str(house)]=load_csv(csv_filename, columns_rename, house)
        print("Saving home_{} dataset(feaher format) to disk".format(house))
        homes[str(house)].to_feather(outputpath+str(house))





def read_channel(filename, app_name):
    """Method to read home channel data from .dat file into panda dataframe
        Args:
                filename: path to a specific channel_(m) from house(n)
        return:
                [pandas.Dataframe] of a signle channel_(m) from house(n)
    """
    channel_to_read = pd.read_feather(filename)[['Timestamp',app_name]]
    return channel_to_read


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
    main.index = main['Timestamp']
    meter.index = meter['Timestamp']

    del meter['Timestamp']
    del main['Timestamp']
    
    main = main.sort_index()
    meter = meter.sort_index()

    meter = meter[meter.index.isin(main.index)]
    main = main[main.index.isin(meter.index)]
    
    return main, meter

def generate_clean_data(path, appliance, window_size, threshold, proportion=[1,1],  test=False, test_on='All'):
    
    activation_proportion = proportion[0]
    non_activation_proportion = proportion[1]
    aggregate_channels = []
    individual_channels = []
    aggregate_channels_test = []
    individual_channels_test = []
    activations = []
    non_activations = []
    
    houses = []
    for key, values in label.items():
        if app_name in values:
            houses.append(key)

    for house in houses:
        csv_filename = path + "house_"+house
        #print(csv_filename)
        iam = read_channel(filename=csv_filename, app_name=appliance)
        aggregate = read_channel(filename=csv_filename, app_name="Aggregate")
        print("Reading house_: {}".format(house))
        aggregate, iam = resample_data(aggregate, iam)
        aggregate, iam = np.array(aggregate['Aggregate']), np.array(iam[appliance])
        
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

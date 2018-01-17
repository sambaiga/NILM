
import pandas as pd
#import dask.dataframe as pd
from os import walk
import numpy as np
import torch
import random
from sys import stdout
from collections import OrderedDict




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
                        'Appliance2':'Unknown',
                        'Appliance3':'Unknown',
                        'Appliance4':'Computer Site',
                        'Appliance5':'Microwave',
                        'Appliance6':'Kettle',
                        'Appliance7':'Toaster',
                        'Appliance8':'Television',
                        'Appliance9':'Unknown' },
                  
                  '13':{'Time': 'Timestamp', 
                        'Appliance1': 'Television Site',
                        'Appliance2':'Freezer',
                        'Appliance3':'Washing Machine',
                        'Appliance4':'Dishwasher',
                        'Appliance5':'Unknown',
                        'Appliance6':'Network Site',
                        'Appliance7':'Microwave',
                        'Appliance8':'Microwave',
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
    df.set_index('Timestamp', inplace=True)
    #df = df.tz_localize('GMT').tz_convert(tz)
    df = df.sort_index()

    return df



def read_channel(filename, house, columns_rename):
    """Method to read home channel data from .csv file into panda dataframe
        Args:
                filename: path to a specific channel_(m) from house(n)
        return:
                [pandas.Dataframe] of a signle channel_(m) from house(n)
    """
        
    channel_to_read = load_csv(filename, columns_rename, house, tz="Europe/London")
    
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
        csv_filename = path + 'CLEAN_House' + str(house) + '.csv'
        print(csv_filename)
        homes[str(house)]=read_channel(csv_filename, house, columns_rename)
        print("Saving home_{} dataset on disk".format(house))
        homes[str(house)].to_feather("/home/ibcn079/data/REFIT/house_"+str(house))
    return homes



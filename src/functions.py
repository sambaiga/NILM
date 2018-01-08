import sys
sys.path.insert(0, '../refit')
import scipy
import numpy as np
from scipy.signal import wiener, medfilt
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
import time
from datetime import datetime, timezone
from refit_loader import RefitLoader
import DataProvider
import argparse
import refit_meta
import refit_helpers
import nilm_helpers



scaler = MinMaxScaler(feature_range=(-1,1))


def load_dataset(appliance, houses,datadir, windowlength, history=None, process=True):
	"""
	load data set and perform data preprocessing
	"""
	#print("Loading data")
	data_loader = RefitLoader(appliance, houses, datadir, history)
	appliance_data_list, aggregate_data_list, _, timestamps_list = data_loader.get_merged_data()

	appliance_data = np.hstack(appliance_data_list)
	aggregate_data = np.hstack(aggregate_data_list)
	timestamps     = np.hstack(timestamps_list)

	#print("Find usable index")
	#data_indices = nilm_helpers.usable_indices(timestamps, windowlength)

	#print("Clean mains data")
	aggregate_data = refit_helpers.clean_mains(aggregate_data)
	
	
	return aggregate_data[:, np.newaxis], appliance_data[:, np.newaxis], timestamps[:, np.newaxis]


def get_sequnce_instances(data, window_size):
    timeseries = np.asarray(data)
    assert 0 < window_size < timeseries.shape[0]
    X = np.array([timeseries[start:start + window_size] \
        for start in range(0, timeseries.shape[0] - window_size)])
    return X.squeeze()


def data_generator(provider, kwag):
	while 1:
		for batch in provider.feed(**kwag):
			X, y = batch
		
		yield X, y


def get_datesstring(timestamps):
	"""
	Transform a date string into unix timestam
	"""
	time = [datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S') 
			for x in timestamps]
	return time


def inputs_normalize(inputs_batch):
    return (inputs_batch - refit_meta.mains_mean) / refit_meta.mains_std

def targets_normalize(appliance_name, targets_batch):
	"""
	Normalize appliance power signal
	"""
	return (targets_batch - refit_meta.mean_on_power[appliance_name])/refit_meta.std_on_power[appliance_name]

def predictions_transform(appliance_name, predictions):
	"""
	transform prediction output
	"""
	prediction=np.maximum(predictions *refit_meta.std_on_power[appliance_name] + refit_meta.mean_on_power[appliance_name], 0) 
	return  


def generator(arrays, batch_size):
    """Generate batches, one with respect to each array's first axis."""
    starts = [0] * len(arrays)  # pointers to where we are in iteration
    while True:
        batches = []
        for i, array in enumerate(arrays):
            start = starts[i]
            stop = start + batch_size
            diff = stop - array.shape[0]
            if diff <= 0:
                batch = array[start:stop]
                starts[i] += batch_size
            else:
                batch = np.concatenate((array[start:], array[:diff]))
                starts[i] = diff
            batches.append(batch)
        yield batches
       

def input_process_list(data):
    results = []
    for array in data:
        array = get_variable_power_component(array)
        array = get_variant_power(array, alpha=0.1)
        array = get_differential_power(array)
        
        results.append(array)
    return  results  


def target_process_list(data):
    results = []
    for array in data:
        array = linear2mu(array, mu=255)
        results.append(array.flatten())
    return  results     

def linear_quantization(data, w):
    step = 1/(2**(w-1))
    data = np.copy(data)
    idx = np.where(np.abs(data) >= 1)
    data[idx] = np.sign(data[idx])
    # linear uniform quantization
    return  (step * np.floor(data/step + 1/2))      


def quantize_list(data):
    results = []
    for array in data:
        array = linear_quantization(array, 2)
        results.append(array)
    return results    
        

def uniform_quantization(data, w):
    step = 1/(2**(w-1))
    data = np.copy(data)
    idx = np.where(np.abs(data) >= 1)
    data[idx] = np.sign(data[idx])
    # linear uniform quantization
    return  (step * np.floor(data/step + 1/2))    


def get_variable_power_component(data):
	"""
	Remove always ON component from aggregate power
	:arg
		data: aggregate power
	:return
		always_on_power: The constant power removed from aggregate power data
		always_on_power: The variable power after removing the always ON constant power 
	"""
	threshold=np.percentile(data,0.5)
	always_on_power = np.where(data> threshold,threshold,0)
	variable_power = data - always_on_power    

	return variable_power

def smooth_power(data, kernel=9,type="wainer"):
	"""
	Smooth the power signal using either meadian filter or wainer filter
	:arg
		data:  power signal
		kernel:  size of the median filter window
		type: type of filter to use weather wainer or median
	:return
		smoothed_data: The smoothed power generated
	"""
	if type=='wainer':
		smoothed_data = wiener(data, kernel)
	else:
		smoothed_data = medfilt(data, kernel)
			
	return smoothed_data       



def get_variant_power(data, alpha=0.01):
	"""
	Generate variant power which reduce noise that may impose negative inluence on  pattern identification
	:arg
		data:  power signal
		alpha[0,0.9]:  reflection rate
	:return
		variant_power: The variant power generated
	"""
	variant_power = np.zeros(len(data))
	for i in range(1,len(data)):
		d = data[i]-variant_power[i-1]
		variant_power[i] = variant_power[i-1] + alpha*d
		
	return  variant_power  

def get_sequence_varpower(data):
	"""
	Remove always ON component from aggregate power
	:arg
		data: aggregate power
	:return
		always_on_power: The constant power removed from aggregate power data
		always_on_power: The variable power after removing the always ON constant power 
	"""
	N, D = data.shape
	dat = data.flatten()
	threshold=np.percentile(dat,0.5)
	always_on_power = np.where(dat> threshold,threshold,0)
	variable_power = dat - always_on_power       

	return  always_on_power.reshape(N,D),  variable_power.reshape(N,D)


def get_sequence_vardpower(data):
	N, D = data.shape
	dat = get_variant_power(data.flatten())
	dat = get_differential_power(dat)
	return  dat.reshape(N,D)

def get_differential_power(data):
	"""
	Generate differential power:=p[t+1]-p[t]
	:arg
		data:  power signal
	:return
		differential_power
	"""
	differential_power = np.ediff1d(data, to_end=None, to_begin=0)   
	
	return differential_power

def get_sequence_vd_power(data):
	N, D = data.shape
	data = get_variant_power(data.flatten())
	dat = get_differential_power(data.flatten())
	return  data.reshape(N,D)


def uniform_midrange_quantizer(data, w=2):
	"""
	linear midrange quantization on the power signal
	source: http://dsp-nbsphinx.readthedocs.io/en/nbsphinx-experiment/quantization/linear_uniform_characteristic.html
	:arg
		data:  power signal
		w: number of bits
	:return
		quantized_signal: linear uniform quantized signal
	"""
	step = 1/(2**(w-1))
	data = np.copy(data)
	idx = np.where(np.abs(data) >= 1)
	data[idx] = np.sign(data[idx])
	
	quantized_signal = step * np.floor(data/step + 1/2)

	return  quantized_signal


def uniform_midrise_quantizer(data, w):
	"""
	linear midrise quantization on the power signal
	source: http://dsp-nbsphinx.readthedocs.io/en/nbsphinx-experiment/quantization/linear_uniform_characteristic.html
	:arg
		data:  power signal
		w: number of bits
	:return
		quantized_signal: linear uniform quantized signal
	"""
	step = 1/(2**(w-1))
	# limiter
	data = np.copy(data)
	idx = np.where(np.abs(data) >= 1)
	data[idx] = np.sign(data[idx])
	
	quantized_signal = step * (np.floor(data/step) + .5)
	return quantized_signal   



def data_normalize(data, data_min, data_max):
	"""
	Normalize data in a range of [-1, 1]
	:arg
		data:  power signal
		data_min: minimum value in your data
		data_max: maximum value in your data
	:return
		normalized_signal: in range of [-1,1]
	"""
 
	result = 2*(data - data_min)/(data_max - data_min)
	return result - 1

def inverse_normalize_data(y, data_min, data_max):
	"""
	Recover normalized data in range of [-1, 1] to its valid scale
	:arg
		y:  normalized power signal
		data_min: minimum value in your data
		data_max: maximum value in your data
	:return
		normalized_signal: in range of [-1,1]
	"""
	result = (y + 1)*(data_max - data_min)/2
	return result + data_min    


def mu_law(data,  mu=255):
  """
  Perform mu_law quantization
  :arg
	normalized_data:  normalized power signal in range of [-1,1]
	mu: scale
  :return
	mu_data: in range of [-1,1]
  """
  mu = float(mu)
  if data.ndim ==1:
  	data = data.reshape(len(data),1)
  data=scaler.fit_transform(data)
  mu_data = np.sign(data) * np.log(1 + mu*np.abs(data))/np.log(1 + mu)
  
  return mu_data
  
  
  
def mu_law_inverse(y,  mu=255):
  """
  Perform inverse of mu_law quantization
  :arg
	y:  mu_quantized power signal in range of [-1,1]
	mu: scale
  :return
	mu_recover_data: in range of [-1,1]
  """
  
  mu = float(mu)
   
  mu_recover_data= np.sign(y)*(1/mu)*((1+mu)**np.abs(y)-1)
	
  mu_recover_data=scaler.inverse_transform(mu_recover_data)
  return mu_recover_data 


def A_law(data, A=87.6):
  """
  Perform A_law quantization
  :arg
	normalized_data:  normalized power signal in range of [-1,1]
	A: scale
  :return
	A_data: in range of [-1,1]
  """
   
  data=scaler.fit_transform(data)   
  y = np.zeros_like(data)
  idx = np.where(np.abs(data) < 1/A)
  y[idx] = A*np.abs(data[idx]) / (1 + np.log(A))
  idx = np.where(np.abs(data) >= 1/A)
  y[idx] = (1 + np.log(A*np.abs(data[idx]))) / (1 + np.log(A))
  
  A_data = np.sign(data)*y
  return A_data
  
def A_law_inverse(y, A=87.6):
  """
  Perform inverse of A_law quantization
  :arg
	y: A_quantized power signal in range of [-1,1]
	A: scale
  :return
	A_recover_data: in range of [-1,1]
  """
  x = np.zeros_like(y)
  idx = np.where(np.abs(y) < 1/(1+np.log(A)))
  x[idx] = np.abs(y[idx])*(1+np.log(A)) / A
  idx = np.where(np.abs(y) >= 1/(1+np.log(A)))
  x[idx] = np.exp(np.abs(y[idx])*(1+np.log(A))-1)/A
  A_recover_data = np.sign(y)*x
  A_recover_data=scaler.inverse_transform(A_recover_data)
  return  A_recover_data


def linear2mu(x, mu=255):
    """targets_process
    From Joao
    x should be normalized between -1 and 1
    Converts an array according to mu-law and discretizes it
    Note:
        mu2linear(linear2mu(x)) != x
        Because we are compressing to 8 bits here.
        They will sound pretty much the same, though.
    :usage:
        >>> bitrate, samples = scipy.io.wavfile.read('orig.wav')
        >>> norm = __normalize(samples)[None, :]  # It takes 2D as inp
        >>> mu_encoded = linear2mu(2.*norm-1.)  # From [0, 1] to [-1, 1]
        >>> print mu_encoded.min(), mu_encoded.max(), mu_encoded.dtype
        0, 255, dtype('int16')
        >>> mu_decoded = mu2linear(mu_encoded)  # Back to linear
        >>> print mu_decoded.min(), mu_decoded.max(), mu_decoded.dtype
        -1, 0.9574371, dtype('float32')
    """
    if x.ndim==1:
        x = x.reshape(-1,1)
    x = scaler.fit_transform(x)    
    x_mu = np.sign(x) * np.log(1 + mu*np.abs(x))/np.log(1 + mu)
    return ((x_mu + 1)/2 * mu).astype('int16')

def mu2linear(x, mu=255):
    """
    From Joao with modifications
    Converts an integer array from mu to linear
    For important notes and usage see: linear2mu
    """
    mu = float(mu)
    x = x.astype('float32')
    y = 2. * (x - (mu+1.)/2.) / (mu+1.)
    x_recovered = np.sign(y) * (1./mu) * ((1. + mu)**np.abs(y) - 1.)
    
    return scaler.inverse_transform(x_recovered)

import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.insert(0, '../refit')
sys.path.insert(0, '../uk_dale')
sys.path.insert(0, '../src')
import numpy as np
from refit_loader import RefitLoader
import uk_dale_loader 
import refit_meta
import refit_helpers
import nilm_helpers
import uk_dale_loader
import uk_dale_helpers
import uk_dale_meta
from ploting import *
latexify()
 

def shuffle_data(dataset, val_fraction):
	train = []
	validation = []
	for data in dataset:

		permutation = np.random.permutation(data.shape[0])
		datas= data[permutation]
		train.append(data[:int(len(data)*(1-val_fraction))])
		validation.append(data[:int(len(data)*val_fraction)])
		
	return np.hstack(train) , np.hstack(validation)  


def shuffle_data2(dataset, val_fraction):
	
	permutation = np.random.permutation(dataset.shape[0])
	dataset = dataset[permutation]
	train = np.hstack([data[:int(len(data)*(1-val_fraction))] for data in dataset])
	val = np.hstack([data[:int(len(data)*val_fraction)] for data in dataset])
		
	return train , val




def load_testset(appliance_name=None, save_path=None, dataset=None):

	test_x = save_path+dataset+'/_{0}_test_mains.npy'.format(appliance_name)
	test_y = save_path+dataset+'/_{0}_test.npy'.format(appliance_name)
	test_t = save_path+dataset+'/_{0}_test_indices.npy'.format(appliance_name)


	print("Load test dataset")
	test_set_x = np.load(test_x)
	test_set_y = np.load(test_y)
	test_indices = np.load(test_t)

	print('test set:', test_set_x.shape, test_set_y.shape)
	print('test set usable: ', len(test_indices))


	return test_set_x, test_set_y, test_indices



def load_trainset(appliance_name=None, save_path=None, dataset=None):

	tra_x = save_path+dataset+'/_{0}_train_mains.npy'.format(appliance_name)
	val_x = save_path+dataset+'/_{0}_val_mains.npy'.format(appliance_name)
 
	tra_y = save_path+dataset+'/_{0}_train.npy'.format(appliance_name)
	val_y = save_path+dataset+'/_{0}_val.npy'.format(appliance_name)

	tra_t = save_path+dataset+'/_{0}_train_indices.npy'.format(appliance_name)
	val_t = save_path+dataset+'/_{0}_val_incices.npy'.format(appliance_name)

	
	print("Load dataset")
	tra_set_x = np.load(tra_x)
	tra_set_y = np.load(tra_y)  
	val_set_x = np.load(val_x) 
	val_set_y = np.load(val_y)
	tra_indices = np.load(tra_t)
	val_indices = np.load(val_t)

	print('training set:', tra_set_x.shape, tra_set_y.shape)
	print('training set usable: ', len(tra_indices))
	print('validation set:', val_set_x.shape, val_set_y.shape)
	print('validation set usable: ', len(val_indices))


	return tra_set_x, tra_set_y, val_set_x,  val_set_y, tra_indices, val_indices


   





def save_data(appliance_name=None,save_path=None, val_fraction=0.2, train_houses=None, \
			 valid_houses=None, test_houses=None, history=None, dataset=None, separate_valid_houses=False):

	if dataset=="refit":

		data_path ='../refit/refit_np/'

		windowlength = refit_meta.pointnet_window_length[appliance_name]
		if train_houses is None:
			train_houses = refit_meta.train_houses
		if valid_houses is None:    
			valid_houses = refit_meta.valid_houses
		if test_houses is None:
			test_houses  = refit_meta.valid_houses

		if separate_valid_houses:

			print("Load train and validation sets")
			tra_loader = RefitLoader(appliance_name, train_houses, data_path, history)
			val_loader = RefitLoader(appliance_name, valid_houses, data_path, history)

			tra_appliance_data_list, tra_aggregate_data_list, _, tra_timestamps_list   =  tra_loader.get_merged_data()
			val_appliance_data_list, val_aggregate_data_list, _, val_timestamps_list   =  val_loader.get_merged_data()
			tra_set_y = np.hstack(tra_appliance_data_list)
			val_set_y = np.hstack(val_appliance_data_list)
			tra_set_x = np.hstack(tra_aggregate_data_list)
			val_set_x = np.hstack(val_aggregate_data_list)
			tra_timestamps = np.hstack(tra_timestamps_list)
			val_timestamps = np.hstack(val_timestamps_list)

		else:
			print("Load train sets")
			train_houses += valid_houses
			loader = RefitLoader(appliance_name, train_houses, data_path, history)
			appliance_data_list, aggregate_data_list, _, timestamps_list = loader.get_merged_data()

			print("Generate train and validation set")
			tra_set_y, val_set_y = shuffle_data(appliance_data_list, val_fraction)
			tra_set_x, val_set_x = shuffle_data(aggregate_data_list, val_fraction)
			tra_timestamps, val_timestamps = shuffle_data(timestamps_list, val_fraction)
		
		print("Load test sets")
		test_loader = RefitLoader(appliance_name, test_houses, data_path, history)
		test_appliance_data_list, test_aggregate_data_list, _,test_timestamps_list =  test_loader.get_merged_data()
		test_set_y = np.hstack(test_appliance_data_list)
		test_set_x = np.hstack(test_aggregate_data_list)
		test_timestamps = np.hstack(test_timestamps_list)



	if dataset=="uk_dale":

		datadir = '../uk_dale/uk_dale_np/'
		period  = 6
		windowlength = refit_meta.pointnet_window_length[appliance_name]

		if train_houses is None:
			train_houses = [1,3,4,5]
		if test_houses is None:
			test_houses  = [2]

		print("Load train set")
		
		appliance_data_list, aggregate_data_list, timestamps_list = uk_dale_loader.get_resampled_data(appliance=appliance_name, houses=train_houses, \
																				new_interval=period, path=datadir, history_length=history)

		print("Generate train and validation set")

		tra_set_y, val_set_y = shuffle_data(appliance_data_list, val_fraction)
		tra_set_x, val_set_x = shuffle_data(aggregate_data_list, val_fraction)
		tra_timestamps, val_timestamps = shuffle_data(timestamps_list, val_fraction)

		print("Load test set")
		test_set_y_list, test_set_x_list, test_timestamps_list = uk_dale_loader.get_resampled_data(appliance=appliance_name, houses=test_houses, \
																				new_interval=period, path=datadir, history_length=history)

		test_set_y = np.hstack(test_set_y_list)
		test_set_x = np.hstack(test_set_x_list)
		test_timestamps = np.hstack(test_timestamps_list)


	print("Find usable indices")
	tra_indices = nilm_helpers.usable_indices(tra_timestamps, windowlength)
	val_indices = nilm_helpers.usable_indices(val_timestamps, windowlength)
	test_indices = nilm_helpers.usable_indices(test_timestamps, windowlength)


	print('training set:', tra_set_x.shape, tra_set_y.shape)
	print('training set usable: ', len(tra_indices))
	print('validation set:', val_set_x.shape, val_set_y.shape)
	print('validation set usable: ', len(val_indices))
	print('Test set:', test_set_x.shape, test_set_y.shape)
	print('Test set usable: ', len(test_indices))


	if save_path is not None:

		print("Saving results") 
		np.save(save_path+dataset+'/_{0}_train_mains.npy'.format(appliance_name),tra_set_x) 
		np.save(save_path+dataset+'/_{0}_train.npy'.format(appliance_name),tra_set_y) 
		np.save(save_path+dataset+'/_{0}_train_indices.npy'.format(appliance_name),tra_indices)

		np.save(save_path+dataset+'/_{0}_val_mains.npy'.format(appliance_name),val_set_x) 
		np.save(save_path+dataset+'/_{0}_val.npy'.format(appliance_name),val_set_y) 
		np.save(save_path+dataset+'/_{0}_val_incices.npy'.format(appliance_name),val_indices) 

		np.save(save_path+dataset+'/_{0}_test_mains.npy'.format(appliance_name),test_set_x) 
		np.save(save_path+dataset+'/_{0}_test.npy'.format(appliance_name),test_set_y) 
		np.save(save_path+dataset+'/_{0}_test_indices.npy'.format(appliance_name),test_indices)  

		
  
if __name__ == '__main__':

  train_houses = [1]
  test_houses  = [2]
  val_houses  =  [3]
  appliance_name="Fridge"
  save_path ='../data/'
  history = None


  """

  appliances =  ['Kettle', 'Microwave', 'Fridge', 'Dishwasher', 'Washing Machine'] 
  for appliance in appliances:
	print("Saving {} data ".format(appliance))
	save_data(appliance_name=appliance,save_path=save_path, val_fraction=0.2, train_houses=None, \
			 valid_houses=None, test_houses=None, history=history, dataset="refit", separate_valid_houses=False)

  """           



  #tra_set_x, tra_set_y, val_set_x,  val_set_y, tra_indices, val_indices = load_trainset(appliance_name=appliance_name, save_path=save_path, dataset="refit")

  #test_set_x, test_set_y, test_indices=load_testset(appliance_name=appliance_name, save_path=save_path, dataset="uk_dale")
  
  """
  print("ploting")
  plt.subplot(2, 1, 1)
  plt.plot(test_set_x[0:100])
  plt.subplot(2, 1, 2)
  plt.plot(test_set_y[0:100])
  plt.savefig("../figure/test.pdf")     
  """
  #plt.subplot(2, 1, 1)
  #plt.plot(val_set_x[0:1000])
  #plt.subplot(2, 1, 2)
  #plt.plot(val_set_y[0:1000])
  #plt.savefig("../figure/valset.pdf")                                                                               


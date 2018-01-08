import numpy as np


class DataProvider(object):
	
    def __init__(self, inputs, targets,  batch_size, input_window_size, shuffle_order=True, rng=None):
        
        self.inputs = inputs
        self.targets = targets
        self.batch_size = batch_size
        self.input_window_size = input_window_size
        self.rng = rng
        self.shuffle_order = shuffle_order
        
        inputs, targets = inputs.flatten(), targets.flatten()
        assert inputs.size == targets.size
        
        arrays = [inputs, targets]
        starts = [0] * len(arrays)
        
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
            
        self.input_batch = batches[0]  
        self.target_batch = batches[1]  
            
        if rng is None:
            rng = np.random.RandomState()
        self.rng = rng   
        
        self.current_batch = 0
        self.num_batches = self.inputs.size // self.batch_size
        

        self.new_epoch()
        
    def new_epoch(self):
        self.current_batch = 0 
        
    def __iter__(self):
        return self

    def __next__(self):

        if self.current_batch >= self.num_batches:
            self.new_epoch()
            raise StopIteration()

        if self.current_batch == 16:
            pass    
        
        inputs_batch = get_sequnce_instances(self.input_batch, self.input_window_size)
        targets_batch = self.target_batch
        
        return inputs_batch, targets_batch
        
    def generator(self):
        while True:
            for batch in self:
                yield batch   




if __name__ == '__main__':
	import sys
	sys.path.insert(0, '../refit')
	import refit_loader
	import refit_meta
	from functions import *

	appliance_name = 'Fridge'
	datadir ='../refit/refit_np/'
	history = 100000
	batchsize = 1000
	windowlength = refit_meta.pointnet_window_length[appliance_name]


	tra_x, tra_y , tra_t = load_dataset(appliance_name, refit_meta.train_houses, datadir, windowlength, history)

	tra_provider = DataProvider(tra_x, tra_y, batch_size=batchsize,input_window_size=windowlength)

	data=next(tra_provider)

	print(data[0].shape)


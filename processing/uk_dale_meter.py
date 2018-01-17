house_nums = (1, 2, 3, 4, 5)

sampling_interval = 6  # seconds

used_appliances = ['Kettle', 'Microwave', 'Fridge', 'Dishwasher', 'Washing_Machine']

appliances = {1: ['Aggregate', 'Washing Machine', 'Dishwasher', 'Kettle', 'Fridge', 'Microwave'],
                2: ['Aggregate', 'Kettle', 'Washing Machine', 'Fridge', 'Microwave', 'Dishwasher'],
                3: ['Aggregate', 'Kettle'],
                4: ['Aggregate', 'Fridge'],
                5: ['Aggregate', 'Kettle', 'Fridge', 'Dishwasher', 'Microwave']}
params_appliance = {'Kettle':{'windowlength':599,
							  'on_power_threshold':2000,
							  'max_on_power':3998,
							 'mean':700,
							 'std':1000,
							 's2s_length':128},
					'Microwave':{'windowlength':599,
							  'on_power_threshold':200,
							  'max_on_power':3969,
								'mean':500,
								'std':800,
								's2s_length':128},
					'Fridge':{'windowlength':599,
							  'on_power_threshold':50,
							  'max_on_power':3323,
							 'mean':200,
							 'std':400,
							 's2s_length':512},
					'Dishwasher':{'windowlength':599,
							  'on_power_threshold':10,
							  'max_on_power':3964,
								  'mean':700,
								  'std':1000,
								  's2s_length':1536},
					'Washing Machine':{'windowlength':599,
							  'on_power_threshold':20,
							  'max_on_power':3999,
									  'mean':400,
									  'std':700,
									  's2s_length':2000}}
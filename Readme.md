## Requirements
	python
	PyTorch
	pandas
	tqdm
	numpy
	scikit_learn
	matploblib
	numba


## Dataset 
#### Download the public data
	 MOOC/Reddit(http://snap.stanford.edu/jodie) 
	 Amazon(http://jmcauley.ucsd.edu/data/amazon/)

#### Preprocess the data
	 graph file:  `***.csv` 
	 	source id, destination id, time, label 

	 addtional attributes: `ml_***.npy` 

## Running the experiments
### Run graph augmentation
	python generateNeg.py --data reddit 	

### Run anomaly detection
	python main.py -d dataset --g_neg False/True --n_degree history_length partner_size --mask mask_num_source mask_num_desti


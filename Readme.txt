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
	 graph file:  `***.csv` 
	 	source id, destination id, time, label 

	 addtional attributes: `ml_***.npy` 

## Run graph augmentation
	python generateNeg.py --data reddit 	

## Run anomaly detection
	python main.py -d dataset --g_neg False/True --n_degree history_length partner_size --mask mask_num_source mask_num_desti


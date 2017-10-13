import numpy as np
from scipy.sparse import csr_matrix
import sys
from sklearn.datasets import load_svmlight_file
import random
from datetime import datetime
import math 

def main():
	traindatafile = sys.argv[1];
	modelname = sys.argv[2]
	# The training file is in libSVM format
	tr_data = load_svmlight_file(traindatafile);
	Xtr = tr_data[0]; # Training features in sparse format
	Ytr = tr_data[1]; # Training labels
	w = np.load(modelname)
	# We have n data points each in d-dimensions
	n, d = Xtr.get_shape();
	
	# The labels are named 1 and 2 in the data set. Convert them to our standard -1 and 1 labels
	Ytr = 2*(Ytr - 1.5);
	Ytr = Ytr.astype(int);
	Ytr = Ytr.reshape(n,1);
	
	# Optional: densify the features matrix.
	# Warning: will slow down computations
	Xtr = Xtr.toarray();
	acc = (np.sign((w*Xtr).sum(axis=1)) == (Ytr.ravel())).sum()
	percac = acc/(n*0.01)
	print(n)
	print (acc)
	print(percac)
	#acc = (Ytr == (np.sign(w*Xtr).sum(axis=1).ravel()))
#	print(acc)
	
main()


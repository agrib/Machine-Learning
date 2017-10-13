import numpy as np
from scipy.sparse import csr_matrix
import sys
from sklearn.datasets import load_svmlight_file
import random
from datetime import datetime
import math

def main():

	# Get training file name from the command line
	traindatafile = sys.argv[1];
	# For how many iterations do we wish to execute GD?
	n_iter = int(sys.argv[2]);
	# After how many iterations do we want to timestamp?
	spacing = int(sys.argv[3]);
	
	# The training file is in libSVM format
	tr_data = load_svmlight_file(traindatafile);

	Xtr = tr_data[0]; # Training features in sparse format
	Ytr = tr_data[1]; # Training labels
	
	# We have n data points each in d-dimensions
	n, d = Xtr.get_shape();
	
	
	# The labels are named 1 and 2 in the data set. Convert them to our standard -1 and 1 labels
	Ytr = 2*(Ytr - 1.5);
	Ytr = Ytr.astype(int);

	
	# Optional: densify the features matrix.
	# Warning: will slow down computations
	Xtr = Xtr.toarray();
	
	# Initialize model
	# For primal GD, you only need to maintain w
	# Note: if you have densified the Xt matrix then you can initialize w as a NumPy array
	w = np.zeros(d)
	
	# We will take a timestamp after every "spacing" iterations
	time_elapsed = np.zeros(int(math.ceil(n_iter/spacing)));
	tick_vals = np.zeros(int(math.ceil(n_iter/spacing)));
	obj_val = np.zeros(int(math.ceil(n_iter/spacing)));
	
	tick = 0;
	
	ttot = 0.0;
	t_start = datetime.now();
	x=np.array(Xtr)
	y=np.array(Ytr)[np.newaxis]
	ytran=np.transpose(y)	
	yx=ytran*x
	
	
	for t in range(n_iter):
		### Doing primal GD ###
		g=np.zeros(d)
		# Compute gradient
		
		wxy=w*yx
		#r1 =[1*n]
		r1=np.sum(wxy,axis=1)
		
		#r1 =hingeloss
		r1=1-r1
		#r2 = requred rows in yx
		r2=r1>=0
		r2tran=np.transpose(np.array(r2)[np.newaxis])
		
		grad=r2tran*yx
		g=np.sum(grad,axis=0)

		g=w-g

		
		
		g.reshape(1,d); # Reshaping since model is a row vector
		
		# Calculate step lenght. Step length may depend on n and t
		eta = 1/math.sqrt(t+1)*0.0001
		#eta=0.00001
		
		# Update the model
		w = w - eta * g;
		
		# Use the averaged model if that works better (see [\textbf{SSBD}] section 14.3)
		# wbar = ...;
		
		# Take a snapshot after every few iterations
		# Take snapshots after every spacing = 5 or 10 GD iterations since they are slow
	'''	if t%spacing == 0:
			# Stop the timer - we want to take a snapshot
			t_now = datetime.now();
			delta = t_now - t_start;
			time_elapsed[tick] = ttot + delta.total_seconds();
			ttot = time_elapsed[tick];
			tick_vals[tick] = tick;

			r1=r1*r2
			hingeloss=np.sum(r1)
			regu=np.sum(w*w)
			regu=regu/2
			obj_val[tick] = hingeloss+regu # Calculate the objective value f(w) for the current model w^t or the current averaged model \bar{w}^t
			
			
			tick = tick+1;
			# Start the timer again - training time!
			t_start = datetime.now();
		
			
	
	# Choose one of the two based on whichever works better for you
	#w_final = w.toarray();
	# w_final = wbar.toarray();
	
	theoriticalval=n*d*spacing*tick_vals
        np.save("obective_GD.npy",obj_val)
	np.save("timeelapsed_GD.npy",time_elapsed)
	np.save("theoriticalval_GD.npy",theoriticalval)
  '''	
	np.save("model_GD.npy", w);
		
if __name__ == '__main__':
    main()

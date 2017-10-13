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
	# For how many iterations do we wish to execute SCD?
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
	# For dual SCD, you will need to maintain d_alpha and w
	# Note: if you have densified the Xt matrix then you can initialize w as a NumPy array
	#w = csr_matrix((1, d));
	w=np.zeros((d,))
	d_alpha = np.zeros((n,));
	
	# We will take a timestamp after every "spacing" iterations
	time_elapsed = np.zeros(int(math.ceil(n_iter/spacing)))
	tick_vals = np.zeros(int(math.ceil(n_iter/spacing)));
	obj_val = np.zeros(int(math.ceil(n_iter/spacing)));
	
	tick = 0;
	
	ttot = 0.0;
	t_start = datetime.now();
	x=np.array(Xtr)
	y=np.array(Ytr)
	###calculateQ
	q=x*x
	q=np.sum(q,axis=1)
	q=y*q
	
	q=y*q	
	
		
	for t in range(n_iter):		
		### Doing dual SCD ###
		
		
		# Choose a random coordinate from 1 to n
	    	i_rand = np.random.randint(1,n);

		#####find grad
		g=np.sum(w*x[i_rand])
		g=g*y[i_rand]
		g=g-1
	
		if(d_alpha[i_rand]==0):
			pg=min(g,0)
		if(d_alpha[i_rand]==1):
			pg=max(g,0)
		if(d_alpha[i_rand]<1):
			if(d_alpha[i_rand]>0):
					pg=g
		
		if(pg!=0):		

			# Store the old and compute the new value of alpha along that coordinate
			d_alpha_old = d_alpha[i_rand];
			d_alpha[i_rand] = min(max(d_alpha[i_rand]-g/q[i_rand],0),1)
		
			# Update the model - takes only O(d) time!
			
			k = y[i_rand]*x[i_rand]
			w=w+(d_alpha[i_rand] - d_alpha_old)*k
		
		# Take a snapshot after every few iterations
		# Take snapshots after every spacing = 5000 or so SCD iterations since they are fast
		'''if t%spacing == 0:
			# Stop the timer - we want to take a snapshot
			t_now = datetime.now();
			delta = t_now - t_start;
			time_elapsed[tick] = ttot + delta.total_seconds();
			ttot = time_elapsed[tick];
			tick_vals[tick] = tick;
			r=y.reshape(n,1)*x
			
			r=d_alpha.reshape(n,1)*r
			r=np.sum(r,axis=0)
			val=np.dot(r,r.reshape(d,1))
			val=val*0.5
			####calculate hinge loss
			wx=w*x
			ywx=y.reshape(n,1)*wx
			r3=np.sum(ywx,axis=1)
			r3=1-r3
			r4=r3>=0
			r3=r3*r4
			hinge=np.sum(r3)	
						
			obj_val[tick] =val+hinge # Calculate the objective value f(w) for the current model w^t
			tick = tick+1;
			# Start the timer again - training time!
			t_start = datetime.now();
			
	
	
			theoriticalval=d*spacing*tick_vals
	
			np.save("obective_SCD.npy",obj_val)
			np.save("timeelapsed_SCD.npy",time_elapsed)
			np.save("theoriticalval_SCD.npy",theoriticalval)	
		  '''
	np.save("model_SCD.npy", w);	
if __name__ == '__main__':
    main()

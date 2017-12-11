################# README ######################
for stochastic LBFGS (results of figure 6 top row)
run Stochastic_LBFGS_LeNet_5.py -m=20 -batch=1024 

for stochastic LBFGS (results of figure 6 bottom row)
run Stochastic_LBFGS_LeNet_5.py  -m=20 -batch=512

for stochastic LBFGS (results of figure 7 top row)
run Stochastic_LBFGS_LeNet_5.py -m=15 -batch=1024 

for stochastic LBFGS (results of figure 7 bottom row)
run Stochastic_LBFGS_LeNet_5.py  -m=15 -batch=512

for robust L-BFGS (results of figure 8 top row -- note that batch is used for internal reason, the gradients are computed on all data)
run Robust_LBFGS_LeNet_5.py -m=4 -batch=1024

for robust L-BFGS (results of figure 8 bottom row)
Robust_LBFGS_one_minibatch_for_some_iterations.py -m=4 -batch=1024

for SciPy L-BFGS (results of figure 9)
LBFGS_LeNet_5_SciPy.py -batch=1024

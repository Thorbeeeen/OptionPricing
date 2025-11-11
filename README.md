This Project is a simple and small simulation of the Black-Scholes equation in pure python which relies on the Finite-Difference and Runge-Kutta-Dormand-Prince method (RKDP). The current implementation of RKDP however does not yet utilize the the "first same as last"  property of RKDP. 

The calculation side of the project leverages Numba and is done in parallel with the Visualisation in matplotlib. In this way it is possible to check parts of the result while the calculation is not yet completely finished.   

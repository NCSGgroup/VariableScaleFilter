The most common filter for GRACE(-FO) time variable gravity field is Gaussian filter.
However, traditional Gaussian filter does not consider the variance of noise in high and low latitudes,
or the anisotropy of the noise.
In this context, we establish this software to numerically realize a new design of filter called variable-scale (VS)
filter,
which can take both the variance and the anisotropy of noise into consideration.

Tips for the usage of the software:

1. A successful implemention of the software requires a Python environment, and Anaconda is recommened.
2. The python source files are stored in the directory of '\pysrc', and the example data are also provided in the
   directory of '\data'.
3. Just running the demo in the directory of '\demo' to see the result, and an output from us is also provided in the
   directory of '\results' for a verification of your installation.
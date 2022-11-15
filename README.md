The most common smoother for GRACE(-FO) time variable gravity field is Gaussian smoother.
However, traditional Gaussian smoother does not consider the variance of noise in high and low latitudes,
or the anisotropy of the noise.
In this context, we establish this software to numerically realize a new design of filter called variable-scale (VS)
filter,
which can take both the variance and the anisotropy of noise into consideration.
Fig.1 shows the comparison of our new smoother and traditional Gaussian one.
Please see our paper 
_An Over-smoothed GRACE Gravity Field in High-latitudes Imbalances the Ocean Budget_
for details.

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://raw.githubusercontent.com/NCSGgroup/VariableScaleFilter/main/img/plot_vs_filter/demo_pic.png">
    <br>
    <div style="
    display: inline-block;
    padding: 2px;">Fig. 1 GRACE EWH after applying different smoothers </div>
</center>

Tips for the usage of the software:

1. A successful implemention of the software requires a Python environment, and Anaconda is recommended.
2. The python source files are stored in the directory of '\pysrc', and the example data are also provided in the
   directory of '\data'.
3. Just running the demo in the directory of '\demo' to see the result, and an output from us is also provided in the
   directory of '\results' for a verification of your installation.

# Team Pymoo - ISCSO 2019
## Julian Blank, Yash Vesikar
### Michigan State Universtity

The following are instruction to install and run the optimizer for the ISCSO competition.

1. Clone this repository onto a machine with Matlab installed.

   ```git clone https://github.com/julesy89/pymoo-iscso19.git```

2. If you have [Anaconda](https://www.anaconda.com/distribution/) installed you may want to create a new conda enviornment to install the necessary dependencies. 
   
   ```conda create -n <name> python=3.6```

   Otherwise you may want to install Anaconda, or use some other form of virtual enviornment for python like [virtualenv](https://virtualenv.pypa.io/en/latest/installation/). **Must be python 3.5+**.

3. Install all the required dependencies.
   
   ```pip install -r requirements.txt```

4. Follow instructions [here](https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html) to install Python-Matlab dependency.

5. Now you can run either the Jupyer notebook in `final/FinalSubmission.ipynb` or run the file `final/FinalSubmission.py` to see the best result. The final best result is stored within the result object `res`. 
To see the final solution variables use:
`res.pop.get("X")[np.argmin(res.pop.get("F")[:, 0])]`

1. Neural Network weights Optimization
==============================================================================================

To run the algorithm on the neural network, first use the command "source init.sh" in the jython directory. This will export the ABIGAL.jar in the CLASSPATH.
Then simply run "jython waveform.py".

2. Optimization Problem
======================================================================================================================================
First use the command "source init.sh" in the jython directory. This will export the ABIGAL.jar in the CLASSPATH.

    a. First problem - Traveling salesman
    --------------------------------------

    To use the algorithm on the first problem, use the "jython travelingsalesman.py" in the "travelingsalesman" directory

    b.Second problem - knapsack
    ---------------------------

    To use the algorithm on the first problem, use the "jython knapsack.py" in the "knapsack" directory
    
     c. Third - countones
    ----------------------------

    To use the algorithm on the first problem, use the "jython countones.py" in the "countones" directory


3. Plotting the result
========================================================================================================================================

To plot the result you'll need matplotlib.pyplot and python3.

Go to the "python" directory and run "python3 plot.py plot_type ../output/filename.csv" with the following value for the parameters:
    - accuracy : To plot the accuracy vs iteration count
    - compute_time : To plot the compute time vs iteration count
    - accuracy_compute_time : To plot the accuracy vs the compute time
filename: The csv file that comes from the previous scripts (co_out.csv, ks_out.csv, tsp_out.csv)

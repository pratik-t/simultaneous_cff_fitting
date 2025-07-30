## Exposition:
The `scripts` directory contains executable Python scripts that perform a particular action within the entire workflow of CFF extraction. We would like  to feature scripts that (i) generate replicas, (ii) preprocess the data used to train, validate, and test replicas, (iii) perform an entire local fit, (iv) perform a global fit, and (v) visualize the results of all the training.

*Note: We do not yet have all of the necessary scripts available for use yet!* 

### Available Scripts:

1. readme_maker.py
Raw data is extracted from papers reporting experimental results and put into the `data` directory. Run `readme_maker.py` to create a human-readable Markdown file that describes the experiment and the collected observables and their uncertainties.

2. replica_data.py
This script will generate the training, validation, and testing data that a *given* replica will see. That is, if you choose to use $N_{\text{replicas}} = 100$, this script will run $100$ times, generating the relevant datasets for each replica to use to train and fit.

3. train_local_fit.py
Run `train_local_fit.py` to run the entire Replica Method on a *given observable*. We will later figure out how to incorporate *all* the observables.

4. generate_pseudodata.py
This script generates a `.csv` file that contains the values of the 8 CFFs we want to extract based on a given GPD model.

5. local_predictions.py
*Once you have run `train_local_fit.py`, you can run this script to generate predictions for the CFFs.
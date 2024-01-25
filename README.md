# interpretable-DL4BGP
This project aims to discuss the importance of interpretability tools in machine learning models for the real-time prediction of blood glucose levels.
To analyze the trained LSTM with SHAP:
- create a folder "data";
- download the OhioT1DM dataset that is available at http://smarthealth.cs.ohio.edu/OhioT1DM-dataset.html;
- create .txt files named 'ohio(patID)\_Training.txt' and 'ohio(patID)\_Testing.txt' for the training and test set, respectively;

Also, you will need to preprocess the original raw data (".xml" format) by following these steps:

1. Create a time vector, i.e. Time (dd-MMM-yyyy hh:mm:ss), with a 5-min resolution;
2. Continuous Glucose Monitoring (CGM) data should be aligned on a uniform 5-min grid (i.e, the time vector described at step 1. In cases of data gap, missing CGM values are replaced with NaN. The measurement unit should be mg/dL;
3. Carbs data should be aligned on the uniform time grid. This results in a time series where all the values are 0, expect the meal timestamp where the patient reported the estimated amount of carbs.
4. Insulin bolus should be formatted similarly to carbs, with values set to 0 for all timestamps expect for the time when the patient injected an insulin bolus;
5. Insulin basal should be aligned on the uniform CGM time grid;

In summary, each .txt file should contains the following variables:
    - Time (dd-MMM-yyyy hh:mm:ss);
    - CGM (mg/dL);
    - basal_insulin (U/h) and bolus_insulin (U).

Finally, the processed data should be saved with the following filenames: 'ohio588_Training.txt' and 'ohio588_Test.txt'  

Note: this is a proof-of-concept work focusing only on the subject ID 588, selected from the OhioT1DM dataset.

# DATASOL-package
This package is made for Seagate.
The package allows the user to detect anomalies in the data.

To Install the package: 
   ```pip install datasol```
   
## Creating_Dataframe function will clean the data from unneeded columns, can transfer categorical to numeric, fill nones with knn imputer , and save the file afterwards.
input- dataframe (pandas), columns that you want to keep and date column the rest are optional
By setting the next options True:
categorical_to_numeric is categorical columns to numeric.

fillna is filling the Na values with knn imputer.

save is saving the file to csv.

output- None if save put on True else DataFrame


## Load_Dataframe function will read csv file and return data frame after cleaning.
input - filedir , columns that you want to keep and date column.

output- DataFrame

## Anomaly_Detetion function will create a list for each column with anomaly data.
input - dataframe works better if you use the output dataframe from previos functions, time_step should be calculted by the user example:
Prepare training data
Get data values from the training timeseries data file and normalize the value data. We have a value for every 5 mins for 14 days.

24 * 60 / 5 = 288 timesteps per day
288 * 14 = 4032 data points in total

*The right amount of time steps will provide better results for the anomaly detetion.*

model settiegs- epochs, batch_size, patience, layer1-4

epochs -An epoch is when all the training data is used at once and is defined as the total number of iterations of all the training data in one cycle for training the machine learning model.

batch_size - number of samples we want to pass into the training loop at each iteration.

patience -many tries the model will have to improve before early stopping.

layers 1-4 - in the base model there is 4 layer you can change how many filters each layer will have to improve preformance.

threshold is set to 99% in another words the 1% data that is problomatic and have high chance to be an anomaly.

visual if set to true it will return plot of the data and the anomaly.


the function will return anomaly list.




   

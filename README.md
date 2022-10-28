# DATASOL-package
This package is made for Seagate.
The package allows the user to detect anomalies in the data.

To Install the package: 
   ```pip install datasol```
   
## Creating_Dataframe function will clean the data from unneeded columns, can transfer categorical to numeric, fill NAs with knn imputer, and save the file afterward.
input- data frame (pandas), columns that you want to keep, and date column the rest are optional
By setting the next options True:
categorical_to_numeric is categorical columns to numeric.

fillna is filling the Na values with knn imputer.

save is saving the file to CSV.

output- None if save put on True else Data Frame


## Load_Dataframe function will read the CSV file and return the data frame after cleaning.
input – file directory, columns that you want to keep, and date column.

output- Data Frame

## Anomaly_Detetion function will create a list for each column with anomaly data.
input - data frameworks better if you use the output data frame from previous functions, time_step should be calculated by the user example:
Prepare training data
Get data values from the training time series data file and normalize the value data. We have a value for every 5 mins for 14 days.

24 * 60 / 5 = 288 timesteps per day
288 * 14 = 4032 data points in total

*The right amount of time steps will provide better results for anomaly detection. *

model settings- epochs, batch_size, patience, layer1-4

epochs -An epoch is when all the training data is used at once and is defined as the total number of iterations of all the training data in one cycle for training the machine learning model.

batch_size – the number of samples we want to pass into the training loop at each iteration.

patience -After many tries, the model will have to improve before early stopping.

layers 1-4 - in the base model there is 4 layer you can change how many filters each layer will have to improve performance.

The threshold is set to 99% in other words the 1% data that is problematic and has a high chance to be an anomaly.

visual if set to true will return a plot of the data and the anomaly.


the function will return the anomaly list.



Original data frame:
![image](https://user-images.githubusercontent.com/80150303/198687972-4914febc-fc3f-4c47-b626-e295d1c37034.png)

After Creating_Dataframe
![image](https://user-images.githubusercontent.com/80150303/198688034-8d94cc9d-827c-46b5-90a1-122b0ec74374.png)

outpu of Anomaly_Detetion
![image](https://user-images.githubusercontent.com/80150303/198688233-baa2bdb9-72c7-44f3-9662-bc59da0f8899.png)
![outputlab](https://user-images.githubusercontent.com/80150303/198688254-38f6d1c4-32cb-45f8-9d91-7783a959440f.png)






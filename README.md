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


## Load_Dataframe function will read csv file and return data frame after cleaning.
input - filedir , columns that you want to keep and date column.

## Anomaly_Detetion

   

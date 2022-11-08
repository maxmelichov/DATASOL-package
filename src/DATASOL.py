import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
# from pyts.decomposition import SingularSpectrumAnalysis
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
import warnings
warnings.filterwarnings("ignore")
class External():
    def data_engineering(df,col):
        # train_size =int(len(df)*0.5)
        train, test = df,df#df.iloc[0:train_size],df.iloc[train_size:len(df)]

        scaler = StandardScaler()
        scaler = scaler.fit(train[[col]])
        train[col]= scaler.transform(train[[col]])
        test[col]= scaler.transform(test[[col]])
        # print(train.head())
        return train,test

    def create_dataset(X,y,time_steps=1):
        Xs, ys = [], []
        for i in range(len(X) - time_steps):
            v = X.iloc[i:(i + time_steps)].values
            Xs.append(v)
            ys.append(y.iloc[i + time_steps])
        return np.array(Xs), np.array(ys)

    def train_test_split(train,test,TIME_STEPS,col):
        x_train,y_train = External.create_dataset(train[[col]],train[col],TIME_STEPS)
        x_test,y_test = External.create_dataset(test[[col]],test[col],TIME_STEPS)
        return x_train,x_test,y_train,y_test

    def get_model(x_train,layer1 = 32 , layer2= 16, layer3 = 16, layer4 =32):
        model = keras.Sequential(
        [
            tf.keras.layers.Input(shape=(x_train.shape[1], x_train.shape[2])),
            tf.keras.layers.Conv1D(
                filters=layer1, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.Conv1D(
                filters=layer2, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            tf.keras.layers.Conv1DTranspose(
                filters=layer3, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.Conv1DTranspose(
                filters=layer4, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            tf.keras.layers.Conv1DTranspose(filters=1, kernel_size=7, padding="same"),
        ]
        )
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mae")
        # model.summary()
        return model

def Creating_Dataframe(df, columns, date_col, categorical_to_numeric =False,fillna =False,save = False):
    date = df[date_col]
    df.drop(date_col, axis =1 ,inplace =True)
    for col in df.columns:
        if col not in columns:
            df.drop(col, axis =1 ,inplace =True)
# checks if the df in null or not
    assert not df.empty, "The DataFrame is empty"
    if categorical_to_numeric:
        for col in df.columns:
            if len(df[col].value_counts()) <50:
                df.drop(col, axis = 1 , inplace = True)
            elif not np.issubdtype(df[col].dtype, np.number):
                df[col] = pd.to_numeric(df[col], errors='coerce')
    if fillna:
        imputer = KNNImputer(n_neighbors=5)
        knn = imputer.fit(df)
        df=pd.DataFrame(knn.transform(df),columns=df.columns)
    df['Date'] = date
    if save:
        df.to_csv("save.csv")
    else:
        df.Date = pd.to_datetime(df.Date)
        # Set Index
        df = df.set_index('Date')
        return df
    
def Load_Dataframe(file_name,columns,date_col='Date'):
    df= pd.read_csv(file_name,parse_dates=[date_col], index_col=date_col)
    for col in df.columns:
        if col not in columns:
            df.drop(col, axis =1 ,inplace =True)
    return df


def Anomaly_Detetion(df,TIME_STEPS= 288,epochs = 50,batch_size= 128,patience=1,layer1 = 32 , layer2= 16, layer3 = 16, layer4 =32,threshold=99,visual =False):
    anomaly= []
    ig,axs=plt.subplots(df.shape[1],figsize=(50,15))

    for i,col in enumerate(df.columns):
        print("START",col)
        final = df[[col]]
        train,test = External.data_engineering(final,col)
        print("10%",col)

        x_train,x_test,y_train,y_test = External.train_test_split(train,test,TIME_STEPS,col)
        print(x_train.shape, x_test.shape,y_train.shape,y_test.shape)
        print("20%",col)

        model = External.get_model(x_train,layer1 = layer1 , layer2= layer2, layer3 = layer3, layer4 =layer4)
        history = model.fit(x_train,y_train, epochs =epochs, batch_size=batch_size,shuffle = False,
        callbacks = [keras.callbacks.EarlyStopping(monitor = 'loss', patience = patience, restore_best_weights = True)])
        print("90%",col)

        x_test_pred = model.predict(x_test)
        test_mae_loss = np.mean(np.abs(x_test_pred - x_test), axis=1)
        THRESHOLD = np.percentile(test_mae_loss, threshold)
        test_score_df = pd.DataFrame(index=test[TIME_STEPS:].index)
        test_score_df['loss'] = test_mae_loss
        test_score_df['threshold'] = THRESHOLD
        test_score_df['anomaly'] = test_score_df.loss > test_score_df.threshold
        test_score_df[col] = test[TIME_STEPS:][col]
        anomalies = test_score_df[test_score_df.anomaly == True]
        anomaly.append(test_score_df[test_score_df.anomaly == True])
        anomalies.head()
        print("Number of anomaly samples: ", np.sum(anomalies))
        print("Indices of anomaly samples: ", np.where(anomalies))
        print("100%",col)
        print('Finish',col)
        if visual:
            if df.shape[1] != 1:
                Visualization(test,THRESHOLD,TIME_STEPS,axs[i],test_mae_loss)
            elif df.shape[1] == 1:
                Visualization(test,THRESHOLD,TIME_STEPS,axs,test_mae_loss)   
                
        anomaly[i].drop(col,inplace = True,axis =1)
    return anomaly

def Visualization(test,threshold,TIME_STEPS,axs,test_mae_loss):
    test_mae_loss = test_mae_loss.reshape((-1))
    anomalies = test_mae_loss > threshold
    anomalous_data_indices = []
    for data_idx in range(TIME_STEPS - 1, len(test) - TIME_STEPS + 1):
        if np.all(anomalies[data_idx - int(TIME_STEPS/2) + 1 ]):
            anomalous_data_indices.append(data_idx)
    df_subset = test.iloc[anomalous_data_indices]
    test.plot(legend=False,ax=axs)
    df_subset.plot(legend=False, color="r",ax=axs)
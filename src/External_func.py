
import numpy as np 
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
import warnings
warnings.filterwarnings("ignore")

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
    x_train,y_train = create_dataset(train[[col]],train[col],TIME_STEPS)
    x_test,y_test = create_dataset(test[[col]],test[col],TIME_STEPS)
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
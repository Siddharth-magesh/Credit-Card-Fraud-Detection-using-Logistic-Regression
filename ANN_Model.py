import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

creditcard = pd.read_csv(r'creditcard.csv')
x = creditcard.drop(columns='Class', axis=1)
y = creditcard['Class']
n_features = x.shape[1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = keras.Sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(n_features,)),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation='sigmoid')
])


model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.binary_crossentropy, 
    metrics=['accuracy']
)

model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=128,
    verbose=2
)

y_pred = model.predict(x_test)
y_pred = (y_pred > 0.5).astype(int).flatten()

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

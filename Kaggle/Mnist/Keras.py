import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

mnist = tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()
x_test = tf.keras.utils.normalize(x_test,axis=1)
x_train=tf.keras.utils.normalize(x_train,axis=1)
model= tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=5)

val_loss,val_acc=model.evaluate(x_test,y_test)
print(val_loss,val_acc)
test = pd.read_csv('test.csv')
test = test.iloc[: , :]
X_testr = [28000][28][28]
for k in range(0,28000):
    for i in range(0,28):
        for j in range(0,28):
            X_testr[k][i][j]=test[k : 28*i+j]
            

model.save('keras_model')
new_model= tf.keras.models.load_model('keras_model')

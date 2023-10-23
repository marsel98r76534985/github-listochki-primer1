import tensorflow as tf
from tensorflow import keras
(x_train, y_train),(x_test, y_test)=keras.datasets.cifar10.load_data()
x_train, x_test=x_train / 255.0, x_test/255.0
model=keras.Sequential([
    keras.layers.Flatten(input_shape=(32,32,3)),
    keras.layers.Dense(128,activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10)
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.fit(x_train,y_train,epochs=10)
test_loss,test_acc=model.evaluate(x_test,y_test,verbose=2)
print (f'Точность на тестовых данных:{test_acc}')

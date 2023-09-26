import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense , Dropout , Flatten , MaxPooling2D , Conv2D
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

ap = argparse.ArgumentParser()
ap.add_argument('--mode' , help='train/display')
mode = ap.parse_args().mode 

def plot_history(model_history):
    fig,axs = plt.subplots(1 , 2 , figsize = (15,5))


train_data = 'data/train'
val_data = 'data/test'

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(
    train_data,
    target_size = (48,48),
    batch_size = 64,
    color_mode = 'grayscale', 
    class_mode = 'categorical', 
    )
val_genrator = val_datagen.flow_from_directory(
    val_data,
    target_size=(48,48),
    batch_size = 64 ,
    color_mode= 'grayscale',
    class_mode='categorical',
    )

model = Sequential([
    Conv2D(32 , kernel_size = (3, 3) , activation='relu', input_shape=(48,48,1)),
    Conv2D(64 , kernel_size = (3, 3) , activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    Conv2D(128 , kernel_size=(3, 3) , activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128 , kernel_size=(2, 2) , activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.25),
    
    Flatten(),
    Dense(1024 , activation='relu',) ,
    Dropout(0.5) ,
    Dense(7 , activation = 'softmax')
])

if mode == 'train':
    model.compile(loss='categorical_crossentropy' , optimizer=Adam(lr=0.0001, decay = 1e-6) , metrics=['accuracy'])
    model_info = model.fit_generator(
        train_generator,
        steps_per_epoch=28709//64,
        epochs = 50,
        validation_data=val_genrator, 
        validation_steps= 7178 // 64,
    )
    model.save_weights('model.h5')
    
elif mode == 'test':
    model.load_weights('model.h5')
    cv2.ocl.setUseOpenCL(False)
    
    emotion_cats = {0:'Angry' ,
                    1:'Disgusted' , 
                    2:'Fearful' ,
                    3:'Happy' , 
                    4:'Neutral' ,
                    5:'Sad',
                    6:'Surprised' , 
                    }

    cap = cv2.VideoCapture(0)
    
    while True:
        ret , frame = cap.read()
        if not ret:
            break
        facecasc = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray , scaleFactor=1.3 , minNeighbors=5)
        
        for (x , y , w , h) in faces:
            cv2.rectangle(frame , (x , y-50), (x+w , y+h+10) , (255 , 0 , 0) , 2)
            roi_gray = gray[y:y +h , x:x+w]
            croped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray , (48,48)) , -1), 0)
            prediction = model.predict(croped_img)
            maxindex = int(np.argmax(prediction))
            cv2.putText(frame , emotion_cats[maxindex] , (x+20 , y-60) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (255 , 255 , 255 ) , 2, cv2.LINE_AA)
            
        cv2.imshow('Video' , cv2.resize(frame , (1600 , 960) , interpolation = cv2.INTER_CUBIC))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break 
    cv2.release()
    cv2.destroyAllWindows()
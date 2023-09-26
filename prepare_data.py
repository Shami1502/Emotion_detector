import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import os

def to_int(s):
    n = 0
    for i in s:
        n = n*10 + ord(i) - ord('0')
    return n 

outer_dirs = ['test' , 'train']
inner_dirs = ['angry' , 'disgusted' , 'fearful' , 'happy' , 'sad' , 'surprised' , 'neutral']
os.makedirs('data' , exist_ok=True)

for i in outer_dirs:
    os.makedirs(os.path.join('data' , i) , exist_ok=True)
    for j in inner_dirs:
        os.makedirs(os.path.join('data' , i , j) , exist_ok=True)
        
angry , angry_test = 0 , 0
disgusted , disgusted_test = 0 , 0
fearful , fearful_test = 0 , 0
happy , happy_test = 0 , 0
neutral , neutral_test = 0 , 0
sad , sad_test = 0 , 0
surprised , surprised_test = 0 , 0

df = pd.read_csv('fer2013.csv')
mat = np.zeros((48,48) , dtype = np.uint8)
print('saving images')

for i in tqdm(range(len(df))):
    txt = df['pixels'][i]
    words = txt.split()
    for j in range(2304):
        xind = j//48
        yind = j%48
        mat[xind][yind] = to_int(words[j])
    
    img = Image.fromarray(mat)
    
    if i < 28709:
        if df['emotion'][i] == 0:
            img.save('data/train/angry/im'+str(angry)+'.png')
            angry += 1
        elif df['emotion'][i] == 1:
            img.save('data/train/disgusted/im'+str(disgusted)+'.png')
            disgusted += 1
        elif df['emotion'][i] == 2:
            img.save('data/train/fearful/im'+str(fearful)+'.png')
            fearful += 1
        elif df['emotion'][i] == 3:
            img.save('data/train/happy/im'+str(happy)+'.png')
            happy += 1
        elif df['emotion'][i] == 4:
            img.save('data/train/sad/im'+str(sad)+'.png')
            sad += 1
        elif df['emotion'][i] == 5:
            img.save('data/train/surprised/im'+str(surprised)+'.png')
            surprised += 1
        elif df['emotion'][i] == 6:
            img.save('data/train/neutral/im'+str(neutral)+'.png')
            neutral += 1       
    else:
        if df['emotion'][i] == 0:
            img.save('data/test/angry/im'+str(angry_test)+'.png')
            angry += 1
        elif df['emotion'][i] == 1:
            img.save('data/test/disgusted/im'+str(disgusted_test)+'.png')
            disgusted += 1
        elif df['emotion'][i] == 2:
            img.save('data/test/fearful/im'+str(fearful_test)+'.png')
            fearful += 1
        elif df['emotion'][i] == 3:
            img.save('data/test/happy/im'+str(happy_test)+'.png')
            happy += 1
        elif df['emotion'][i] == 4:
            img.save('data/test/sad/im'+str(sad_test)+'.png')
            sad += 1
        elif df['emotion'][i] == 5:
            img.save('data/test/surprised/im'+str(surprised_test)+'.png')
            surprised += 1
        elif df['emotion'][i] == 6:
            img.save('data/test/neutral/im'+str(neutral_test)+'.png')
            neutral += 1 
print('Done!')
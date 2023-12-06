from statistics import mode
from tkinter.tix import COLUMN
import cv2 as cv
from matplotlib import pyplot as plt 
import os,shutil
import numpy as np 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
"""
etiketleme


target = ['CLL','FL','MCL']
a=0
for i in target:   
    path = 'C:\\Users\\Pc\\Downloads\\archive\\'+i
    print(i)
    dosyalar= os.listdir(path)
    for file in dosyalar:
        print(file)
        a=a+1

print(a)
"""
import tensorflow as tf
"""
train_data=ImageDataGenerator(rescale=1/255)
train_dataset=train_data.flow_from_directory('C:\\Users\\Pc\\Downloads\\archive\\', target_size=(150,150), batch_size = 10, class_mode= 'categorical')
print(train_dataset.class_indices)
print(train_dataset.classes)
"""
"""

Etiketleme

categories= ['CLL', 'FL', 'MCL']
dir='C:\\Users\\Pc\\Downloads\\archive\\'
data=[]

for category in categories:
    path=os.path.join(dir,category)
    label= categories.index(category)
    for img6 in os.listdir(path):
        imgpath=os.path.join(path,img6)
        image1=cv.imread(imgpath,0)
        image1 = cv.resize(image1, (700,700))
        img_array=np.array(image1).flatten()
        print(img_array,"                      ",label)
        data.append([img_array,label])
print(len(data))

features=[]
labels=[]
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
for feature,label in data:
    features.append(feature)
    labels.append(label)
"""
"""
Resimleri resize

resim3 =cv.imread("C:/Users/Pc/Downloads/archive/CLL/sj-03-476_007.tif")
resim3= cv.resize(resim3, (600,600))
resim4 =cv.imread("C:/Users/Pc/Downloads/archive/CLL/sj-03-476_007.tif",cv.COLOR_BGR2RGB)
resim4= cv.resize(resim4, (600,600))
"""
from skimage import transform
"""
K-MEANS KÜMELEME VE MASKELEME 


transformed = transform.rotate(resim4, angle=90)



pixel_vals = resim4[:,:,0].reshape((-1,1))

# Convert to float type
pixel_vals = np.float32(pixel_vals)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1.0)
 
# then perform k-means clustering wit h number of clusters defined as 3
#also random centres are initially choosed for k-means clustering
k = 3
retval, labels, centers = cv.kmeans(pixel_vals, k, None, criteria, 10, cv.KMEANS_PP_CENTERS)
 
# convert data into 8-bit values
centers = np.uint8(centers)
segmented_data = centers[labels.flatten()]
 
# reshape data into the original image dimensions
segmented_image = segmented_data.reshape((resim4[:,:,0].shape))

ret3,th3 = cv.threshold(segmented_image,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
cv.imshow("th3",th3)

cv.waitKey(0)
cv.destroyAllWindows()

plt.imshow(segmented_image)
plt.show()
print("deneme boyut", segmented_image.shape)

result = cv.bitwise_and(resim3, resim3, mask=th3)
print("pikseller", result)
cv.imshow("maskeli",result)


cv.waitKey(0)
cv.destroyAllWindows()
"""

"""
Eğitim

x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.20, random_state=10)

print(y_train)
print(y_test)
print(len(y_train))
print(len(y_test))


from sklearn.naive_bayes import GaussianNB
print("deneme")
naive_bayes=GaussianNB()
print("deneme1")
naive_bayes.fit(x_train, y_train)
print("deneme2")
prediction3=naive_bayes.predict(x_test)
print("deneme3")
accuracy3=naive_bayes.score(x_test,y_test)
print("deneme4")
print('naive bayes Accuracy: ', accuracy3)

"""


from PIL import Image, ImageStat
"""
im= Image.open('C:/Users/Pc/Downloads/archive/CLL/sj-05-1396-R3_006.tif')
stat = ImageStat.Stat(im)
print(stat.mean)
"""
"""
categories= ['CLL', 'FL', 'MCL']
dir='C:\\Users\\Pc\\Downloads\\archive\\'
data=[]

for category in categories:
    path=os.path.join(dir,category)
    label= categories.index(category)
    for img6 in os.listdir(path):
        imgpath=os.path.join(path,img6)
        image1=Image.open(imgpath)
        stat = ImageStat.Stat(image1)
        print(stat.mean)
        img_array=np.array(image1).flatten()
        print(img_array,"                      ",label)
        data.append([img_array,label])
print(len(data))

"""

def calcEntropy(img):
    entropy = []

    hist = cv.calcHist([img], [0], None, [256], [0, 255])
    total_pixel = img.shape[0] * img.shape[1]

    for item in hist:
        probability = item / total_pixel
        if probability == 0:
            en = 0
        else:
            en = -1 * probability * (np.log(probability) / np.log(2))
        entropy.append(en)

    sum_en = np.sum(entropy)
    return sum_en

def calcEntropy_R(img):
    entropy = []

    hist = cv.calcHist([img[:,:,0]], [0], None, [256], [0, 255])
    total_pixel = img.shape[0] * img.shape[1]

    for item in hist:
        probability = item / total_pixel
        if probability == 0:
            en = 0
        else:
            en = -1 * probability * (np.log(probability) / np.log(2))
        entropy.append(en)

    sum_en = np.sum(entropy)
    return sum_en

def calcEntropy_G(img):
    entropy = []

    hist = cv.calcHist([img[:,:,1]], [0], None, [256], [0, 255])
    total_pixel = img.shape[0] * img.shape[1]

    for item in hist:
        probability = item / total_pixel
        if probability == 0:
            en = 0
        else:
            en = -1 * probability * (np.log(probability) / np.log(2))
        entropy.append(en)

    sum_en = np.sum(entropy)
    return sum_en

def calcEntropy_B(img):
    entropy = []

    hist = cv.calcHist([img[:,:,2]], [0], None, [256], [0, 255])
    total_pixel = img.shape[0] * img.shape[1]

    for item in hist:
        probability = item / total_pixel
        if probability == 0:
            en = 0
        else:
            en = -1 * probability * (np.log(probability) / np.log(2))
        entropy.append(en)

    sum_en = np.sum(entropy)
    return sum_en


"""
img2 = cv.imread("C:/Users/Pc/Downloads/archive/CLL/sj-05-1396-R3_006.tif", cv.IMREAD_GRAYSCALE)

entropy2 = calcEntropy(img2)
print(entropy2)
"""


"""

ASIL KODLAR 




categories= ['CLL', 'FL', 'MCL']
dir='C:\\Users\\Pc\\Downloads\\archive\\'
data=[]
entropy=[]
x=1
while x<3:
    x=x+1
    for category in categories:
        path=os.path.join(dir,category)
        label= categories.index(category)
        for img6 in os.listdir(path):
            imgpath=os.path.join(path,img6)
            image2=Image.open(imgpath)
            stat = ImageStat.Stat(image2)
            print(stat.mean)
            image1=cv.imread(imgpath, cv.IMREAD_GRAYSCALE)
            entropy2 = calcEntropy(image1)
            image2=cv.imread(imgpath,cv.COLOR_BGR2RGB)
            entropy3 = calcEntropy_R(image2)
            entropy4 = calcEntropy_G(image2)
            entropy5 = calcEntropy_B(image2)
            img_array=np.array(image1).flatten()
            print(entropy2)
            print(img_array,"                      ",label)
            entropy.append([entropy2,entropy3,entropy4,entropy5,stat.mean,label])
            data.append([img_array,label])
            
print(len(data))

features=[]
labels=[]
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
for feature,entropy,entropy1,entropy2,feature1,label in entropy:
    features.append([feature,entropy,entropy1,entropy2,feature1[0],feature1[1],feature1[2]])
    print(feature," ",feature1)
    labels.append(label)

x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.20, random_state=10)

print(y_train)
print(y_test)
print(len(y_train))
print(len(y_test))


from sklearn.naive_bayes import GaussianNB, MultinomialNB
print("deneme")
naive_bayes=MultinomialNB()
print("deneme1")
naive_bayes.fit(x_train, y_train)
print("deneme2")
prediction3=naive_bayes.predict(x_test)
print("deneme3")
accuracy3=naive_bayes.score(x_test,y_test)
print("deneme4")
print('naive bayes Accuracy: ', accuracy3)

from sklearn.ensemble import RandomForestClassifier
randomfc= RandomForestClassifier(n_estimators=30, random_state=10)
randomfc.fit(x_train, y_train)
prediction2=randomfc.predict(x_test)
accuracy2=randomfc.score(x_test,y_test)

print('random forest Accuracy: ', accuracy2)
print('random forest Prediction: ', categories[prediction2[60]])
print("etiketi: ", y_test[60])



model=SVC(C=7) 
model.fit(x_train, y_train)

prediction=model.predict(x_test)
accuracy=model.score(x_test,y_test)

print('SVC Accuracy: ', accuracy)
print('SVC Prediction: ', categories[prediction[60]])

"""

"""

RGB ORTALAMALARINI ALMA 

image = cv.imread('C:/Users/Pc/Downloads/archive/CLL/sj-05-1396-R3_006.tif', cv.COLOR_BGR2RGB)
image = cv.resize(image, (600,600))
feature_matrix = np.zeros((600,600)) 
print(feature_matrix.shape)
for i in range(0,image.shape[0]):
    for j in range(0,image.shape[1]):
        feature_matrix[i][j] = ((int(image[i,j,0]) + int(image[i,j,1]) + int(image[i,j,2]))/3)
features = np.reshape(feature_matrix, (600*600)) 
print(features.shape)
print(features)

"""

"""

ÖNEMLİ 2 


feature_matrix = np.zeros((600,600)) 
categories= ['CLL', 'FL', 'MCL']
dir='C:\\Users\\Pc\\Downloads\\archive\\'
data=[]
entropy=[]
x=1
while x<3:
    x=x+1
    for category in categories:
        path=os.path.join(dir,category)
        label= categories.index(category)
        for img6 in os.listdir(path):
            imgpath=os.path.join(path,img6)
            image1=cv.imread(imgpath, cv.COLOR_BGR2RGB)
            image = cv.resize(image1, (600,600))
            for i in range(0,image.shape[0]):
                for j in range(0,image.shape[1]):
                    feature_matrix[i][j] = ((int(image[i,j,0]) + int(image[i,j,1]) + int(image[i,j,2]))/3)
            features = np.reshape(feature_matrix, (600*600))
            img_array=np.array(features).flatten()
            print(img_array,"                      ",label)
            data.append([img_array,label])
            
print(len(data))

features=[]
labels=[]
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
for feature,label in data:
    features.append([feature,label])
    print(feature, "  ", label)
    labels.append(label)
"""


"""
categories= ['CLL', 'FL', 'MCL']
dir='C:\\Users\\Pc\\Downloads\\archive\\'
data=[]
features1=[]
labels=[]
feature_matrix = np.zeros((400,400)) 
x=1
while x<3:
    x=x+1
    for category in categories:
        path=os.path.join(dir,category)
        label= categories.index(category)
        for img6 in os.listdir(path):
            imgpath=os.path.join(path,img6)
            image2=cv.imread(imgpath,cv.COLOR_BGR2RGB)
            image = cv.resize(image2, (400,400))
            for i in range(0,image.shape[0]):
                for j in range(0,image.shape[1]):
                    feature_matrix[i][j] = round((int(image[i,j,0]) + int(image[i,j,1]) + int(image[i,j,2]))/3)
            img_array=np.array(feature_matrix).flatten()
            print(img_array,"                      ",label)
            data.append([img_array,label])
            
print(len(data))
for feature,label in data:
    features1.append(feature)
    labels.append(label)

image = cv.imread('C:/Users/Pc/Downloads/archive/CLL/sj-05-1396-R3_006.tif', cv.COLOR_BGR2RGB)

image = cv.resize(image, (400,400))
feature_matrix = np.zeros((400,400)) 
print(feature_matrix.shape)
for i in range(0,image.shape[0]):
    for j in range(0,image.shape[1]):
        feature_matrix[i][j] = ((int(image[i,j,0]) + int(image[i,j,1]) + int(image[i,j,2]))/3)
        feature_matrix[i][j] = int(feature_matrix[i][j])
features = np.reshape(feature_matrix, (400*400)) 
print(features.shape)
print(features)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(features1, labels, test_size=0.20, random_state=10)

from sklearn.ensemble import RandomForestClassifier
randomfc= RandomForestClassifier(n_estimators=30, random_state=10)
randomfc.fit(x_train, y_train)
prediction2=randomfc.predict(x_test)
accuracy2=randomfc.score(x_test,y_test)

print('random forest Accuracy: ', accuracy2)
print('random forest Prediction: ', categories[prediction2[60]])
print("etiketi: ", y_test[60])
"""

from PIL import Image
 
# Giving The Original image Directory
# Specified

import imutils
"""
Döndürülmüş görsel gösterimi


resim = cv.imread("C:/Users/Pc/Downloads/archive/CLL/sj-05-1396-R3_006.tif")
resim = cv.resize(resim, (400,400))
 
Rotated_image = imutils.rotate(resim, angle=90)
Rotated1_image = imutils.rotate(resim, angle=180)
Rotated2_image = imutils.rotate(resim, angle=270)
# display the image using OpenCV of
# angle 45
plt.subplot(1,3,1)
plt.imshow(Rotated_image,cmap=plt.cm.gray)
plt.subplot(1,3,2)
plt.imshow(Rotated1_image,cmap=plt.cm.gray)
plt.subplot(1,3,3)
plt.imshow(Rotated2_image,cmap=plt.cm.gray)
plt.show()
"""


"""
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression


model = LogisticRegression()
rfe = RFE(model, 100)
fit = rfe.fit(x_train, y_train)
print("Num Features: %s" % (fit.n_features_))
print("Selected Features: %s" % (fit.support_))
print("Feature Ranking: %s" % (fit.ranking_))

accuracy2=rfe.score(x_test,y_test)

print('random forest Accuracy: ', accuracy2)
"""


"""

Maske ve AÇI


angle =[90,180,270,360]
categories= ['CLL', 'FL', 'MCL']
dir='C:\\Users\\Pc\\Downloads\\archive\\'
entropy10=[]
for category in categories:
        path=os.path.join(dir,category)
        label= categories.index(category)
        for img6 in os.listdir(path):
            imgpath=os.path.join(path,img6)                       
            image1=cv.imread(imgpath, cv.COLOR_BGR2RGB)
            image1 = cv.resize(image1, (400,400))
            for x in angle:
                Rotated = imutils.rotate(image1, angle=x)
                
                transformed = transform.rotate(Rotated, angle=90)
                pixel_vals = Rotated[:,:,0].reshape((-1,1))
                pixel_vals = np.float32(pixel_vals)
                criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.85)
                k = 3
                retval, labels, centers = cv.kmeans(pixel_vals, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
                centers = np.uint8(centers)
                segmented_data = centers[labels.flatten()]
                segmented_image = segmented_data.reshape((Rotated[:,:,0].shape))
                ret3,th3 = cv.threshold(segmented_image,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
                result = cv.bitwise_and(Rotated, Rotated, mask=th3)
                img_array=np.array(result).flatten()
                print(img_array)
                entropy10.append([img_array,label])

print(len(entropy10))
print(np.ndim(entropy10))
features=[]
labels=[]
from sklearn.model_selection import train_test_split

for feature,label in entropy10:
    features.append(feature)
    labels.append(label)

x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.30, random_state=10)

print(y_train)
print(y_test)
print(len(y_train))
print(len(y_test))

from sklearn.ensemble import RandomForestClassifier
randomfc= RandomForestClassifier()
randomfc.fit(x_train, y_train)
prediction2=randomfc.predict(x_test)
accuracy2=randomfc.score(x_test,y_test)
from sklearn import metrics
print('random forest Accuracy: ', accuracy2)
print('random forest Accuracy1: ', metrics.accuracy_score(y_test,prediction2))
print('random forest Accuracy2: ', metrics.confusion_matrix(y_test,prediction2))
print('random forest Prediction: ', categories[prediction2[60]])
print("etiketi: ", y_test[60])
"""
import pandas as pd

import pywt
"""
 
ROTASYONLA DWT
angle =[90,180,270,360]
categories= ['CLL', 'FL', 'MCL']
dir='C:\\Users\\Pc\\Downloads\\archive\\'
entropy10=[]
for category in categories:
        path=os.path.join(dir,category)
        label= categories.index(category)
        for img6 in os.listdir(path):
            imgpath=os.path.join(path,img6)                       
            image1=cv.imread(imgpath, cv.IMREAD_GRAYSCALE)
            image1 = cv.resize(image1, (400,400))
            image2=cv.imread(imgpath,cv.COLOR_BGR2RGB)
            image2 = cv.resize(image2, (400,400))
            for x in angle:
                Rotated = imutils.rotate(image1, angle=x)
                Rotated2 = imutils.rotate(image2, angle=x)

                deneme2 = pywt.dwt2(Rotated, 'db3', mode='periodization')

                cA,(cH,cV,cD) = deneme2
                print(cA.shape)
                print(cA)
                deneme3 = pywt.idwt2(deneme2, 'db3', mode='periodization')
                img= np.uint8(deneme3)
                img_array=np.array(img).flatten()
                print(img_array)
                entropy10.append([img_array,label])


print(len(entropy10))
print(np.ndim(entropy10))
features=[]
labels=[]
from sklearn.model_selection import train_test_split

for feature,label in entropy10:
    features.append(feature)
    labels.append(label)

x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.30, random_state=10)

print(y_train)
print(y_test)
print(len(y_train))
print(len(y_test))

from sklearn.ensemble import RandomForestClassifier
randomfc= RandomForestClassifier()
randomfc.fit(x_train, y_train)
prediction2=randomfc.predict(x_test)
accuracy2=randomfc.score(x_test,y_test)
from sklearn import metrics
print('random forest Accuracy: ', accuracy2)
print('random forest Accuracy1: ', metrics.accuracy_score(y_test,prediction2))
print('random forest Accuracy2: ', metrics.confusion_matrix(y_test,prediction2))
"""


"""
#GLCM hesabı feture extraction

from skimage.feature import greycomatrix, greycoprops
def calc_glcm_all_agls(img, label, props, dists=[5], agls=[0, np.pi/4, np.pi/2, 3*np.pi/4], lvl=256, sym=True, norm=True):
    
    glcm = greycomatrix(img, 
                        distances=dists, 
                        angles=agls, 
                        levels=lvl,
                        symmetric=sym, 
                        normed=norm)
    feature = []
    glcm_props = [propery for name in props for propery in greycoprops(glcm, name)[0]]
    for item in glcm_props:
            feature.append(item)
    feature.append(label) 
    
    return feature
entropy3=[]
categories= ['CLL', 'FL', 'MCL']
dir='C:\\Users\\Pc\\Downloads\\archive\\'
entropy10=[]

for category in categories:
        path=os.path.join(dir,category)
        label= categories.index(category)
        for img6 in os.listdir(path):
            imgpath=os.path.join(path,img6)                       
            image1=cv.imread(imgpath, cv.IMREAD_GRAYSCALE)
            image1 = cv.resize(image1, (600,600))
            equalized1 = cv.equalizeHist(image1)
            entropy2 = calcEntropy(equalized1)
            entropy3.append(entropy2)
            entropy10.append([equalized1,label])

for category in categories:
        path=os.path.join(dir,category)
        label= categories.index(category)
        for img6 in os.listdir(path):
            imgpath=os.path.join(path,img6)                       
            image1=cv.imread(imgpath, cv.IMREAD_GRAYSCALE)
            image1 = cv.resize(image1, (600,600))
            entropy2 = calcEntropy(image1)
            entropy3.append(entropy2)
            entropy10.append([image1,label])


for category in categories:
        path=os.path.join(dir,category)
        label= categories.index(category)
        for img6 in os.listdir(path):
            imgpath=os.path.join(path,img6)                       
            image1=cv.imread(imgpath, cv.IMREAD_GRAYSCALE)
            image1 = cv.resize(image1, (600,600))
            clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(6, 6))
            equalized = clahe.apply(image1)
            entropy2 = calcEntropy(equalized)
            entropy3.append(entropy2)
            entropy10.append([equalized,label])

print(len(entropy10))
print(np.ndim(entropy10))
features=[]
labels=[]
from sklearn.model_selection import train_test_split

for feature,label in entropy10:
    features.append(feature)
    labels.append(label)
properties = ['dissimilarity', 'correlation', 'homogeneity', 'contrast', 'ASM', 'energy']

glcm_all_agls = []

for img, label1 in zip(features, labels): 
    glcm_all_agls.append(
            calc_glcm_all_agls(img, 
                                label1, 
                                props=properties)
                            )
 
columns = []
angles = ['0', '45', '90','135']
for name in properties :
    for ang in angles:
        columns.append(name + "_" + ang)
        
columns.append("label")

import pandas as pd 

# Create the pandas DataFrame for GLCM features data

glcm_df = pd.DataFrame(glcm_all_agls, 
                      columns = columns)

#glcm_df["entropy"] = entropy3.copy()
glcm_df.head(15)

print(glcm_df.head(200))

#Bu ya da   print(glcm_df.iloc[:,-1])

print(glcm_df.loc[:,"label"])
labelss=glcm_df.loc[:,"label"]
#three= glcm_df.pop("label")
glcm_df.drop('label',axis=1, inplace=True)
print(glcm_df)

#Bu da olur ->  glcm_df.iloc[:,:]
#glcm_df.filter(regex="[^label]")
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(abs(glcm_df), labelss, test_size=0.20, random_state=10)

print(y_train)
print(y_test)
print(len(y_train))
print(len(y_test))

from sklearn.ensemble import RandomForestClassifier
randomfc= RandomForestClassifier()
randomfc.fit(x_train, y_train)
prediction2=randomfc.predict(x_test)
accuracy2=randomfc.score(x_test,y_test)
from sklearn import metrics
print('random forest Accuracy: ', accuracy2)
print('random forest Accuracy1: ', metrics.accuracy_score(y_test,prediction2))
print('random forest Accuracy2: ', metrics.confusion_matrix(y_test,prediction2))


from sklearn.naive_bayes import GaussianNB, MultinomialNB
print("deneme")
naive_bayes=GaussianNB()
print("deneme1")
naive_bayes.fit(x_train, y_train)
print("deneme2")
prediction3=naive_bayes.predict(x_test)
print("deneme3")
accuracy3=naive_bayes.score(x_test,y_test)
print("deneme4")
print('naive bayes Accuracy: ', accuracy3)
print('random forest Accuracy1: ', metrics.accuracy_score(y_test,prediction3))
print('random forest Accuracy2: ', metrics.confusion_matrix(y_test,prediction3))

"""
"""


"""
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # dimesion reduction
        rotation_range=5,  # randomly rotate images in the range 5 degrees
        zoom_range = 0.1, # Randomly zoom image 10%
        width_shift_range=0.1,  # randomly shift images horizontally 10%
        height_shift_range=0.1,  # randomly shift images vertically 10%
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(x_train)

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
#history = model.fit_generator(x_train,y_train,batch_size=120,epochs=15)
history = model.fit(x_train, y_train,batch_size=120,epochs=10,validation_split=0.1)

test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)
print("Accuracy: ", test_acc)

model_feat = Model(inputs=model.input,outputs=model.get_layer('flatten').output)

feat_train = model_feat.predict(x_train)
print(feat_train.shape)

feat_test = model_feat.predict(x_test)
print(feat_test.shape)

from sklearn.svm import SVC

svm = SVC(kernel='rbf')

svm.fit(feat_train,y_train)

print('fitting done !!!')

score=svm.score(feat_test,y_test)

print(score)
"""

"""


VGG16 ile olan



categories= ['CLL', 'FL', 'MCL']
dir='C:\\Users\\Pc\\Downloads\\archive\\'
labels1=[]
features1=[]
for category in categories:
        path=os.path.join(dir,category)
        label= categories.index(category)
        for img6 in os.listdir(path):
            imgpath=os.path.join(path,img6)                       
            image1=cv.imread(imgpath, cv.IMREAD_COLOR)
            image1 = cv.resize(image1, (227,227))
            image1=cv.cvtColor(image1, cv.COLOR_RGB2BGR)
            features1.append(image1/255)
            labels1.append(label)
                
                

from sklearn.model_selection import train_test_split


features= np.array(features1)
labels =np.array(labels1)

x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.20, random_state=10)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

print(y_train)
print(y_test)
print(len(x_train))
print(len(x_test))
print(len(y_train))
print(len(y_test))



import glob

from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from keras.layers.normalization import BatchNormalization

from keras.applications.vgg16 import VGG16
#from keras.applications.vgg19 import VGG19
from keras.utils import to_categorical


VGG_Model = VGG16(weights='imagenet', include_top=False, input_shape=(300,300,3))

for layer in VGG_Model.layers:
    layer.trainable = False

VGG_Model.summary()



feature_extractor =VGG_Model.predict(x_train)
print(feature_extractor.shape[0])
print(feature_extractor.shape[1])
print(np.ndim(feature_extractor))

features11 = feature_extractor.reshape(feature_extractor.shape[0],-1)

X_for_RF = features11
print(X_for_RF)
print("X_for_RF:", np.ndim(X_for_RF))
print(np.ndim(y_train))
print(feature_extractor.shape[0])
print(feature_extractor.shape[1])
tablo = pd.DataFrame(X_for_RF)
print(tablo)
"""

"""
class convers_pca():
    def __init__(self, no_of_components):
        self.no_of_components = no_of_components
        self.eigen_values = None
        self.eigen_vectors = None
        
    def transform(self, x):
        return np.dot(x - self.mean, self.projection_matrix.T)
    
    def inverse_transform(self, x):
        return np.dot(x, self.projection_matrix) + self.mean
    
    def fit(self, x):
        self.no_of_components = x.shape[1] if self.no_of_components is None else self.no_of_components
        self.mean = np.mean(x, axis=0)
        
        cov_matrix = np.cov(x - self.mean, rowvar=False)
        
        self.eigen_values, self.eigen_vectors = np.linalg.eig(cov_matrix)
        self.eigen_vectors = self.eigen_vectors.T
        
        self.sorted_components = np.argsort(self.eigen_values)[::-1]
        
        self.projection_matrix = self.eigen_vectors[self.sorted_components[:self.no_of_components]]self.explained_variance = self.eigen_values[self.sorted_components]
        self.explained_variance_ratio = self.explained_variance / self.eigen_values.sum()

from sklearn.preprocessing import StandardScaler
std = StandardScaler()
transformed = StandardScaler().fit_transform(tablo.values)

pca = convers_pca(no_of_components=100)
pca.fit(transformed)
"""
"""
from sklearn.ensemble import RandomForestClassifier
randomfc= RandomForestClassifier(n_estimators=50, random_state=42)
randomfc.fit(X_for_RF, y_train)
x_test_feature = VGG_Model.predict(x_test)
X_test_features = x_test_feature.reshape(x_test_feature.shape[0], -1)
print("X_test_features: ", X_test_features)
prediction2=randomfc.predict(X_test_features)
#accuracy2=randomfc.score(x_test,y_test)
from sklearn import metrics
#print('random forest Accuracy: ', accuracy2)
print('random forest Accuracy1: ', metrics.accuracy_score(y_test,prediction2))
print('random forest Accuracy2: ', metrics.confusion_matrix(y_test,prediction2))
"""

"""
Pre-Trained   Transfer Learning

VGG16 + PCA() + ML -> ASIL KODLAR

angle =[90,180,270,360]
#categories= ['CLL', 'FL', 'MCL']
categories= ['MCL', 'CLL']
dir='C:\\Users\\Pc\\Downloads\\archive\\'
labels1=[]
features1=[]
for category in categories:
        path=os.path.join(dir,category)
        label= categories.index(category)
        for img6 in os.listdir(path):
            imgpath=os.path.join(path,img6)                       
            image1=cv.imread(imgpath, cv.IMREAD_COLOR)
            image1 = cv.resize(image1, (300,300))
            for x in angle:
                Rotated = imutils.rotate(image1, angle=x)
            #image1=cv.cvtColor(image1, cv.COLOR_RGB2BGR)
                features1.append(Rotated/255)
                labels1.append(label)
                
                

from sklearn.model_selection import train_test_split


features= np.array(features1)
labels =np.array(labels1)

x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.20)

# , random_state=10
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

print(y_train)
print(y_test)
print(len(x_train))
print(len(x_test))
print(len(y_train))
print(len(y_test))



import glob

from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D,Input
from keras.layers.normalization import BatchNormalization

from keras.applications.vgg16 import VGG16
#from keras.applications.resnet50 import ResNet50
#from keras.applications.densenet import DenseNet201
#from keras.applications.mobilenet import MobileNet

#from keras.applications.vgg19 import VGG19
from keras.utils import to_categorical


VGG_Model = VGG16(weights='imagenet', include_top=False, input_shape=(300,300,3))

for layer in VGG_Model.layers:
    layer.trainable = False

VGG_Model.summary()



train_feature_extractor =VGG_Model.predict(x_train)
print(train_feature_extractor.shape[0])
print(train_feature_extractor.shape[1])
print(np.ndim(train_feature_extractor))

train_features = train_feature_extractor.reshape(train_feature_extractor.shape[0],-1)

test_feature_extractor=VGG_Model.predict(x_test)
test_features = test_feature_extractor.reshape(test_feature_extractor.shape[0], -1)

from sklearn.decomposition import PCA


pca_test = PCA(n_components=0.4) 
pca_test.fit(train_features)


#n_PCA_components = 20
#pca = PCA(n_components=n_PCA_components)
pca = PCA()
#31 seçildiğinde en iyi sonucu verdi.
train_PCA = pca.fit_transform(train_features)
test_PCA = pca.transform(test_features) #Make sure you are just transforming, not fitting. 

from sklearn.ensemble import RandomForestClassifier
RF_model = RandomForestClassifier(n_estimators = 100)

# , random_state = 10  -> eliydi çıkardım.
# Train the model on training data
RF_model.fit(train_PCA, y_train) #For sklearn no one hot encoding

prediction_RF = RF_model.predict(test_PCA)
#Inverse le transform to get original label back. 
#Print overall accuracy
from sklearn import metrics
print ("RF Accuracy = ", metrics.accuracy_score(y_test, prediction_RF))
#Confusion Matrix - verify accuracy of each class
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, prediction_RF)
print(cm)
report = metrics.classification_report(y_test, prediction_RF)
print(report)



from sklearn.neighbors import KNeighborsClassifier
knnmodel= KNeighborsClassifier(n_neighbors=5, metric='minkowski')
#ya da 10 yaz -> daha önce öyle yaptım.

knnmodel.fit(train_PCA, y_train)
prediction_knnmodel = knnmodel.predict(test_PCA)
print ("KNN Accuracy = ", metrics.accuracy_score(y_test, prediction_knnmodel))
#Confusion Matrix - verify accuracy of each class
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, prediction_knnmodel)
print(cm)
report = metrics.classification_report(y_test, prediction_knnmodel)
print(report)



from sklearn.naive_bayes import GaussianNB
naive_bayes=GaussianNB()
naive_bayes.fit(train_PCA, y_train)
prediction_naivebayes = naive_bayes.predict(test_PCA)

print ("Naive Bayes Accuracy = ", metrics.accuracy_score(y_test, prediction_naivebayes))
#Confusion Matrix - verify accuracy of each class
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, prediction_naivebayes)
print(cm)
report = metrics.classification_report(y_test, prediction_naivebayes)
print(report)



from sklearn.tree import DecisionTreeClassifier
decision= DecisionTreeClassifier()
decision.fit(train_PCA, y_train)
prediction_decision = decision.predict(test_PCA)
print ("Decision Tree Accuracy = ", metrics.accuracy_score(y_test, prediction_decision))
#Confusion Matrix - verify accuracy of each class
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, prediction_decision)
print(cm)
report = metrics.classification_report(y_test, prediction_decision)
print(report)
"""

#1221. satırdaydı.
"""
plt.plot(np.cumsum(pca_test.explained_variance_ratio_))
plt.xlabel("Number of components")
plt.ylabel("Cum variance")
plt.show()
print(pca_test.n_components_)
"""

"""

#GLCM + PCA()  -> Texture Feature  (Test_size=0.2)

from sklearn.decomposition import PCA
from skimage.feature import greycomatrix, greycoprops
def calc_glcm_all_agls(img, label, props, dists=[5], agls=[0, np.pi/4, np.pi/2, 3*np.pi/4], lvl=256, sym=True, norm=True):
    
    glcm = greycomatrix(img, 
                        distances=dists, 
                        angles=agls, 
                        levels=lvl,
                        symmetric=sym, 
                        normed=norm)
    feature = []
    glcm_props = [propery for name in props for propery in greycoprops(glcm, name)[0]]
    for item in glcm_props:
            feature.append(item)
    feature.append(label) 
    
    return feature
entropy3=[]
categories= ['FL','CLL']
dir='C:\\Users\\Pc\\Downloads\\archive\\'
entropy10=[]

for category in categories:
        path=os.path.join(dir,category)
        label= categories.index(category)
        for img6 in os.listdir(path):
            imgpath=os.path.join(path,img6)                       
            image1=cv.imread(imgpath, cv.IMREAD_GRAYSCALE)
            image1 = cv.resize(image1, (400,400))
            equalized1 = cv.equalizeHist(image1)
            entropy2 = calcEntropy(equalized1)
            entropy3.append(entropy2)
            entropy10.append([equalized1,label])

for category in categories:
        path=os.path.join(dir,category)
        label= categories.index(category)
        for img6 in os.listdir(path):
            imgpath=os.path.join(path,img6)                       
            image1=cv.imread(imgpath, cv.IMREAD_GRAYSCALE)
            image1 = cv.resize(image1, (400,400))
            entropy2 = calcEntropy(image1)
            entropy3.append(entropy2)
            entropy10.append([image1,label])


for category in categories:
        path=os.path.join(dir,category)
        label= categories.index(category)
        for img6 in os.listdir(path):
            imgpath=os.path.join(path,img6)                       
            image1=cv.imread(imgpath, cv.IMREAD_GRAYSCALE)
            image1 = cv.resize(image1, (400,400))
            clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(6, 6))
            equalized = clahe.apply(image1)
            entropy2 = calcEntropy(equalized)
            entropy3.append(entropy2)
            entropy10.append([equalized,label])

print(len(entropy10))
print(np.ndim(entropy10))
features=[]
labels=[]
from sklearn.model_selection import train_test_split

for feature,label in entropy10:
    features.append(feature)
    labels.append(label)
properties = ['dissimilarity', 'correlation', 'homogeneity', 'contrast', 'ASM', 'energy']

glcm_all_agls = []

for img, label1 in zip(features, labels): 
    glcm_all_agls.append(
            calc_glcm_all_agls(img, 
                                label1, 
                                props=properties)
                            )
 
columns = []
angles = ['0', '45', '90','135']
for name in properties :
    for ang in angles:
        columns.append(name + "_" + ang)
        
columns.append("label")

import pandas as pd 

# Create the pandas DataFrame for GLCM features data

glcm_df = pd.DataFrame(glcm_all_agls, 
                      columns = columns)

#glcm_df["entropy"] = entropy3.copy()
glcm_df.head(15)

print(glcm_df.head(100))

#Bu ya da   print(glcm_df.iloc[:,-1])

print(glcm_df.loc[:,"label"])
labelss=glcm_df.loc[:,"label"]
#three= glcm_df.pop("label")
glcm_df.drop('label',axis=1, inplace=True)
print(glcm_df)

#Bu da olur ->  glcm_df.iloc[:,:]
#glcm_df.filter(regex="[^label]")
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(abs(glcm_df), labelss, test_size=0.20, random_state=10)

print(y_train)
print(y_test)
print(len(y_train))
print(len(y_test))

pca = PCA()
#31 seçildiğinde en iyi sonucu verdi.
train_PCA = pca.fit_transform(x_train)
test_PCA = pca.transform(x_test) 

from sklearn.ensemble import RandomForestClassifier
RF_model = RandomForestClassifier(n_estimators = 100)

# , random_state = 10  -> eliydi çıkardım.
# Train the model on training data
RF_model.fit(train_PCA, y_train) #For sklearn no one hot encoding

prediction_RF = RF_model.predict(test_PCA)
#Inverse le transform to get original label back. 
#Print overall accuracy
from sklearn import metrics
print ("RF Accuracy = ", metrics.accuracy_score(y_test, prediction_RF))
#Confusion Matrix - verify accuracy of each class
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, prediction_RF)
print(cm)
report = metrics.classification_report(y_test, prediction_RF)
print(report)



from sklearn.neighbors import KNeighborsClassifier
knnmodel= KNeighborsClassifier(n_neighbors=5, metric='minkowski')
#ya da 10 yaz -> daha önce öyle yaptım.

knnmodel.fit(train_PCA, y_train)
prediction_knnmodel = knnmodel.predict(test_PCA)
print ("KNN Accuracy = ", metrics.accuracy_score(y_test, prediction_knnmodel))
#Confusion Matrix - verify accuracy of each class
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, prediction_knnmodel)
print(cm)
report = metrics.classification_report(y_test, prediction_knnmodel)
print(report)



from sklearn.naive_bayes import GaussianNB
naive_bayes=GaussianNB()
naive_bayes.fit(train_PCA, y_train)
prediction_naivebayes = naive_bayes.predict(test_PCA)

print ("Naive Bayes Accuracy = ", metrics.accuracy_score(y_test, prediction_naivebayes))
#Confusion Matrix - verify accuracy of each class
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, prediction_naivebayes)
print(cm)
report = metrics.classification_report(y_test, prediction_naivebayes)
print(report)



from sklearn.tree import DecisionTreeClassifier
decision= DecisionTreeClassifier()
decision.fit(train_PCA, y_train)
prediction_decision = decision.predict(test_PCA)
print ("Decision Tree Accuracy = ", metrics.accuracy_score(y_test, prediction_decision))
#Confusion Matrix - verify accuracy of each class
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, prediction_decision)
print(cm)
report = metrics.classification_report(y_test, prediction_decision)
print(report)
"""

"""
#GLCM -> Texture Feature  (Test_size=0.2)


from skimage.feature import greycomatrix, greycoprops
def calc_glcm_all_agls(img, label, props, dists=[5], agls=[0, np.pi/4, np.pi/2, 3*np.pi/4], lvl=256, sym=True, norm=True):
    
    glcm = greycomatrix(img, 
                        distances=dists, 
                        angles=agls, 
                        levels=lvl,
                        symmetric=sym, 
                        normed=norm)
    feature = []
    glcm_props = [propery for name in props for propery in greycoprops(glcm, name)[0]]
    for item in glcm_props:
            feature.append(item)
    feature.append(label) 
    
    return feature

categories= ['CLL','FL','MCL']
dir='C:\\Users\\Pc\\Downloads\\archive\\'
entropy10=[]

for category in categories:
        path=os.path.join(dir,category)
        label= categories.index(category)
        for img6 in os.listdir(path):
            imgpath=os.path.join(path,img6)                       
            image1=cv.imread(imgpath, cv.IMREAD_GRAYSCALE)
            image1 = cv.resize(image1, (400,400))
            equalized1 = cv.equalizeHist(image1)
            entropy10.append([equalized1,label])


for category in categories:
        path=os.path.join(dir,category)
        label= categories.index(category)
        for img6 in os.listdir(path):
            imgpath=os.path.join(path,img6)                       
            image1=cv.imread(imgpath, cv.IMREAD_GRAYSCALE)
            image1 = cv.resize(image1, (400,400))
            entropy10.append([image1,label])


for category in categories:
        path=os.path.join(dir,category)
        label= categories.index(category)
        for img6 in os.listdir(path):
            imgpath=os.path.join(path,img6)                       
            image1=cv.imread(imgpath, cv.IMREAD_GRAYSCALE)
            image1 = cv.resize(image1, (400,400))
            clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(6, 6))
            equalized = clahe.apply(image1)
            entropy10.append([equalized,label])

print(len(entropy10))
print(np.ndim(entropy10))
features=[]
labels=[]
from sklearn.model_selection import train_test_split

for feature,label in entropy10:
    features.append(feature)
    labels.append(label)
properties = ['dissimilarity', 'correlation', 'homogeneity', 'contrast', 'ASM', 'energy']

glcm_all_agls = []

for img, label1 in zip(features, labels): 
    glcm_all_agls.append(
            calc_glcm_all_agls(img, 
                                label1, 
                                props=properties)
                            )
 
columns = []
angles = ['0', '45', '90','135']
for name in properties :
    for ang in angles:
        columns.append(name + "_" + ang)
        
columns.append("label")

import pandas as pd 

# Create the pandas DataFrame for GLCM features data

glcm_df = pd.DataFrame(glcm_all_agls, 
                      columns = columns)

#glcm_df["entropy"] = entropy3.copy()
glcm_df.head(15)

print(glcm_df.head(100))

#Bu ya da   print(glcm_df.iloc[:,-1])

print(glcm_df.loc[:,"label"])
labelss=glcm_df.loc[:,"label"]
#three= glcm_df.pop("label")
glcm_df.drop('label',axis=1, inplace=True)
print(glcm_df)

#Bu da olur ->  glcm_df.iloc[:,:]
#glcm_df.filter(regex="[^label]")
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(abs(glcm_df), labelss, test_size=0.20, random_state=10)

print(y_train)
print(y_test)
print(len(y_train))
print(len(y_test))



from sklearn.ensemble import RandomForestClassifier
RF_model = RandomForestClassifier(n_estimators = 100)

# , random_state = 10  -> eliydi çıkardım.
# Train the model on training data
RF_model.fit(x_train, y_train) #For sklearn no one hot encoding

prediction_RF = RF_model.predict(x_test)
#Inverse le transform to get original label back. 
#Print overall accuracy
from sklearn import metrics
print ("RF Accuracy = ", metrics.accuracy_score(y_test, prediction_RF))
#Confusion Matrix - verify accuracy of each class
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, prediction_RF)
print(cm)
report = metrics.classification_report(y_test, prediction_RF)
print(report)



from sklearn.neighbors import KNeighborsClassifier
knnmodel= KNeighborsClassifier(n_neighbors=5, metric='minkowski')
#ya da 10 yaz -> daha önce öyle yaptım.

knnmodel.fit(x_train, y_train)
prediction_knnmodel = knnmodel.predict(x_test)
print ("KNN Accuracy = ", metrics.accuracy_score(y_test, prediction_knnmodel))
#Confusion Matrix - verify accuracy of each class
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, prediction_knnmodel)
print(cm)
report = metrics.classification_report(y_test, prediction_knnmodel)
print(report)



from sklearn.naive_bayes import GaussianNB
naive_bayes=GaussianNB()
naive_bayes.fit(x_train, y_train)
prediction_naivebayes = naive_bayes.predict(x_test)

print ("Naive Bayes Accuracy = ", metrics.accuracy_score(y_test, prediction_naivebayes))
#Confusion Matrix - verify accuracy of each class
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, prediction_naivebayes)
print(cm)
report = metrics.classification_report(y_test, prediction_naivebayes)
print(report)



from sklearn.tree import DecisionTreeClassifier
decision= DecisionTreeClassifier()
decision.fit(x_train, y_train)
prediction_decision = decision.predict(x_test)
print ("Decision Tree Accuracy = ", metrics.accuracy_score(y_test, prediction_decision))
#Confusion Matrix - verify accuracy of each class
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, prediction_decision)
print(cm)
report = metrics.classification_report(y_test, prediction_decision)
print(report)
"""

"""
#GLCM  + PCA() -> Texture Feature  (Test_size=0.2)

from sklearn.decomposition import PCA
from skimage.feature import greycomatrix, greycoprops
def calc_glcm_all_agls(img, label, props, dists=[5], agls=[0, np.pi/4, np.pi/2, 3*np.pi/4], lvl=256, sym=True, norm=True):
    
    glcm = greycomatrix(img, 
                        distances=dists, 
                        angles=agls, 
                        levels=lvl,
                        symmetric=sym, 
                        normed=norm)
    feature = []
    glcm_props = [propery for name in props for propery in greycoprops(glcm, name)[0]]
    for item in glcm_props:
            feature.append(item)
    feature.append(label) 
    
    return feature
entropy3=[]
categories= ['FL','CLL']
dir='C:\\Users\\Pc\\Downloads\\archive\\'
entropy10=[]

for category in categories:
        path=os.path.join(dir,category)
        label= categories.index(category)
        for img6 in os.listdir(path):
            imgpath=os.path.join(path,img6)                       
            image1=cv.imread(imgpath, cv.IMREAD_GRAYSCALE)
            image1 = cv.resize(image1, (400,400))
            equalized1 = cv.equalizeHist(image1)
            entropy2 = calcEntropy(equalized1)
            entropy3.append(entropy2)
            entropy10.append([equalized1,label])

for category in categories:
        path=os.path.join(dir,category)
        label= categories.index(category)
        for img6 in os.listdir(path):
            imgpath=os.path.join(path,img6)                       
            image1=cv.imread(imgpath, cv.IMREAD_GRAYSCALE)
            image1 = cv.resize(image1, (400,400))
            entropy2 = calcEntropy(image1)
            entropy3.append(entropy2)
            entropy10.append([image1,label])


for category in categories:
        path=os.path.join(dir,category)
        label= categories.index(category)
        for img6 in os.listdir(path):
            imgpath=os.path.join(path,img6)                       
            image1=cv.imread(imgpath, cv.IMREAD_GRAYSCALE)
            image1 = cv.resize(image1, (400,400))
            clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(6, 6))
            equalized = clahe.apply(image1)
            entropy2 = calcEntropy(equalized)
            entropy3.append(entropy2)
            entropy10.append([equalized,label])

print(len(entropy10))
print(np.ndim(entropy10))
features=[]
labels=[]
from sklearn.model_selection import train_test_split

for feature,label in entropy10:
    features.append(feature)
    labels.append(label)
properties = ['dissimilarity', 'correlation', 'homogeneity', 'contrast', 'ASM', 'energy']

glcm_all_agls = []

for img, label1 in zip(features, labels): 
    glcm_all_agls.append(
            calc_glcm_all_agls(img, 
                                label1, 
                                props=properties)
                            )
 
columns = []
angles = ['0', '45', '90','135']
for name in properties :
    for ang in angles:
        columns.append(name + "_" + ang)
        
columns.append("label")

import pandas as pd 

# Create the pandas DataFrame for GLCM features data

glcm_df = pd.DataFrame(glcm_all_agls, 
                      columns = columns)

#glcm_df["entropy"] = entropy3.copy()
glcm_df.head(15)

print(glcm_df.head(100))

#Bu ya da   print(glcm_df.iloc[:,-1])

print(glcm_df.loc[:,"label"])
labelss=glcm_df.loc[:,"label"]
#three= glcm_df.pop("label")
glcm_df.drop('label',axis=1, inplace=True)
print(glcm_df)

#Bu da olur ->  glcm_df.iloc[:,:]
#glcm_df.filter(regex="[^label]")
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(abs(glcm_df), labelss, test_size=0.20, random_state=10)

print(y_train)
print(y_test)
print(len(y_train))
print(len(y_test))

pca = PCA()
#31 seçildiğinde en iyi sonucu verdi.
train_PCA = pca.fit_transform(x_train)
test_PCA = pca.transform(x_test) 

from sklearn.ensemble import RandomForestClassifier
RF_model = RandomForestClassifier(n_estimators = 100)

# , random_state = 10  -> eliydi çıkardım.
# Train the model on training data
RF_model.fit(train_PCA, y_train) #For sklearn no one hot encoding

prediction_RF = RF_model.predict(test_PCA)
#Inverse le transform to get original label back. 
#Print overall accuracy
from sklearn import metrics
print ("RF Accuracy = ", metrics.accuracy_score(y_test, prediction_RF))
#Confusion Matrix - verify accuracy of each class
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, prediction_RF)
print(cm)
report = metrics.classification_report(y_test, prediction_RF)
print(report)



from sklearn.neighbors import KNeighborsClassifier
knnmodel= KNeighborsClassifier(n_neighbors=5, metric='minkowski')
#ya da 10 yaz -> daha önce öyle yaptım.

knnmodel.fit(train_PCA, y_train)
prediction_knnmodel = knnmodel.predict(test_PCA)
print ("KNN Accuracy = ", metrics.accuracy_score(y_test, prediction_knnmodel))
#Confusion Matrix - verify accuracy of each class
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, prediction_knnmodel)
print(cm)
report = metrics.classification_report(y_test, prediction_knnmodel)
print(report)



from sklearn.naive_bayes import GaussianNB
naive_bayes=GaussianNB()
naive_bayes.fit(train_PCA, y_train)
prediction_naivebayes = naive_bayes.predict(test_PCA)

print ("Naive Bayes Accuracy = ", metrics.accuracy_score(y_test, prediction_naivebayes))
#Confusion Matrix - verify accuracy of each class
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, prediction_naivebayes)
print(cm)
report = metrics.classification_report(y_test, prediction_naivebayes)
print(report)



from sklearn.tree import DecisionTreeClassifier
decision= DecisionTreeClassifier()
decision.fit(train_PCA, y_train)
prediction_decision = decision.predict(test_PCA)
print ("Decision Tree Accuracy = ", metrics.accuracy_score(y_test, prediction_decision))
#Confusion Matrix - verify accuracy of each class
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, prediction_decision)
print(cm)
report = metrics.classification_report(y_test, prediction_decision)
print(report)
"""



"""
#(Data Augmentation yapılarak) GLCM -> Texture Feature  (Test_size=0.2)


from skimage.feature import greycomatrix, greycoprops
def calc_glcm_all_agls(img, label, props, dists=[5], agls=[0, np.pi/4, np.pi/2, 3*np.pi/4], lvl=256, sym=True, norm=True):
    
    glcm = greycomatrix(img, 
                        distances=dists, 
                        angles=agls, 
                        levels=lvl,
                        symmetric=sym, 
                        normed=norm)
    feature = []
    glcm_props = [propery for name in props for propery in greycoprops(glcm, name)[0]]
    for item in glcm_props:
            feature.append(item)
    feature.append(label) 
    
    return feature

categories= ['MCL','CLL']
dir='C:\\Users\\Pc\\Downloads\\archive\\'
entropy10=[]

angle1 =[90,180,270,360]

for category in categories:
        path=os.path.join(dir,category)
        label= categories.index(category)
        for img6 in os.listdir(path):
            imgpath=os.path.join(path,img6)                       
            image1=cv.imread(imgpath, cv.IMREAD_GRAYSCALE)
            image1 = cv.resize(image1, (400,400))
            equalized1 = cv.equalizeHist(image1)
            for x in angle1:
                Rotated = imutils.rotate(equalized1, angle=x)
                entropy10.append([Rotated,label])

for category in categories:
        path=os.path.join(dir,category)
        label= categories.index(category)
        for img6 in os.listdir(path):
            imgpath=os.path.join(path,img6)                       
            image1=cv.imread(imgpath, cv.IMREAD_GRAYSCALE)
            image1 = cv.resize(image1, (400,400))
            for x in angle1:
                Rotated = imutils.rotate(image1, angle=x)
                entropy10.append([Rotated,label])


for category in categories:
        path=os.path.join(dir,category)
        label= categories.index(category)
        for img6 in os.listdir(path):
            imgpath=os.path.join(path,img6)                       
            image1=cv.imread(imgpath, cv.IMREAD_GRAYSCALE)
            image1 = cv.resize(image1, (400,400))
            clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(6, 6))
            equalized = clahe.apply(image1)
            for x in angle1:
                Rotated = imutils.rotate(equalized, angle=x)
                entropy10.append([Rotated,label])
            


print(len(entropy10))
print(np.ndim(entropy10))
features=[]
labels=[]
from sklearn.model_selection import train_test_split

for feature,label in entropy10:
    features.append(feature)
    labels.append(label)
properties = ['dissimilarity', 'correlation', 'homogeneity', 'contrast', 'ASM', 'energy']

glcm_all_agls = []

for img, label1 in zip(features, labels): 
    glcm_all_agls.append(
            calc_glcm_all_agls(img, 
                                label1, 
                                props=properties)
                            )
 
columns = []
angles = ['0', '45', '90','135']
for name in properties :
    for ang in angles:
        columns.append(name + "_" + ang)
        
columns.append("label")

import pandas as pd 

# Create the pandas DataFrame for GLCM features data

glcm_df = pd.DataFrame(glcm_all_agls, 
                      columns = columns)

#glcm_df["entropy"] = entropy3.copy()
glcm_df.head(15)

print(glcm_df.head(100))

#Bu ya da   print(glcm_df.iloc[:,-1])

print(glcm_df.loc[:,"label"])
labelss=glcm_df.loc[:,"label"]
#three= glcm_df.pop("label")
glcm_df.drop('label',axis=1, inplace=True)
print(glcm_df)

#Bu da olur ->  glcm_df.iloc[:,:]
#glcm_df.filter(regex="[^label]")
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(abs(glcm_df), labelss, test_size=0.20)

print(y_train)
print(y_test)
print(len(y_train))
print(len(y_test))



from sklearn.ensemble import RandomForestClassifier
RF_model = RandomForestClassifier(n_estimators = 100)

# , random_state = 10  -> eliydi çıkardım.
# Train the model on training data
RF_model.fit(x_train, y_train) #For sklearn no one hot encoding

prediction_RF = RF_model.predict(x_test)
#Inverse le transform to get original label back. 
#Print overall accuracy
from sklearn import metrics
print ("RF Accuracy = ", metrics.accuracy_score(y_test, prediction_RF))
#Confusion Matrix - verify accuracy of each class
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, prediction_RF)
print(cm)
report = metrics.classification_report(y_test, prediction_RF)
print(report)



from sklearn.neighbors import KNeighborsClassifier
knnmodel= KNeighborsClassifier(n_neighbors=5, metric='minkowski')
#ya da 10 yaz -> daha önce öyle yaptım.

knnmodel.fit(x_train, y_train)
prediction_knnmodel = knnmodel.predict(x_test)
print ("KNN Accuracy = ", metrics.accuracy_score(y_test, prediction_knnmodel))
#Confusion Matrix - verify accuracy of each class
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, prediction_knnmodel)
print(cm)
report = metrics.classification_report(y_test, prediction_knnmodel)
print(report)



from sklearn.naive_bayes import GaussianNB
naive_bayes=GaussianNB()
naive_bayes.fit(x_train, y_train)
prediction_naivebayes = naive_bayes.predict(x_test)

print ("Naive Bayes Accuracy = ", metrics.accuracy_score(y_test, prediction_naivebayes))
#Confusion Matrix - verify accuracy of each class
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, prediction_naivebayes)
print(cm)
report = metrics.classification_report(y_test, prediction_naivebayes)
print(report)



from sklearn.tree import DecisionTreeClassifier
decision= DecisionTreeClassifier()
decision.fit(x_train, y_train)
prediction_decision = decision.predict(x_test)
print ("Decision Tree Accuracy = ", metrics.accuracy_score(y_test, prediction_decision))
#Confusion Matrix - verify accuracy of each class
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, prediction_decision)
print(cm)
report = metrics.classification_report(y_test, prediction_decision)
print(report)


"""





angle =[90,180,270,360]
#categories= ['CLL', 'FL', 'MCL']
categories= ['CLL', 'FL','MCL']
dir='C:\\Users\\Pc\\Downloads\\archive\\'
labels1=[]
features1=[]
for category in categories:
        path=os.path.join(dir,category)
        label= categories.index(category)
        for img6 in os.listdir(path):
            imgpath=os.path.join(path,img6)                       
            image1=cv.imread(imgpath, cv.IMREAD_COLOR)
            image1 = cv.resize(image1, (300,300))
            for x in angle:
                Rotated = imutils.rotate(image1, angle=x)
            #image1=cv.cvtColor(image1, cv.COLOR_RGB2BGR)
                features1.append(Rotated/255)
                labels1.append(label)
                
                

from sklearn.model_selection import train_test_split


features= np.array(features1)
labels =np.array(labels1)

x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.20,random_state=10)

# , random_state=10
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

print(y_train)
print(y_test)
print(len(x_train))
print(len(x_test))
print(len(y_train))
print(len(y_test))



import glob

from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D,Input
from keras.layers.normalization import BatchNormalization

from keras.applications.vgg16 import VGG16
#from keras.applications.resnet50 import ResNet50
#from keras.applications.densenet import DenseNet201
#from keras.applications.mobilenet import MobileNet

#from keras.applications.vgg19 import VGG19
from keras.utils import to_categorical


VGG_Model = VGG16(weights='imagenet', include_top=False, input_shape=(300,300,3))

for layer in VGG_Model.layers:
    layer.trainable = False

VGG_Model.summary()



train_feature_extractor =VGG_Model.predict(x_train)
print(train_feature_extractor.shape[0])
print(train_feature_extractor.shape[1])
print(np.ndim(train_feature_extractor))

train_features = train_feature_extractor.reshape(train_feature_extractor.shape[0],-1)

test_feature_extractor=VGG_Model.predict(x_test)
test_features = test_feature_extractor.reshape(test_feature_extractor.shape[0], -1)

from sklearn.decomposition import PCA


pca_test = PCA(n_components=0.4) 
pca_test.fit(train_features)


#n_PCA_components = 20
#pca = PCA(n_components=n_PCA_components)
pca = PCA()
#31 seçildiğinde en iyi sonucu verdi.
train_PCA = pca.fit_transform(train_features)
test_PCA = pca.transform(test_features) #Make sure you are just transforming, not fitting. 

from sklearn.ensemble import RandomForestClassifier
RF_model = RandomForestClassifier(n_estimators = 100)

# , random_state = 10  -> eliydi çıkardım.
# Train the model on training data
RF_model.fit(train_PCA, y_train) #For sklearn no one hot encoding

prediction_RF = RF_model.predict(test_PCA)
#Inverse le transform to get original label back. 
#Print overall accuracy
from sklearn import metrics
print ("RF Accuracy = ", metrics.accuracy_score(y_test, prediction_RF))
#Confusion Matrix - verify accuracy of each class
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, prediction_RF)
print(cm)
report = metrics.classification_report(y_test, prediction_RF)
print(report)



from sklearn.neighbors import KNeighborsClassifier
knnmodel= KNeighborsClassifier(n_neighbors=5, metric='minkowski')
#ya da 10 yaz -> daha önce öyle yaptım.

knnmodel.fit(train_PCA, y_train)
prediction_knnmodel = knnmodel.predict(test_PCA)
print ("KNN Accuracy = ", metrics.accuracy_score(y_test, prediction_knnmodel))
#Confusion Matrix - verify accuracy of each class
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, prediction_knnmodel)
print(cm)
report = metrics.classification_report(y_test, prediction_knnmodel)
print(report)



from sklearn.naive_bayes import GaussianNB
naive_bayes=GaussianNB()
naive_bayes.fit(train_PCA, y_train)
prediction_naivebayes = naive_bayes.predict(test_PCA)

print ("Naive Bayes Accuracy = ", metrics.accuracy_score(y_test, prediction_naivebayes))
#Confusion Matrix - verify accuracy of each class
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, prediction_naivebayes)
print(cm)
report = metrics.classification_report(y_test, prediction_naivebayes)
print(report)



from sklearn.tree import DecisionTreeClassifier
decision= DecisionTreeClassifier()
decision.fit(train_PCA, y_train)
prediction_decision = decision.predict(test_PCA)
print ("Decision Tree Accuracy = ", metrics.accuracy_score(y_test, prediction_decision))
#Confusion Matrix - verify accuracy of each class
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, prediction_decision)
print(cm)
report = metrics.classification_report(y_test, prediction_decision)
print(report)





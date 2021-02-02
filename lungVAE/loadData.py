import matplotlib.image as mpimg
from skimage.transform import rescale, resize, downscale_local_mean
import numpy as np
import csv
import torch

class loadData(object):
    def __init__(self, imgPath1,imgPath2,matchFile,trainRatio):
        self.imgPath1 = imgPath1
        self.imgPath2 = imgPath2
        self.matchFile = matchFile
        self.trainRatio = trainRatio
        self.dataN = []
        self.dataP = []

        self.trainN =None
        self.testN = None
        self.trainP = None
        self.testP = None

        self.loadNoise()
        self.loadPure()

    def loadNoise(self):
        with open(self.matchFile) as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            next(spamreader)

            for row in spamreader:
                path1 = row[0].replace('.png','')

                img1 = mpimg.imread(self.imgPath1+'/'+path1+'/image.png')
                #img1 = mpimg.imread(self.imgPath2+'/'+path2)
                img1 = rescale(img1, 128/1024, anti_aliasing=False)
                img1 = np.array(img1)
                #tempImg = np.ones((img1.shape[0],img1.shape[1]))
                #img1 = tempImg - img1
                img1 = img1.astype(np.float32)
                #img1 = img1.reshape((1,img1.shape[0],img1.shape[1]))
                self.dataN.append(img1)

        #separate pairedData to train and test set
        length = len(self.dataN)
        self.trainN = self.dataN[0:int(length*self.trainRatio)]
        self.trainN = np.array(self.trainN)
        self.testN = self.dataN[int(length*self.trainRatio):length]
        self.testN = np.array(self.testN)

        csvfile.close()


    def loadPure(self):
        with open(self.matchFile) as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            next(spamreader)

            for row in spamreader:
                path2 = row[1]
                img1 = mpimg.imread(self.imgPath2+'/'+path2)
                #img1 = mpimg.imread(self.imgPath2+'/'+path2)
                img1 = rescale(img1, 128/1024, anti_aliasing=False)
                img1 = img1 * 100
                img1 = np.array(img1)
                img1[img1>0.1] = 1
                img1 = img1.astype(np.float32)
                #img1 = img1.reshape((1,img1.shape[0],img1.shape[1]))
                self.dataP.append(img1)

        #separate pairedData to train and test set
        length = len(self.dataP)
        self.trainP = self.dataP[0:int(length*self.trainRatio)]
        self.trainP = np.array(self.trainP)
        self.testP = self.dataP[int(length*self.trainRatio):length]
        self.testP = np.array(self.testP)

        csvfile.close()

if __name__ == "__main__":
    loadImg = loadData('./images','masks','./train.csv',0.8)
    print('done')
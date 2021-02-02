import matplotlib.image as mpimg
from skimage.transform import rescale, resize, downscale_local_mean
import numpy as np
import csv
import torch



class pairImg(object):
    def __init__(self, imgPath1,imgPath2,matchFile,batchSize,trainRatio,block=False):
        self.imgPath1 = imgPath1
        self.imgPath2 = imgPath2
        self.matchFile = matchFile
        self.batchSize = batchSize
        self.pairedData = [] # a python array
        self.trainRatio = trainRatio
        self.train =None
        self.test = None
        self.block = block
        self.loadData()


    def loadData(self):
        with open(self.matchFile) as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            next(spamreader)
            count = 1
            imgBatch1 = []
            imgBatch2 = []
            for row in spamreader:
                img1 = None
                img2 = None
                #preprocess path
                path1 = row[0].replace('.png','')
                path2 = row[1]
                #print(img1)
                #load images
                img2 = mpimg.imread(self.imgPath1+'/'+path1+'/image.png')
                #img2 = mpimg.imread(self.imgPath2+'/'+path2)
                img2 = rescale(img2, 128/1024, anti_aliasing=False)
                #img2 = img2*100
                triple = []
                triple.append(img2)
                #triple.append(img2)
                #triple.append(img2)

                triple = np.array(triple)
                triple = triple.astype(np.float32)
                triple = triple.reshape((1,triple.shape[1],triple.shape[2])) #first D is batch size
                imgBatch2.append(triple)

                #img1 = mpimg.imread(self.imgPath1+'/'+path1+'/image.png')
                img1 = mpimg.imread(self.imgPath2+'/'+path2)
                img1 = rescale(img1, 128/1024, anti_aliasing=False)
                #black img
                img1 = img1*100

                img1 = np.array(img1)
                img1[img1 > 0] = 1
                img1 = img1.astype(np.float32)
                img1 = img1.reshape((1,img1.shape[0],img1.shape[1]))
                imgBatch1.append(img1)
                #pair images
                if count % self.batchSize == 0:
                    pair = []
                    imgBatch1 = np.array(imgBatch1)
                    imgBatch1 = torch.from_numpy(imgBatch1)
                    imgBatch2 = np.array(imgBatch2)
                    imgBatch2 = torch.from_numpy(imgBatch2)
                    pair.append(imgBatch1) #img1 0
                    pair.append(imgBatch2) #img2 1
                    # save images
                    self.pairedData.append(pair)
                    #reInitial batch
                    imgBatch1 = []
                    imgBatch2 = []

                count += 1

        #shuffle pairedData
        np.random.shuffle(self.pairedData)
        #separate pairedData to train and test set
        length = len(self.pairedData)
        self.train = self.pairedData[0:int(length*self.trainRatio)]
        self.test = self.pairedData[int(length*self.trainRatio):length]

        if self.block == True:
            for eachtest in self.test:
                for i,eachpair in enumerate(eachtest):
                    if i == 1:
                        eachpair[:,:,:,64:128] = 0
                        #eachpair[:, :, 64:128,: ] = 1

        '''
        #zero out source 1 in test
        for i in range(len(self.test)):
            #self.test[i][0] = self.test[i][0] * 0
            self.test[i][0][:] = -1
        '''
        pass

if __name__ == "__main__":
    loadImg = pairImg('./images','masks','./train.csv',5,0.8)
    print('done')
import matplotlib.image as mpimg
from skimage.transform import rescale, resize, downscale_local_mean
import matplotlib.pyplot as plt
import numpy as np
import csv
import torch



class pairImg(object):
    def __init__(self, imgPath1,imgPath2,matchFile,batchSize,trainRatio):
        self.imgPath1 = imgPath1
        self.imgPath2 = imgPath2
        self.matchFile = matchFile
        self.batchSize = batchSize
        self.pairedData = [] # a python array
        self.trainRatio = trainRatio
        self.train =None
        self.test = None
        self.loadData()

    def loadData(self):
        with open(self.matchFile) as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            next(spamreader)

            img1A = np.zeros((1024,1024))
            img2A = np.zeros((1024,1024))
            count = 0
            for row in spamreader:
                count += 1
                img1 = None
                img2 = None
                #preprocess path
                path1 = row[0].replace('.png','')
                path2 = row[1]
                img2 = mpimg.imread(self.imgPath2+'/'+path2)
                img1 = mpimg.imread(self.imgPath1+'/'+path1+'/image.png')
                img1A = img1A + img1
                img2A = img2A + img2

            img2avg = img2A/count
            img1avg = img1A/count
            plt.imshow(img2avg)
            plt.savefig('img2avg.png')
            plt.clf()
            plt.imshow(img1avg)
            plt.savefig('img1avg.png')


if __name__ == "__main__":
    loadImg = pairImg('./images','masks','./train.csv',5,0.8)
    print('done')
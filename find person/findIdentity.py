'''
task = are the 2 people the same?
presupun fara restrangerea problemei ca ancora = Obama
=> task devine "is the person Obama?"

TP = the person is Obama and I said it is Obama  * 
FN = the person is Obama and I said it is not Obama 

TN = the person is not Obama and I said it is not Obama
FP = the person is not Obama and I said it is Obama  *

Recall = TP / (TP + FN)
Precision = TP / (TP + FP)

Recall = (the person is Obama and I said it is Obama) / (the person is Obama)
Precision = (the person is Obama and I said it is Obama) / (I said it is Obama)

Recall mare = am gasit multi Obama
Precision mare = am spus putine prostii <=> am spus de putine ori la non Obama ca sunt Obama


thr creste => Recall creste, Precision scade <=> gasim mai multi Obama, spunem mai multe prostii

daca thr mic => spun "da" rar => recall = mic, precision = mare

daca thr mare => spun "da" des => recall mare, precision = 50% (presupunand ca |Obama data| = |Non Obama data|)

'''

import time
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.cluster import KMeans
import numpy as np
import cv2
import torch

def getColorsAndCenters(points, k):
    kmeans = KMeans(n_clusters = k).fit(points)
    return kmeans.labels_, kmeans.cluster_centers_

'''
def newRect(rect, rect2): # poate alta data :(, ar fi fost o idee sa aplic recursiv cautarea
    if rect == None:
        return rect2 
    else:
        x1 = min(rect[0], rect2[0])
        y1 = min(rect[1], rect2[1])
        x2 = max(rect[2], rect2[2])
        y2 = max(rect[3], rect2[3])
        return [x1, y1, x2, y2]
'''


        
def smoothCluster(rects, k):
    assert (len(rects) > 0)
    dif = rects[0][3] - rects[0][1]

    for i in range(len(rects)):
        assert (rects[i][3] - rects[i][1] == dif)
        assert (rects[i][4] - rects[i][2] == dif)

    k = min(k, len(rects))
    points = [(x1, y1) for (pr, x1, y1, x2, y2) in rects]

    # NMS = Non maximum supression

    x = np.array(points)
    colors, centers = getColorsAndCenters(x, k)

    guys = [None for i in range(k)]
    mean = [0 for i in range(k)]
    cntt = [0 for i in range(k)]

    for i in range(len(rects)):
        (pr, x1, y1, x2, y2) = rects[i]
        mean[colors[i]] += pr 
        cntt[colors[i]] += 1

    for i in range(k):
        x1, y1 = centers[i]
        x1 = int(x1)
        y1 = int(y1)
        x2 = x1 + dif
        y2 = y1 + dif
        guys[i] = (mean[i] / cntt[i], x1, y1, x2, y2)
        assert cntt[i] > 0
    
    return guys


def smoothRemoveNonPeople(rects, img):
    face_cascade = cv2.CascadeClassifier('theNets/haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    newRects = []
    for (pr, x1, y1, x2, y2) in rects:
        faces = face_cascade.detectMultiScale(gray[x1 : x2, y1 : y2], 1.1, 4)

        if len(faces) >= 1: # sau poate = 1, who knows?
            newRects.append((pr, x1, y1, x2, y2))

    return newRects

def getPrecision(thr, precisionRecallData):
    precision = None
    closestThr = None


    for (t, pr, re) in precisionRecallData:
        if closestThr == None:
            closestThr = t 
            precision = pr 
        else:
            if abs(t - thr) < abs(closestThr - thr):
                closestThr = t 
                precision = pr

    assert precision != None
    return precision

def findIdendityOptimized(bigImg, identity, model, scales, transform, device, dim, strideFactor, precisionRecallData, precision, maxReturn = None, KmeansK = None):

    maxRecallOverPrecision = None
    chosenThr = None

    for (thr, pr, re) in precisionRecallData:
        if pr >= precision:
            if maxRecallOverPrecision == None or maxRecallOverPrecision < re:
                maxRecallOverPrecision = re
                chosenThr = thr
    
    assert chosenThr != None
    assert identity.shape[0] == dim and identity.shape[1] == dim 

    A = model(transform(identity).unsqueeze(0).to(device))
    sol = []

    for l in scales:
        bg = time.time()


        rects = []
        img = cv2.resize(bigImg, fx = l, fy = l, dsize = (0, 0))
        val = transform(img).to(device)
        stride = int(strideFactor * dim)
        crops = []

        for i in range(0, img.shape[0] - dim, stride):
            for j in range(0, img.shape[1] - dim, stride):
                crops.append((val[:, i : i + dim, j : j + dim], (i, j)))

        cropsDataLoader = DataLoader(crops, batch_size = 64)
        for (X, D) in cropsDataLoader:
            B = model(X)
            for k in range(X.shape[0]):
                dist = ((A - B[k])**2).sum()

                if dist <= chosenThr:
                    rects.append((float(dist), int(D[0][k]), int(D[1][k]), int(D[0][k] + dim), int(D[1][k] + dim)))
        
        del val
        torch.cuda.empty_cache() # mic imporvement la memorie

        print(time.time() - bg, ":", l, len(rects))

        if KmeansK != None:
            rects = smoothCluster(rects, KmeansK)

        for i in range(len(rects)):
            dist, x1, y1, x2, y2 = rects[i]
            x1 = int(x1 / l)
            y1 = int(y1 / l)
            x2 = int(x2 / l)
            y2 = int(y2 / l)
            sol.append((dist, x1, y1, x2, y2))
        
    sol.sort()
    for i in range(len(sol)):
        dist, x1, y1, x2, y2 = sol[i]
        sol[i] = (getPrecision(dist, precisionRecallData), x1, y1, x2, y2)

    if maxReturn != None:
        if len(sol) > maxReturn:
            sol = sol[:maxReturn]
        
    

    return sol
        
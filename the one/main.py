from dataloader import loadIdentities, gen
from train import generateTripletsDumb, train
from network import NN
import torch.optim as optim
import torch 
import cv2
import torchvision.transforms as transforms
import random

device = "cuda"
#device = "cpu"

dataset = loadIdentities(cnt = 5696 - 5650 * 0, H = 150, W = 150, verbose = True) 
model = NN.loadModel().to(device)
learningRate = 1e-4
optimizer = optim.Adam(model.parameters(), lr = learningRate)
margin = 0.2

def printParameters(model):
    sol = 0
    for parameter in model.parameters():
        prod = 1
        for x in parameter.shape:
            prod *= x
        sol += prod
        
        print(parameter.shape, prod)
    print("total params =", sol)

printParameters(model) # 2838080 params

def test(threshold, cntTriplets, repeat, verbose = False):
    TP, FP, TN, FN = 0, 0, 0, 0

    for i in range(repeat):
        # generate triplets
        triplets = generateTripletsDumb(dataset, cntTriplets, device)
        X = torch.cat(triplets).to(device)

        del triplets
        #torch.cuda.empty_cache() 
        Y = model(X)

        for index in range(cntTriplets):
            a = Y[3 * index]
            b = Y[3 * index + 1]
            c = Y[3 * index + 2]

        
            distAB = ((a - b)**2).sum()
            distAC = ((a - c)**2).sum()

            if distAB <= threshold:
                TP += 1
            else:
                FN += 1
            
            if distAC <= threshold:
                FP += 1
            else:
                TN += 1
                
            del a 
            del b
            del c
            #torch.cuda.empty_cache() 

        
        del X
        del Y
        #torch.cuda.empty_cache() 

    precision = TP / (TP + FP + 1e-4)
    recall = TP / (TP + FN + 1e-4)
    f1 = 2 / (1 / (precision + 1e-4) + 1 / (recall + 1e-4) + 1e-4)

    if verbose == True:
        print(threshold, ":", "precision =", precision, "| recall =", recall, "| f1 score =", f1)

    return precision, recall

def draw(cnt):
    import matplotlib.pyplot as plt
    plt.style.use('seaborn-whitegrid')
    import numpy as np
    
    xs = []
    ys = []

    for th in np.arange(0, 5, 0.01):
        x, y = test(th, 1, repeat = 100, verbose = True)
        xs.append(x)
        ys.append(y)

    plt.plot(xs, ys, '.', color = 'black')
    plt.show()

    exit(0)

'''

search(big img, small img, model, precision)

vector of scales = [1, x, 1/x, ... xmax = 2 sau 4] ca parametru 

stride = procent al lui (small img rescaled) 10% of height | stride parametru in search





'''

def sameImg(img1, img2, threshold):


    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # (x - mean) / std  

    kek1, img1 = gen(img1)
    kek2, img2 = gen(img2)

    assert kek1 == True
    assert kek2 == True

    img1 = transform(cv2.resize(img1, (150, 150))).unsqueeze(0).to(device)
    img2 = transform(cv2.resize(img2, (150, 150))).unsqueeze(0).to(device)

    A = model(img1)
    B = model(img2)

    distAB = ((A - B)**2).sum()
    return distAB <= threshold

def trainTest():
    print(sameImg(cv2.imread("testData/eu1.jpg"), cv2.imread("testData/eu2.jpg"), 1.25))
    print(sameImg(cv2.imread("testData/eu1.jpg"), cv2.imread("testData/medi1.jpg"), 1.25))
    print(sameImg(cv2.imread("testData/eu1.jpg"), cv2.imread("testData/medi2.jpg"), 1.25))

'''trainTest()

exit(0)'''
draw(100)

train(model = model, dataset = dataset, epochs = 1000000, optimizer = optimizer, margin = margin, device = device, cntTriplets = 50, repeat = 1)
 
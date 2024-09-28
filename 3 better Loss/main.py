from dataloader import loadIdentities, gen
from network import NN
import torch.optim as optim
import time
import random
import torch 
import torch.nn.functional as F
import numpy as np

import torch.nn as nn

device = "cuda"
#device = "cpu"

dataset = loadIdentities(cnt = 5749 - 5700 * 1, H = 255, W = 255, verbose = True) # cnt max = 5749
model = NN.loadModel().to(device)
#model = NN().to(device)
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

printParameters(model)

#exit(0)

# 2838080


def generateTripletsDumb(dataset, cntTriplets, device):
    allTriplets = []
    
    for tripledId in range(cntTriplets):
        # find idendity1 of anchor, positive
        identidy1 = random.randint(0, len(dataset) - 1)
        while len(dataset[identidy1]) < 2:
            identidy1 = random.randint(0, len(dataset) - 1)

        # find identidy 2 of negative
        identidy2 = random.randint(0, len(dataset) - 1)
        while identidy2 == identidy1:
            identidy2 = random.randint(0, len(dataset) - 1)

        # find anchor and positive
        anchorID = random.randint(0, len(dataset[identidy1]) - 1)
        positiveID = random.randint(0, len(dataset[identidy1]) - 1)
        while positiveID == anchorID:
            positiveID = random.randint(0, len(dataset[identidy1]) - 1)
        
        # find negative
        negativeID = random.randint(0, len(dataset[identidy2]) - 1)

        # push them to all triplets
        allTriplets.append(dataset[identidy1][anchorID].to(device))
        allTriplets.append(dataset[identidy1][positiveID].to(device))
        allTriplets.append(dataset[identidy2][negativeID].to(device))

    return allTriplets 

def train(model, dataset, optimizer, epochs, device, cntTriplets, repeat, margin):
    for epoch in range(1, epochs + 1):
        startEpoch = time.time()

        # generate triplets
        triplets = generateTripletsDumb(dataset, cntTriplets, device)
        X = torch.cat(triplets).to(device)

        del triplets
        torch.cuda.empty_cache() 
        losses = np.zeros(repeat)

        interEpoch = time.time()

        for rp in range(repeat):
            Y = model(X)
            loss = 0
            for index in range(cntTriplets):
                a = Y[3 * index]
                b = Y[3 * index + 1]
                c = Y[3 * index + 2]

        
                distAB = ((a - b)**2).sum()
                distAC = ((a - c)**2).sum()

                del a 
                del b
                del c
                torch.cuda.empty_cache() 
                loss += F.relu(distAB - distAC + margin)
            

            loss /= cntTriplets 
            losses[rp] = loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            del Y
            torch.cuda.empty_cache() 
                
        finishEpoch = time.time()
        torch.save(model.state_dict(), model.name)
        print("finished epoch", epoch, "in", '%.4f'%(finishEpoch - startEpoch), "seconds", "loss = ", losses.mean(), "|", '%.4f'%(finishEpoch - interEpoch))
        #print(epoch, losses)
        del X
        torch.cuda.empty_cache() 



    pass


def genThreshold(cntTriplets):
    m1 = 0
    m2 = 0
    t1 = 0
    t2 = 0

    for i in range(1000):
        print("start =", i)
    
        # generate triplets
        triplets = generateTripletsDumb(dataset, cntTriplets, device)
        X = torch.cat(triplets).to(device)

        del triplets
        torch.cuda.empty_cache() 
        Y = model(X)

        for index in range(cntTriplets):
            a = Y[3 * index]
            b = Y[3 * index + 1]
            c = Y[3 * index + 2]

        
            distAB = ((a - b)**2).sum()
            distAC = ((a - c)**2).sum()

            
            t1 += 1
            t2 += 1

            m1 += distAB.item()
            m2 += distAC.item()

            del a 
            del b
            del c
            torch.cuda.empty_cache() 
        del X
        del Y
        torch.cuda.empty_cache() 

    m1 /= t1 
    m2 /= t2
    return (m1 + m2) / 2

def test(threshold, cntTriplets):
    g1 = 0
    g2 = 0
    t1 = 0
    t2 = 0

    for i in range(1000):
    
        # generate triplets
        triplets = generateTripletsDumb(dataset, cntTriplets, device)
        X = torch.cat(triplets).to(device)

        del triplets
        torch.cuda.empty_cache() 
        Y = model(X)

        for index in range(cntTriplets):
            a = Y[3 * index]
            b = Y[3 * index + 1]
            c = Y[3 * index + 2]

        
            distAB = ((a - b)**2).sum()
            distAC = ((a - c)**2).sum()

            
            t1 += 1
            t2 += 1

            if distAB <= threshold:
                g1 += 1

            if distAC > threshold:
                g2 += 1

            del a 
            del b
            del c
            torch.cuda.empty_cache() 

        
        del X
        del Y
        torch.cuda.empty_cache() 
    print(threshold, ":", g1 / t1, g2 / t2)
# choose t = 0.82 sau 0.83

import torchvision.transforms as transforms
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # (x - mean) / std  

import cv2
img1 = cv2.imread("obama1.png")
img2 = cv2.imread("obama2.png")
img3 = cv2.imread("virgil1.png")

cv2.imshow("obama1", img1)
cv2.imshow("obama2", img2)
cv2.imshow("virgil1", img3)
cv2.waitKey(0)
cv2.destroyAllWindows()

img1 = cv2.resize(gen(img1), (255, 255))
img2 = cv2.resize(gen(img2), (255, 255))
img3 = cv2.resize(gen(img3), (255, 255))

cv2.imshow("obama2", img1)
cv2.imshow("obama1", img2)
cv2.imshow("virgil1", img3)
cv2.waitKey(0)

A = transform(img1).unsqueeze(0).to(device)
B = transform(img2).unsqueeze(0).to(device)
C = transform(img3).unsqueeze(0).to(device)

A = model(A)
B = model(B)
C = model(C)

distAB = ((A - B)**2).sum()
distAC = ((A - C)**2).sum()

print(distAB)
print(distAC)

print(distAB <= 0.83)
print(distAC > 0.83)

exit(0)
test(0.80, 1)
test(0.81, 1)
test(0.82, 1)
test(0.83, 1)
test(0.84, 1)
test(0.85, 1)
test(0.86, 1)
test(0.87, 1)
test(0.88, 1)
test(0.89, 1)
test(0.90, 1)
'''
choose t = 0.82 sau 0.83 
0.8 : 0.939 0.957
0.81 : 0.939 0.931
0.82 : 0.955 0.95
0.83 : 0.96 0.943
0.84 : 0.957 0.943
0.85 : 0.969 0.942
0.86 : 0.943 0.929
0.87 : 0.963 0.941
0.88 : 0.968 0.936
0.89 : 0.952 0.921
0.9 : 0.968 0.921
'''
'''
0 : 0.0 1.0
0.1 : 0.002 1.0
0.2 : 0.098 1.0
0.3 : 0.266 0.998
0.4 : 0.516 0.995
0.5 : 0.711 0.993
0.6 : 0.815 0.984
0.7 : 0.886 0.98
0.8 : 0.944 0.955
0.9 : 0.971 0.93
1.0 : 0.983 0.896
1.1 : 0.989 0.892
1.2 : 0.996 0.829
1.3 : 0.994 0.801
1.4 : 0.998 0.731
1.5 : 0.998 0.721
'''
# 1.207756953779608 : 0.994 0.829 values
exit(0)
# trained for 13473 for 0.3744 seconds

train(model = model, dataset = dataset, epochs = 1000000, optimizer = optimizer, margin = margin, device = device, cntTriplets = 50, repeat = 1)

'''
finished epoch 13458 in 0.3882 seconds loss =  3.232747258152813e-05 | 0.3232
finished epoch 13459 in 0.3778 seconds loss =  0.0036259498447179794 | 0.3230
finished epoch 13460 in 0.3837 seconds loss =  0.008651360869407654 | 0.3337
finished epoch 13461 in 0.3866 seconds loss =  0.009541715495288372 | 0.3351
finished epoch 13462 in 0.3854 seconds loss =  0.003437510458752513 | 0.3438
finished epoch 13463 in 0.3844 seconds loss =  0.01745423674583435 | 0.3438
finished epoch 13464 in 0.3936 seconds loss =  0.006356960162520409 | 0.3479
finished epoch 13465 in 0.3778 seconds loss =  0.0 | 0.3270
finished epoch 13466 in 0.3785 seconds loss =  0.014007491990923882 | 0.3322
finished epoch 13467 in 0.3843 seconds loss =  0.0033168538939207792 | 0.3438
finished epoch 13468 in 0.3941 seconds loss =  0.0041534979827702045 | 0.3537
finished epoch 13469 in 0.4018 seconds loss =  0.009465144947171211 | 0.3436
finished epoch 13470 in 0.4053 seconds loss =  0.015207096934318542 | 0.3428
finished epoch 13471 in 0.4056 seconds loss =  0.007992786355316639 | 0.3408
finished epoch 13472 in 0.3951 seconds loss =  0.0252251997590065 | 0.3366
finished epoch 13473 in 0.3744 seconds loss =  0.0018772640032693744 | 0.3191
'''
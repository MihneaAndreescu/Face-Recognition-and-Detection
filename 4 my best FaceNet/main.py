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

dataset = loadIdentities(cnt = 5696 - 5650 * 1, H = 150, W = 150, verbose = True) # cnt max = 5696
#exit(0)
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

def draw():
    import matplotlib.pyplot as plt
    plt.style.use('seaborn-whitegrid')
    import numpy as np
    
    x = np.linspace(0, 10, 30)
    y = np.sin(x)

    plt.plot(x, y, 'o', color = 'black')
    plt.show()

    exit(0)
#draw()
'''
#t = genThreshold(1)
test(0.0, 1)
test(0.1, 1)
test(0.2, 1)
test(0.3, 1)
test(0.4, 1)
test(0.5, 1)
test(0.6, 1)
test(0.7, 1)
test(0.8, 1)
test(0.9, 1)
test(1.0, 1)
test(1.1, 1)
test(1.2, 1)
test(1.3, 1)
test(1.4, 1)
test(1.5, 1)
test(1.6, 1)
test(1.7, 1)
test(1.8, 1)
test(1.9, 1)
test(2.0, 1)
test(2.1, 1)
test(2.2, 1)
test(2.3, 1)
test(2.4, 1)
test(2.5, 1)
test(2.6, 1)
test(2.7, 1)
test(2.8, 1)
test(2.9, 1)
test(3.0, 1)
exit(0)'''
'''
0.0 : 0.0 1.0
0.1 : 0.0 1.0
0.2 : 0.018 1.0
0.3 : 0.07 1.0
0.4 : 0.182 1.0
0.5 : 0.326 1.0
0.6 : 0.644 0.999
0.7 : 0.661 0.994
0.8 : 0.739 0.99
0.9 : 0.765 0.983
1.0 : 0.911 0.975
1.1 : 0.921 0.972
1.2 : 0.918 0.965
1.3 : 0.95 0.938
1.4 : 0.956 0.938
1.5 : 0.964 0.927
1.6 : 0.969 0.887
1.7 : 0.972 0.858
1.8 : 0.968 0.829
1.9 : 0.971 0.799
2.0 : 0.973 0.765
2.1 : 0.978 0.756
2.2 : 0.974 0.715
2.3 : 0.968 0.701
2.4 : 0.969 0.694
2.5 : 0.981 0.631
2.6 : 0.997 0.6
2.7 : 1.0 0.58
2.8 : 1.0 0.56
2.9 : 1.0 0.527
3.0 : 0.999 0.511
'''


import torchvision.transforms as transforms
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # (x - mean) / std  

import cv2
img1 = cv2.imread("eu1.jpg")
img2 = cv2.imread("eu2.jpg")
img3 = cv2.imread("medi1.jpg")

cv2.imshow("A", img1)
cv2.imshow("P", img2)
cv2.imshow("N", img3)
cv2.waitKey(0)
cv2.destroyAllWindows()

kek1, img1 = gen(img1)
kek2, img2 = gen(img2)
kek3, img3 = gen(img3)

assert kek1 == True
assert kek2 == True
assert kek3 == True

img1 = cv2.resize(img1, (150, 150))
img2 = cv2.resize(img2, (150, 150))
img3 = cv2.resize(img3, (150, 150))

cv2.imshow("A", img1)
cv2.imshow("P", img2)
cv2.imshow("N", img3)
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

print(distAB <= 1.25) # o sa apara True pe ecran daca in primele doua poze e ac om False otherwise
print(distAC <= 1.25) # o sa apara True pe ecran daca in prima si a treia poze e ac om False otherwise

# True
# False
exit(0)

'''
#train(model = model, dataset = dataset, epochs = 1000000, optimizer = optimizer, margin = margin, device = device, cntTriplets = 50, repeat = 1)

# started training at 9:37 AM
# finished training at 12:20 PM
# trained for 22104 epochs
'''
'''
finished epoch 22092 in 0.4153 seconds loss =  0.006838600616902113 | 0.3933
finished epoch 22093 in 0.4133 seconds loss =  8.00758607510943e-06 | 0.3910
finished epoch 22094 in 0.4238 seconds loss =  0.009997661225497723 | 0.4123
finished epoch 22095 in 0.4348 seconds loss =  0.0053011104464530945 | 0.4129
finished epoch 22096 in 0.4414 seconds loss =  0.0039910972118377686 | 0.4154
finished epoch 22097 in 0.4298 seconds loss =  0.0 | 0.4038
finished epoch 22098 in 0.4271 seconds loss =  0.014038294553756714 | 0.4011
finished epoch 22099 in 0.4195 seconds loss =  0.005048039834946394 | 0.3968
finished epoch 22100 in 0.4127 seconds loss =  0.0 | 0.3871
finished epoch 22101 in 0.4262 seconds loss =  0.0 | 0.4037
finished epoch 22102 in 0.4205 seconds loss =  0.007115745451301336 | 0.3940
finished epoch 22103 in 0.4368 seconds loss =  0.004207410849630833 | 0.4068
finished epoch 22104 in 0.4220 seconds loss =  0.0009324327111244202 | 0.3957
'''
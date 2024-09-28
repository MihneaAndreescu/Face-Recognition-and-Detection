from dataloader import loadIdentities
from network import NN
import torch.optim as optim
import time
import random
import torch 
import torch.nn.functional as F
import numpy as np

import torch.nn as nn
print(torch.cuda.is_available())
exit(0)
device = "cuda"
#device = "cpu"

dataset = loadIdentities(cnt = 5749, H = 445, W = 445, verbose = True) # cnt max = 5749
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

# 6718130


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
                loss += F.relu(distAB - distAC + margin) # smecherie aici :)) eroare dubioasa la max(x, 0)
            
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

            if i  % 100 == 0:
                print(g1 / t1, g2 / t2)
        del X
        del Y
        torch.cuda.empty_cache() 
    

thr = genThreshold(1) 
print(thr)
test(thr, 1) 
exit(0)

# trained for 4598 epochs
train(model = model, dataset = dataset, epochs = 10000, optimizer = optimizer, margin = margin, device = device, cntTriplets = 5, repeat = 1)

import time
import random
import torch 
import torch.nn.functional as F
import numpy as np

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
        
        del X
        torch.cuda.empty_cache() 



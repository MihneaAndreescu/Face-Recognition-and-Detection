import cv2
import os
from os import listdir
import torchvision.transforms as transforms

def loadIdentities(cnt, H, W, verbose = False):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # (x - mean) / std  
    dataset = []

    people = [person for person in listdir("data")]

    assert cnt <= len(people)

    for i in range(cnt):
        dataset.append([])
        names = [person for person in listdir(os.path.join("data", people[i]))]
        
        for imgName in names:
            img = cv2.imread(os.path.join("data", people[i], imgName), cv2.IMREAD_COLOR)
            img = cv2.resize(img, (H, W))
            
            dataset[i].append((transform(img).unsqueeze(0)))

        if verbose == True:
            print("loaded", i + 1, "out of", cnt, "identities")
    return dataset


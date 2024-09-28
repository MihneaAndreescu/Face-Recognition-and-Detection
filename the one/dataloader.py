import cv2
import os
from os import listdir
import torchvision.transforms as transforms
import shutil

def gen(img):
    face_cascade = cv2.CascadeClassifier('theNets/haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) == 0:
        return False, img 
    
    maxArea = -1
    for (x, y, w, h) in faces:
        if h * w > maxArea:
            maxArea = h * w

    for (x, y, w, h) in faces:
        if h * w == maxArea:
            img = img[x : x + w, y : y + h]
            return True, img



def prepData():
    people = [person for person in listdir("trainData")]

    for i in range(len(people)):
        names = [person for person in listdir(os.path.join("trainData", people[i]))]
        good = True

        for imgName in names:
            img = cv2.imread(os.path.join("trainData", people[i], imgName), cv2.IMREAD_COLOR)
            working, img = gen(img)
    
            if working == False:
                good = 0
                break

            cv2.imwrite(os.path.join("trainData", people[i], imgName), img)
        
        if good == False:
            shutil.rmtree(os.path.join("trainData", people[i]))

            
def loadIdentities(cnt, H, W, verbose = False):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # (x - mean) / std  
    dataset = []

    people = [person for person in listdir("trainData")]

    for i in range(cnt):
        dataset.append([])
        names = [person for person in listdir(os.path.join("trainData", people[i]))]

        for imgName in names:
            img = cv2.resize(cv2.imread(os.path.join("trainData", people[i], imgName), cv2.IMREAD_COLOR), (H, W))
            dataset[i].append((transform(img).unsqueeze(0)))

        if verbose == True:
            print("loaded", i + 1, "out of", cnt, "identities")
    
    return dataset

import cv2
import os
from os import listdir
import torchvision.transforms as transforms

bad = 0

def gen(img):

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) == 0:
        global bad 
        bad += 1
        return img 

    assert(len(faces) >= 1)
    mxArea = -1
    for (x, y, w, h) in faces:
        if h * w > mxArea:
            mxArea = h * w
    for (x, y, w, h) in faces:
        if h * w != mxArea:
            continue
        img = img[x : x + w, y : y + h]
        return img
    assert (0)

def prepData():
    global bad
    bad = 0
    people = [person for person in listdir("data")]

    for i in range(len(people)):
        names = [person for person in listdir(os.path.join("data", people[i]))]
        

        for imgName in names:
            img = cv2.imread(os.path.join("data", people[i], imgName), cv2.IMREAD_COLOR)

            img = gen(img)
            cv2.imwrite(os.path.join("data", people[i], imgName), img)
            

        print("done", i, "out of", len(people), "| bad =", bad)

def loadIdentities(cnt, H, W, verbose = False):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # (x - mean) / std  
    dataset = []

    people = [person for person in listdir("data")]

    assert cnt <= len(people)

    for i in range(cnt):
        dataset.append([])
        names = [person for person in listdir(os.path.join("data", people[i]))]
        
        ttt = 1000

        for imgName in names:
            img = cv2.imread(os.path.join("data", people[i], imgName), cv2.IMREAD_COLOR)
            img = cv2.resize(img, (H, W))
            
            dataset[i].append((transform(img).unsqueeze(0)))

        if verbose == True:
            print("loaded", i + 1, "out of", cnt, "identities")
    return dataset

#prepData()
#exit(0)
#dataset = loadIdentities(cnt = 5749, H = 445, W = 445, verbose = True) # cnt max = 5749
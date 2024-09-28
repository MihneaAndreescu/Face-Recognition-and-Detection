import cv2
import os
from os import listdir
import torchvision.transforms as transforms
import shutil

bad = 0

def gen(img):

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) == 0:
        global bad 
        bad += 1
        return 0, img 

    assert(len(faces) >= 1)
    mxArea = -1
    for (x, y, w, h) in faces:
        if h * w > mxArea:
            mxArea = h * w
    for (x, y, w, h) in faces:
        if h * w != mxArea:
            continue
        img = img[x : x + w, y : y + h]
        return 1, img
    assert (0)

def prepData():
    global bad
    bad = 0
    people = [person for person in listdir("data")]

    for i in range(len(people)):
        names = [person for person in listdir(os.path.join("data", people[i]))]
        
        #print(os.path.join("data", people[i]))
        #exit(0)
        good = True

        for imgName in names:
            img = cv2.imread(os.path.join("data", people[i], imgName), cv2.IMREAD_COLOR)

            kek, img = gen(img)
    
            if kek == 0:
                good = 0
                break

            cv2.imwrite(os.path.join("data", people[i], imgName), img)
        
        if good == False:
            print("deleted", os.path.join("data", people[i]), "I'm at", i)
            shutil.rmtree(os.path.join("data", people[i]))

            
            #print("error")

        print("done", i, "out of", len(people), "| bad =", bad)

def loadIdentities(cnt, H, W, verbose = False):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # (x - mean) / std  
    dataset = []

    people = [person for person in listdir("data")]

    #print(len(people))
    #exit(0)
    assert cnt <= len(people)
    
    mx1 = 0
    mx2 = 0

    s1 = 0
    s2 = 0
    c1 = 0
    c2 = 0
    
    bad1 = 0
    bad2 = 0

    for i in range(cnt):
        dataset.append([])
        names = [person for person in listdir(os.path.join("data", people[i]))]
        
        ttt = 1000


        for imgName in names:
            img = cv2.imread(os.path.join("data", people[i], imgName), cv2.IMREAD_COLOR)
            s1 += img.shape[0]
            s2 += img.shape[1]
            c1 += 1
            c2 += 1
            mx1 = max(mx1, img.shape[0])
            mx2 = max(mx2, img.shape[1])
            
            bad1 += (img.shape[0] > 150)
            bad2 += (img.shape[1] > 150)

            print(i, ":", img.shape, "|", mx1, mx2, "|", s1 / c1, s2 / c2, bad1, bad2)
            

            img = cv2.resize(img, (H, W))
            
            dataset[i].append((transform(img).unsqueeze(0)))

        continue
        if verbose == True:
            print("loaded", i + 1, "out of", cnt, "identities")
    return dataset

#prepData()
#exit(0)
#dataset = loadIdentities(cnt = 5696, H = 445, W = 445, verbose = True) # cnt max = 5696
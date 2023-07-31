import os
import glob

def train_prep(file="train.txt", path="/home/as-hunt/Etra-Space/white-thirds/train/"):
    filoo = open(path + file, 'w')
    for image in os.listdir(path):
        if image.endswith(".jpg"):
            print(image)
            filoo.write(path + image + "\n")
    filoo.close()

def test_prep(file="test.txt", path="/home/as-hunt/Etra-Space/white-thirds/test/"):
    filoo = open(path + file, 'w')
    for image in os.listdir(path):
        if image.endswith(".jpg"):
            print(image)
            filoo.write(path + image + "\n")
    filoo.close()

def val_prep(file="valid.txt", path="/home/as-hunt/Etra-Space/white-thirds/valid/"):
    filoo = open(path + file, 'w')
    for image in os.listdir(path):
        if image.endswith(".jpg"):
            print(image)
            filoo.write(path + image + "\n")
    filoo.close()


def allDaPrep():
    train_prep()
    test_prep()
    val_prep()

if __name__ == "__main__":
    allDaPrep()

import os
import random


def writeFile(rootdirs, filePath):
    file = open(filePath, 'w')
    rootdir = rootdirs
    list = os.listdir(rootdir)
    list.sort()
    print(list)
    filenames = []
    for i in range(10):
        f = []
        path = os.path.join(rootdir, list[i])
        subList = os.listdir(rootdir + "" + os.path.basename(path))
        for j in range(0, len(subList)):
            f.append("/"+list[i]+"/"+subList[j] + " " + str(i))
        filenames.append(f)
    random_list = []
    for i in range(10):
        random_list += filenames[i]
    random.shuffle(random_list)
    for i in range(len(random_list)):
        file.write(random_list[i] + "\n")
    file.close()


if __name__ == '__main__':
    writeFile('./cifar-10/test/', 'cifar10_test_jpg_10000.txt')
    writeFile('./cifar-10/train/', 'cifar10_train_jpg_60000.txt')

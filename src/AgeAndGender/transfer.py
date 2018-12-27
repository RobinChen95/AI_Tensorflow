# coding: utf-8

import glob
import os
import random

AGE = True

age_table = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
sex_table = ['f', 'm']  # f:女; m:男

face_set_fold = 'Adience'

fold_0_data = os.path.join(face_set_fold, 'fold_0_data.txt')
fold_1_data = os.path.join(face_set_fold, 'fold_1_data.txt')
fold_2_data = os.path.join(face_set_fold, 'fold_2_data.txt')
fold_3_data = os.path.join(face_set_fold, 'fold_3_data.txt')
fold_4_data = os.path.join(face_set_fold, 'fold_4_data.txt')

face_image_set = os.path.join(face_set_fold, 'aligned')


def parse_data(fold_x_data):
    data_set = []

    with open(fold_x_data, 'r') as f:
        line_one = True
        for line in f:
            tmp = []
            if line_one == True:
                line_one = False
                continue

            tmp.append(line.split('\t')[0])
            tmp.append(line.split('\t')[1])
            tmp.append(line.split('\t')[3])
            tmp.append(line.split('\t')[4])

            file_path = os.path.join(face_image_set, tmp[0])
            if os.path.exists(file_path):
                filenames = glob.glob(file_path + "/*.jpg")
                for filename in filenames:
                    if tmp[1] in filename:
                        break
                if AGE:
                    if tmp[2] in age_table:
                        data_set.append([filename, age_table.index(tmp[2])])
                else:
                    if tmp[3] in sex_table:
                        data_set.append([filename, sex_table.index(tmp[3])])

    return data_set


def write_txt(isTrain=True):
    if isTrain:
        if AGE:
            filename = os.path.join(face_set_fold, "age_train.txt")
        else:
            filename = os.path.join(face_set_fold, "gender_train.txt")
        data_set_0 = parse_data(fold_0_data)
        data_set_1 = parse_data(fold_1_data)
        data_set_2 = parse_data(fold_2_data)
        data_set_3 = parse_data(fold_3_data)
        data_set = data_set_0 + data_set_1 + data_set_2 + data_set_3
    else:
        if AGE:
            filename = os.path.join(face_set_fold, "age_text.txt")
        else:
            filename = os.path.join(face_set_fold, "gender_text.txt")
        data_set_4 = parse_data(fold_4_data)
        data_set = data_set_4
    random.shuffle(data_set)
    with open(filename, "w") as f:
        for p in data_set:
            f.write(p[0] + " " + str(p[1]) + "\n")


def main():
    write_txt()
    write_txt(False)


if __name__ == '__main__':
    main()

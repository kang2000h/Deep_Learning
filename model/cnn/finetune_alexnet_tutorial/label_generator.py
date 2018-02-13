import os
import random
import argparse
from glob import glob

parser = argparse.ArgumentParser()
parser.add_argument('--path_list', type=str, nargs='*', help="data to use to making your model")
parser.add_argument('--train_rate', type=float,
                    help="the rate of train set for total dataset")

args = parser.parse_args()

import numpy as np

def main():
    train_list = []
    val_list = []
    for ind, each_class_path in enumerate(args.path_list):
        ob_files = glob(os.path.join(each_class_path, '*'))
        temp = [(files, ind) for files in ob_files]
        print(ind, "class num : ", len(temp))

        random.shuffle(temp)

        train_list += temp[:int(float(len(temp))*args.train_rate)]
        val_list += temp[int(float(len(temp))*args.train_rate):]

    print("train_list", len(train_list))
    print("val_list", len(val_list))

    input_ = input("Do you want to save the paths?(Y/n)")

    if input_ is 'Y' or input_ is 'y':
        save_dir = input("Let me know save directory you want to save #")
        if save_dir is not None:
            with open("train.txt", "w") as f:
                [f.write(p_l[0]+" "+str(p_l[1])+'\n') for p_l in train_list]
            with open("val.txt", "w") as f:
                [f.write(p_l[0]+" "+str(p_l[1])+'\n') for p_l in val_list]
        else:
            print("save_dir is None.")
    else :
        return


if __name__=="__main__":
    main()

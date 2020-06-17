import os
import sys
import numpy as np


def Main():

    folder_path = sys.argv[1]

    for sequence in ['train', 'test']:

        if sequence == 'train':

            train_folder_path = os.path.join(folder_path, 'train')
            img_folder_path = os.path.join(train_folder_path, 'img')
            img_files_list = os.listdir(img_folder_path)

            total_files_size = len(img_files_list)
            val_size = int(0.15 * total_files_size)
            val_idx = sorted(np.random.choice(total_files_size, val_size, replace=False).tolist())

            traintxt = open(os.path.join(train_folder_path, 'trainval.txt'), 'w')
            valtxt = open(os.path.join(train_folder_path, 'val.txt'), 'w')

            # Remove file type
            for idx in range(len(img_files_list)):
                file_name, file_extension = os.path.splitext(img_files_list[idx])
                if idx in val_idx:
                    valtxt.write(file_name+'\n')
                else:
                    traintxt.write(file_name+'\n')
            traintxt.close()
            valtxt.close()

        else:

            test_folder_path = os.path.join(folder_path, 'test')
            img_folder_path = os.path.join(test_folder_path, 'img')
            img_files_list = os.listdir(img_folder_path)

            testtxt = open(os.path.join(test_folder_path, 'test.txt'), 'w')

            # Remove file type
            for idx in range(len(img_files_list)):
                file_name, file_extension = os.path.splitext(img_files_list[idx])
                testtxt.write(file_name + '\n')


if __name__ == '__main__':

    Main()

import numpy as np
from operator import itemgetter
import os
import code

if __name__ == '__main__':
    
    cls_labels_dict = np.load('cls_labels.npy', allow_pickle=True).item()
    
    # Extract Train Images
    train_img_list = [img_name.strip() for img_name in open('train_id_ori.txt').readlines()]
    train_npy_list = []
    for id in train_img_list:
        train_npy_list.append([id, np.sum(cls_labels_dict[id]), cls_labels_dict[id]])
    train_npy_sorted = sorted(train_npy_list, key=itemgetter(1))[::-1]
    train_npy_final = train_npy_sorted[:50]
    train_classes = np.zeros(20)
    for lst in train_npy_final:
        train_classes += lst[2]
    train_indexes = np.nonzero(train_classes < 5)[0]
    train_num_iter = 5 
    for index in train_indexes:
        for num in range(train_num_iter):
            for lst in train_npy_sorted:
                if lst not in train_npy_final:
                    labels = lst[2]
                    if labels[index] == 1:
                        train_npy_final.append(lst)
                        train_classes += labels
                        break
    # Check no duplicates exist for train
    check_duplicate_ids_for_train = len(set([lst[0] for lst in train_npy_final])) < 100
    if check_duplicate_ids_for_train: 
        print(f"Duplicates Exist in Images")
        exit
    print(f'Train Classes = {train_classes}\n')
    # Write 
    train_file_name = 'train_aug_id.txt'
    if not os.path.exists(train_file_name):
        with open(train_file_name, 'w') as train_file:
            for lst in train_npy_final:
                train_file.write(lst[0])
                train_file.write('\n')
            
    # Extract Val Images
    val_img_list = [img_name.strip() for img_name in open('val_id_ori.txt').readlines()]
    val_npy_list = []
    for id in val_img_list:
        val_npy_list.append([id, np.sum(cls_labels_dict[id]), cls_labels_dict[id]])
    val_npy_sorted = sorted(val_npy_list, key=itemgetter(1))[::-1]
    val_npy_final = val_npy_sorted[:25]
    val_classes = np.zeros(20)
    for lst in val_npy_final:
        val_classes += lst[2]
    print(f'Val Classes = {val_classes}\n')
    # Write 
    val_file_name = 'val_id.txt'
    if not os.path.exists(val_file_name):
        with open(val_file_name, 'w') as val_file:
            for lst in val_npy_final:
                val_file.write(lst[0])
                val_file.write('\n')

    # code.interact(local=dict(globals(), **locals()))

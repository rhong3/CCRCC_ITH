"""
Prepare training and testing datasets as CSV dictionaries 2.0

Created on 04/26/2019; modified on 11/06/2019

@author: RH
"""
import os
import pandas as pd
import sklearn.utils as sku
import numpy as np
import re


# pair tiles of 10x, 5x, 2.5x of the same area
def paired_tile_ids_in(patient, slide, label, root_dir):
    dira = os.path.isdir(root_dir + 'level1')
    dirb = os.path.isdir(root_dir + 'level2')
    dirc = os.path.isdir(root_dir + 'level3')
    if dira and dirb and dirc:
        fac = 1000
        ids = []
        for level in range(1, 4):
            dirr = root_dir + 'level{}'.format(str(level))
            for id in os.listdir(dirr):
                if '.png' in id:
                    x = int(float(id.split('x-', 1)[1].split('-', 1)[0]) / fac)
                    y = int(float(re.split('.p', id.split('y-', 1)[1])[0]) / fac)
                    ids.append([patient, slide, label, level, dirr + '/' + id, x, y])
        ids = pd.DataFrame(ids, columns=['Patient_ID', 'Slide_ID', 'label', 'level', 'path', 'x', 'y'])
        idsa = ids.loc[ids['level'] == 1]
        idsa = idsa.drop(columns=['level'])
        idsa = idsa.rename(index=str, columns={"path": "L1path"})
        idsb = ids.loc[ids['level'] == 2]
        idsb = idsb.drop(columns=['Patient_ID', 'Slide_ID', 'label', 'level'])
        idsb = idsb.rename(index=str, columns={"path": "L2path"})
        idsc = ids.loc[ids['level'] == 3]
        idsc = idsc.drop(columns=['Patient_ID', 'Slide_ID', 'label', 'level'])
        idsc = idsc.rename(index=str, columns={"path": "L3path"})
        idsa = pd.merge(idsa, idsb, on=['x', 'y'], how='left', validate="many_to_many")
        idsa['x'] = idsa['x'] - (idsa['x'] % 2)
        idsa['y'] = idsa['y'] - (idsa['y'] % 2)
        idsa = pd.merge(idsa, idsc, on=['x', 'y'], how='left', validate="many_to_many")
        idsa = idsa.drop(columns=['x', 'y'])
        idsa = idsa.dropna()
        idsa = sku.shuffle(idsa)
    else:
        print('Pass: ', root_dir)
        idsa = pd.DataFrame(columns=['Patient_ID', 'Slide_ID', 'label', 'L1path', 'L2path', 'L3path'])

    return idsa


# Prepare label at per patient level
def big_image_sum(label_col, path, ref_file):
    ref = pd.read_csv(ref_file, header=0)
    big_images = []
    ref = ref.loc[ref[label_col].notna()]
    for idx, row in ref.iterrows():
        big_images.append([row['Patient_ID'], row['Slide_ID'], row['Source'],
                           path + "/CCRCC/{}/{}/".format(str(row['Patient_ID']), row['Slide_ID_tag']), row[label_col]])
    datapd = pd.DataFrame(big_images, columns=['Patient_ID', 'Slide_ID', 'source', 'path', 'label'])
    datapd = datapd.dropna()

    return datapd


# seperate into training and testing; each type is the same separation ratio on big images
# test and train csv files contain tiles' path.
def set_sep(alll, path, cut=0.4):
    trlist = []
    telist = []
    valist = []

    for lb in list(alll.label.unique()):
        sub = alll[alll['label'] == lb]
        unq = list(sub.Slide_ID.unique())
        np.random.shuffle(unq)
        validation = unq[:np.max([int(len(unq) * cut / 2), 1])]
        valist.append(sub[sub['Slide_ID'].isin(validation)])
        test = unq[np.max([int(len(unq) * cut / 2), 1]):np.max([int(len(unq) * cut), 2])]
        telist.append(sub[sub['Slide_ID'].isin(test)])
        train = unq[np.max([int(len(unq) * cut), 2]):]
        trlist.append(sub[sub['Slide_ID'].isin(train)])

    test = pd.concat(telist)
    train = pd.concat(trlist)
    validation = pd.concat(valist)

    test.to_csv(path + '/te_sample_raw.csv', header=True, index=False)
    train.to_csv(path + '/tr_sample_raw.csv', header=True, index=False)
    validation.to_csv(path + '/va_sample_raw.csv', header=True, index=False)

    test_tiles = pd.DataFrame(columns=['Patient_ID', 'Slide_ID', 'label', 'L1path', 'L2path', 'L3path'])
    train_tiles = pd.DataFrame(columns=['Patient_ID', 'Slide_ID', 'label', 'L1path', 'L2path', 'L3path'])
    validation_tiles = pd.DataFrame(columns=['Patient_ID', 'Slide_ID', 'label', 'L1path', 'L2path', 'L3path'])
    for idx, row in test.iterrows():
        tile_ids = paired_tile_ids_in(row['Patient_ID'], row['Slide_ID'], row['label'], row['path'])
        test_tiles = pd.concat([test_tiles, tile_ids])
    for idx, row in train.iterrows():
        tile_ids = paired_tile_ids_in(row['Patient_ID'], row['Slide_ID'], row['label'], row['path'])
        train_tiles = pd.concat([train_tiles, tile_ids])
    for idx, row in validation.iterrows():
        tile_ids = paired_tile_ids_in(row['Patient_ID'], row['Slide_ID'], row['label'], row['path'])
        validation_tiles = pd.concat([validation_tiles, tile_ids])

    # No shuffle on test set
    train_tiles = sku.shuffle(train_tiles)
    validation_tiles = sku.shuffle(validation_tiles)
    test_tiles = test_tiles.sort_values(by=['Slide_ID'], ascending=True)

    test_tiles.to_csv(path+'/te_sample_full.csv', header=True, index=False)
    train_tiles.to_csv(path+'/tr_sample_full.csv', header=True, index=False)
    validation_tiles.to_csv(path+'/va_sample_full.csv', header=True, index=False)



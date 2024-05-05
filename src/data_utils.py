import numpy as np
import pandas as pd
import os

from sklearn.model_selection import StratifiedKFold

from src.utils import filter_data, upsample_data

import ast


def get_metadata(n_folds):
    base_dir = 'data'
    train_dir = base_dir + '/train_audio/'
    test_dir = base_dir + '/test_soundscapes/'
    unlabeled_dir = base_dir + '/unlabeled_soundscapes/'

    class_names = sorted(os.listdir(train_dir))
    n_classes = len(class_names)
    class_labels = list(range(n_classes))
    label2name = dict(zip(class_labels, class_names))
    name2label = {v:k for k,v in label2name.items()}

    def get_label_from_name(name):
        if name not in name2label.keys():
            return None
        return name2label[name]

    metadata = pd.read_csv(base_dir + '/train_metadata.csv')
    metadata['filepath'] = train_dir + metadata.filename
    metadata['target'] = metadata.primary_label.map(name2label)
    metadata['secondary_targets'] = metadata.secondary_labels.map(lambda x: [get_label_from_name(name) for name in ast.literal_eval(x)])
    metadata['filename'] = metadata.filepath.map(lambda x: x.split('/')[-1])
    metadata['xc_id'] = metadata.filepath.map(lambda x: x.split('/')[-1].split('.')[0])

    # Mark samples of classes with less samples than n_folds
    metadata = filter_data(metadata, thr=n_folds)

    # Mark k-fold index in metadata
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)
    metadata = metadata.reset_index(drop=True)
    metadata["fold"] = -1
    for fold, (train_idx, val_idx) in enumerate(skf.split(metadata, metadata['primary_label'])):
        metadata.loc[val_idx, 'fold'] = fold

    cols = ["primary_label", "secondary_labels", "filepath", "target", "secondary_targets", "cv", "fold"]
    return metadata[cols]


def get_metadata_from_csv(filepath):
    base_dir = 'data'
    train_dir = base_dir + '/train_audio/'
    test_dir = base_dir + '/test_soundscapes/'
    unlabeled_dir = base_dir + '/unlabeled_soundscapes/'

    class_names = sorted(os.listdir(train_dir))
    n_classes = len(class_names)
    class_labels = list(range(n_classes))
    label2name = dict(zip(class_labels, class_names))
    name2label = {v:k for k,v in label2name.items()}

    def get_label_from_name(name):
        if name not in name2label.keys():
            return None
        return name2label[name]
    
    metadata = pd.read_csv(filepath)
    metadata['secondary_targets'] = metadata.secondary_labels.map(lambda x: [get_label_from_name(name) for name in ast.literal_eval(x)])
    return metadata


def get_fold(metadata, fold, up_thr=None):
    train_df = metadata.query("fold!=@fold | ~cv").reset_index(drop=True)
    valid_df = metadata.query("fold==@fold & cv").reset_index(drop=True)

    if up_thr is not None:
        train_df_up = upsample_data(train_df, thr=up_thr)
        train_df = train_df_up.reset_index(drop=True)

    class_weights = train_df['target'].count()/np.maximum(1, np.bincount(train_df['target']))
    class_weights = class_weights/class_weights.max()

    print(f"Num Train: {len(train_df)}, {len(train_df['target'].unique())} classes | \
Num Valid: {len(valid_df)}, {len(valid_df['target'].unique())} classes")

    return train_df, valid_df, class_weights
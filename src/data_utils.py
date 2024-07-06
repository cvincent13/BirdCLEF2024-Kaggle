import numpy as np
import pandas as pd
import os

from sklearn.model_selection import StratifiedKFold

from src.utils import filter_data, upsample_data, filter_classes

import ast

dupes = [
    ("asbfly/XC347563.ogg", "asbfly/XC341611.ogg"),
    ("grewar3/XC507658.ogg", "grewar3/XC184040.ogg"),
    
    ('asbfly/XC724266.ogg', 'asbfly/XC724148.ogg'),
    ('barswa/XC575749.ogg', 'barswa/XC575747.ogg'),
    ('bcnher/XC669544.ogg', 'bcnher/XC669542.ogg'),
    ('bkskit1/XC350251.ogg', 'bkskit1/XC350249.ogg'),
    ('blhori1/XC417215.ogg', 'blhori1/XC417133.ogg'),
    ('blhori1/XC743616.ogg', 'blhori1/XC537503.ogg'),
    ('blrwar1/XC662286.ogg', 'blrwar1/XC662285.ogg'),
    ('brakit1/XC743675.ogg', 'brakit1/XC537471.ogg'),
    ('brcful1/XC197746.ogg', 'brcful1/XC157971.ogg'),
    ('brnshr/XC510751.ogg', 'brnshr/XC510750.ogg'),
    ('btbeat1/XC665307.ogg', 'btbeat1/XC513403.ogg'),
    ('btbeat1/XC743618.ogg', 'btbeat1/XC683300.ogg'),
    ('btbeat1/XC743619.ogg', 'btbeat1/XC683300.ogg'),
    ('btbeat1/XC743619.ogg', 'btbeat1/XC743618.ogg'),
    ('categr/XC787914.ogg', 'categr/XC438523.ogg'),
    ('cohcuc1/XC253418.ogg', 'cohcuc1/XC241127.ogg'),
    ('cohcuc1/XC423422.ogg', 'cohcuc1/XC423419.ogg'),
    ('comgre/XC202776.ogg', 'comgre/XC192404.ogg'),
    ('comgre/XC602468.ogg', 'comgre/XC175341.ogg'),
    ('comgre/XC64628.ogg', 'comgre/XC58586.ogg'),
    ('comior1/XC305930.ogg', 'comior1/XC303819.ogg'),
    ('comkin1/XC207123.ogg', 'comior1/XC207062.ogg'),
    ('comkin1/XC691421.ogg', 'comkin1/XC690633.ogg'),
    ('commyn/XC577887.ogg', 'commyn/XC577886.ogg'),
    ('commyn/XC652903.ogg', 'commyn/XC652901.ogg'),
    ('compea/XC665320.ogg', 'compea/XC644022.ogg'),
    ('comsan/XC385909.ogg', 'comsan/XC385908.ogg'),
    ('comsan/XC643721.ogg', 'comsan/XC642698.ogg'),
    ('comsan/XC667807.ogg', 'comsan/XC667806.ogg'),
    ('comtai1/XC126749.ogg', 'comtai1/XC122978.ogg'),
    ('comtai1/XC305210.ogg', 'comtai1/XC304811.ogg'),
    ('comtai1/XC542375.ogg', 'comtai1/XC540351.ogg'),
    ('comtai1/XC542379.ogg', 'comtai1/XC540352.ogg'),
    ('crfbar1/XC615780.ogg', 'crfbar1/XC615778.ogg'),
    ('dafbab1/XC188307.ogg', 'dafbab1/XC187059.ogg'),
    ('dafbab1/XC188308.ogg', 'dafbab1/XC187068.ogg'),
    ('dafbab1/XC188309.ogg', 'dafbab1/XC187069.ogg'),
    ('dafbab1/XC197745.ogg', 'dafbab1/XC157972.ogg'),
    ('eaywag1/XC527600.ogg', 'eaywag1/XC527598.ogg'),
    ('eucdov/XC355153.ogg', 'eucdov/XC355152.ogg'),
    ('eucdov/XC360303.ogg', 'eucdov/XC347428.ogg'),
    ('eucdov/XC365606.ogg', 'eucdov/XC124694.ogg'),
    ('eucdov/XC371039.ogg', 'eucdov/XC368596.ogg'),
    ('eucdov/XC747422.ogg', 'eucdov/XC747408.ogg'),
    ('eucdov/XC789608.ogg', 'eucdov/XC788267.ogg'),
    ('goflea1/XC163901.ogg', 'bladro1/XC163901.ogg'),
    ('goflea1/XC208794.ogg', 'bladro1/XC208794.ogg'),
    ('goflea1/XC208795.ogg', 'bladro1/XC208795.ogg'),
    ('goflea1/XC209203.ogg', 'bladro1/XC209203.ogg'),
    ('goflea1/XC209549.ogg', 'bladro1/XC209549.ogg'),
    ('goflea1/XC209564.ogg', 'bladro1/XC209564.ogg'),
    ('graher1/XC357552.ogg', 'graher1/XC357551.ogg'),
    ('graher1/XC590235.ogg', 'graher1/XC590144.ogg'),
    ('grbeat1/XC304004.ogg', 'grbeat1/XC303999.ogg'),
    ('grecou1/XC365426.ogg', 'grecou1/XC365425.ogg'),
    ('greegr/XC247286.ogg', 'categr/XC197438.ogg'),
    ('grewar3/XC743681.ogg', 'grewar3/XC537475.ogg'),
    ('grnwar1/XC197744.ogg', 'grnwar1/XC157973.ogg'),
    ('grtdro1/XC651708.ogg', 'grtdro1/XC613192.ogg'),
    ('grywag/XC459760.ogg', 'grywag/XC457124.ogg'),
    ('grywag/XC575903.ogg', 'grywag/XC575901.ogg'),
    ('grywag/XC650696.ogg', 'grywag/XC592019.ogg'),
    ('grywag/XC690448.ogg', 'grywag/XC655063.ogg'),
    ('grywag/XC745653.ogg', 'grywag/XC745650.ogg'),
    ('grywag/XC812496.ogg', 'grywag/XC812495.ogg'),
    ('heswoo1/XC357155.ogg', 'heswoo1/XC357149.ogg'),
    ('heswoo1/XC744698.ogg', 'heswoo1/XC665715.ogg'),
    ('hoopoe/XC631301.ogg', 'hoopoe/XC365530.ogg'),
    ('hoopoe/XC631304.ogg', 'hoopoe/XC252584.ogg'),
    ('houcro1/XC744704.ogg', 'houcro1/XC683047.ogg'),
    ('houspa/XC326675.ogg', 'houspa/XC326674.ogg'),
    ('inbrob1/XC744708.ogg', 'inbrob1/XC744706.ogg'),
    ('insowl1/XC305214.ogg', 'insowl1/XC301142.ogg'),
    ('junbab2/XC282587.ogg', 'junbab2/XC282586.ogg'),
    ('labcro1/XC267645.ogg', 'labcro1/XC265731.ogg'),
    ('labcro1/XC345836.ogg', 'labcro1/XC312582.ogg'),
    ('labcro1/XC37773.ogg', 'labcro1/XC19736.ogg'),
    ('labcro1/XC447036.ogg', 'houcro1/XC447036.ogg'),
    ('labcro1/XC823514.ogg', 'gybpri1/XC823527.ogg'),
    ('laudov1/XC185511.ogg', 'grewar3/XC185505.ogg'),
    ('laudov1/XC405375.ogg', 'laudov1/XC405374.ogg'),
    ('laudov1/XC514027.ogg', 'eucdov/XC514027.ogg'),
    ('lblwar1/XC197743.ogg', 'lblwar1/XC157974.ogg'),
    ('lewduc1/XC261506.ogg', 'lewduc1/XC254813.ogg'),
    ('litegr/XC403621.ogg', 'bcnher/XC403621.ogg'),
    ('litegr/XC535540.ogg', 'litegr/XC448898.ogg'),
    ('litegr/XC535552.ogg', 'litegr/XC447850.ogg'),
    ('litgre1/XC630775.ogg', 'litgre1/XC630560.ogg'),
    ('litgre1/XC776082.ogg', 'litgre1/XC663244.ogg'),
    ('litspi1/XC674522.ogg', 'comtai1/XC674522.ogg'),
    ('litspi1/XC722435.ogg', 'litspi1/XC721636.ogg'),
    ('litspi1/XC722436.ogg', 'litspi1/XC721637.ogg'),
    ('litswi1/XC443070.ogg', 'litswi1/XC440301.ogg'),
    ('lobsun2/XC197742.ogg', 'lobsun2/XC157975.ogg'),
    ('maghor2/XC197740.ogg', 'maghor2/XC157978.ogg'),
    ('maghor2/XC786588.ogg', 'maghor2/XC786587.ogg'),
    ('malpar1/XC197770.ogg', 'malpar1/XC157976.ogg'),
    ('marsan/XC383290.ogg', 'marsan/XC383288.ogg'),
    ('marsan/XC733175.ogg', 'marsan/XC716673.ogg'),
    ('mawthr1/XC455222.ogg', 'mawthr1/XC455211.ogg'),
    ('orihob2/XC557991.ogg', 'orihob2/XC557293.ogg'),
    ('piebus1/XC165050.ogg', 'piebus1/XC122395.ogg'),
    ('piebus1/XC814459.ogg', 'piebus1/XC792272.ogg'),
    ('placuc3/XC490344.ogg', 'placuc3/XC486683.ogg'),
    ('placuc3/XC572952.ogg', 'placuc3/XC572950.ogg'),
    ('plaflo1/XC615781.ogg', 'plaflo1/XC614946.ogg'),
    ('purher1/XC467373.ogg', 'graher1/XC467373.ogg'),
    ('purher1/XC827209.ogg', 'purher1/XC827207.ogg'),
    ('pursun3/XC268375.ogg', 'comtai1/XC241382.ogg'),
    ('pursun4/XC514853.ogg', 'pursun4/XC514852.ogg'),
    ('putbab1/XC574864.ogg', 'brcful1/XC574864.ogg'),
    ('rewbul/XC306398.ogg', 'bkcbul1/XC306398.ogg'),
    ('rewbul/XC713308.ogg', 'asbfly/XC713467.ogg'),
    ('rewlap1/XC733007.ogg', 'rewlap1/XC732874.ogg'),
    ('rorpar/XC199488.ogg', 'rorpar/XC199339.ogg'),
    ('rorpar/XC402325.ogg', 'comior1/XC402326.ogg'),
    ('rorpar/XC516404.ogg', 'rorpar/XC516402.ogg'),
    ('sbeowl1/XC522123.ogg', 'brfowl1/XC522123.ogg'),
    ('sohmyn1/XC744700.ogg', 'sohmyn1/XC743682.ogg'),
    ('spepic1/XC804432.ogg', 'spepic1/XC804431.ogg'),
    ('spodov/XC163930.ogg', 'bladro1/XC163901.ogg'),
    ('spodov/XC163930.ogg', 'goflea1/XC163901.ogg'),
    ('spoowl1/XC591485.ogg', 'spoowl1/XC591177.ogg'),
    ('stbkin1/XC266782.ogg', 'stbkin1/XC266682.ogg'),
    ('stbkin1/XC360661.ogg', 'stbkin1/XC199815.ogg'),
    ('stbkin1/XC406140.ogg', 'stbkin1/XC406138.ogg'),
    ('vefnut1/XC197738.ogg', 'vefnut1/XC157979.ogg'),
    ('vefnut1/XC293526.ogg', 'vefnut1/XC289785.ogg'),
    ('wemhar1/XC581045.ogg', 'comsan/XC581045.ogg'),
    ('wemhar1/XC590355.ogg', 'wemhar1/XC590354.ogg'),
    ('whbbul2/XC335671.ogg', 'whbbul2/XC335670.ogg'),
    ('whbsho3/XC856465.ogg', 'whbsho3/XC856463.ogg'),
    ('whbsho3/XC856468.ogg', 'whbsho3/XC856463.ogg'),
    ('whbsho3/XC856468.ogg', 'whbsho3/XC856465.ogg'),
    ('whbwat1/XC840073.ogg', 'whbwat1/XC840071.ogg'),
    ('whbwoo2/XC239509.ogg', 'rufwoo2/XC239509.ogg'),
    ('whcbar1/XC659329.ogg', 'insowl1/XC659329.ogg'),
    ('whiter2/XC265271.ogg', 'whiter2/XC265267.ogg'),
    ('whtkin2/XC197737.ogg', 'whtkin2/XC157981.ogg'),
    ('whtkin2/XC430267.ogg', 'whtkin2/XC430256.ogg'),
    ('whtkin2/XC503389.ogg', 'comior1/XC503389.ogg'),
    ('whtkin2/XC540094.ogg', 'whtkin2/XC540087.ogg'),
    ('woosan/XC184466.ogg', 'marsan/XC184466.ogg'),
    ('woosan/XC545316.ogg', 'woosan/XC476064.ogg'),
    ('woosan/XC587076.ogg', 'woosan/XC578599.ogg'),
    ('woosan/XC742927.ogg', 'woosan/XC740798.ogg'),
    ('woosan/XC825766.ogg', 'grnsan/XC825765.ogg'),
    ('zitcis1/XC303866.ogg', 'zitcis1/XC302781.ogg'),
]

duplicates = [a[0] for a in dupes]


def get_full_data(base_dir, up_thr=None, weight_power=0.5, min_thr=0):
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
    metadata['filename_long'] = metadata.filepath.map(lambda x: '/'.join(x.split('/')[-2:]))
    metadata['xc_id'] = metadata.filepath.map(lambda x: x.split('/')[-1].split('.')[0])

    # filter duplicates
    metadata = metadata[~metadata['filename_long'].isin(duplicates)]

    cols = ["primary_label", "secondary_labels", "filepath", "target", "secondary_targets"]
    train_df = metadata[cols]

    if min_thr > 0:
        train_df = filter_classes(train_df, min_thr=min_thr)

    elif up_thr is not None:
        train_df_up = upsample_data(train_df, thr=up_thr)
        train_df = train_df_up.reset_index(drop=True)

    class_weights = train_df['target'].count()/np.maximum(1, np.bincount(train_df['target']))
    class_weights = ((class_weights/class_weights.max())**weight_power).astype(np.float32)

    print(f"Num Train: {len(train_df)}, {len(train_df['target'].unique())} classes")

    return train_df, class_weights


def get_metadata(n_folds, base_dir):
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


def get_metadata_from_csv(filepath, base_dir):
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


def get_fold(metadata, fold, up_thr=None, weight_power=0.5):
    fold = fold
    train_df = metadata.query("fold!=@fold | ~cv").reset_index(drop=True)
    valid_df = metadata.query("fold==@fold & cv").reset_index(drop=True)

    if up_thr is not None:
        train_df_up = upsample_data(train_df, thr=up_thr)
        train_df = train_df_up.reset_index(drop=True)

    class_weights = train_df['target'].count()/np.maximum(1, np.bincount(train_df['target']))
    class_weights = ((class_weights/class_weights.max())**weight_power).astype(np.float32)

    print(f"Num Train: {len(train_df)}, {len(train_df['target'].unique())} classes | \
Num Valid: {len(valid_df)}, {len(valid_df['target'].unique())} classes")

    return train_df, valid_df, class_weights
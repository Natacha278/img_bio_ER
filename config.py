import os

ROOT_DIR = os.environ["HOME"]

CURRENT_DIR = os.path.abspath(os.getcwd())

DATASET_FOLDER = "/datasets"

_FER_DATASET_PATH = ROOT_DIR + DATASET_FOLDER

DATASET_BIOVID = 'BIOVID'
BIO_DATASET  = 'physio_processed'
BIOVID_PATH = _FER_DATASET_PATH + '/Biovid'
BIOVID_SUBS_PATH = _FER_DATASET_PATH + '/Biovid/sub_img_red_classes'
BIOVID_EDA_PATH =  _FER_DATASET_PATH + '/Biovid/' + BIO_DATASET
BIOVID_REDUCE_LABEL_PATH = _FER_DATASET_PATH + '/Biovid/sub_labels.txt'

MODEL_BIO_PATH = ROOT_DIR + ""
MODEL_FUS_PATH = ROOT_DIR + ""

TRAIN_SOURCE_AND_TARGET = 'Train both source and target'
TRAIN_ONLY_TARGET = 'Train only target'


# ---------------------  xxx  xxxx --------------------- #


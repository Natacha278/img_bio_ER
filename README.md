#Emotion Recognition 

# Installation

## Dependencies

### Using conda

```
conda create --name <env> --file <this file>
```


### Using pip

```
pip install -r req.txt
```


## Biovid Pain and Heat Dataset

```
Biovid datasets PartA can be downloaded from here: (https://www.nit.ovgu.de/BioVid.html#PubACII17)
```

## Multi source Adaptation to Target Domain

```
CUDA_VISIBLE_DEVICES=0 python methods/multi_mod_msda.py --sub_domains_datasets_path=$BIOVID_DATASET_PATH --sub_domains_label_path=$BIOVID_DATASET_LABEL_PATH --pain_db_root_path=$BIOVID_ROOT_FOLDER_PATH
```


```

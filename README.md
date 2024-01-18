# vaik-patch-core-demo

--------

## Result
| key | oss model | vbd-fine-tuning model |
| :---: |:---------:|:---------------------:|
| instance_auroc_mean |  93.65%   |        95.02%         |
| full_pixel_auroc |  92.42%   |        90.05%         |
| anomaly_pixel_auroc |  91.79%   |        88.23%         |
| reliable_good_ratio_mean |  65.45%   |        70.00%         |


--------

## Run

### PreInstall
- NVIDIA Docker

### concat model
```shell
cd dataset/models
cat wide_resnet50_2_VBD_L.pth* > wide_resnet50_2_VBD_L.pth
```

### Experiment by oss model
```shell
./experiment.sh ./dataset/anomaly_experiment.v.1.s ${HOME}/.anomaly_experiment.v.1_log 256 256 1024 1024 0.1 10 128
```

### Experiment by oss vbd-fine-tuning model
```shell
./experiment.sh ./dataset/anomaly_experiment.v.1.s ${HOME}/.anomaly_experiment.v.1_log 256 256 1024 1024 0.1 10 128 ./dataset/models
```

### parse log
```shell
pip install -r requirements.txt
python3 parse_log.py --input_dir_path ${HOME}/.anomaly_experiment.v.1_log --output_csv_path ${HOME}/.anomaly_experiment.v.1_log/experiment.log
```

--------

## Valid Dataset Catalog

```shell
├── bottle
│   ├── ground_truth
│   │   ├── broken_large
│   │   ├── broken_small
│   │   └── contamination
│   ├── test
│   │   ├── broken_large
│   │   ├── broken_small
│   │   ├── contamination
│   │   └── good
│   └── train
│       └── good
├── carpet
│   ├── ground_truth
│   │   ├── color
│   │   ├── cut
│   │   ├── hole
│   │   └── thread
│   ├── test
│   │   ├── color
│   │   ├── cut
│   │   ├── good
│   │   ├── hole
│   │   └── thread
│   └── train
│       └── good
├── felix1
│   ├── ground_truth
│   │   ├── color
│   │   ├── cut
│   │   ├── hole
│   │   ├── oil
│   │   └── protrude
│   ├── test
│   │   ├── color
│   │   ├── cut
│   │   ├── good
│   │   ├── hole
│   │   ├── oil
│   │   └── protrude
│   └── train
│       └── good
├── felix2
│   ├── ground_truth
│   │   ├── color
│   │   ├── cut
│   │   ├── hole
│   │   ├── oil
│   │   └── protrude
│   ├── test
│   │   ├── color
│   │   ├── cut
│   │   ├── good
│   │   ├── hole
│   │   ├── oil
│   │   └── protrude
│   └── train
│       └── good
├── grid
│   ├── ground_truth
│   │   ├── bent
│   │   ├── broken
│   │   ├── glue
│   │   └── thread
│   ├── test
│   │   ├── bent
│   │   ├── broken
│   │   ├── glue
│   │   ├── good
│   │   └── thread
│   └── train
│       └── good
├── gum
│   ├── ground_truth
│   │   ├── cut
│   │   ├── hole
│   │   ├── liquid
│   │   └── press
│   ├── test
│   │   ├── cut
│   │   ├── good
│   │   ├── hole
│   │   ├── liquid
│   │   └── press
│   └── train
│       └── good
├── hazelnut
│   ├── ground_truth
│   │   ├── color
│   │   ├── crack
│   │   ├── cut
│   │   └── hole
│   ├── test
│   │   ├── color
│   │   ├── crack
│   │   ├── cut
│   │   ├── good
│   │   └── hole
│   └── train
│       └── good
├── kabayaki1
│   ├── ground_truth
│   │   ├── color
│   │   ├── cut
│   │   ├── glue
│   │   └── hole
│   ├── test
│   │   ├── color
│   │   ├── cut
│   │   ├── glue
│   │   ├── good
│   │   └── hole
│   └── train
│       └── good
├── kabayaki2
│   ├── ground_truth
│   │   ├── color
│   │   ├── cut
│   │   ├── glue
│   │   └── hole
│   ├── test
│   │   ├── color
│   │   ├── cut
│   │   ├── glue
│   │   ├── good
│   │   └── hole
│   └── train
│       └── good
├── leather
│   ├── ground_truth
│   │   ├── color
│   │   ├── cut
│   │   ├── fold
│   │   ├── glue
│   │   └── poke
│   ├── test
│   │   ├── color
│   │   ├── cut
│   │   ├── fold
│   │   ├── glue
│   │   ├── good
│   │   └── poke
│   └── train
│       └── good
├── screw
│   ├── ground_truth
│   │   ├── manipulated_front
│   │   ├── scratch_head
│   │   ├── scratch_neck
│   │   └── thread_side
│   ├── test
│   │   ├── good
│   │   ├── manipulated_front
│   │   ├── scratch_head
│   │   ├── scratch_neck
│   │   └── thread_side
│   └── train
│       └── good
├── toothbrush
│   ├── ground_truth
│   │   └── defective
│   ├── test
│   │   ├── defective
│   │   └── good
│   └── train
│       └── good
├── transistor
│   ├── ground_truth
│   │   ├── bent_lead
│   │   ├── cut_lead
│   │   └── damaged_case
│   ├── test
│   │   ├── bent_lead
│   │   ├── cut_lead
│   │   ├── damaged_case
│   │   └── good
│   └── train
│       └── good
├── umai1
│   ├── ground_truth
│   │   ├── color
│   │   ├── cut
│   │   ├── different
│   │   ├── hole
│   │   └── open
│   ├── test
│   │   ├── color
│   │   ├── cut
│   │   ├── different
│   │   ├── good
│   │   ├── hole
│   │   └── open
│   └── train
│       └── good
├── umai2
│   ├── ground_truth
│   │   ├── color
│   │   ├── cut
│   │   ├── different
│   │   ├── hole
│   │   └── open
│   ├── test
│   │   ├── color
│   │   ├── cut
│   │   ├── different
│   │   ├── good
│   │   ├── hole
│   │   └── open
│   └── train
│       └── good
├── wood
│   ├── ground_truth
│   │   ├── color
│   │   ├── hole
│   │   ├── liquid
│   │   └── scratch
│   ├── test
│   │   ├── color
│   │   ├── good
│   │   ├── hole
│   │   ├── liquid
│   │   └── scratch
│   └── train
│       └── good
└── zipper
    ├── ground_truth
    │   ├── broken_teeth
    │   ├── fabric_interior
    │   ├── rough
    │   └── split_teeth
    ├── test
    │   ├── broken_teeth
    │   ├── fabric_interior
    │   ├── good
    │   ├── rough
    │   └── split_teeth
    └── train
        └── good

```

## Comment
- Although not public, there are food packages as well.
- Although not public, there are high resolutions image as well.
- Although not public, there are also anomaly images that are twice as large. 

## Feature
- Pretrain by segmentation model
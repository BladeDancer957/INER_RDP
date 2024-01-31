# Task Relation Distillation and Prototypical Pseudo Label for Incremental Named Entity Recognition (CIKM2023)

This repository contains all of our source code. We sincerely thank the help of [Zheng et al.'s repository](https://github.com/zzz47zzz/CFNER).

## Overview of the directory
- *bert-base-cased/*: the directory of configurations and PyTorch pretrained model for bert-base-cased
- *config/* : the directory of configurations for our CPFD method
- *datasets/* : the directory of datasets
- *experiments/* : the directory of training logs from different runs
- *src/* : the directory of the source code
- *main_CL.py* : the python file to be executed
```
.
├── bert-base-cased
├── config
│   ├── conll2003
│   ├── ontonotes5
│   ├── i2b2
├── datasets
│   └── NER_data
│       ├── conll2003
│       ├── i2b2
│       └── ontonotes5
├── experiments
│   └── result_analyze.py
|   └── xxx.pth
├── main_CL.py
└── src
    ├── config.py
    ├── dataloader.py
    ├── model.py
    ├── trainer.py
    ├── utils_plot.py
    └── utils.py
```

### Step 1: Prepare your environments
Reference environment settings:
```
python             3.7.13
torch              1.12.1+cu116
transformers       4.14.1
```

Download [bert-base-cased](https://huggingface.co/bert-base-cased/tree/main) to the directory of *bert-base-cased/*

Download [base models](https://drive.google.com/file/d/1FMBVDbyYrPHAnek8MWL-cjG7YVnTVZ5I/view?usp=sharing) to the directory of *experiments/*

### Step 2: Run main_CL.py
Specify your configurations (e.g., *./config/i2b2/fg_8_pg_2/RDP.yaml*) and run the following command 
```
CUDA_VISIBLE_DEVICES=0 nohup python3 -u main_CL.py --exp_name i2b2_8-2_RDP --exp_id 1 --cfg config/i2b2/fg_8_pg_2/RDP.yaml 2>&1 &
```
Then, the results as well as the model checkpoint will be saved automatically in the directory *./experiments/i2b2_8-2_RDP/1/* 


## Citation

```
@inproceedings{zhang2023rdp,
  title={Task Relation Distillation and Prototypical Pseudo Label for Incremental Named Entity Recognition},
  author={Zhang, Duzhen and Li, Hongliu and Cong, Wei and Xu, Rongtao and Dong, Jiahua and Chen, Xiuyi},
  booktitle={Proceedings of the 32th ACM International Conference on Information \& Knowledge Management},
  year={2023}
}
```

# lightgbm
LightGBM example


Expected folder structure:
```shell
$OUTPUT_DIR
└── data
    ├── raw             # raw data
    │   ├── train.txt
    │   ├── vali.txt
    │   └── test.txt
    └── preprocessed    # preprocessed data
        ├── train.parquet
        ├── vali.parquet
        └── test.parquet
```

```shell
PYTHONPATH=src python src/main.py --help

PYTHONPATH=src python src/main.py preprocess-data
```

# lightgbm

Hyperparameter optimization with Optuna.

We use a single fold of the [MSLR-WEB30k](https://www.microsoft.com/en-us/research/project/mslr/) dataset,
and store it in `$OUTPUT_DIR/data/raw`. Expected folder structure:

```shell
$OUTPUT_DIR
├── db.sqlite3
└── data
    ├── raw             # raw data (MSLR-WEB30k)
    │   ├── train.txt
    │   ├── vali.txt
    │   └── test.txt
    └── preprocessed    # preprocessed data
        ├── train.parquet
        ├── vali.parquet
        └── test.parquet
```

```shell
export output

PYTHONPATH=src python src/main.py --help

PYTHONPATH=src python src/main.py preprocess-data

PYTHONPATH=src python src/main.py hyperparameter-optimization

optuna-dashboard sqlite:///db.sqlite3
```

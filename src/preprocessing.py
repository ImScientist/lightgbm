import os
import logging
import pandas as pd

logger = logging.getLogger(__name__)


def preprocess_data(src_dir: str, dst_dir: str) -> None:
    """ Preprocess a single fold of the Microsoft learning to rank dataset

    https://www.microsoft.com/en-us/research/project/mslr/

    The first column refers to the target variable (relevance of the result).
    The second column refers to the query id.
    The remaining columns refer to the features (136 in total).

    Parameters
    ----------
    src_dir: source data directory; contains train.txt, vali.txt, test.txt
    dst_dir: preprocessed data directory;

    The following files will be created:
        dst_dir
        ├── train.parquet
        ├── vali.parquet
        └── test.parquet
    """

    logger.info(f'Preprocessing fold {src_dir}')

    os.makedirs(dst_dir, exist_ok=True)

    files = ['train.txt', 'vali.txt', 'test.txt']

    kwargs = {'delimiter': " ", 'header': None, 'usecols': list(range(138))}

    for f in files:
        path_src = os.path.join(src_dir, f)
        path_dst = os.path.join(dst_dir, f.replace('.txt', '.parquet'))

        df = pd.read_csv(path_src, **kwargs)

        df[0] = df[0].astype(int)
        df[1] = df[1].map(lambda x: x.split(':')[1]).astype(int)

        for i in range(2, 138):
            df[i] = df[i].map(lambda x: x.split(':')[1]).astype(float)

        df.sort_values([1]).reset_index(drop=True).to_parquet(path_dst)


def get_data(data_dir: str) -> tuple[dict, dict, dict]:
    """ Get preprocessed data """

    result = {}

    for ds, file in [['tr', 'train.parquet'],
                     ['va', 'vali.parquet'],
                     ['te', 'test.parquet']]:
        path = os.path.join(data_dir, file)

        df = pd.read_parquet(path)
        y = df.iloc[:, 0]
        queries = df.iloc[:, 1]
        sizes = queries.value_counts().sort_index().tolist()
        df.drop([0, 1], axis=1, inplace=True)
        df.columns = list(range(1, 137))

        result[ds] = {}
        result[ds]['x'] = df
        result[ds]['y'] = y
        result[ds]['queries'] = queries
        result[ds]['sizes'] = sizes

    return result['tr'], result['va'], result['te']

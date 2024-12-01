# lightgbm

Hyperparameter optimization with Optuna.

- Data: we use a single fold of the [MSLR-WEB30k](https://www.microsoft.com/en-us/research/project/mslr/) dataset,
  and store it in `$DATA_DIR_RAW`. Expected folder structure:
  ```shell
  $DATA_DIR_RAW
  ├── train.txt
  ├── vali.txt
  └── test.txt
  ``` 

- Build image:
  ```shell
  docker build -t ranking .
  ```

- Create a volume `myvolume` that contains the preprocessed data. The `preprocess-data` command preprocess the three
  txt-files and stores them as parquet files in the newly created volume.
  ```shell
  docker run --rm \
    -v $DATA_DIR_RAW:/data/raw \
    -v myvolume:/data/preprocessed \
    ranking:latest preprocess-data \
      --data-dir-raw /data/raw \
      --data-dir-preprocessed /data/preprocessed
  
  # Check that the volume contains data
  docker run --rm \
    -v myvolume:/volumes/myvolume \
    bash:4.4 bash -c "ls -l volumes/myvolume"
    
  #You should see the following 3 files
  #-rw-r--r--    1 root     root      61084833 Nov 19 23:10 test.parquet
  #-rw-r--r--    1 root     root     172122198 Nov 19 23:10 train.parquet
  #-rw-r--r--    1 root     root      60236192 Nov 19 23:10 vali.parquet
  ```

- Create a postgres database:
  ```shell
  helm repo add bitnami https://charts.bitnami.com/bitnami
  helm repo update
  helm install my-psql-release oci://registry-1.docker.io/bitnamicharts/postgresql
  ```

- Run optimization on a single container. To connect to your database from outside the cluster we
  use `kubectl port-forward`. To connect to the localhost of the machine inside a Docker container we use
  the `--network=host` option and replace the `localhost` with `host.docker.internal` (this depends on the machine on
  which Docker is running [documentation](https://docs.docker.com/engine/network/drivers/host/)).
  ```shell
  # Execute in a new tab
  kubectl port-forward --namespace default svc/my-psql-release-postgresql 5432:5432

  export POSTGRES_PASSWORD=$(kubectl get secret --namespace default my-psql-release-postgresql -o jsonpath="{.data.postgres-password}" | base64 -d)

  docker run -it --rm --network=host \
    -v myvolume:/data/preprocessed \
    ranking:latest \
    hyperparameter-optimization \
      --data-dir-preprocessed /data/preprocessed \
      --storage "postgresql://postgres:${POSTGRES_PASSWORD}@host.docker.internal:5432/postgres" \
      --study-name optimize_ranking_v4
  ```

- Check the results:
  ```shell
  optuna-dashboard postgresql://postgres:${POSTGRES_PASSWORD}@localhost:5432/postgres
  ```

- Clean the postgres helm chart:
  ```shell
  helm uninstall --namespace default my-psql-release
  
  # TODO: check and, eventually, clean the persistent volumes
  ```

- Docker image for CUDA-enabled LightGBM:
  ```shell
  docker build -t lightgbm-gpu -f Dockerfile.gpu .
  
  # Start a jupyter server; password: keras
  docker run -it --rm \
    --runtime=nvidia --gpus=all --name=testlightgbm -p 8888:8888 \
    -v $DATA_DIR_RAW:/data/raw \
    -v myvolume:/data/preprocessed \
    -v "$(pwd)/src:/home/src" \
    lightgbm-gpu:latest
  ``` 
    - If you want to make an OpenCL-based build targeting a wide range of GPUs you have to replace
      `-DUSE_CUDA=1` with `-DUSE_GPU=1` in the `Dockerfile.gpu`.
    - To enable training on the GPU add to lgb_parameters dict the option `'device_type': 'cuda'` or
      `'device_type': 'gpu'` depending on the image build. 
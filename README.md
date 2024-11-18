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

### Rest

PostgreSQL docker image:

```shell
docker run --name some-postgres -p 5432:5432 -e POSTGRES_PASSWORD=mysecretpassword -d postgres 

psql "postgres://postgres:mysecretpassword@localhost:5432/postgres"
```

PostgreSQL helm chart:

```shell
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo update
helm install my-psql-release oci://registry-1.docker.io/bitnamicharts/postgresql

export POSTGRES_PASSWORD=$(kubectl get secret --namespace default my-psql-release-postgresql -o jsonpath="{.data.postgres-password}" | base64 -d)

# create a pod that connects to the postgres db
kubectl run my-psql-release-postgresql-client \
  --rm --tty -i --restart='Never' --namespace default \
  --image docker.io/bitnami/postgresql:17.1.0-debian-12-r0 \
  --env="PGPASSWORD=$POSTGRES_PASSWORD" \
  --command -- psql --host my-psql-release-postgresql -U postgres -d postgres -p 5432




Pulled: registry-1.docker.io/bitnamicharts/postgresql:16.2.1
Digest: sha256:8204d160ca00beec7ca03e2cfb626226481e1b8bf566cdcc286d9b2484f58e92
NAME: my-psql-release
LAST DEPLOYED: Tue Nov 19 00:12:07 2024
NAMESPACE: default
STATUS: deployed
REVISION: 1
TEST SUITE: None
NOTES:
CHART NAME: postgresql
CHART VERSION: 16.2.1
APP VERSION: 17.1.0

** Please be patient while the chart is being deployed **

PostgreSQL can be accessed via port 5432 on the following DNS names from within your cluster:

    my-psql-release-postgresql.default.svc.cluster.local - Read/Write connection

To get the password for "postgres" run:

    export POSTGRES_PASSWORD=$(kubectl get secret --namespace default my-psql-release-postgresql -o jsonpath="{.data.postgres-password}" | base64 -d)

To connect to your database run the following command:

    kubectl run my-psql-release-postgresql-client --rm --tty -i --restart='Never' --namespace default --image docker.io/bitnami/postgresql:17.1.0-debian-12-r0 --env="PGPASSWORD=$POSTGRES_PASSWORD" \
      --command -- psql --host my-psql-release-postgresql -U postgres -d postgres -p 5432

    > NOTE: If you access the container using bash, make sure that you execute "/opt/bitnami/scripts/postgresql/entrypoint.sh /bin/bash" in order to avoid the error "psql: local user with ID 1001} does not exist"

To connect to your database from outside the cluster execute the following commands:

    kubectl port-forward --namespace default svc/my-psql-release-postgresql 5432:5432 &
    PGPASSWORD="$POSTGRES_PASSWORD" psql --host 127.0.0.1 -U postgres -d postgres -p 5432

WARNING: The configured password will be ignored on new installation in case when previous PostgreSQL release was deleted through the helm command. In that case, old PVC will have an old password, and setting it through helm won't take effect. Deleting persistent volumes (PVs) will solve the issue.

WARNING: There are "resources" sections in the chart not set. Using "resourcesPreset" is not recommended for production. For production installations, please set the following values according to your workload needs:
  - primary.resources
  - readReplicas.resources
+info https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/
```

Clean charts:

```shell
helm uninstall --namespace default my-psql-release

TODO: check and, eventually, clean the persistent volumes
```

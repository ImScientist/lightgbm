import optuna
import pandas as pd
import lightgbm as lgb

from joblib import parallel_backend
from optuna.trial import TrialState

from preprocessing import get_data


def logging_callback(study, frozen_trial):
    previous_best_value = study.user_attrs.get("previous_best_value", None)
    if previous_best_value != study.best_value:
        study.set_user_attr("previous_best_value", study.best_value)
        print(
            "Trial {} finished with best value: {} "
            "and parameters: {}. ".format(frozen_trial.number,
                                          frozen_trial.value,
                                          frozen_trial.params)
        )


def objective_with_pruning_callback(trial, data_dir: str):
    """ Optuna objective

    In this implementation optuna can stop the training if the score is not good enough,
    I guess.
    """

    data_tr, data_va, data_te = get_data(data_dir=data_dir)

    ds_tr = lgb.Dataset(
        data=data_tr['x'],
        label=data_tr['y'],
        group=data_tr['sizes'],
        reference=None)

    ds_va = lgb.Dataset(
        data=data_va['x'],
        label=data_va['y'],
        group=data_va['sizes'],
        reference=ds_tr)

    lgbm_parameters = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'n_estimators': 150,
        'num_leaves': trial.suggest_categorical('num_leaves', choices=[128, 256]),
        'max_depth': trial.suggest_categorical('max_depth', choices=[6, 8, 10]),
        # 'learning_rate': trial.suggest_categorical('learning_rate', choices=[.02, .05]),
        'learning_rate': .03,
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-4, .1, log=True),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 30, 150),
        'verbose': 0,
        'early_stopping_rounds': 10,
        'first_metric_only': True,
        'ndcg_eval_at': [5, 10]}

    # Add a callback for pruning.
    pruning_callback = optuna.integration.LightGBMPruningCallback(
        trial,
        metric='ndcg@5',
        valid_name='val')

    gbm = lgb.train(
        params=lgbm_parameters,
        train_set=ds_tr,
        valid_sets=[ds_tr, ds_va],
        valid_names=['train', 'val'],
        callbacks=[lgb.log_evaluation(period=10, show_stdv=False),
                   pruning_callback])

    return gbm.best_score["val"]["ndcg@5"]


def hyperparameter_optimization(data_dir: str):
    """ Ex with Optuna """

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        storage='sqlite:///db.sqlite3',
        pruner=optuna.pruners.PercentilePruner(
            percentile=50,
            n_startup_trials=8,
            n_warmup_steps=50,
            interval_steps=50,
            n_min_trials=8),
        study_name='optimize_ranking',
        direction='maximize',
        load_if_exists=False)

    with parallel_backend('multiprocessing'):
        study.optimize(
            lambda x: objective_with_pruning_callback(x, data_dir=data_dir),
            n_trials=20,
            n_jobs=4,
            gc_after_trial=True,
            # callbacks=[logging_callback]
        )

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # Explored parameters
    trials = study.get_trials(deepcopy=False)
    trials_params = pd.DataFrame([trial.params for trial in trials])
    trials_params['score'] = [tr.value for tr in trials]
    print('\n', trials_params)

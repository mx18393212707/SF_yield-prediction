import pickle
from pathlib import Path
from typing import Tuple
import numpy as np
from xgboost import XGBRegressor, DMatrix, callback
from sklearn.metrics import mean_squared_error, r2_score
import wandb
import os
import sys
from scipy.stats import pearsonr


def save_results(
    set_name: str,
    split_id: str,
    file_name: str,
    ground_truth: np.ndarray,
    prediction: np.ndarray
) -> None:
    results_dir = Path("scripts", "training", "xgb_init_results")
    results_dir.mkdir(parents=True, exist_ok=True)
    file_path = results_dir / f"{set_name}_{split_id}_{file_name}_results.csv"
    
    with open(file_path, "w+") as f:
        f.write("set,split,type,ground_truth,prediction\n")
        for gt, pred in zip(ground_truth, prediction):
            f.write(f"{set_name},{split_id},{file_name},{gt},{pred}\n")


def save_model(
    model: XGBRegressor,
    model_type: str,
    dataset: str,
    split_id: str,
    name: str,
    save_dir: str = "models"
) -> None:
    save_path = Path(save_dir, f"{model_type}_{dataset}_{split_id}_{name}.pkl")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, "wb") as f:
        pickle.dump(model, f)
    print(f"模型已保存至：{save_path}")


def log_metrics(y_true: np.ndarray, y_pred: np.ndarray, set_type: str):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    pearson_r = pearsonr(y_true, y_pred)[0]  # 提取皮尔逊相关系数
    wandb.log({
        f"{set_type}_mse": mse,
        f"{set_type}_r2": r2,
        f"{set_type}_pearson_r": pearson_r
    })


def load_split1_data():
    """加载第一种划分数据 (train/valid: 512/128)"""
    train_data = pickle.load(open("data/split_data/split1_train_512.pkl", "rb"))
    valid_data = pickle.load(open("data/split_data/split1_valid_128.pkl", "rb"))
    
    X_train, y_train, _ = train_data
    X_valid, y_valid, _ = valid_data
    return X_train, y_train, X_valid, y_valid, None, None  # 无测试集


def load_split2_data():
    """加载第二种划分数据 (train/valid/test: 384/128/128)"""
    train_data = pickle.load(open("data/split_data/split2_train_384.pkl", "rb"))
    valid_data = pickle.load(open("data/split_data/split2_valid_128.pkl", "rb"))
    test_data = pickle.load(open("data/split_data/split2_test_128.pkl", "rb"))
    
    X_train, y_train, _ = train_data
    X_valid, y_valid, _ = valid_data
    X_test, y_test, _ = test_data
    return X_train, y_train, X_valid, y_valid, X_test, y_test


class WandBCallback(callback.TrainingCallback):
    def __init__(self, X_train: np.ndarray, y_train: np.ndarray, X_valid: np.ndarray, y_valid: np.ndarray):
        super().__init__()
        self.dtrain = DMatrix(X_train, label=y_train)
        self.dvalid = DMatrix(X_valid, label=y_valid)
        self.y_train = y_train
        self.y_valid = y_valid

    def after_iteration(self, model, epoch, evals_log):
        y_train_pred = model.predict(self.dtrain, iteration_range=(0, epoch))
        y_valid_pred = model.predict(self.dvalid, iteration_range=(0, epoch))
        log_metrics(self.y_train, y_train_pred, "train")
        log_metrics(self.y_valid, y_valid_pred, "valid")
        return False


def predict_SF_split1():
    """处理第一种划分 (train/valid: 512/128)"""
    X_train, y_train, X_valid, y_valid, _, _ = load_split1_data()
    
    wandb.init(project="SF_yield_prediction", name="init_split1_training")
    wandb.config.update({
        "n_estimators": 999999,
        "learning_rate": 0.01,
        "max_depth": 15,
        "min_child_weight": 8,
        "colsample_bytree": 0.2125,
        "subsample": 1.0,
        "random_state": 42,
        "early_stopping_rounds": 10,
        "tree_method":"hist",
        "n_jobs":-1
    })

    model = XGBRegressor(
        n_estimators=999999,
        learning_rate=0.01,
        max_depth=15,
        min_child_weight=8,
        colsample_bytree=0.2125,
        subsample=1.0,
        random_state=42,
        eval_metric="rmse",
        tree_method="hist",
        n_jobs=-1
    )

    wandb_callback = WandBCallback(X_train, y_train, X_valid, y_valid)

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False,
        early_stopping_rounds=10,
        callbacks=[wandb_callback]
    )
    
    save_model(
        model=model,
        model_type="xgboost",
        dataset="SF",
        split_id="split1",
        name="init"
    )

    y_valid_pred = model.predict(X_valid, iteration_range=(0, model.best_ntree_limit))
    save_results("SF", "split1", "valid", y_valid, y_valid_pred)
    
    wandb.log({"final_valid_r2": r2_score(y_valid, y_valid_pred)})
    wandb.finish()


def predict_SF_split2():
    """处理第二种划分 (train/valid/test: 384/128/128)"""
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_split2_data()
    
    wandb.init(project="SF_yield_prediction", name="init_split2_training")
    wandb.config.update({
        "n_estimators": 999999,
        "learning_rate": 0.01,
        "max_depth": 15,
        "min_child_weight": 8,
        "colsample_bytree": 0.2125,
        "subsample": 1.0,
        "random_state": 42,
        "early_stopping_rounds": 10,
        "tree_method": "hist",
        "n_jobs": -1
    })

    model = XGBRegressor(
        n_estimators=999999,
        learning_rate=0.01,
        max_depth=15,
        min_child_weight=8,
        colsample_bytree=0.2125,
        subsample=1.0,
        random_state=42,
        eval_metric="rmse",
        tree_method="hist",
        n_jobs=-1
    )

    wandb_callback = WandBCallback(X_train, y_train, X_valid, y_valid)

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False,
        early_stopping_rounds=10,
        callbacks=[wandb_callback]
    )
    
    save_model(
        model=model,
        model_type="xgboost",
        dataset="SF",
        split_id="split2",
        name="init"
    )

    # 验证集和测试集结果
    y_valid_pred = model.predict(X_valid, iteration_range=(0, model.best_ntree_limit))
    y_test_pred = model.predict(X_test, iteration_range=(0, model.best_ntree_limit))
    
    save_results("SF", "split2", "valid", y_valid, y_valid_pred)
    save_results("SF", "split2", "test", y_test, y_test_pred)
    
    wandb.log({
        "final_valid_r2": r2_score(y_valid, y_valid_pred),
        "final_test_r2": r2_score(y_test, y_test_pred)
    })
    wandb.finish()


def main():
    predict_SF_split1()
    predict_SF_split2()


if __name__ == "__main__":
    main()
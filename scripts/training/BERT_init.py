import typer
import torch
import pkg_resources
import sys
sys.path.append('.')  # 将当前目录（项目根目录）添加到Python模块搜索路径
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import pandas as pd
from typing import List, Tuple

from rxnfp.models import SmilesClassificationModel
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr
import sklearn

import wandb

from scripts.encoding.preprocess_and_split import load_and_split_data

app = typer.Typer()

try:
    import wandb
    wandb_available = True
except ImportError:
    wandb_available = False
    

# 定义你的数据划分配置（与load_and_split_data的num_split参数对应）
DATA_SPLITS = [
    (2, "split1"),   # num_split=2，对应train/valid划分
    (3, "split2")    # num_split=3，对应train/valid/test划分
]



def calculate_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """计算并返回所有要求的评估指标"""
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    pearson = pearsonr(y_true, y_pred)[0]
    return {"mse": mse, "r2": r2, "pearson_r": pearson}


def launch_training(num_split: int, name: str , base_model: str, dropout: float, learning_rate: float,standardize=True):
    """
    split1: 512/128    train_df=train_df, test_df=val_df
    split2: 384/128/128   train_df=train_df+val_df, test_df=test_df
    
    The fun(train_model) of [SmilesClassificationModel(class)] evalueate r2 on test_df

    """
    project = "SF_yield_prediction"
    model_args = {
        "wandb_project": project, "num_train_epochs": 10, "overwrite_output_dir": True,
        "learning_rate": learning_rate, "gradient_accumulation_steps": 1,
        "regression": True, "num_labels": 1, "fp16": False,
        "evaluate_during_training": True, "manual_seed": 42,
        "max_seq_length": 300, "train_batch_size": 16, "warmup_ratio": 0.00,
        "config": {"hidden_dropout_prob": dropout}
    }
    
    # 加载并分割数据（直接使用你的函数）
    if num_split == 2:
        train_df, val_df = load_and_split_data(num_split=2, standardize=standardize)
        train_data = train_df
        eval_df = val_df
    else:
        train_df, val_df, test_df = load_and_split_data(num_split=3, standardize=standardize)
        train_data = train_df.append(val_df, ignore_index=True)
        eval_df = test_df
    
    train_data.columns = ['text', 'labels']
    eval_df.columns = ['text', 'labels']    
        
    
    
    if wandb_available: wandb.init(name=f"init_{base_model}_{name}_std-{standardize}", project=project, reinit=True)


    model_path = pkg_resources.resource_filename("rxnfp", f"models/transformers/bert_{base_model}")
    pretrained_bert = SmilesClassificationModel("bert", model_path, num_labels=1, args=model_args, use_cuda=torch.cuda.is_available())
    pretrained_bert.train_model(train_data, output_dir=f"{base_model}_init_std-{standardize}_results", eval_df=eval_df, r2=sklearn.metrics.r2_score)




    # 训练集指标计算
    train_pred , _ = pretrained_bert.predict(train_data["text"])
    train_metrics = calculate_metrics(train_data["labels"], train_pred)
    wandb.log({f"train_{k}": v for k, v in train_metrics.items()})

    # 测试集指标计算

    test_pred , _ = pretrained_bert.predict(eval_df["text"].tolist())
    test_true = eval_df["labels"].tolist()
    
    test_r2 = sklearn.metrics.r2_score(test_true, test_pred)
    test_pearson_r, _ = pearsonr(test_true, test_pred)
    test_mse = mean_squared_error(test_true, test_pred)
    wandb.log({'test_r2': test_r2, 'test_pearson_r': test_pearson_r, 'test_mse': test_mse})

    if wandb_available:
        wandb.join()


@app.command()
def pretrained1true():
    launch_training(
                num_split=2,
                name="split1",
                base_model="pretrained",
                dropout=0.7987,
                learning_rate=0.00009659,
                standardize=True
        )
    
@app.command()
def pretrained1false():
    launch_training(
                num_split=2,
                name="split1",
                base_model="pretrained",
                dropout=0.7987,
                learning_rate=0.00009659,
                standardize=False
        )
@app.command()
def pretrained2true():
    launch_training(
                num_split=3,
                name="split2",
                base_model="pretrained",
                dropout=0.7987,
                learning_rate=0.00009659,
                standardize=True
        )
    
@app.command()
def pretrained2false():
    launch_training(
                num_split=3,
                name="split2",
                base_model="pretrained",
                dropout=0.7987,
                learning_rate=0.00009659,
                standardize=False
        )
    
@app.command()
def ft1true():
    launch_training(
                num_split=2,
                name="split1",
                base_model="ft",
                dropout=0.7304,
                learning_rate=0.0000976,
                standardize=True
        )
    
@app.command()       
def ft1false():
    launch_training(
                num_split=2,
                name="split1",
                base_model="ft",
                dropout=0.7304,
                learning_rate=0.0000976,
                standardize=False
        )
    
@app.command()
def ft2true():
    launch_training(
                num_split=3,
                name="split2",
                base_model="ft",
                dropout=0.7304,
                learning_rate=0.0000976,
                standardize=True
        )
    
@app.command()       
def ft2false():
    launch_training(
                num_split=3,
                name="split2",
                base_model="ft",
                dropout=0.7304,
                learning_rate=0.0000976,
                standardize=False
        )
    

    
if __name__ == '__main__':
    app()
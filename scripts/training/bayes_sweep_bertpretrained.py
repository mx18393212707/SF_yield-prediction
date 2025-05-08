# 执行该脚本后，wandb实际使用的lr和记录的lr不是同一个值，但优化过程仍然有效
import sys
sys.path.append('.')

from scripts.encoding.preprocess_and_split import load_and_split_data
from rxnfp.models import SmilesClassificationModel
import typer
import torch
import pkg_resources
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import pandas as pd
from typing import List, Tuple
from sklearn.utils import shuffle
import sklearn
import os
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_squared_error

try:
    import wandb
    wandb_available = True
except ImportError:
    wandb_available = False

app = typer.Typer()

def calculate_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """计算并返回所有要求的评估指标"""
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    pearson = pearsonr(y_true, y_pred)[0]
    return {"mse": mse, "r2": r2, "pearson_r": pearson}


def sweep(num_split: int , base_model: str , standardize=True):
     
    # 数据加载
    if num_split == 2:
        split_name = "split1"
        train_df, val_df = load_and_split_data(num_split=2, standardize=standardize)

    else:
        split_name = "split2"
        train_df, val_df, test_df = load_and_split_data(num_split=3, standardize=standardize)

    
    train_df.columns = ['text', 'labels']
    val_df.columns = ['text', 'labels'] 
    
        
    # 输出目录配置
    output_folder = f"bayes-opt_BERT-{base_model}_{split_name}_standardize-{standardize}"
    os.makedirs(output_folder, exist_ok=True)
    project_name = f"SF_Bayes-Optimization_BERT-{base_model}_{split_name}_standardize-{standardize}"


    def train():
        
        wandb.init()
        print("HyperParams=>>", wandb.config)
        print(f"Using learning rate: {wandb.config.learning_rate}")  # 添加日志输出

        model_args = {
            'wandb_project': project_name,
            'num_train_epochs': wandb.config.num_train_epochs,
            'overwrite_output_dir': True,
            'gradient_accumulation_steps': wandb.config.gradient_accumulation_steps,
            "warmup_ratio": wandb.config.warmup_ratio,
            "train_batch_size": 16,
            'regression': True,
            "num_labels": 1,
            "fp16": False,
            "evaluate_during_training": True,
            "max_seq_length": wandb.config.max_seq_length,
            "config": {'hidden_dropout_prob': wandb.config.dropout_rate},
            'learning_rate': wandb.config.learning_rate,
        }
        
        model_path =  pkg_resources.resource_filename("rxnfp", f"models/transformers/bert_{base_model}")
        pretrained_bert = SmilesClassificationModel("bert", model_path, num_labels=1, args=model_args, use_cuda=torch.cuda.is_available())
        pretrained_bert.train_model(train_df, output_dir=output_folder, eval_df=val_df, r2=sklearn.metrics.r2_score)
      
        # 训练集指标计算
        train_pred , _ = pretrained_bert.predict(train_df["text"])
        train_metrics = calculate_metrics(train_df["labels"], train_pred)
        wandb.log({f"train_{k}": v for k, v in train_metrics.items()})

        # 测试集指标计算

        val_pred , _ = pretrained_bert.predict(val_df["text"].tolist())
        val_true = val_df["labels"].tolist()
        
        val_r2 = sklearn.metrics.r2_score(val_true, val_pred)
        val_pearson_r, _ = pearsonr(val_true, val_pred)
        val_mse = mean_squared_error(val_true, val_pred)
        wandb.log({'val_r2': val_r2, 'val_pearson_r': val_pearson_r, 'val_mse': val_mse})


    # 贝叶斯优化配置（保持不变）
    sweep_config = {
        'method': 'bayes',
        'metric': {'name': 'r2', 'goal': 'maximize'},
        'parameters': {
            'learning_rate': {'min': 1e-6, 'max': 1e-4},
            'dropout_rate': {'min': 0.05, 'max': 0.8},
            'num_train_epochs': {'min': 5, 'max': 25, 'distribution': 'int_uniform'},
            'warmup_ratio': {'min': 0.0, 'max': 0.3},
            'gradient_accumulation_steps': {'values': [1, 2, 4]},
            'max_seq_length': {'values': [200, 250, 300]}
        },
        'early_terminate': {
            'type': 'hyperband',
            'min_iter': 3,
            'eta': 2,  # 缩减因子
            'max_iter': 15  # 最大迭代次数
        }
    }

    sweep_id = wandb.sweep(sweep_config, project=project_name)
    
    wandb.agent(sweep_id= sweep_id,function=train)
    
    
@app.command()
def pretrained1true():
    sweep(num_split=2, base_model="pretrained", standardize=True)
    
@app.command()
def pretrained1false():
    sweep(num_split=2, base_model="pretrained", standardize=False)
    
@app.command()
def pretrained2true():
    sweep(num_split=3, base_model="pretrained", standardize=True)
    
@app.command()
def pretrained2false():
    sweep(num_split=3, base_model="pretrained", standardize=False)
    
@app.command()
def ft1true():
    sweep(num_split=2, base_model="ft", standardize=True)
    
@app.command()
def ft1false():
    sweep(num_split=2, base_model="ft", standardize=False)
    
@app.command()
def ft2true():
    sweep(num_split=3, base_model="ft", standardize=True)

@app.command()
def ft2false():
    sweep(num_split=3, base_model="ft", standardize=False)

if __name__ == '__main__':
    app()   
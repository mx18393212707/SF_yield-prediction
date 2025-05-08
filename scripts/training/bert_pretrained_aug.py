import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from data.data_split import load_and_split_data
from data.augmentation import do_random_permutations_on_df,do_randomizations_on_df
from rxnfp.models import SmilesClassificationModel
from scipy.stats import pearsonr
import typer
import torch
import importlib.resources
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import pandas as pd
from sklearn.utils import shuffle
import sklearn
import os
from transformers import EarlyStoppingCallback

try:
    import wandb
    wandb_available = True
except ImportError:
    wandb_available = False

app = typer.Typer()

# 数据增强：合用。smiles1次，排列组合1/2次。
def pre_experiment(do_aug=True,num_split=2,standardize=True,shuffle_data=False, num_shuffles=1, n_permutations = 1, random_type="rotated", seed = 42,output_folder='SF_aug_experiment_results',project_name = "必填",num_hidden_layers=12):
    if num_split == 2:
        train_df,test_df = load_and_split_data(num_split,standardize)
    elif num_split == 3:
        train_df,val_df,test_df= load_and_split_data(num_split,standardize)


    # 获取当前脚本文件的目录
    current_script_dir = os.path.dirname(os.path.abspath(__file__))

        # 构建输出文件夹的绝对路径
    output_folder = os.path.join(current_script_dir, output_folder)

    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    all_results = []

    for shuffle_index in range(num_shuffles):
        # 根据是否打乱数据设置项目名称和输出目录
        if shuffle_data:
            project = f'{project_name}_{shuffle_index}'
            output_dir = os.path.join(output_folder,f'outputs_divide_{project_name}_{shuffle_index}')
        else:
            project = f'{project_name}_unshuffled'
            output_dir = os.path.join(output_folder,f'outputs_no_aug_numhiddenlayer_adj_{project_name}_unshuffled')

        model_args = {
            'wandb_project': project,
            'num_train_epochs': 10,
            'overwrite_output_dir': True,
            'learning_rate': 0.00009659,
            'gradient_accumulation_steps': 1,
            'regression': True,
            "num_labels": 1,
            "fp16": False,
            "evaluate_during_training": True,
            'manual_seed': 42,
            "max_seq_length": 300,
            "train_batch_size": 16,
            "warmup_ratio": 0.00,
            "config": {'hidden_dropout_prob': 0.7987},
            'dataloader_num_workers': 12,  

            #'attention_probs_dropout_prob': 0.2,  # 新增注意力Dropout
            #'layer_norm_eps': 1e-5,  # 增强数值稳定性
        }
        
        model_args.update({
            # 新增
            "weight_decay": 0.01,  # L2正则化
            "config": {'hidden_dropout_prob': 0.5},

            "use_early_stopping": True,  # 启用早停
            "early_stopping_patience": 3,  # 早停耐心值
            "early_stopping_metric": "eval_loss",  # 监控指标
            "early_stopping_metric_minimize": True,  # 是否最小化指标

            "learning_rate": 3e-5,  # 初始学习率
            #"scheduler": "cosine",  # 余弦退火调度器
            "warmup_steps": 0.2,  # 预热步数    
        })

        # 修改model_args中的config
        model_args['config'].update({
            'num_hidden_layers': num_hidden_layers # 原12层 → 6层    # hidden_size 原256 不改变
            #'num_attention_heads': 8 # 原4 ，不改动
        })

        if wandb_available:
            if shuffle_data:
                wandb.init(name=f'shuffled_{shuffle_index}', project="pre_split2unshuffled_bert_pretrained", dir=output_folder)
            else:
                wandb.init(name=f'unshuffled_adj_numhiddenlayer_{num_hidden_layers}_{n_permutations}_{random_type}', project="pre_split2unshuffled_bert_pretrained", dir=output_folder)

        package = importlib.resources.files("rxnfp")
        resource = package / "models" / "transformers" / "bert_pretrained"
        model_path = str(resource)

        if num_split == 3:
            new_train_df = pd.concat([train_df,val_df])
            # 如果需要打乱数据，则对训练集和验证集进行打乱
            if shuffle_data:
                train_df_shuffled = shuffle(new_train_df, random_state=shuffle_index)
            else:
                train_df_shuffled = new_train_df
        elif num_split == 2:
            if shuffle_data:
                train_df_shuffled = shuffle(train_df, random_state=shuffle_index)
            else:
                train_df_shuffled = train_df

        
        aug_1 = do_random_permutations_on_df(train_df_shuffled, n_permutations=n_permutations,random_type=random_type,seed=seed)
        combined_train_df = pd.concat([train_df_shuffled, aug_1], ignore_index=True)

        
        pretrained_bert = SmilesClassificationModel("bert", model_path, num_labels=1, args=model_args, use_cuda=torch.cuda.is_available())
        
        if do_aug:
            pretrained_bert.train_model(combined_train_df, output_dir=output_dir, eval_df=test_df, r2=sklearn.metrics.r2_score)
        else:
            pretrained_bert.train_model(train_df_shuffled, output_dir=output_dir, eval_df=test_df, r2=sklearn.metrics.r2_score)


            
        # 在测试集上进行预测
        predictions, _ = pretrained_bert.predict(test_df['text'].tolist())
        true_labels = test_df['labels'].tolist()

        # 计算评估指标
        r2 = sklearn.metrics.r2_score(true_labels, predictions)
        pearson_r, _ = pearsonr(true_labels, predictions)

        # 使用 wandb 记录指标
        if wandb_available:
            wandb.log({'test_r2': r2, 'test_pearson_r': pearson_r})

        if wandb_available:
            wandb.join()


@app.command()
def augsplit2unshuffled():
    for i in [3,6,9,10,11,12]:
        pre_experiment(do_aug=True,shuffle_data=False, standardize=True, num_split=2, project_name="model_adj_split2unshuffled_bert_pretrained", n_permutations=1, random_type="rotated", seed=42,num_hidden_layers=i)

@app.command()
def noaugsplit2unshuffled():
    pre_experiment(do_aug=False,shuffle_data=False, standardize=True, num_split=2, project_name="model_adj_split2unshuffled_bert_pretrained", seed=42)




if __name__ == '__main__':
    app()
__all__ = ['load_and_split_data']

import pandas as pd
from rdkit import Chem
import numpy as np
import os


def load_and_split_data(num_split=2, standardize=True):
    """
    加载并处理SF640数据集，支持两种数据划分方式
    :param num_split: 分割方式（2或3），2对应train/valid，3对应train/valid/test
    :param standardize: 是否对标签进行标准化（使用训练集均值/标准差）
    :return: 分割后的数据集（DataFrame格式）
    """
    # 1. 加载原始数据
    file_path = os.path.join(os.path.dirname(__file__), '../../data/SF640_A_B_C_lbl.pkl')
    with open(file_path, 'rb') as f:
        data = pd.read_pickle(f)  # 假设data是字典类型，键为反应SMILES，值为包含各字段的字典

    # 2. 初始化数据容器
    entries = []
    max_reactants = 0
    max_inter_reactants = 0
    max_products = 0

    for reaction_smiles, entry in data.items():
        # 解析原始数据字段
        label = entry['label'][0]
        predict = entry['predict']['DRFP']['random']

        # 解析反应SMILES格式：反应物>中间体.中间体>产物.产物
        try:
            reactants_part, inter_part, products_part = reaction_smiles.split('>')
        except ValueError:
            print(f"无效的SMILES格式: {reaction_smiles}")
            continue  # 跳过格式错误的条目

        # 拆分各部分
        reactants = [reactants_part]  # 反应物通常只有一个前体
        inter_reactants = inter_part.split('.') if inter_part else []
        products = products_part.split('.') if products_part else []

        # 记录最大数量用于列定义
        max_reactants = max(max_reactants, len(reactants))
        max_inter_reactants = max(max_inter_reactants, len(inter_reactants))
        max_products = max(max_products, len(products))

        entries.append({
            'smiles': reaction_smiles,
            'labels': label,
            'predict': predict,
            'reactants': reactants,
            'inter_reactants': inter_reactants,
            'products': products
        })

    # 3. 构建结构化DataFrame
    df = pd.DataFrame(entries)

    # 生成固定列名
    reactant_cols = [f'reactant_{i + 1}' for i in range(max_reactants)]
    inter_cols = [f'inter_{i + 1}' for i in range(max_inter_reactants)]
    product_cols = [f'product_{i + 1}' for i in range(max_products)]

    # 展开列表到多列
    for col, prefix, max_len in [
        ('reactants', 'reactant', max_reactants),
        ('inter_reactants', 'inter', max_inter_reactants),
        ('products', 'product', max_products)
    ]:
        for i in range(max_len):
            df[f'{prefix}_{i + 1}'] = df[col].apply(lambda x: x[i] if i < len(x) else np.nan)

    # 4. SMILES规范化与反应式生成
    def canonicalize_smiles(smi):
        """安全的SMILES规范化函数"""
        mol = Chem.MolFromSmiles(smi)
        return Chem.MolToSmiles(mol) if mol is not None else smi

    def build_reaction_smiles(row):
        """构建反应SMILES：反应物.中间体>>产物.产物"""
        reactants = [canonicalize_smiles(row[col]) for col in reactant_cols if not pd.isna(row[col])]
        intermediates = [canonicalize_smiles(row[col]) for col in inter_cols if not pd.isna(row[col])]
        products = [canonicalize_smiles(row[col]) for col in product_cols if not pd.isna(row[col])]

        reactant_str = '.'.join(reactants + intermediates)
        product_str = '.'.join(products)
        return f"{reactant_str}>>{product_str}" if product_str else np.nan

    df['text'] = df.apply(build_reaction_smiles, axis=1)

    # 5. 数据分割
    def perform_split():
        total_samples = len(df)
        if total_samples < 512 and num_split == 2:
            raise ValueError("数据总量不足，无法按照 num_split=2 进行划分。")
        if total_samples < 512 and num_split == 3:
            raise ValueError("数据总量不足，无法按照 num_split=3 进行划分。")

        if num_split == 2:
            split_points = [0, 512]  # 修正：添加起始索引0，分割为[0, 512, total_samples]
        elif num_split == 3:
            split_points = [0, 384, 512]  # 修正：三段划分的起始索引

        datasets = []
        for i in range(num_split):
            start = split_points[i]  # 直接使用split_points中的起始索引（已包含0）
            end = split_points[i + 1] if i+1 < num_split else total_samples  # 处理最后一个数据集的结束索引
            datasets.append(df.iloc[start:end][['text', 'labels']].copy())

        train_df = datasets[0]
        val_df = datasets[1]
        if num_split == 3:
            test_df = datasets[2]

        # 标签标准化（仅使用训练集统计量）
        if standardize:
            train_mean = train_df['labels'].mean()
            train_std = train_df['labels'].std()
            if train_std == 0:
                raise ValueError("训练集标签的标准差为 0，无法进行标准化。")

            for dset in datasets:
                dset['labels'] = (dset['labels'] - train_mean) / train_std

        return (train_df, val_df) if num_split == 2 else (train_df, val_df, test_df)

    return perform_split()
    
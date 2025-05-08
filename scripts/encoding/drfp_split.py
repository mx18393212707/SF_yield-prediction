import pickle
import os  # 添加目录操作模块


def split_data_1(X, y, smiles):
    """
    第一种划分方式：train/valid = 512/128
    前 512 个样本为 train，后 128 个样本为 valid
    """
    train_X = X[:512]
    train_y = y[:512]
    train_smiles = smiles[:512]

    valid_X = X[512:]
    valid_y = y[512:]
    valid_smiles = smiles[512:]

    return (train_X, train_y, train_smiles), (valid_X, valid_y, valid_smiles)


def split_data_2(X, y, smiles):
    """
    第二种划分方式：train/valid/test = 384/128/128
    前 384 个样本为 train，中间 128 个为 valid，最后 128 个为 test
    """
    train_X = X[:384]
    train_y = y[:384]
    train_smiles = smiles[:384]

    valid_X = X[384:512]  # 384+128=512，正确切片区间
    valid_y = y[384:512]
    valid_smiles = smiles[384:512]

    test_X = X[512:]  # 512+128=640，刚好到数据集末尾
    test_y = y[512:]
    test_smiles = smiles[512:]

    return (train_X, train_y, train_smiles), (valid_X, valid_y, valid_smiles), (test_X, test_y, test_smiles)


def save_split_data(data, output_dir, split_name):
    """
    保存划分好的数据到 pickle 文件（修正路径处理）
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{split_name}.pkl")  # 正确拼接路径
    with open(file_path, "wb+") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    input_filepath = "data/SF640_A_B_C_lbl-2048-3-true.pkl"
    output_dir = "data/split_data"  # 修正：使用目录名（不带结尾斜杠）

    try:
        with open(input_filepath, 'rb') as f:
            X, y, smiles = pickle.load(f)

        # 第一种划分方式（train/valid）
        train_1, valid_1 = split_data_1(X, y, smiles)
        save_split_data(train_1, output_dir, "split1_train_512")
        save_split_data(valid_1, output_dir, "split1_valid_128")

        # 第二种划分方式（train/valid/test）
        train_2, valid_2, test_2 = split_data_2(X, y, smiles)
        save_split_data(train_2, output_dir, "split2_train_384")
        save_split_data(valid_2, output_dir, "split2_valid_128")
        save_split_data(test_2, output_dir, "split2_test_128")

        print("数据划分并保存成功。")

    except FileNotFoundError:
        print(f"未找到文件: {input_filepath}")
    except Exception as e:
        print(f"发生错误: {e}")
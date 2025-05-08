import sys
sys.path.append('.')  

from scripts.encoding.preprocess_and_split import load_and_split_data



train_df, val_df = load_and_split_data(num_split=2, standardize=True)




train_df, val_df, test_df = load_and_split_data(num_split=3, standardize=True)
train_data = train_df.append(val_df, ignore_index=True)
eval_df = test_df
print("train_data",train_data.iloc[380:390])
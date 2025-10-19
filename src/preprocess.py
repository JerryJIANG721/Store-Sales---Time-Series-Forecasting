# preprocess.py
import pandas as pd
from pathlib import Path

# 以脚本所在目录为基准，稳妥获取 data 目录
SCRIPT_DIR = Path(__file__).resolve().parent        # ...\src
DATA_DIR   = SCRIPT_DIR.parent / 'data'             # ...\data

print("脚本目录：", SCRIPT_DIR)
print("数据目录：", DATA_DIR)

# 读取 CSV
stores_df = pd.read_csv(DATA_DIR / 'stores.csv')
train_df  = pd.read_csv(DATA_DIR / 'train.csv')

# 想要转换为 category 的列（只要在表里存在就转）
stores_cat_cols = ['type', 'city', 'state', 'cluster']   # 你可按需删减
train_cat_cols  = ['family']                             # Kaggle 常见列名

# 安全转换：仅对存在的列转换，避免 KeyError
for col in stores_cat_cols:
    if col in stores_df.columns:
        stores_df[col] = stores_df[col].astype('category')
    else:
        print(f"⚠️ stores.csv 未找到列：{col}")

for col in train_cat_cols:
    if col in train_df.columns:
        train_df[col] = train_df[col].astype('category')
    else:
        print(f"⚠️ train.csv 未找到列：{col}")

# 查看结果
print("\n=== stores_df.info() ===")
print(stores_df.info())
print("\n=== train_df.info() ===")
print(train_df.info())

# 如果需要，保存处理后的文件
# (DATA_DIR / 'stores_processed.csv').write_text(stores_df.to_csv(index=False), encoding='utf-8')
# (DATA_DIR / 'train_processed.csv').write_text(train_df.to_csv(index=False), encoding='utf-8')

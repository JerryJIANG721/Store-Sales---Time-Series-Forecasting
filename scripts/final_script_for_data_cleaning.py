import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# === 自动定位项目根目录 ===
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CLEAN_DIR = BASE_DIR / "clean_data"

CLEAN_DIR.mkdir(exist_ok=True)

# === 读取 train / test ===
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")

# === train: 转换日期 ===
train['date'] = pd.to_datetime(train['date'], errors='coerce')
train.to_csv(CLEAN_DIR / "train_clean.csv", index=False)

# === train: 创建日期特征 ===
train['year'] = train['date'].dt.year
train['month'] = train['date'].dt.month
train['weekday'] = train['date'].dt.weekday
train.to_csv(CLEAN_DIR / "train_clean.csv", index=False)

# === train: 合并节假日 ===
holidays = pd.read_csv(DATA_DIR / "holidays_events.csv")
holidays['date'] = pd.to_datetime(holidays['date'], errors='coerce')
train = train.merge(holidays[['date', 'type', 'description']], on='date', how='left')
train['is_holiday'] = train['type'].notnull().astype(int)
train['holiday_type'] = train['type'].fillna('None')
train['holiday_name'] = train['description'].fillna('')
train.to_csv(CLEAN_DIR / "train_clean.csv", index=False)

# === test: 转换日期 ===
test['date'] = pd.to_datetime(test['date'], errors='coerce')
test.to_csv(CLEAN_DIR / "test_clean.csv", index=False)

# === test: 创建日期特征 ===
test['year'] = test['date'].dt.year
test['month'] = test['date'].dt.month
test['weekday'] = test['date'].dt.weekday
test.to_csv(CLEAN_DIR / "test_clean.csv", index=False)

# === test: 合并节假日 ===
test = test.merge(holidays[['date', 'type', 'description']], on='date', how='left')
test['is_holiday'] = test['type'].notnull().astype(int)
test['holiday_type'] = test['type'].fillna('None')
test['holiday_name'] = test['description'].fillna('')
test.to_csv(CLEAN_DIR / "test_clean.csv", index=False)

# === 加载原始数据并转换为 category ===
train_df  = pd.read_csv(DATA_DIR / "train.csv")
stores_df = pd.read_csv(DATA_DIR / "stores.csv")
cat_cols_train = ['family']
cat_cols_stores = ['type', 'city', 'state', 'cluster']
for col in cat_cols_train:
    if col in train_df.columns:
        train_df[col] = train_df[col].astype('category')
for col in cat_cols_stores:
    if col in stores_df.columns:
        stores_df[col] = stores_df[col].astype('category')

# === 生成交叉特征 ===
train_merged = train_df.merge(stores_df[['store_nbr', 'type']], on='store_nbr', how='left')
train_merged['store_type_item_family'] = (
    train_merged['type'].astype(str) + "_" + train_merged['family'].astype(str)
)

# === 连续促销天数 ===
def mark_promo_streak(group: pd.DataFrame) -> pd.Series:
    streak = pd.Series(0, index=group.index)
    current = 0
    for i in range(len(group)):
        if group.loc[group.index[i], "onpromotion"]:
            current += 1
        else:
            current = 0
        streak.iloc[i] = current
    return streak

train_df = train_df.sort_values(["store_nbr", "family", "date"])
train_df["promo_streak"] = (
    train_df.groupby(["store_nbr", "family"], group_keys=False)
            .apply(mark_promo_streak)
)

# === 清洗 oil 与 transactions ===
file_oil = pd.read_csv(DATA_DIR / "oil.csv")
file_transactions = pd.read_csv(DATA_DIR / "transactions.csv")
file_oil.fillna({"dcoilwtico": file_oil["dcoilwtico"].mean()}, inplace=True)
file_transactions.fillna({
    "store_nbr": file_transactions["store_nbr"].mean(),
    "transactions": file_transactions["transactions"].mean()
}, inplace=True)
file_oil.to_csv(CLEAN_DIR / "oil_clean.csv", index=False)
file_transactions.to_csv(CLEAN_DIR / "transactions_clean.csv", index=False)

# === 聚合交易量 ===
transaction = pd.read_csv(CLEAN_DIR / "transactions_clean.csv")
result = transaction.groupby(["store_nbr", "date"])["transactions"].sum().reset_index()
result.to_csv(CLEAN_DIR / "result.csv", index=False)

# === 生成 is_promo 特征 ===
file_test = pd.read_csv(CLEAN_DIR / "test_clean.csv")
file_train = pd.read_csv(CLEAN_DIR / "train_clean.csv")
file_test['is_promo'] = (file_test['onpromotion'] > 0).astype(int)
file_train['is_promo'] = (file_train['onpromotion'] > 0).astype(int)
file_test.to_csv(CLEAN_DIR / "test_clean.csv", index=False)
file_train.to_csv(CLEAN_DIR / "train_clean.csv", index=False)

# === 合并 stores 信息 ===
train = pd.read_csv(CLEAN_DIR / "train_clean.csv")
stores = pd.read_csv(DATA_DIR / "stores.csv")
train_merged = train.merge(stores, on="store_nbr", how="left")
train_merged.to_csv(CLEAN_DIR / "train_with_store_info.csv", index=False)

# === 合并 oil 信息 ===
train = pd.read_csv(CLEAN_DIR / "train_with_store_info.csv")
oil = pd.read_csv(DATA_DIR / "oil.csv")
train['date'] = pd.to_datetime(train['date'])
oil['date'] = pd.to_datetime(oil['date'])
train = train.merge(oil[['date','dcoilwtico']], on='date', how='left')
train.to_csv(CLEAN_DIR / "train_with_store_oil.csv", index=False)

# === 生成 city_family 特征 ===
file_train = pd.read_csv(CLEAN_DIR / "train_clean.csv")
file_stores = pd.read_csv(DATA_DIR / "stores.csv")
file_test = pd.read_csv(CLEAN_DIR / "test_clean.csv")
merged_train = file_train.merge(file_stores, on='store_nbr', how='left')
file_train['city_family'] = merged_train['city'] + '_' + merged_train['family']
file_train.to_csv(CLEAN_DIR / "train_clean.csv", index=False)

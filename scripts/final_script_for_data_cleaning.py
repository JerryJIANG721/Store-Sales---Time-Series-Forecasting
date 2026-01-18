import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

# ====== Data Cleaning Script ======

# === find the path of project root ===
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CLEAN_DIR = BASE_DIR / "clean_data"

CLEAN_DIR.mkdir(exist_ok=True)

# === read train / test ===
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")

# === train: transform date ===
train['date'] = pd.to_datetime(train['date'], errors='coerce')
train.to_csv(CLEAN_DIR / "train_clean.csv", index=False)

# === train: create date features ===
train['year'] = train['date'].dt.year
train['month'] = train['date'].dt.month
train['weekday'] = train['date'].dt.weekday
train.to_csv(CLEAN_DIR / "train_clean.csv", index=False)

# === train: combine holidays ===
holidays = pd.read_csv(DATA_DIR / "holidays_events.csv")
holidays['date'] = pd.to_datetime(holidays['date'], errors='coerce')
train = train.merge(holidays[['date', 'type', 'description']], on='date', how='left')
train['is_holiday'] = train['type'].notnull().astype(int)
train['holiday_type'] = train['type'].fillna('None')
train['holiday_name'] = train['description'].fillna('')
train.to_csv(CLEAN_DIR / "train_clean.csv", index=False)

# === handle sales column: negative values and outliers (99% quantile truncation) ===
train = pd.read_csv(CLEAN_DIR / "train_clean.csv", low_memory=False)
if "sales" in train.columns:
    train["sales"] = pd.to_numeric(train["sales"], errors="coerce")
    median_sales = train["sales"].median()
    train["sales"] = train["sales"].fillna(median_sales)
    upper_limit = train["sales"].quantile(0.99)
    train["sales"] = train["sales"].clip(lower=0, upper=upper_limit)
train.to_csv(CLEAN_DIR / "train_clean.csv", index=False)

# === normalize continuous variables (sales, transactions) ===
scaler = MinMaxScaler()
train = pd.read_csv(CLEAN_DIR / "train_clean.csv", low_memory=False)

if "sales" in train.columns:
    train["sales_norm"] = scaler.fit_transform(train[["sales"]])

try:
    transactions = pd.read_csv(CLEAN_DIR / "transactions_clean.csv", low_memory=False)
    if "transactions" in transactions.columns:
        transactions["transactions_norm"] = scaler.fit_transform(transactions[["transactions"]])
        transactions.to_csv(CLEAN_DIR / "transactions_clean.csv", index=False)
except FileNotFoundError:
    pass

train.to_csv(CLEAN_DIR / "train_clean.csv", index=False)

# === test: transform date ===
test['date'] = pd.to_datetime(test['date'], errors='coerce')
test.to_csv(CLEAN_DIR / "test_clean.csv", index=False)

# === test: create date features ===
test['year'] = test['date'].dt.year
test['month'] = test['date'].dt.month
test['weekday'] = test['date'].dt.weekday
test.to_csv(CLEAN_DIR / "test_clean.csv", index=False)

# === test: combine holidays ===
test = test.merge(holidays[['date', 'type', 'description']], on='date', how='left')
test['is_holiday'] = test['type'].notnull().astype(int)
test['holiday_type'] = test['type'].fillna('None')
test['holiday_name'] = test['description'].fillna('')
test.to_csv(CLEAN_DIR / "test_clean.csv", index=False)

# === load original data and transform to category ===
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

# === generate cross features ===
train_merged = train_df.merge(stores_df[['store_nbr', 'type']], on='store_nbr', how='left')
train_merged['store_type_item_family'] = (
    train_merged['type'].astype(str) + "_" + train_merged['family'].astype(str)
)

# === on promotion date in a roll ===
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

# === clean oil and transactions ===
file_oil = pd.read_csv(DATA_DIR / "oil.csv")
file_transactions = pd.read_csv(DATA_DIR / "transactions.csv")
file_oil.fillna({"dcoilwtico": file_oil["dcoilwtico"].mean()}, inplace=True)
file_transactions.fillna({
    "store_nbr": file_transactions["store_nbr"].mean(),
    "transactions": file_transactions["transactions"].mean()
}, inplace=True)
file_oil.to_csv(CLEAN_DIR / "oil_clean.csv", index=False)
file_transactions.to_csv(CLEAN_DIR / "transactions_clean.csv", index=False)

# === merge transactions ===
transaction = pd.read_csv(CLEAN_DIR / "transactions_clean.csv")
result = transaction.groupby(["store_nbr", "date"])["transactions"].sum().reset_index()
result.to_csv(CLEAN_DIR / "result.csv", index=False)

# === create is_promo feature ===
file_test = pd.read_csv(CLEAN_DIR / "test_clean.csv")
file_train = pd.read_csv(CLEAN_DIR / "train_clean.csv")
file_test['is_promo'] = (file_test['onpromotion'] > 0).astype(int)
file_train['is_promo'] = (file_train['onpromotion'] > 0).astype(int)
file_test.to_csv(CLEAN_DIR / "test_clean.csv", index=False)
file_train.to_csv(CLEAN_DIR / "train_clean.csv", index=False)

# === combine stores information ===
train = pd.read_csv(CLEAN_DIR / "train_clean.csv")
stores = pd.read_csv(DATA_DIR / "stores.csv")
train_merged = train.merge(stores, on="store_nbr", how="left")
train_merged.to_csv(CLEAN_DIR / "train_with_store_info.csv", index=False)

# === combine oil information ===
train = pd.read_csv(CLEAN_DIR / "train_with_store_info.csv")
oil = pd.read_csv(DATA_DIR / "oil.csv")
train['date'] = pd.to_datetime(train['date'])
oil['date'] = pd.to_datetime(oil['date'])
train = train.merge(oil[['date','dcoilwtico']], on='date', how='left')
train.to_csv(CLEAN_DIR / "train_with_store_oil.csv", index=False)

# === create city_family feature ===
file_train = pd.read_csv(CLEAN_DIR / "train_clean.csv")
file_stores = pd.read_csv(DATA_DIR / "stores.csv")
file_test = pd.read_csv(CLEAN_DIR / "test_clean.csv")
merged_train = file_train.merge(file_stores, on='store_nbr', how='left')
file_train['city_family'] = merged_train['city'] + '_' + merged_train['family']
file_train.to_csv(CLEAN_DIR / "train_clean.csv", index=False)



# ===== Visualization Script ======


# === Plot daily sales trends for the Top 10 stores (aggregated from train.csv) & Plot trend of the Top 10 best-selling product categories ===

data = pd.read_csv(CLEAN_DIR / "train_clean.csv")
data["date"] = pd.to_datetime(data["date"])

daily_store = data.groupby(["date", "store_nbr"])["sales"].sum().reset_index()

top_stores = data.groupby("store_nbr")["sales"].sum().nlargest(10).index

daily_store_top = daily_store[daily_store["store_nbr"].isin(top_stores)]

top_families = data.groupby("family")["sales"].sum().nlargest(10).index

df_top_family = data[data["family"].isin(top_families)]

daily_family = df_top_family.groupby(["date", "family"])["sales"].sum().reset_index()

data["weekday_num"] = data["date"].dt.weekday
data["weekday_name"] = data["date"].dt.day_name()

weekday_order = [0, 1, 2, 3, 4, 5, 6]

weekday_sales = data.groupby("weekday_num")["sales"].mean().reindex(weekday_order).reset_index()

weekday_map = {0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"}
weekday_sales["weekday"] = weekday_sales["weekday_num"].map(weekday_map)

data["month"] = data["date"].dt.month

month_sales = data.groupby("month")["sales"].mean().reset_index()

data["quarter"] = data["date"].dt.quarter

quarter_sales = data.groupby("quarter")["sales"].mean().reset_index()



fig, axs = plt.subplots(5,1,figsize=(15,50))

# plot1

for store, sub in daily_store_top.groupby("store_nbr"):
    axs[0].plot(sub["date"], sub["sales"], label=f"Store {store}", alpha=0.8)
axs[0].set_title("Tendance of sales everyday(Top 10 stores)")
axs[0].set_xlabel("Date")
axs[0].set_ylabel("Sales")
axs[0].legend(title="Store number")


# plot2

for fam, sub in daily_family.groupby("family"):
    axs[1].plot(sub["date"], sub["sales"], label=fam, alpha=0.8)

axs[1].set_title("Tendance of top 10 hot sale products")
axs[1].set_xlabel("Date")
axs[1].set_ylabel("Sales")
axs[1].legend(title="Type")


# plot3

axs[2].bar(weekday_sales["weekday"], weekday_sales["sales"])
axs[2].set_title("Average sales between different weeks")
axs[2].set_xlabel("Week")
axs[2].set_ylabel("Average sales")


# plot4

axs[3].bar(month_sales["month"], month_sales["sales"])
axs[3].set_xticks(range(1, 13))
axs[3].set_title("Average sales between different months")
axs[3].set_xlabel("Months")
axs[3].set_ylabel("Average sales")


# plot5


axs[4].bar(quarter_sales["quarter"], quarter_sales["sales"])
axs[4].set_xticks([1, 2, 3, 4])
axs[4].set_title("Average sales between different quarters")
axs[4].set_xlabel("Quarters")
axs[4].set_ylabel("Average sales")

plt.savefig("../figures/figures for tendance and average sales.jpg",dpi=300,bbox_inches='tight')
plt.show()


# === Plot sales trends for each product category (aggregated by item_family) ===
# === Plot holiday vs non-holiday sales comparison (holidays_events.csv) ===
# === Plot heatmap of transactions vs sales volume (transactions.csv and train.csv) ===
# === Plot scatter plot of oil prices vs sales volume (oil.csv and sales data`) ===

data["date"] = pd.to_datetime(data["date"])

df_family = (
    data.groupby(["date", "family"])["sales"]
            .sum()
            .reset_index()
)

plt.figure(figsize=(15, 8))

for fam in df_family["family"].unique():
    sub = df_family[df_family["family"] == fam]
    plt.plot(sub["date"], sub["sales"], label=fam)

plt.title("Sales Trend by Item Family")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()

plt.savefig("../figures/item_family_sales.jpg", dpi=300, bbox_inches="tight")

holidays_df = pd.read_csv(DATA_DIR / "holidays_events.csv")
holidays_df["date"] = pd.to_datetime(holidays_df["date"])
data["date"] = pd.to_datetime(data["date"])

df = data.merge(
    holidays_df[["date", "type", "locale", "description"]],
    on="date",
    how="left"
)

holiday_types = ["Holiday", "Event", "Additional"]
df_holiday = df[df["type"].isin(holiday_types)]

daily_sales = df.groupby("date")["sales"].sum().reset_index()

holiday_sales = (
    df_holiday.groupby("date")["sales"].sum().reset_index()
)

plt.figure(figsize=(15, 6))

plt.plot(daily_sales["date"], daily_sales["sales"],
         label="All Days", alpha=0.4)

plt.plot(holiday_sales["date"], holiday_sales["sales"],
         label="Holiday Days", linewidth=2)

plt.title("Holiday vs Regular Sales Trend")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.tight_layout()
plt.show()

plt.savefig("../figures/holiday_vs_regular_sales.jpg", dpi=300, bbox_inches="tight")



transactions_df = pd.read_csv(CLEAN_DIR / "transactions.csv")
transactions_df["date"] = pd.to_datetime(transactions_df["date"])
data["date"] = pd.to_datetime(data["date"])

sales_daily = (
    data.groupby("date")["sales"]
            .sum()
            .reset_index()
)

df_ts = sales_daily.merge(transactions_df, on="date", how="left")

df_ts["year"] = df_ts["date"].dt.year
df_ts["month"] = df_ts["date"].dt.month

pivot_trans = df_ts.pivot_table(
    index="year", columns="month",
    values="transactions", aggfunc="mean"
)

pivot_sales = df_ts.pivot_table(
    index="year", columns="month",
    values="sales", aggfunc="mean"
)

def plot_heatmap(pivot_df, title, label, filename):
    plt.figure(figsize=(12, 6))
    data = pivot_df.values
    im = plt.imshow(data, cmap="viridis", aspect="auto")
    cbar = plt.colorbar(im)
    cbar.set_label(label)

    plt.xticks(
        np.arange(12),
        ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],
        rotation=45
    )
    plt.yticks(np.arange(len(pivot_df.index)), pivot_df.index)

    plt.title(title)
    plt.xlabel("Month")
    plt.ylabel("Year")
    plt.tight_layout()
    plt.savefig(f"../figures/{filename}", dpi=300, bbox_inches="tight")
    plt.show()

plot_heatmap(
    pivot_trans,
    "Heatmap of Average Transactions",
    "Avg Transactions",
    "transactions_heatmap.jpg"
)

plot_heatmap(
    pivot_sales,
    "Heatmap of Average Sales",
    "Avg Sales",
    "sales_heatmap.jpg"
)




oil_df = pd.read_csv(DATA_DIR / "oil.csv")
oil_df["date"] = pd.to_datetime(oil_df["date"])

data["date"] = pd.to_datetime(data["date"])

sales_daily = (
    data.groupby("date")["sales"]
            .sum()
            .reset_index()
)

df_oil = sales_daily.merge(oil_df, on="date", how="left")

plt.figure(figsize=(10, 6))

plt.scatter(df_oil["dcoilwtico"], df_oil["sales"], alpha=0.4)

plt.title("Oil Price vs Sales (Scatter Plot)")
plt.xlabel("Oil Price (dcoilwtico)")
plt.ylabel("Sales")
plt.tight_layout()
plt.show()
plt.savefig("../figures/oil_vs_sales_scatter.jpg", dpi=300, bbox_inches="tight")


# ===  Plot the total sales over time (aggregated from train.csv) ===
# === Plot total sales by store type (store_type × total sales) ===
# === Plot the impact of promotions on sales (using the onpromotion column) ===


train = pd.read_csv(CLEAN_DIR / "train_final.csv")
daily_sales = train.groupby('date')['sales'].sum().reset_index()
plt.figure(figsize=(12, 6))
plt.plot(daily_sales['date'], daily_sales['sales'])
plt.xticks(daily_sales['date'][::30], rotation=45)
plt.xlabel('date')
plt.ylabel('Total sales')
plt.title('Total sales volume over time')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("../figures/Total sales volume over time.jpg", format='jpg', dpi=300)
plt.show()


store_sales = train.groupby('type_y')['sales'].sum().reset_index()
plt.figure(figsize=(10, 6))
plt.bar(store_sales['type_y'], store_sales['sales'])

plt.xlabel('store type')
plt.ylabel('Total sales')
plt.title('Total sales volume of different store types')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("../figures/Total sales volume of different store types.jpg", format='jpg', dpi=300)
plt.show()


train['date'] = pd.to_datetime(train['date'])
daily_summary = train.groupby('date').agg({'sales':'sum', 'onpromotion':'sum'}).reset_index()
fig, ax1 = plt.subplots(figsize=(14,6))
ax2 = ax1.twinx()
ax1.plot(daily_summary['date'], daily_summary['sales'], color='blue', label='Total sales', linewidth=2)
ax1.set_ylabel('Total sales', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax2.plot(daily_summary['date'], daily_summary['onpromotion'], color='red', label='Number of promotonal items', linewidth=2)
ax2.set_ylabel('Number of promotonal items', color='red')
ax2.tick_params(axis='y', labelcolor='red')
ax1.set_xlabel('data')
ax1.set_xticks(daily_summary['date'][::30])
ax1.set_xticklabels(daily_summary['date'][::30].dt.strftime('%Y-%m-%d'), rotation=45)
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')
fig.tight_layout()
plt.title('Impact of promotions on sales')
plt.savefig("../figures/Impact of promotions on sales.jpg", format='jpg', dpi=300)
plt.show()



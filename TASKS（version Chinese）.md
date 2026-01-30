- [ ] 未完成任务
- [x] 已完成任务

---

## ZXY

- [x] 检查 `train.csv` 和 `test.csv` 的数据类型及缺失值（用 `df.info()` 和 `df.isnull().sum()`）
- [x] 转换 `date` 列为 datetime 类型（`pd.to_datetime()`）
- [x] 创建日期特征：year、month、weekday（基于 `date` 列）
- [x] 创建节假日特征（结合 `holidays_events.csv`，标记是否节假日、节假日类型）
- [x] 聚合 `transactions.csv`，计算每个 store 每天交易总量
- [x] 合并 `stores.csv` 到主表，加入 store_type、city 信息
- [x] 合并 `oil.csv` 到主表，加入当日油价特征
- [x] 合并 `holidays_events.csv` 到主表，加入节假日相关信息
- [x] 检查 `sales` 列的负值或异常值，并处理
- [x] 绘制总销售量随时间变化折线图（基于 `train.csv` 聚合数据）
- [x] 绘制不同店铺类型销售柱状图（`store_type` × 总销售）
- [x] 绘制促销对销售的影响图（结合 `onpromotion` 列）
- [x] 构建简单基线模型（前一天销售或滑动平均预测）
- [x] 划分训练集和验证集（例如按日期切分）
- [x] 对销售量做 log(1+sales) 平滑处理
- [x] 输出初步预测图表（基线模型预测）
- [ ] 保存清洗后的训练集和测试集 CSV 文件
- [ ] 编写清洗和初步分析脚本，保证可复现
- [ ] 集中整理最终文件完善project
- [ ] 撰写数据清理部分的报告

---

## ZZN

- [x] 检查分类列并转换为 category（如 `store_type`, `item_family`）
- [x] 生成交叉特征（store_type × item_family）
- [x] 标记连续促销天数（基于 `is_promo` 列）
- [x] 归一化连续变量（如销售量、交易量）
- [x] 绘制每个商品类别销售趋势（基于 `item_family` 聚合）
- [x] 绘制节假日销售对比图（`holidays_events.csv`）
- [x] 绘制交易量与销售量热力图（`transactions.csv` 与 `train.csv`）
- [x] 绘制油价与销售量散点图（`oil.csv` 与销售数据）
- [x] 构建时间序列模型（SARIMA）预测销售
- [x] 训练 LightGBM
- [x] 对验证集生成预测
- [x] 编写建模和可视化脚本，保证可复现
- [ ] 集中整理最终文件完善project
- [ ] 撰写可视化部分的报告，以及观测到的现象

---

## JL

- [x] 检查缺失值并填充（`oil.csv`, `transactions.csv`）
- [x] 生成是否促销特征（`onpromotion` 列）
- [x] 生成城市 × 商品交叉特征（`city` × `item_family`）
- [x] 保存清洗后的 CSV 文件
- [x] 绘制TOP10店铺每日销售趋势（`train.csv` 聚合）
- [x] 绘制 top 10 热销商品类别趋势
- [x] 分析周中销售差异
- [x] 分析月份/季度销售差异
- [x] 构建基线模型
- [x] 训练时间序列模型（Prophet）
- [x] 训练机器学习模型（XGBoost）
- [x] 对验证集生成预测
- [x] 对测试集生成预测
- [ ] 格式化为 `submission.csv`
- [x] 绘制预测与实际对比图
- [ ] 编写可复现脚本（清洗 + EDA + 建模整合）
- [ ] 集中整理最终文件完善project
- [ ] 撰写训练以及预测部分的报告以及预测结果

---

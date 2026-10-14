# Kaggle 房价预测（House Prices: Advanced Regression Techniques）

基于 Ames 住宅数据集的房价回归预测练习项目，包含从探索性数据分析（EDA）、缺失值处理、特征编码到模型训练与调参的完整流程。

- 比赛地址：https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques
- 数据规模：`train.csv` 1460 × 81（含 `SalePrice`），`test.csv` 1459 × 80

## 目录结构

```
.
├── data_description.txt            官方字段说明字典（处理缺失值/编码时的主要依据）
├── kaggle房价预测（初学）.ipynb     第一版：EDA 为主 + XGBoost 基线模型
├── kaggle房价预测（完善）.ipynb     第二版：规范化预处理 + Optuna 多模型自动调参
├── train.csv                       训练集（含 SalePrice）
├── test.csv                        测试集（无 SalePrice）
├── submission.csv                  初学版提交结果（XGBoost）
└── submission_gbr.csv              完善版提交结果（GradientBoostingRegressor）
```

## 两份 Notebook 的区别

| | `kaggle房价预测（初学）.ipynb` | `kaggle房价预测（完善）.ipynb` |
| --- | --- | --- |
| 侧重点 | 熟悉数据、可视化探索 | 完整且可复现的建模流程 |
| EDA | missingno 缺失矩阵、类别/数值分布对比、箱线图、相关性热力图 | 按特征类型分类统计、偏度计算、相关性热力图 |
| 缺失值 | 类别型高频缺失填 `"None"`、低频填众数；数值型离散填众数、连续填均值 | 类别型按缺失量分档填充；`GarageYrBlt` 增加 `HasGarage` 指示列 + 众数填充；`LotFrontage` 用 `LotArea`/`1stFlrSF` 训练随机森林回归填充 |
| 目标变量 | `log1p` 变换（偏度 1.88 → 0.12） | `log1p` 变换，并对偏度 > 1 的连续特征同样做对数化处理 |
| 编码 | `get_dummies`（仅类别列） | `get_dummies`（全表） |
| 调参 | Optuna，10 trials，仅 XGBoost | Optuna，100 trials，同时搜索 Ridge / GBR / XGBoost / LightGBM |
| 最终模型 | `XGBRegressor` | `GradientBoostingRegressor`（交叉验证得分最优） |
| 输出 | `submission.csv` | `submission_gbr.csv` |

## 处理流程（完善版）

1. **导入与合并**：`train` 与 `test` 纵向拼接后统一处理，删除无预测价值的 `Id` 列。
2. **特征分类**：按 dtype 分为数值型/类别型，再按取值个数（`>= 20`）细分为连续型与离散型。
3. **缺失值分析**：分别统计训练/测试集缺失率，结合 `data_description.txt` 区分「结构性缺失」（如 `PoolQC` 缺失表示没有泳池）与「随机缺失」。
4. **缺失值填充**：
   - 类别型：缺失数 ≥ 80 的列填 `"None"`，其余填众数；
   - `GarageYrBlt`：新增 `HasGarage` 指示列，原列填众数；
   - `LotFrontage`：以 `LotArea`、`1stFlrSF` 为特征训练 `RandomForestRegressor` 回归填充；
   - 其余数值型：离散填众数，连续填均值。
5. **正态化**：`SalePrice` 与偏度 > 1 的连续特征做 `np.log1p` 变换。
6. **编码与切分**：`pd.get_dummies` 独热编码后，按原始行数切回 train / test。
7. **建模调参**：`optuna` + 10 折 `cross_val_score`（`neg_mean_squared_error`）搜索超参数，各模型 100 次试验。
8. **预测输出**：最优模型（GBR）在全量训练集上拟合，`np.expm1` 还原对数尺度后写出 `submission_gbr.csv`。

## 环境依赖

Python 3.x，主要依赖：

```
pandas numpy matplotlib seaborn scikit-learn
xgboost lightgbm optuna missingno
```

初学版额外用到 `missingno` 做缺失值可视化。建议在虚拟环境中安装后，用 Jupyter Notebook / VS Code 打开 `.ipynb` 顺序执行；两份 Notebook 都假设 `train.csv`、`test.csv` 位于同一目录。

## 运行方式

```bash
# 1. 安装依赖
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm optuna missingno

# 2. 启动 Jupyter
jupyter notebook
```

打开 `kaggle房价预测（完善）.ipynb` 依次运行全部单元格即可复现 `submission_gbr.csv`。

## 数据集说明

`train.csv` / `test.csv` 来自 Kaggle 官方比赛，共 79 个特征（不含 `Id` 与 `SalePrice`），涵盖房屋面积、质量评级、建造/翻新年份、地下室与车库情况、社区位置等维度。每个字段的详细含义参见 `data_description.txt`。

## 备注

- 初学版 Notebook 保留了大量绘图输出，文件体积较大（约 4.6 MB），在 GitHub 上预览可能较慢。
- 两份 Notebook 相互独立，可单独运行；完善版为推荐阅读的主线流程。

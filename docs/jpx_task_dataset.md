### JPX 東京證券交易所股價預測：任務與資料集說明（含子集採樣方法）

本文件整理本專案所對應的 Kaggle「JPX Tokyo Stock Exchange Prediction」任務與資料集，並給出可再現的子集採樣（subset sampling）流程，方便撰寫論文與重現實驗。


## 任務概述

- **目標**：對每日全市場股票進行橫截面排序，構建多空組合（多頭前 N 檔、空頭後 N 檔），以衡量超額表現。
- **模型輸出**：對每支股票在每個交易日輸出分數（或機率），再做每日橫截面標準化後進行排序。
- **評估**：以樣本外（OOS）期間的長短組合日報酬序列計算夏普值（Sharpe）。


## 資料集介紹

- **來源**：Kaggle JPX（需下載官方資料）。
- **主要檔案**：
  - `train_files/stock_prices.csv`：日頻價格資料與目標 `Target`。
  - `stock_list.csv`：股票基本資訊（含 `Universe0` 篩選旗標）。
- **常用欄位**：
  - `Date`（日期）、`SecuritiesCode`（股票代碼）、`Close`（收盤價）、`Volume`（成交量）、`AdjustmentFactor`（復權因子）、`SupervisionFlag`、`Target`（比賽定義的預測目標）。
- **調整收盤價**：以復權因子累乘得到 `AdjClose = Close * cumprod(AdjustmentFactor)`（見程式中的 `add_adjusted_close`）。


## 特徵工程（概略）

- 基於 `AdjClose` 計算：
  - 單期報酬 `ret1`，動能 `mom{1,2,5,10,20}`；波動度 `vol{5,10,20}`（以 `ret1` 的 rolling std）。
  - 交易額 `dv = Close * Volume`，及其 rolling/對數特徵（`dv_mean{5,10,20}`, `vol_mean{5,10,20}`, `log_vol`, `log_dv`）。
- 橫截面處理（逐日）：
  - 分位縮尾（winsorize 1%~99%），再做 Z-score 標準化（避免極端值影響）。


## 標籤定義與訓練樣本選取（弱監督）

- 每個交易日針對 `Target` 做橫截面排序，取上下各 `per_side` 檔形成二元標籤：
  - 底部樣本標記 `label_bin=0`，頂部樣本標記 `label_bin=1`；中間區域不納入訓練（提升訊噪比）。
- 程式中以 `select_outlier_training_samples(..., per_side=200 或 300)` 實作。


## 時序切分與序列構建

- **切分**：以日期做 8:2 的時間切分（前 80% 為訓練、後 20% 為驗證/OOS），確保沒有洩漏未來資訊。
- **序列**：每檔個股依時間滑動視窗產生長度 `seq_len=20` 的特徵序列，用於 LSTM/Transformer/QLSTM 等時間序列模型。


## 子集採樣方法（重點）

為了降低運算量、便於快速迭代或論文實驗的「公平可再現」對比，本專案提供兩種「交易日子集抽樣」方式，均在特徵工程前先對 `stock_prices.csv` 以日期層級抽樣：

1) **步距抽樣（stride）**  
   - 以固定步距 \(k\) 在全部交易日中取樣：選擇每第 \(k\) 個交易日（例如 `k=11`，約取 9% 的交易日）。
   - 適合快速縮小資料量，同時保留長期時間跨度。

2) **分年分層比例抽樣（fraction per year）**  
   - 先按年分桶，對每一年獨立隨機抽取固定比例（例如 10%）；確保跨年均衡，避免集中於特定年份。
   - 可用亂數種子 `seed` 控制可再現性。

兩者皆回傳被抽中的交易日集合，並用來過濾原始 `prices`。抽樣完成後可選擇將子集儲存為一個「自足資料夾」，包含子集化後的 `train_files/stock_prices.csv` 與原樣複製的 `stock_list.csv`，便於分享與重現。


## 再現與操作指引

以下以 `scripts/qlstm_pipeline.py` 為例；同類旗標也存在於 `qnn_pipeline.py`。

- 路徑假設：
  - 原始資料：`/media/meow/Transcend/kaggle_jpx_stock/data`
  - 輸出：`/media/meow/Transcend/kaggle_jpx_stock/outputs/...`

- 以「步距 11」抽樣並另存子集：

```bash
python /media/meow/Transcend/kaggle_jpx_stock/scripts/qlstm_pipeline.py \
  --data-dir /media/meow/Transcend/kaggle_jpx_stock/data \
  --output-dir /media/meow/Transcend/kaggle_jpx_stock/outputs/qlstm_stride11 \
  --date-stride 11 \
  --subset-save-dir /media/meow/Transcend/kaggle_jpx_stock/data_subset_stride11
```

- 以「每年 10%」比例抽樣（可重現 seed=42）並另存子集：

```bash
python /media/meow/Transcend/kaggle_jpx_stock/scripts/qlstm_pipeline.py \
  --data-dir /media/meow/Transcend/kaggle_jpx_stock/data \
  --output-dir /media/meow/Transcend/kaggle_jpx_stock/outputs/qlstm_frac10 \
  --date-fraction 0.1 \
  --seed 42 \
  --subset-save-dir /media/meow/Transcend/kaggle_jpx_stock/data_subset_frac10
```

- 只抽樣、不另存，也可省略 `--subset-save-dir`，直接在同一流程中完成特徵工程、訓練與評估。

- 子集資料夾結構（範例 `data_subset_stride11/`）：
  - `data_subset_stride11/train_files/stock_prices.csv`（抽樣後的價量與 Target）
  - `data_subset_stride11/stock_list.csv`（原樣複製，保持資料自足性）


## 評估與樣本外（OOS）成績（範例）

本專案各流程在完成打分與每日選股後，會輸出：

- `daily_selection.csv`：每日入選之多頭/空頭名單與分數。
- `daily_long_short_returns.csv`：基於 `Target` 的長短組合日報酬。
- `summary.txt`：摘要 OOS 夏普值與期間。

目前部分輸出（示例）：

- `outputs/qlstm_stride11/summary.txt`：Sharpe (OOS) ≈ 0.4667，OOS 期間 2021-11-15 ~ 2021-12-01（示兩個交易日）。
- `outputs/qlstm_50epoch_stride11/summary.txt`：Sharpe (OOS) ≈ 0.3331（示兩個交易日）。
- `outputs/qasa_50epoch_stride11/summary.txt`：Sharpe (OOS) ≈ 0.2659，OOS 期間 2020-12-21 ~ 2021-12-01（示 22 個交易日）。

實際數值與期間請以當前輸出檔為準。


## 重現步驟（論文建議撰寫）

1. 準備資料：下載 Kaggle JPX 官方資料至 `data/`，包含 `train_files/stock_prices.csv` 與 `stock_list.csv`。
2. 選擇子集抽樣策略：
   - 步距抽樣（例如 `--date-stride 11`），或
   - 分年分層比例抽樣（例如 `--date-fraction 0.1 --seed 42`）。
3. 執行任一流程（QLSTM/QNN/Transformer 等），指定 `--output-dir`，必要時指定 `--subset-save-dir` 產生可分享子集。
4. 產出 `daily_selection.csv`、`daily_long_short_returns.csv`、`summary.txt` 並在論文中報告 OOS 夏普值與期間。
5. 報告實作細節：特徵工程、橫截面縮尾與標準化、極端樣本監督、時間切分與序列長度等。


## 版本相依與注意事項

- 若使用 Pandas 新版，`pct_change` 的 `fill_method` 預設變動會引發警告；本專案已改為顯式 `fill_method=None`（避免隱式前填充）。
- 量子層（PennyLane）在當前設定下於 CPU 執行，與 PyTorch 張量裝置需一致。
- 如需完全可重現，請在論文中記錄：隨機種子、抽樣策略與參數、特徵集合、序列長度、模型超參數、訓練 epoch 與批量大小、評估期間與指標定義。


—— 完 ——



# F1 Pitstop Predictor

Binary classification model to predict whether an F1 driver will pit on the next lap.

**Target:** `PitNextLap` | **Dataset:** f1_strategy_dataset_v4.csv | **Best model:** XGBoost

---

## Project Structure

```
f1_pitstop_predictor/
├── data/
│   └── f1_strategy_dataset_v4.csv     
├── notebooks/
│   ├── 01_EDA.ipynb                     
│   ├── 02_BaselineModel.ipynb           
│   ├── 03_ComparisonModel.ipynb         
│   └── 04_FinalModel.ipynb              
├── src/
│   ├── preprocess.py                    
│   ├── feature_engineering.py          
│   ├── train.py                        
│   └── evaluate.py                      
├── outputs/
│   ├── figures/                         
│   └── models/                        
├── config.py                            
├── requirements.txt
└── README.md
```

## Setup

```bash
# 1. Clone / create the folder
cd f1_pitstop_predictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place the dataset
#    Copy f1_strategy_dataset_v4.csv into data/

# 4. Open VS Code in this folder (important for paths)
code .

# 5. Run notebooks in order: 01 → 02 → 03 → 04
```

## Key Design Decisions

| Decision | Reason |
|---|---|
| Target = `PitNextLap` not `PitStop` | Predicting future is more useful than explaining past |
| Year-based train/val/test split | Laps are time-ordered; random split = data leakage |
| `PitStop` dropped from features | Direct leakage into target |
| `scale_pos_weight` in XGBoost | 74.5% / 25.5% class imbalance |
| Threshold tuning | Default 0.5 is suboptimal for imbalanced classes |
| SHAP for interpretation | Required by professor + great for presentation |

## Results Summary

| Model | F1 | ROC-AUC |
|---|---|---|
| Logistic Regression |  | |
| Random Forest |  |  |
| XGBoost (base) |  |  |
| XGBoost + Eng |  |  |
| XGBoost (tuned) |  |  |



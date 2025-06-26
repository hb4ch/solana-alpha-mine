import polars as pl
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, TimeSeriesSplit, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from sklearn.metrics import classification_report, roc_auc_score
from scipy.stats import uniform, randint
from data_loader import load_and_preprocess_raw_data
from features import engineer_features
import logging

LOGS_DIR = "logs"
MODELS_DIR = "models"
DATA_BASE_PATH = "crypto_tick"
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "model_training.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def generate_target_variable(df: pl.DataFrame, market_to_predict: str = 'SOL_USDT',
                             horizon_seconds: int = 3600,
                             tp_pct: float = 0.004, sl_pct: float = 0.002) -> pl.DataFrame:
    logger.info(f"Generating target variable for {market_to_predict} with {horizon_seconds}s horizon...")
    
    df_market = df.filter(pl.col('market') == market_to_predict)
    if df_market.is_empty():
        logger.warning(f"No data for {market_to_predict} to generate labels.")
        return df

    time_diffs = df_market['ts_utc'].diff().median()
    ticks_per_second = 1 / (time_diffs.total_seconds() if time_diffs else 5)
    horizon_steps = int(horizon_seconds * ticks_per_second) or 1

    df_market = df_market.with_columns([
        (pl.col('mid_price') * (1 + tp_pct)).alias('tp_price'),
        (pl.col('mid_price') * (1 - sl_pct)).alias('sl_price')
    ])

    rolling_max_price = pl.col('mid_price').rolling_max(window_size=horizon_steps).shift(-horizon_steps + 1)
    rolling_min_price = pl.col('mid_price').rolling_min(window_size=horizon_steps).shift(-horizon_steps + 1)

    tp_hit = rolling_max_price >= pl.col('tp_price')
    sl_hit = rolling_min_price <= pl.col('sl_price')

    df_market = df_market.with_columns(
        pl.when(tp_hit & ~sl_hit).then(1).otherwise(0).alias('label')
    )

    logger.info(f"Target variable generation complete. Label distribution:\n{df_market['label'].value_counts()}")
    
    return df.join(df_market.select(['ts_utc', 'market', 'label']), on=['ts_utc', 'market'], how='left').with_columns(pl.col('label').fill_null(-1))

def train_model(df: pl.DataFrame, features_list: list, target_col: str = 'label',
                market_to_predict: str = 'SOL_USDT'):
    logger.info("Starting model training...")

    df_train_val = df.filter((pl.col('market') == market_to_predict) & (pl.col(target_col) != -1)).drop_nulls(subset=features_list + [target_col])

    if df_train_val.is_empty() or df_train_val['label'].n_unique() < 2:
        logger.error(f"Not enough data or classes for {market_to_predict} to train.")
        return None, None

    X = df_train_val.select(features_list)
    y = df_train_val[target_col]

    X_dev, X_holdout_test, y_dev, y_holdout_test = train_test_split(X, y, test_size=0.20, shuffle=False)
    
    logger.info(f"Development set size: {X_dev.shape}, Hold-out Test set size: {X_holdout_test.shape}")

    scaler = StandardScaler()
    X_dev_scaled = scaler.fit_transform(X_dev.to_numpy())

    param_dist = {
        'n_estimators': randint(50, 200), 'learning_rate': uniform(0.01, 0.1),
        'num_leaves': randint(20, 50), 'max_depth': [-1, 10, 20, 30],
        'min_child_samples': randint(20, 100), 'subsample': uniform(0.7, 0.3),
        'colsample_bytree': uniform(0.7, 0.3)
    }

    random_search = RandomizedSearchCV(
        estimator=lgb.LGBMClassifier(random_state=42, class_weight='balanced', 
                                     n_jobs=-1, device='gpu'),
        param_distributions=param_dist, n_iter=10, cv=TimeSeriesSplit(n_splits=3),
        scoring='roc_auc', verbose=2, n_jobs=-1, random_state=42
    )
    
    random_search.fit(X_dev_scaled, y_dev.to_numpy())

    logger.info(f"Best parameters for LGBM: {random_search.best_params_}")
    logger.info(f"Best AUC from RandomizedSearchCV (LGBM): {random_search.best_score_:.4f}")

    final_model = random_search.best_estimator_
    X_holdout_test_scaled = scaler.transform(X_holdout_test.to_numpy())
    
    test_preds_proba = final_model.predict_proba(X_holdout_test_scaled)[:, 1]
    test_preds_label = final_model.predict(X_holdout_test_scaled)
    
    logger.info("Hold-out Test Set Performance:")
    logger.info(f"\n{classification_report(y_holdout_test.to_numpy(), test_preds_label)}")
    try:
        logger.info(f"Hold-out Test AUC: {roc_auc_score(y_holdout_test.to_numpy(), test_preds_proba):.4f}")
    except ValueError as e:
        logger.warning(f"Could not calculate AUC for hold-out test set: {e}")

    model_filename = os.path.join(MODELS_DIR, f"{market_to_predict}_lgbm_model.joblib")
    scaler_filename = os.path.join(MODELS_DIR, f"{market_to_predict}_scaler.joblib")
    features_filename = os.path.join(MODELS_DIR, f"{market_to_predict}_features.joblib")
    
    joblib.dump(final_model, model_filename)
    joblib.dump(scaler, scaler_filename)
    joblib.dump(features_list, features_filename)
    
    logger.info(f"Model, scaler, and feature list saved.")

    return final_model, scaler

def run_training_pipeline(markets_to_load: list, target_market: str,
                          features_list: list,
                          tp_pct: float, sl_pct: float, training_horizon_seconds: int):
    logger.info("Starting ML training pipeline...")

    df_raw = load_and_preprocess_raw_data(base_path=DATA_BASE_PATH, markets=markets_to_load)
    if df_raw.is_empty():
        return

    df_featured = engineer_features(df_raw)
    
    final_features_to_use = features_list + [col for col in df_featured.columns if col.startswith(('phase_', 'volregime_')) and col not in features_list]
    final_features_to_use = [f for f in final_features_to_use if f in df_featured.columns]
    logger.info(f"Final features for training: {final_features_to_use}")
    
    df_processed = df_featured.drop_nulls(subset=final_features_to_use)
    if df_processed.is_empty():
        logger.error("DataFrame empty after feature engineering and NaN drop.")
        return

    df_final = generate_target_variable(
        df_processed, market_to_predict=target_market,
        horizon_seconds=training_horizon_seconds, tp_pct=tp_pct, sl_pct=sl_pct
    )
    
    if 'label' not in df_final.columns or df_final.filter(pl.col('market') == target_market)['label'].is_in([0, 1]).sum() == 0:
        logger.error(f"No valid labels for {target_market}. Exiting.")
        return

    train_model(df_final, features_list=final_features_to_use, target_col='label', market_to_predict=target_market)

    logger.info("ML training pipeline finished.")

if __name__ == "__main__":
    logger.info("model_trainer.py executed directly. To run the full pipeline, run main.py.")

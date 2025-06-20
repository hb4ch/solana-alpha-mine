# src/model_trainer.py
import pandas as pd
import numpy as np
import joblib
import os
import ast # For parsing stringified lists
from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier # Will be replaced by LGBM
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb # Import LightGBM
from sklearn.metrics import classification_report, roc_auc_score

# Assuming data_loader.py is in the same directory or accessible via PYTHONPATH
from data_loader import load_and_preprocess_raw_data
from features import engineer_features

# --- Configuration ---
# Define paths relative to the project root or make them configurable
LOGS_DIR = "logs"
MODELS_DIR = "models"
DATA_BASE_PATH = "crypto_tick" # Base path for your crypto data

# Ensure directories exist
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# --- Logging Setup (Basic) ---
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "model_training.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# --- 3. Target Variable Generation ---
def generate_target_variable(df: pd.DataFrame, market_to_predict: str = 'SOL_USDT',
                             horizon_seconds: int = 3600,
                             tp_pct: float = 0.004, sl_pct: float = 0.002) -> pd.DataFrame:
    """
    Generates the target variable using a vectorized Triple Barrier Method.
    Focuses on predicting for `market_to_predict`.
    """
    logger.info(f"Generating target variable for {market_to_predict} with {horizon_seconds}s horizon (vectorized)...")
    
    df_market = df[df['market'] == market_to_predict].copy()
    if df_market.empty:
        logger.warning(f"No data found for target market {market_to_predict} to generate labels.")
        return df

    # Calculate the number of steps to look forward based on time
    # This is an approximation. Assumes relatively constant sampling frequency.
    time_diffs = df_market.index.to_series().diff().median()
    if pd.isna(time_diffs) or time_diffs.total_seconds() == 0:
        # Fallback if median can't be calculated (e.g., too few data points)
        # Assuming 5-second ticks if otherwise unknown
        ticks_per_second = 1 / 5 
    else:
        ticks_per_second = 1 / time_diffs.total_seconds()
    
    logger.info(f"time_diff mean amongst tick is f{time_diffs.total_seconds()}")

    horizon_steps = int(horizon_seconds * ticks_per_second)
    if horizon_steps == 0:
        logger.warning("Horizon results in 0 steps. Target generation might not be meaningful.")
        horizon_steps = 1

    # Calculate barriers
    df_market['tp_price'] = df_market['mid_price'] * (1 + tp_pct)
    df_market['sl_price'] = df_market['mid_price'] * (1 - sl_pct)

    # Find future barrier hits
    # Use rolling windows on future prices to find if a barrier was hit
    # shift(-horizon_steps) brings future data to the current row
    future_prices = df_market['mid_price'].shift(-horizon_steps)
    
    # Rolling max/min over the horizon to find if TP/SL was ever touched
    rolling_max_price = df_market['mid_price'].rolling(window=horizon_steps, min_periods=1).max().shift(-horizon_steps + 1)
    rolling_min_price = df_market['mid_price'].rolling(window=horizon_steps, min_periods=1).min().shift(-horizon_steps + 1)

    # Check for barrier touches
    tp_hit = rolling_max_price >= df_market['tp_price']
    sl_hit = rolling_min_price <= df_market['sl_price']

    # Default label is 0 (timeout or SL hit)
    df_market['label'] = 0
    # Set label to 1 where TP is hit and it's not preceded by an SL hit.
    # This is a simplification; a true vectorized first-touch is complex.
    # We assume if both are hit, we check which one is more likely or prioritize one.
    # A common simplification: if TP is hit at any point in the window, label it 1.
    # This favors finding upward movements but might mislabel short-lived spikes.
    df_market.loc[tp_hit, 'label'] = 1
    # If SL is hit, it should be 0. We can enforce this.
    df_market.loc[sl_hit, 'label'] = 0
    # A more robust (but still simplified) logic: if TP is hit AND SL is NOT hit, label 1.
    df_market['label'] = np.where(tp_hit & ~sl_hit, 1, 0)

    logger.info(f"Target variable generation complete. Label distribution:\n{df_market['label'].value_counts(normalize=True)}")
    
    # Merge label back to the original dataframe
    df = df.join(df_market[['label']])
    df['label'] = df['label'].fillna(-1)
    
    # Cleanup barrier columns
    df.drop(columns=['tp_price', 'sl_price'], inplace=True, errors='ignore')
    df_market.drop(columns=['tp_price', 'sl_price'], inplace=True, errors='ignore')

    return df


# --- 4. Model Training ---
def train_model(df: pd.DataFrame, features_list: list, target_col: str = 'label',
                market_to_predict: str = 'SOL_USDT'):
    """Trains a RandomForest model."""
    logger.info("Starting model training...")

    # Filter for the target market for training, and ensure labels are not -1 (not applicable)
    df_train_val = df[(df['market'] == market_to_predict) & (df[target_col] != -1)].copy()
    df_train_val = df_train_val.dropna(subset=features_list + [target_col]) # Drop rows with NaNs in features or target

    if df_train_val.empty or len(df_train_val['label'].unique()) < 2:
        logger.error(f"Not enough data or classes for market {market_to_predict} to train. "
                     f"Shape: {df_train_val.shape}, Unique labels: {df_train_val['label'].unique()}")
        return None, None

    X = df_train_val[features_list]
    y = df_train_val[target_col].astype(int) # Ensure target is int for classifier

    from sklearn.model_selection import TimeSeriesSplit

    # We'll use TimeSeriesSplit on the combined training + original validation data (X_train_val, y_train_val from before)
    # For a final hold-out test set, we still need one.
    # Let's define X_dev (development set = train + val) and X_holdout_test, y_holdout_test
    X_dev, X_holdout_test, y_dev, y_holdout_test = train_test_split(
        X, y, test_size=0.20, shuffle=False # Chronological hold-out test set
    )
    
    logger.info(f"Development set size: {X_dev.shape}, Hold-out Test set size: {X_holdout_test.shape}")

    n_splits = 3 # Reduced n_splits for faster CV during iteration
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # fold_auc_scores = [] # This loop was for default RF, now handled by GridSearchCV
    
    # Scaling should happen inside the cross-validation loop to prevent leakage from val_fold to train_fold
    # However, for simplicity in this step, we'll scale X_dev once.
    # A more rigorous approach fits scaler on each training fold.
    scaler = StandardScaler()
    X_dev_scaled = scaler.fit_transform(X_dev) # Fit scaler on the entire development set
    # X_holdout_test_scaled will be transformed using this scaler later

    # The TimeSeriesSplit loop for default RF params is removed as GridSearchCV will handle CV.
    # logger.info(f"Performing TimeSeriesSplit with {n_splits} splits...")
    # ... (previous loop code removed) ...
    # if fold_auc_scores:
    #     logger.info(f"Average TimeSeriesSplit AUC: {np.mean(fold_auc_scores):.4f} (+/- {np.std(fold_auc_scores):.4f})")
    # else:
    #     logger.warning("No AUC scores recorded from TimeSeriesSplit.")

    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import uniform, randint

    # Hyperparameter Tuning with RandomizedSearchCV using TimeSeriesSplit
    logger.info("Starting hyperparameter tuning with RandomizedSearchCV and TimeSeriesSplit for LGBM...")
    
    # Define parameter distributions for RandomizedSearch
    param_dist = {
        'n_estimators': randint(50, 200),
        'learning_rate': uniform(0.01, 0.1),
        'num_leaves': randint(20, 50),
        'max_depth': [-1, 10, 20, 30],
        'min_child_samples': randint(20, 100),
        'subsample': uniform(0.7, 0.3), # range is [loc, loc + scale]
        'colsample_bytree': uniform(0.7, 0.3)
    }

    # Use RandomizedSearchCV
    # n_iter controls the number of parameter combinations tried.
    random_search = RandomizedSearchCV(
        estimator=lgb.LGBMClassifier(random_state=42, class_weight='balanced', n_jobs=-1),
        param_distributions=param_dist,
        n_iter=10,  # Try 10 random combinations
        cv=tscv,
        scoring='roc_auc',
        verbose=2,
        n_jobs=-1,
        random_state=42
    )
    
    # For faster tuning, we can fit RandomizedSearchCV on a subsample of the data
    logger.info(f"Subsampling development data for faster hyperparameter tuning...")
    n_samples_for_tuning = min(30000, len(X_dev_scaled)) # Use up to 30k samples for tuning
    
    # Use stratified sampling if possible to maintain label distribution, otherwise random
    try:
        from sklearn.model_selection import train_test_split as stratified_sampler
        X_tuning_sample, _, y_tuning_sample, _ = stratified_sampler(
            X_dev_scaled, y_dev, train_size=n_samples_for_tuning, stratify=y_dev, random_state=42
        )
        logger.info(f"Using stratified sample of size {len(X_tuning_sample)} for tuning.")
    except ValueError: # Happens if a class has too few members for stratification
        logger.warning("Could not use stratified sampling for tuning, falling back to random sampling.")
        indices = np.random.choice(np.arange(len(X_dev_scaled)), size=n_samples_for_tuning, replace=False)
        X_tuning_sample = X_dev_scaled[indices]
        y_tuning_sample = y_dev.iloc[indices]
        logger.info(f"Using random sample of size {len(X_tuning_sample)} for tuning.")

    random_search.fit(X_tuning_sample, y_tuning_sample)

    logger.info(f"Best parameters found for LGBM: {random_search.best_params_}")
    logger.info(f"Best AUC score from RandomizedSearchCV (LGBM): {random_search.best_score_:.4f}")

    # Use the best estimator found by RandomizedSearchCV
    final_model = random_search.best_estimator_
    
    # No need to explicitly fit final_model again if GridSearchCV's refit=True (default)
    # final_model.fit(X_dev_scaled, y_dev) # This is done by GridSearchCV

    # Evaluate on the hold-out test set
    X_holdout_test_scaled = scaler.transform(X_holdout_test) # Use the scaler fitted on X_dev
    
    test_preds_proba = final_model.predict_proba(X_holdout_test_scaled)[:, 1]
    test_preds_label = final_model.predict(X_holdout_test_scaled)
    
    logger.info("Hold-out Test Set Performance (with final model):")
    logger.info(f"\n{classification_report(y_holdout_test, test_preds_label)}")
    try:
        logger.info(f"Hold-out Test AUC: {roc_auc_score(y_holdout_test, test_preds_proba):.4f}")
    except ValueError as e:
        logger.warning(f"Could not calculate AUC for hold-out test set (possibly only one class predicted): {e}")

    # Save the final model and scaler (fitted on X_dev)
    model = final_model # Assign to 'model' for consistency with return type if needed later


    # Save model, scaler, and feature list
    model_filename = os.path.join(MODELS_DIR, f"{market_to_predict}_lgbm_model.joblib")
    scaler_filename = os.path.join(MODELS_DIR, f"{market_to_predict}_scaler.joblib")
    features_filename = os.path.join(MODELS_DIR, f"{market_to_predict}_features.joblib")
    
    joblib.dump(model, model_filename)
    joblib.dump(scaler, scaler_filename)
    joblib.dump(features_list, features_filename)
    
    logger.info(f"Model saved to {model_filename}")
    logger.info(f"Scaler saved to {scaler_filename}")
    logger.info(f"Feature list saved to {features_filename}")

    return model, scaler

# --- Main Pipeline Orchestration ---
def run_training_pipeline(markets_to_load: list, target_market: str,
                          features_list: list,
                          tp_pct: float, sl_pct: float, training_horizon_seconds: int):
    """Orchestrates the full training pipeline."""
    logger.info("Starting ML training pipeline...")

    # 1. Load and preprocess data
    df_raw = load_and_preprocess_raw_data(base_path=DATA_BASE_PATH, markets=markets_to_load)
    if df_raw.empty:
        return

    # 2. Feature Engineering
    df_featured = engineer_features(df_raw)
    
    # Dynamically add one-hot encoded columns to the feature list
    final_features_to_use = features_list.copy()
    for col in df_featured.columns:
        if col.startswith('phase_') or col.startswith('volregime_'):
            if col not in final_features_to_use:
                final_features_to_use.append(col)
    
    # Ensure all selected features actually exist in the dataframe
    final_features_to_use = [f for f in final_features_to_use if f in df_featured.columns]
    logger.info(f"Final list of features for training: {final_features_to_use}")
    
    # Drop rows where essential features might be NaN
    df_processed = df_featured.dropna(subset=final_features_to_use)
    if df_processed.empty:
        logger.error("DataFrame is empty after feature engineering and NaN drop. Cannot proceed.")
        return

    # 3. Target Variable Generation
    df_final = generate_target_variable(
        df_processed,
        market_to_predict=target_market,
        horizon_seconds=training_horizon_seconds,
        tp_pct=tp_pct,
        sl_pct=sl_pct
    )
    
    if 'label' not in df_final.columns or df_final[df_final['market'] == target_market]['label'].isin([0, 1]).sum() == 0:
        logger.error(f"No valid labels for target market {target_market}. Exiting.")
        return

    # 4. Model Training
    train_model(df_final, features_list=final_features_to_use, target_col='label', market_to_predict=target_market)

    logger.info("ML training pipeline finished.")


if __name__ == "__main__":
    # This block is for testing the trainer directly.
    # The main execution logic is in main.py
    logger.info("model_trainer.py executed directly. To run the full pipeline, run main.py.")
    
    # Example of how to run the pipeline directly for testing:
    # from main import FINAL_FEATURES_TO_USE, STRATEGY_CONFIG
    # ml_params = STRATEGY_CONFIG['ml'][1]
    # run_training_pipeline(
    #     markets_to_load=['BTC_USDT', 'ETH_USDT', 'SOL_USDT', 'BNB_USDT'],
    #     target_market='SOL_USDT',
    #     features_list=FINAL_FEATURES_TO_USE,
    #     tp_pct=ml_params['tp_pct'],
    #     sl_pct=ml_params['sl_pct'],
    #     training_horizon_seconds=ml_params['horizon_seconds']
    # )

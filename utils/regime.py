
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings

def classify_regimes_with_kmeans(market_data_df: pd.DataFrame, k: int = 7) -> pd.DataFrame:
    """
    Applies K-Means clustering to a DataFrame to determine market regimes.

    Args:
        market_data_df (pd.DataFrame): DataFrame containing the market features.
        k (int): The number of clusters (regimes) to find.

    Returns:
        pd.DataFrame: The input DataFrame with an added 'regime' column.
    """
    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.cluster._kmeans")

    features_for_clustering = [
        'rsi_5', 
        'rsi_14', 
        'trend_strength', 
        'rolling_volatility', 
        'volume_zscore',
        'rolling_zscore'
    ]
    
    # Ensure all required columns are present
    missing_cols = [col for col in features_for_clustering if col not in market_data_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns for clustering: {missing_cols}")

    clustering_df = market_data_df[features_for_clustering].copy()
    clustering_df_cleaned = clustering_df.dropna()
    original_indices = clustering_df_cleaned.index

    if clustering_df_cleaned.empty:
        # Return with an empty regime column if no data to cluster
        market_data_df['regime'] = 'Unknown'
        return market_data_df

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(clustering_df_cleaned)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    clusters = kmeans.fit_predict(scaled_features)

    cluster_labels_series = pd.Series(clusters, index=original_indices, name='cluster_label')

    # Merge cluster labels back
    if 'cluster_label' in market_data_df.columns:
        market_data_df = market_data_df.drop(columns=['cluster_label'])
        
    market_data_df = market_data_df.merge(
        cluster_labels_series, 
        left_index=True, 
        right_index=True, 
        how='left'
    )
    market_data_df['cluster_label'] = market_data_df['cluster_label'].fillna(-1).astype(int)

    # Interpret Clusters to map to regime names
    cluster_means = market_data_df[market_data_df['cluster_label'] != -1].groupby(
        'cluster_label'
    )[features_for_clustering].mean()

    # This mapping is derived from the analysis in the notebook.
    # It might need to be dynamically generated or validated if the data changes significantly.
    cluster_to_regime_map = {
        0: 'Ranging - Uptrend Bias', 
        1: 'Uptrend - Impulse', 
        2: 'Ranging - Downtrend Bias', 
        3: 'Downtrend - Impulse', 
        4: 'Uptrend - Overbought',
        5: 'Ranging - Accumulation',
        6: 'Volatile - Choppy',
        -1: 'Unknown'
    }
    
    market_data_df['regime'] = market_data_df['cluster_label'].map(cluster_to_regime_map).fillna('Unknown')
    
    # Drop the intermediate cluster label column
    market_data_df = market_data_df.drop(columns=['cluster_label'])

    return market_data_df

if __name__ == '__main__':
    # Example usage:
    # This block will only run when the script is executed directly
    try:
        # Assuming the script is run from the root of the project
        data_file = 'data/market_features.parquet'
        df = pd.read_parquet(data_file)
        
        # Calculate necessary indicators if they are missing (e.g., ATR, ADX for later steps)
        # For simplicity, this example assumes features are present. In a real pipeline,
        # you would run the feature engineering script first.
        
        df_with_regimes = classify_regimes_with_kmeans(df)
        
        print("Successfully classified regimes:")
        print(df_with_regimes['regime'].value_counts())
        
        print("\nDataFrame head with new 'regime' column:")
        print(df_with_regimes[['datetime', 'close', 'regime']].head())

    except FileNotFoundError:
        print(f"Error: The data file was not found. Make sure you are running this script from the project root")
    except Exception as e:
        print(f"An error occurred: {e}")

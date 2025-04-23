import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class WirelessDataProcessor:
    def __init__(self, random_seed=42):
        self.random_seed = random_seed
        self.dl_scaler = StandardScaler()
        self.ul_scaler = StandardScaler()
        
    def identify_scenario_rows(self, df):
        """
        Identify rows belonging to DL and UL scenarios based on non-empty values.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            tuple: (dl_mask, ul_mask) boolean masks for each scenario
        """
        # DL scenario indicators
        dl_indicators = [
            'throughput_DL',
            'target_DL',
            'jitter_DL',
            'datarate_client',
            'cell_load_DL',
            'UE_DL'
        ]
        
        # UL scenario indicators
        ul_indicators = [
            'throughput_UL',
            'target_UL',
            'jitter_UL',
            'datarate_server',
            'cell_load_UL',
            'UE_UL'
        ]
        
        # Create masks for each scenario
        dl_mask = df[dl_indicators].notna().any(axis=1)
        ul_mask = df[ul_indicators].notna().any(axis=1)
        
        print(f"DL scenario rows: {dl_mask.sum()}")
        print(f"UL scenario rows: {ul_mask.sum()}")
        
        return dl_mask, ul_mask
    
    def get_valid_features(self, df, scenario):
        """
        Get list of features that have sufficient non-empty values for a scenario.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            scenario (str): 'DL' or 'UL'
            
        Returns:
            list: Valid features for the scenario
        """
        # Get scenario mask
        dl_mask, ul_mask = self.identify_scenario_rows(df)
        mask = dl_mask if scenario == 'DL' else ul_mask
        
        # Calculate missing value percentages for the scenario
        missing_pct = df[mask].isnull().mean()
        
        # Keep features with less than 50% missing values in the scenario
        valid_features = missing_pct[missing_pct < 0.5].index.tolist()
        
        # Always include some basic features if they exist
        essential_features = [
            'serving_cell_rsrp_1',
            'serving_cell_rsrq_1',
            'serving_cell_rssi_1',
            'serving_cell_snr_1'
        ]
        
        valid_features.extend([f for f in essential_features 
                             if f in df.columns and f not in valid_features])
        
        return valid_features
    
    def prepare_scenario_data(self, df, scenario='DL'):
        """
        Prepare data for a specific scenario, handling missing values appropriately.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            scenario (str): 'DL' or 'UL'
            
        Returns:
            tuple: (X, y) features and target
        """
        # Get scenario mask
        dl_mask, ul_mask = self.identify_scenario_rows(df)
        mask = dl_mask if scenario == 'DL' else ul_mask
        
        # Get valid features for this scenario
        valid_features = self.get_valid_features(df, scenario)
        
        # Remove target from features if present
        target_col = f'throughput_{scenario}'
        if target_col in valid_features:
            valid_features.remove(target_col)
        
        print(f"\n{scenario} Scenario Features:")
        print(f"Number of valid features: {len(valid_features)}")
        print("Features:", valid_features)
        
        # Select data for this scenario
        scenario_df = df[mask].copy()
        
        # Handle missing values for each feature
        X = scenario_df[valid_features].copy()
        y = scenario_df[target_col]
        
        # Print missing value statistics
        missing_stats = X.isnull().sum()
        if missing_stats.any():
            print("\nMissing value statistics before handling:")
            print(missing_stats[missing_stats > 0])
        
        # Handle missing values based on feature type
        for column in X.columns:
            missing_count = X[column].isnull().sum()
            if missing_count > 0:
                if X[column].dtype in ['int64', 'float64']:
                    # For numeric features, use median
                    X[column].fillna(X[column].median(), inplace=True)
                else:
                    # For categorical features, use mode
                    X[column].fillna(X[column].mode()[0], inplace=True)
        
        return X, y
    
    def process_data(self, csv_path, test_size=0.2):
        """
        Process the wireless measurement data with scenario-specific handling.
        
        Args:
            csv_path (str): Path to CSV file
            test_size (float): Proportion of test set
            
        Returns:
            dict: Processed data for both scenarios
        """
        print("Reading data...")
        df = pd.read_csv(csv_path, delimiter='\t')
        
        # Remove 'Ignored_' prefix from column names
        df.columns = [col.replace('Ignored_', '') for col in df.columns]
        
        processed_data = {}
        
        for scenario in ['DL', 'UL']:
            print(f"\nProcessing {scenario} scenario...")
            
            # Prepare features and target
            try:
                X, y = self.prepare_scenario_data(df, scenario)
                
                if len(X) == 0:
                    print(f"No valid data for {scenario} scenario")
                    continue
                
                # Scale features
                scaler = self.dl_scaler if scenario == 'DL' else self.ul_scaler
                X_scaled = scaler.fit_transform(X)
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y,
                    test_size=test_size,
                    random_state=self.random_seed
                )
                
                processed_data[scenario] = {
                    'X_train': X_train,
                    'X_test': X_test,
                    'y_train': y_train,
                    'y_test': y_test,
                    'feature_names': X.columns.tolist(),
                    'scaler': scaler
                }
                
                print(f"Final dataset shape: {X.shape}")
                print(f"Training samples: {len(X_train)}")
                print(f"Test samples: {len(X_test)}")
                
            except Exception as e:
                print(f"Error processing {scenario} scenario: {str(e)}")
                continue
            
        return processed_data

def process_wireless_data(csv_path, test_size=0.2):
    """
    Process wireless measurement data from CSV file.
    
    Args:
        csv_path (str): Path to CSV file
        test_size (float): Proportion of test set
        
    Returns:
        tuple: (processor, prepared_data)
    """
    processor = WirelessDataProcessor()
    prepared_data = processor.process_data(csv_path, test_size)
    
    # Print summary statistics
    if prepared_data:
        print("\nProcessing Summary:")
        for scenario in prepared_data:
            data = prepared_data[scenario]
            print(f"\n{scenario} Scenario:")
            print(f"Number of features: {len(data['feature_names'])}")
            print(f"Training samples: {data['X_train'].shape[0]}")
            print(f"Test samples: {data['X_test'].shape[0]}")
    
    return processor, prepared_data
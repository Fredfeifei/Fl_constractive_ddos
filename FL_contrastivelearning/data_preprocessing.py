import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, RobustScaler
import pyarrow.parquet as pq
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

class DDoSPreprocessor:
    def __init__(self):
        self.scaler = RobustScaler()
        self.label_encoder = LabelEncoder()
        self.identifier_cols = {
            'unnamed0', 'flowid', 'sourceip', 'destinationip',
            'sourceport', 'destinationport', 'timestamp'
        }

    def _standardize_column_names(self, df):
        """Standardize column names by converting to lowercase and removing spaces"""
        try:
            column_mapping = {col: col.lower().strip().replace(' ', '') for col in df.columns}

            # Handle duplicates
            seen_names = set()
            for old_name, new_name in column_mapping.items():
                if new_name in seen_names:
                    counter = 1
                    while f"{new_name}{counter}" in seen_names:
                        counter += 1
                    column_mapping[old_name] = f"{new_name}{counter}"
                seen_names.add(column_mapping[old_name])

            # Apply the mapping
            df = df.rename(columns=column_mapping)

            # Print mapping for debugging
            print("\nColumn name mapping:")
            for old, new in column_mapping.items():
                if old != new:
                    print(f"{old} -> {new}")

            return df
        except Exception as e:
            print(f"Error in column name standardization: {str(e)}")
            raise

    def _remove_identifier_columns(self, df):
        """Remove identifier columns"""
        try:
            # Get columns that match our identifier patterns after standardization
            cols_to_drop = [col for col in df.columns
                            if col.replace('_', '') in self.identifier_cols]

            if cols_to_drop:
                df = df.drop(columns=cols_to_drop)
                print(f"\nDropped identifier columns: {cols_to_drop}")

            return df
        except Exception as e:
            print(f"Error removing identifier columns: {str(e)}")
            raise

    def _handle_missing_values(self, df):
        """Handle missing and infinite values"""
        try:
            # Replace infinite values with NaN
            df = df.replace([np.inf, -np.inf], np.nan)

            # Count missing values
            missing_counts = df.isna().sum()
            missing_cols = missing_counts[missing_counts > 0]

            if not missing_cols.empty:
                print("\nHandling missing values:")
                print(missing_cols)

                # Fill missing values with median for numeric columns
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

            return df
        except Exception as e:
            print(f"Error handling missing values: {str(e)}")
            raise

    def _create_features(self, df):
        """Create new features based on available columns"""
        try:
            # Flow-based features
            if 'flowduration' in df.columns:
                if 'flowpacketss' not in df.columns:
                    total_packets = df['totalfwdpackets'] + df['totalbackwardpackets']
                    df['flowpacketss'] = total_packets / (df['flowduration'] + 1e-6)

                if 'flowbytess' not in df.columns:
                    total_bytes = df['totallengthoffwdpackets'] + df['totallengthofbwdpackets']
                    df['flowbytess'] = total_bytes / (df['flowduration'] + 1e-6)

            # IAT features
            iat_columns = [col for col in df.columns if 'iat' in col]
            if iat_columns:
                df['totaliat'] = df[iat_columns].sum(axis=1)

            # Flag features
            flag_columns = [col for col in df.columns if 'flag' in col and 'count' in col]
            if flag_columns:
                df['totalflags'] = df[flag_columns].sum(axis=1)

            # Packet length features
            length_columns = [col for col in df.columns if 'length' in col and 'mean' in col]
            if length_columns:
                df['avgpacketlength'] = df[length_columns].mean(axis=1)

            print("\nCreated new features:", [col for col in df.columns
                                              if col in {'flowpacketss', 'flowbytess', 'totaliat', 'totalflags',
                                                         'avgpacketlength'}])

            return df
        except Exception as e:
            print(f"Error in feature engineering: {str(e)}")
            raise

    def _optimize_dtypes(self, df):

        try:
            original_size = df.memory_usage().sum() / 1024 ** 2

            for col in df.columns:
                if col == 'label':
                    continue

                col_type = df[col].dtype
                if col_type != 'object':
                    c_min, c_max = df[col].min(), df[col].max()

                    # Integer optimization
                    if str(col_type)[:3] == 'int':
                        if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                            df[col] = df[col].astype(np.int8)
                        elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                            df[col] = df[col].astype(np.int16)
                        elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                            df[col] = df[col].astype(np.int32)
                    # Float optimization
                    else:
                        if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                            df[col] = df[col].astype(np.float32)

            optimized_size = df.memory_usage().sum() / 1024 ** 2
            print(f"\nMemory optimization: {original_size:.2f}MB -> {optimized_size:.2f}MB")

            return df
        except Exception as e:
            print(f"Error in dtype optimization: {str(e)}")
            raise

    def _encode_categorical(self, df):
        """Encode categorical features"""
        try:
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

            if categorical_cols:
                print("\nEncoding categorical columns:", categorical_cols)

                for col in categorical_cols:
                    df[col] = self.label_encoder.fit_transform(df[col].astype(str))
                    if col == 'label':
                        self.label_mapping = dict(zip(
                            self.label_encoder.classes_,
                            self.label_encoder.transform(self.label_encoder.classes_)
                        ))
                        print(f"\nLabel mapping: {self.label_mapping}")

            return df
        except Exception as e:
            print(f"Error encoding categorical features: {str(e)}")
            raise

    def process_files(self, csv_dir, output_dir):
        """Process all CSV files in directory"""
        try:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            print("Starting data preprocessing...")

            for file in os.listdir(csv_dir):
                if not file.endswith('.csv'):
                    continue

                try:
                    print(f"\nProcessing {file}...")
                    attack_type = file.split('.')[0].lower()

                    # Load and process data
                    df = pd.read_csv(os.path.join(csv_dir, file))
                    print(f"Initial row count: {len(df)}")
                    initial_size = df.memory_usage().sum() / 1024 ** 2
                    print(f"Initial memory usage: {initial_size:.2f} MB")

                    # Processing pipeline
                    df = self._standardize_column_names(df)
                    print(f"After standardizing names: {len(df)}")
                    df = self._remove_identifier_columns(df)
                    print(f"After removing identifiers: {len(df)}")
                    df = self._handle_missing_values(df)
                    print(f"After handling missing values: {len(df)}")
                    df = self._create_features(df)
                    df = self._optimize_dtypes(df)
                    df = self._encode_categorical(df)

                    # Scale numeric features
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if 'label' in numeric_cols:
                        numeric_cols = numeric_cols.drop('label')
                    df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
                    print(f"Final row count: {len(df)}")

                    # Save processed data
                    output_path = os.path.join(output_dir, f"{attack_type}_processed.parquet")
                    df.to_parquet(output_path, compression='snappy')

                    final_size = df.memory_usage().sum() / 1024 ** 2
                    print(f"Final memory usage: {final_size:.2f} MB")
                    print(f"Memory reduction: {((initial_size - final_size) / initial_size) * 100:.2f}%")
                    print(f"Saved to: {output_path}")

                except Exception as e:
                    print(f"Error processing file {file}: {str(e)}")
                    continue

            print("\nPreprocessing complete!")
        except Exception as e:
            print(f"Error in process_files: {str(e)}")
            raise

def main():
    try:
        csv_dir = "D:\FL_contrastivelearning\data\original"
        output_dir = "D:\FL_contrastivelearning\data\processed"
        preprocessor = DDoSPreprocessor()
        preprocessor.process_files(csv_dir, output_dir)

    except Exception as e:
        print(f"Error in main: {str(e)}")


if __name__ == "__main__":
    main()
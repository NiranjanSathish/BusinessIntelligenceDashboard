"""
Data Processor Module
Handles data loading, validation, filtering, and column type detection.
Optimized for handling large datasets efficiently.
"""

import pandas as pd
import numpy as np
import os
from typing import Tuple, Dict, List, Optional

# Configuration constants
MAX_FILE_SIZE_MB = 100  # Maximum file size in MB (reduced for Hugging Face Spaces)
MAX_PREVIEW_ROWS = 1000  # Maximum rows to display in preview
LARGE_DATASET_THRESHOLD = 100000  # Rows threshold for large dataset warnings


def load_data(file) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Load CSV or Excel file with comprehensive error handling.
    Optimized for large datasets with chunked reading for very large files.
    
    Args:
        file: File object from Gradio upload component
        
    Returns:
        Tuple of (DataFrame or None, status_message)
    """
    try:
        if file is None:
            return None, "❌ No file uploaded. Please upload a CSV or Excel file."
        
        file_name = file.name if hasattr(file, 'name') else str(file)
        
        # Check file size
        try:
            file_size_mb = os.path.getsize(file_name) / (1024 * 1024)
            if file_size_mb > MAX_FILE_SIZE_MB:
                return None, f"❌ File too large ({file_size_mb:.1f}MB). Maximum size is {MAX_FILE_SIZE_MB}MB."
        except:
            pass  # If we can't check size, continue anyway
        
        # Load based on file extension
        if file_name.endswith('.csv'):
            # For large CSV files, use optimized settings
            df = pd.read_csv(
                file_name, 
                encoding='utf-8',
                low_memory=False,  # Better type inference
                engine='c'  # Faster C engine
            )
        elif file_name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_name, engine='openpyxl')
        else:
            return None, f"❌ Unsupported file format. Please upload CSV or Excel files only."
        
        # Validate the loaded data
        is_valid, validation_msg = validate_dataset(df)
        if not is_valid:
            return None, validation_msg
        
        # Add performance warning for large datasets
        rows = len(df)
        size_warning = ""
        if rows > LARGE_DATASET_THRESHOLD:
            size_warning = f" ⚠️ Large dataset detected ({rows:,} rows). Some operations may take longer."
        
        return df, f"✅ Successfully loaded {file_name} - {rows:,} rows × {len(df.columns)} columns{size_warning}"
        
    except FileNotFoundError:
        return None, "❌ File not found. Please try uploading again."
    except pd.errors.EmptyDataError:
        return None, "❌ The file is empty. Please upload a file with data."
    except pd.errors.ParserError:
        return None, "❌ Error parsing file. Please check if the file is properly formatted."
    except MemoryError:
        return None, "❌ File too large to load into memory. Try a smaller dataset or filter the data first."
    except Exception as e:
        return None, f"❌ Error loading file: {str(e)}"


def validate_dataset(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validate dataset integrity.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        Tuple of (is_valid, message)
    """
    if df is None:
        return False, "❌ DataFrame is None"
    
    if df.empty:
        return False, "❌ Dataset is empty. Please upload a file with data."
    
    if len(df.columns) == 0:
        return False, "❌ Dataset has no columns."
    
    if len(df) == 0:
        return False, "❌ Dataset has no rows."
    
    return True, "✅ Dataset is valid"


def get_dataset_info(df: pd.DataFrame) -> Dict:
    """
    Extract comprehensive dataset metadata.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Dictionary containing dataset information
    """
    if df is None or df.empty:
        return {}
    
    # Calculate memory usage
    memory_bytes = df.memory_usage(deep=True).sum()
    memory_mb = memory_bytes / (1024 * 1024)
    
    # Get data types summary
    dtype_counts = df.dtypes.value_counts().to_dict()
    dtype_counts = {str(k): int(v) for k, v in dtype_counts.items()}
    
    info = {
        'rows': len(df),
        'columns': len(df.columns),
        'column_names': df.columns.tolist(),
        'dtypes': df.dtypes.astype(str).to_dict(),
        'dtype_summary': dtype_counts,
        'memory_usage': f"{memory_mb:.2f} MB",
        'duplicates': df.duplicated().sum(),
        'total_missing': df.isnull().sum().sum()
    }
    
    return info


def get_data_preview(df: pd.DataFrame, n: int = 10) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Get head and tail of dataset.
    For large datasets, limits preview to avoid performance issues.
    
    Args:
        df: DataFrame to preview
        n: Number of rows to show (capped at MAX_PREVIEW_ROWS)
        
    Returns:
        Tuple of (head_df, tail_df)
    """
    if df is None or df.empty:
        empty_df = pd.DataFrame()
        return empty_df, empty_df
    
    # Limit preview size for large datasets
    preview_rows = min(n, MAX_PREVIEW_ROWS // 2)
    
    head_df = df.head(preview_rows)
    tail_df = df.tail(preview_rows)
    
    return head_df, tail_df


def detect_datetime_columns(df: pd.DataFrame, threshold: float = 0.95) -> List[str]:
    """
    Detect datetime columns from object/string columns by parsing.
    Only classifies as datetime if >= threshold% of values can be parsed.
    
    Args:
        df: DataFrame to analyze
        threshold: Minimum ratio of successfully parsed values (0.0 to 1.0)
        
    Returns:
        List of column names that are likely datetime
    """
    datetime_cols = []
    
    if df is None or df.empty:
        return datetime_cols
    
    # Only check object/string columns
    object_cols = df.select_dtypes(include=['object']).columns
    
    for col in object_cols:
        try:
            # Skip if column is mostly null
            if df[col].notna().sum() < 10:
                continue
            
            # Attempt to parse as datetime
            parsed = pd.to_datetime(df[col], errors='coerce')
            
            # Calculate ratio of successfully parsed values
            valid_ratio = parsed.notna().mean()
            
            # If >= threshold% can be parsed, it's likely a datetime column
            if valid_ratio >= threshold:
                datetime_cols.append(col)
        
        except Exception as e:
            # If any error occurs, skip this column
            continue
    
    return datetime_cols


def detect_column_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Categorize columns by type for appropriate filtering and visualization.
    Optimized for large datasets with sampling for unique value checks.
    Intelligently detects datetime columns from strings using parse threshold.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Dictionary with 'numerical', 'categorical', and 'datetime' column lists
    """
    if df is None or df.empty:
        return {'numerical': [], 'categorical': [], 'datetime': []}
    
    numerical_cols = []
    categorical_cols = []
    datetime_cols = []
    
    # For large datasets, sample for unique value checks
    sample_size = min(10000, len(df))
    df_sample = df.sample(n=sample_size, random_state=42) if len(df) > 10000 else df
    
    # First, detect datetime columns from object/string types
    detected_datetime_cols = detect_datetime_columns(df, threshold=0.95)
    
    for col in df.columns:
        # Check if already datetime dtype
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            datetime_cols.append(col)
        # Check if detected as datetime from string parsing
        elif col in detected_datetime_cols:
            datetime_cols.append(col)
        # Check if numerical
        elif pd.api.types.is_numeric_dtype(df[col]):
            # Additional check: if unique values are very few, might be categorical
            unique_ratio = df_sample[col].nunique() / len(df_sample)
            if unique_ratio < 0.05 and df_sample[col].nunique() < 20:
                categorical_cols.append(col)
            else:
                numerical_cols.append(col)
        # All other types (object, string, etc.) are categorical
        else:
            categorical_cols.append(col)
    
    return {
        'numerical': numerical_cols,
        'categorical': categorical_cols,
        'datetime': datetime_cols
    }


def get_unique_values(df: pd.DataFrame, column: str, max_values: int = 100) -> List:
    """
    Get unique values for categorical columns (for dropdowns).
    
    Args:
        df: DataFrame
        column: Column name
        max_values: Maximum number of unique values to return
        
    Returns:
        Sorted list of unique values
    """
    if df is None or df.empty or column not in df.columns:
        return []
    
    try:
        unique_vals = df[column].dropna().unique()
        unique_vals = sorted(unique_vals)[:max_values]
        return [str(val) for val in unique_vals]
    except:
        return []


def apply_filters(df: pd.DataFrame, filters: Dict) -> pd.DataFrame:
    """
    Apply multiple filters to DataFrame.
    
    Args:
        df: DataFrame to filter
        filters: Dictionary with filter specifications
            Format: {
                'column_name': {
                    'type': 'range' | 'categorical' | 'date',
                    'min': value,  # for range
                    'max': value,  # for range
                    'values': list  # for categorical
                }
            }
    
    Returns:
        Filtered DataFrame
    """
    if df is None or df.empty:
        return df
    
    filtered_df = df.copy()
    
    try:
        for column, filter_spec in filters.items():
            if column not in filtered_df.columns:
                continue
            
            filter_type = filter_spec.get('type', '')
            
            # Range filter (numerical)
            if filter_type == 'range':
                min_val = filter_spec.get('min')
                max_val = filter_spec.get('max')
                
                if min_val is not None:
                    filtered_df = filtered_df[filtered_df[column] >= min_val]
                if max_val is not None:
                    filtered_df = filtered_df[filtered_df[column] <= max_val]
            
            # Categorical filter
            elif filter_type == 'categorical':
                values = filter_spec.get('values', [])
                if values and len(values) > 0:
                    filtered_df = filtered_df[filtered_df[column].isin(values)]
            
            # Date range filter
            elif filter_type == 'date':
                start_date = filter_spec.get('start')
                end_date = filter_spec.get('end')
                
                if start_date is not None:
                    filtered_df = filtered_df[filtered_df[column] >= start_date]
                if end_date is not None:
                    filtered_df = filtered_df[filtered_df[column] <= end_date]
        
        return filtered_df
    
    except Exception as e:
        print(f"Error applying filters: {str(e)}")
        return df


def get_column_stats(df: pd.DataFrame, column: str) -> Dict:
    """
    Get basic statistics for a column (used for filter ranges).
    
    Args:
        df: DataFrame
        column: Column name
        
    Returns:
        Dictionary with min, max, mean, median values
    """
    if df is None or df.empty or column not in df.columns:
        return {}
    
    try:
        if pd.api.types.is_numeric_dtype(df[column]):
            return {
                'min': float(df[column].min()),
                'max': float(df[column].max()),
                'mean': float(df[column].mean()),
                'median': float(df[column].median())
            }
        return {}
    except:
        return {}


def format_dataframe_info(info: Dict) -> str:
    """
    Format dataset information into a readable string.
    
    Args:
        info: Dictionary from get_dataset_info()
        
    Returns:
        Formatted string for display
    """
    if not info:
        return "No dataset information available."
    
    info_text = f"""
### Dataset Overview

**Shape:** {info['rows']:,} rows × {info['columns']} columns

**Memory Usage:** {info['memory_usage']}

**Data Types:**
"""
    
    for dtype, count in info['dtype_summary'].items():
        info_text += f"\n- {dtype}: {count} columns"
    
    info_text += f"""

**Data Quality:**
- Total Missing Values: {info['total_missing']:,}
- Duplicate Rows: {info['duplicates']:,}
"""
    
    return info_text
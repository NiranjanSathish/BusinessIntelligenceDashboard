"""
Transformations Module
Data transformation operations for creating new columns and modifying data.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, List
import re


def create_calculated_column(
    df: pd.DataFrame,
    new_col_name: str,
    col1: str,
    operation: str,
    col2: str
) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Create a new calculated column using two existing columns.
    
    Args:
        df: DataFrame
        new_col_name: Name for the new column
        col1: First column name
        operation: Operation (+, -, *, /)
        col2: Second column name
        
    Returns:
        Tuple of (transformed_df, message)
    """
    if df is None or df.empty:
        return None, "❌ No data available"
    
    if not new_col_name or new_col_name.strip() == "":
        return None, "❌ Please provide a name for the new column"
    
    if new_col_name in df.columns:
        return None, f"❌ Column '{new_col_name}' already exists. Choose a different name."
    
    if col1 not in df.columns or col2 not in df.columns:
        return None, f"❌ Selected columns not found in dataset"
    
    if not pd.api.types.is_numeric_dtype(df[col1]) or not pd.api.types.is_numeric_dtype(df[col2]):
        return None, "❌ Both columns must be numerical for calculations"
    
    try:
        df_copy = df.copy()
        
        if operation == '+':
            df_copy[new_col_name] = df_copy[col1] + df_copy[col2]
        elif operation == '-':
            df_copy[new_col_name] = df_copy[col1] - df_copy[col2]
        elif operation == '*':
            df_copy[new_col_name] = df_copy[col1] * df_copy[col2]
        elif operation == '/':
            df_copy[new_col_name] = df_copy[col1] / df_copy[col2].replace(0, np.nan)
        else:
            return None, f"❌ Invalid operation: {operation}"
        
        message = f"✅ Created column '{new_col_name}' = {col1} {operation} {col2}"
        return df_copy, message
    
    except Exception as e:
        return None, f"❌ Error creating calculated column: {str(e)}"


def extract_date_components(
    df: pd.DataFrame,
    date_column: str,
    components: List[str]
) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Extract date components (year, month, day, quarter, weekday) from datetime column.
    
    Args:
        df: DataFrame
        date_column: Datetime column name
        components: List of components to extract ['year', 'month', 'day', 'quarter', 'weekday']
        
    Returns:
        Tuple of (transformed_df, message)
    """
    if df is None or df.empty:
        return None, "❌ No data available"
    
    if date_column not in df.columns:
        return None, f"❌ Column '{date_column}' not found"
    
    if not components or len(components) == 0:
        return None, "❌ Please select at least one date component"
    
    try:
        df_copy = df.copy()
        
        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(df_copy[date_column]):
            df_copy[date_column] = pd.to_datetime(df_copy[date_column], errors='coerce')
        
        created_cols = []
        
        if 'Year' in components:
            df_copy[f'{date_column}_Year'] = df_copy[date_column].dt.year
            created_cols.append(f'{date_column}_Year')
        
        if 'Month' in components:
            df_copy[f'{date_column}_Month'] = df_copy[date_column].dt.month
            created_cols.append(f'{date_column}_Month')
        
        if 'Day' in components:
            df_copy[f'{date_column}_Day'] = df_copy[date_column].dt.day
            created_cols.append(f'{date_column}_Day')
        
        if 'Quarter' in components:
            df_copy[f'{date_column}_Quarter'] = df_copy[date_column].dt.quarter
            created_cols.append(f'{date_column}_Quarter')
        
        if 'Weekday' in components:
            df_copy[f'{date_column}_Weekday'] = df_copy[date_column].dt.day_name()
            created_cols.append(f'{date_column}_Weekday')
        
        if 'Week' in components:
            df_copy[f'{date_column}_Week'] = df_copy[date_column].dt.isocalendar().week
            created_cols.append(f'{date_column}_Week')
        
        message = f"✅ Created {len(created_cols)} column(s): {', '.join(created_cols)}"
        return df_copy, message
    
    except Exception as e:
        return None, f"❌ Error extracting date components: {str(e)}"


def bin_numerical_column(
    df: pd.DataFrame,
    column: str,
    num_bins: int,
    bin_labels: str,
    new_col_name: str
) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Bin a numerical column into categories.
    
    Args:
        df: DataFrame
        column: Column to bin
        num_bins: Number of bins (2-10)
        bin_labels: Comma-separated labels or "auto"
        new_col_name: Name for the binned column
        
    Returns:
        Tuple of (transformed_df, message)
    """
    if df is None or df.empty:
        return None, "❌ No data available"
    
    if column not in df.columns:
        return None, f"❌ Column '{column}' not found"
    
    if not pd.api.types.is_numeric_dtype(df[column]):
        return None, f"❌ Column '{column}' must be numerical"
    
    if not new_col_name or new_col_name.strip() == "":
        new_col_name = f"{column}_Binned"
    
    if new_col_name in df.columns:
        return None, f"❌ Column '{new_col_name}' already exists"
    
    try:
        df_copy = df.copy()
        
        # Parse labels
        if bin_labels and bin_labels.lower() != "auto":
            labels = [label.strip() for label in bin_labels.split(',')]
            if len(labels) != num_bins:
                return None, f"❌ Number of labels ({len(labels)}) must match number of bins ({num_bins})"
        else:
            labels = None  # Auto-generate labels
        
        # Create bins
        df_copy[new_col_name] = pd.cut(
            df_copy[column],
            bins=num_bins,
            labels=labels,
            duplicates='drop'
        )
        
        # Get bin ranges for message
        bin_edges = pd.cut(df_copy[column], bins=num_bins, retbins=True, duplicates='drop')[1]
        ranges = [f"[{bin_edges[i]:.2f} - {bin_edges[i+1]:.2f}]" for i in range(len(bin_edges)-1)]
        
        message = f"✅ Created '{new_col_name}' with {num_bins} bins: {', '.join(ranges[:3])}{'...' if len(ranges) > 3 else ''}"
        return df_copy, message
    
    except Exception as e:
        return None, f"❌ Error binning column: {str(e)}"


def text_transformation(
    df: pd.DataFrame,
    column: str,
    operation: str,
    new_col_name: str = None
) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Apply text transformations to a column.
    
    Args:
        df: DataFrame
        column: Column to transform
        operation: 'uppercase', 'lowercase', 'title', 'trim'
        new_col_name: Optional new column name (if None, modifies in place)
        
    Returns:
        Tuple of (transformed_df, message)
    """
    if df is None or df.empty:
        return None, "❌ No data available"
    
    if column not in df.columns:
        return None, f"❌ Column '{column}' not found"
    
    try:
        df_copy = df.copy()
        target_col = new_col_name if new_col_name else column
        
        if new_col_name and new_col_name in df.columns:
            return None, f"❌ Column '{new_col_name}' already exists"
        
        # Convert to string
        df_copy[target_col] = df_copy[column].astype(str)
        
        if operation == 'uppercase':
            df_copy[target_col] = df_copy[target_col].str.upper()
        elif operation == 'lowercase':
            df_copy[target_col] = df_copy[target_col].str.lower()
        elif operation == 'title':
            df_copy[target_col] = df_copy[target_col].str.title()
        elif operation == 'trim':
            df_copy[target_col] = df_copy[target_col].str.strip()
        else:
            return None, f"❌ Invalid operation: {operation}"
        
        action = "Created" if new_col_name else "Modified"
        message = f"✅ {action} column '{target_col}' with {operation} operation"
        return df_copy, message
    
    except Exception as e:
        return None, f"❌ Error in text transformation: {str(e)}"


def fill_missing_values(
    df: pd.DataFrame,
    column: str,
    method: str,
    fill_value: str = None
) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Fill missing values in a column.
    
    Args:
        df: DataFrame
        column: Column name
        method: 'mean', 'median', 'mode', 'forward_fill', 'backward_fill', 'constant'
        fill_value: Value to use if method is 'constant'
        
    Returns:
        Tuple of (transformed_df, message)
    """
    if df is None or df.empty:
        return None, "❌ No data available"
    
    if column not in df.columns:
        return None, f"❌ Column '{column}' not found"
    
    missing_before = df[column].isnull().sum()
    
    if missing_before == 0:
        return None, f"ℹ️ Column '{column}' has no missing values"
    
    try:
        df_copy = df.copy()
        
        if method == 'mean' and pd.api.types.is_numeric_dtype(df_copy[column]):
            df_copy[column].fillna(df_copy[column].mean(), inplace=True)
        elif method == 'median' and pd.api.types.is_numeric_dtype(df_copy[column]):
            df_copy[column].fillna(df_copy[column].median(), inplace=True)
        elif method == 'mode':
            mode_value = df_copy[column].mode()[0] if len(df_copy[column].mode()) > 0 else None
            df_copy[column].fillna(mode_value, inplace=True)
        elif method == 'forward_fill':
            df_copy[column].fillna(method='ffill', inplace=True)
        elif method == 'backward_fill':
            df_copy[column].fillna(method='bfill', inplace=True)
        elif method == 'constant':
            if fill_value is None or fill_value.strip() == "":
                return None, "❌ Please provide a fill value"
            df_copy[column].fillna(fill_value, inplace=True)
        else:
            return None, f"❌ Invalid method: {method}"
        
        missing_after = df_copy[column].isnull().sum()
        filled = missing_before - missing_after
        
        message = f"✅ Filled {filled} missing values in '{column}' using {method}"
        return df_copy, message
    
    except Exception as e:
        return None, f"❌ Error filling missing values: {str(e)}"


def get_transformation_preview(
    original_df: pd.DataFrame,
    transformed_df: pd.DataFrame,
    affected_columns: List[str]
) -> pd.DataFrame:
    """
    Create a preview showing before/after for transformed columns.
    
    Args:
        original_df: Original DataFrame
        transformed_df: Transformed DataFrame
        affected_columns: List of affected column names
        
    Returns:
        Preview DataFrame
    """
    if original_df is None or transformed_df is None:
        return pd.DataFrame()
    
    try:
        # Show first 10 rows with relevant columns
        preview_data = {}
        
        for col in affected_columns:
            if col in original_df.columns and col in transformed_df.columns:
                preview_data[f'{col}_BEFORE'] = original_df[col].head(10)
                preview_data[f'{col}_AFTER'] = transformed_df[col].head(10)
            elif col in transformed_df.columns:
                # New column
                preview_data[f'{col}_NEW'] = transformed_df[col].head(10)
        
        return pd.DataFrame(preview_data)
    
    except Exception as e:
        print(f"Error creating preview: {str(e)}")
        return pd.DataFrame()


def get_transformation_summary(original_df: pd.DataFrame, transformed_df: pd.DataFrame) -> str:
    """
    Generate summary of transformations applied.
    
    Args:
        original_df: Original DataFrame
        transformed_df: Transformed DataFrame
        
    Returns:
        Formatted summary string
    """
    if original_df is None or transformed_df is None:
        return "No transformation applied"
    
    original_cols = set(original_df.columns)
    transformed_cols = set(transformed_df.columns)
    
    new_cols = transformed_cols - original_cols
    
    summary = f"""
### 🔧 Transformation Summary

**Original Columns**: {len(original_cols)}
**Current Columns**: {len(transformed_cols)}
**New Columns Added**: {len(new_cols)}

**New Columns**: {', '.join(new_cols) if new_cols else 'None'}
"""
    
    return summary
"""
Filters Module
Dynamic filter generation and application for interactive data exploration.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from data_processor import detect_column_types, get_column_stats


def generate_filter_config(df: pd.DataFrame) -> Dict:
    """
    Generate filter configuration based on DataFrame column types.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Dictionary with filter configurations for each column and excluded columns info
    """
    if df is None or df.empty:
        return {}
    
    column_types = detect_column_types(df)
    filter_config = {
        'filters': {},
        'excluded_categorical': []
    }
    
    # Numerical columns - range filters
    for col in column_types['numerical']:
        stats = get_column_stats(df, col)
        if stats:
            filter_config['filters'][col] = {
                'type': 'range',
                'min': stats['min'],
                'max': stats['max'],
                'current_min': stats['min'],
                'current_max': stats['max']
            }
    
    # Categorical columns - multi-select filters
    for col in column_types['categorical']:
        unique_vals = df[col].dropna().unique().tolist()
        # Limit to 100 unique values for performance
        if len(unique_vals) <= 100:
            filter_config['filters'][col] = {
                'type': 'categorical',
                'values': sorted([str(v) for v in unique_vals]),
                'selected': []
            }
        else:
            # Track excluded columns with their unique value counts
            filter_config['excluded_categorical'].append({
                'column': col,
                'unique_count': len(unique_vals)
            })
    
    # Datetime columns - date range filters
    for col in column_types['datetime']:
        try:
            min_date = df[col].min()
            max_date = df[col].max()
            filter_config['filters'][col] = {
                'type': 'date',
                'min': min_date,
                'max': max_date,
                'current_min': min_date,
                'current_max': max_date
            }
        except:
            continue
    
    return filter_config


def apply_numerical_filter(df: pd.DataFrame, column: str, min_val: float, max_val: float) -> pd.DataFrame:
    """
    Apply range filter to numerical column.
    
    Args:
        df: DataFrame to filter
        column: Column name
        min_val: Minimum value
        max_val: Maximum value
        
    Returns:
        Filtered DataFrame
    """
    if df is None or df.empty or column not in df.columns:
        return df
    
    try:
        filtered_df = df[(df[column] >= min_val) & (df[column] <= max_val)]
        return filtered_df
    except Exception as e:
        print(f"Error applying numerical filter to {column}: {str(e)}")
        return df


def apply_categorical_filter(df: pd.DataFrame, column: str, selected_values: List[str]) -> pd.DataFrame:
    """
    Apply categorical filter with multiple value selection.
    
    Args:
        df: DataFrame to filter
        column: Column name
        selected_values: List of selected values
        
    Returns:
        Filtered DataFrame
    """
    if df is None or df.empty or column not in df.columns:
        return df
    
    if not selected_values or len(selected_values) == 0:
        return df
    
    try:
        # Convert DataFrame column to string for comparison
        filtered_df = df[df[column].astype(str).isin(selected_values)]
        return filtered_df
    except Exception as e:
        print(f"Error applying categorical filter to {column}: {str(e)}")
        return df


def apply_date_filter(df: pd.DataFrame, column: str, start_date, end_date) -> pd.DataFrame:
    """
    Apply date range filter.
    Automatically converts column to datetime if not already.
    
    Args:
        df: DataFrame to filter
        column: Column name
        start_date: Start date
        end_date: End date
        
    Returns:
        Filtered DataFrame
    """
    if df is None or df.empty or column not in df.columns:
        return df
    
    try:
        df_copy = df.copy()
        
        # Ensure column is datetime (convert if needed)
        if not pd.api.types.is_datetime64_any_dtype(df_copy[column]):
            df_copy[column] = pd.to_datetime(df_copy[column], errors='coerce')
        
        filtered_df = df_copy[(df_copy[column] >= start_date) & (df_copy[column] <= end_date)]
        return filtered_df
    except Exception as e:
        print(f"Error applying date filter to {column}: {str(e)}")
        return df


def get_filter_summary(original_df: pd.DataFrame, filtered_df: pd.DataFrame) -> str:
    """
    Generate a summary of filtering results.
    
    Args:
        original_df: Original DataFrame
        filtered_df: Filtered DataFrame
        
    Returns:
        Formatted summary string
    """
    if original_df is None or filtered_df is None:
        return "No data available"
    
    original_rows = len(original_df)
    filtered_rows = len(filtered_df)
    rows_removed = original_rows - filtered_rows
    
    summary = f"""
### 🔍 Filter Results

**Original Data:** {original_rows:,} rows  
**Filtered Data:** {filtered_rows:,} rows  
**Rows Removed:** {rows_removed:,} rows
"""
    
    if filtered_rows == 0:
        summary += "\n⚠️ **Warning:** All rows have been filtered out. Try adjusting your filter criteria."
    
    return summary


def reset_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reset all filters and return original DataFrame.
    
    Args:
        df: Original DataFrame
        
    Returns:
        Original DataFrame (unfiltered)
    """
    return df


def get_active_filters_summary(filter_states: Dict) -> str:
    """
    Generate a summary of currently active filters.
    
    Args:
        filter_states: Dictionary of current filter states
        
    Returns:
        Formatted string of active filters
    """
    if not filter_states:
        return "No filters applied"
    
    active_filters = []
    
    for col, state in filter_states.items():
        filter_type = state.get('type', '')
        
        if filter_type == 'range':
            min_val = state.get('current_min')
            max_val = state.get('current_max')
            original_min = state.get('min')
            original_max = state.get('max')
            
            # Check if filter is actually applied (different from original range)
            if min_val != original_min or max_val != original_max:
                active_filters.append(f"**{col}:** {min_val:.2f} to {max_val:.2f}")
        
        elif filter_type == 'categorical':
            selected = state.get('selected', [])
            if selected:
                active_filters.append(f"**{col}:** {len(selected)} value(s) selected")
        
        elif filter_type == 'date':
            start = state.get('current_min')
            end = state.get('current_max')
            if start and end:
                active_filters.append(f"**{col}:** {start} to {end}")
    
    if not active_filters:
        return "No filters applied"
    
    return "**Active Filters:**\n" + "\n".join(f"- {f}" for f in active_filters)
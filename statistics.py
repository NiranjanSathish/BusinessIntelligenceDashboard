"""
Statistics Module
Generate summary statistics, data profiling, and correlation analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import plotly.graph_objects as go
import plotly.express as px
from data_processor import detect_column_types


def profile_numerical_columns(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Generate comprehensive statistical summary for numerical columns.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        DataFrame with statistical measures for each numerical column
    """
    if df is None or df.empty:
        return None
    
    # Select only numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numerical_cols:
        return None
    
    stats_list = []
    
    for col in numerical_cols:
        try:
            col_data = df[col].dropna()
            
            if len(col_data) == 0:
                continue
            
            stats = {
                'Column': col,
                'Count': len(col_data),
                'Mean': col_data.mean(),
                'Median': col_data.median(),
                'Std Dev': col_data.std(),
                'Min': col_data.min(),
                'Q1 (25%)': col_data.quantile(0.25),
                'Q2 (50%)': col_data.quantile(0.50),
                'Q3 (75%)': col_data.quantile(0.75),
                'Max': col_data.max(),
                'Range': col_data.max() - col_data.min(),
                'IQR': col_data.quantile(0.75) - col_data.quantile(0.25)
            }
            
            stats_list.append(stats)
        except Exception as e:
            print(f"Error processing column {col}: {str(e)}")
            continue
    
    if not stats_list:
        return None
    
    stats_df = pd.DataFrame(stats_list)
    
    # Round numerical values for better display
    numeric_cols = stats_df.select_dtypes(include=[np.number]).columns
    stats_df[numeric_cols] = stats_df[numeric_cols].round(3)
    
    return stats_df


def profile_categorical_columns(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Generate summary for categorical columns.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        DataFrame with categorical column summaries
    """
    if df is None or df.empty:
        return None
    
    # Select categorical columns (object and category types)
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if not categorical_cols:
        return None
    
    stats_list = []
    
    for col in categorical_cols:
        try:
            col_data = df[col].dropna()
            
            if len(col_data) == 0:
                continue
            
            value_counts = col_data.value_counts()
            mode_value = value_counts.index[0] if len(value_counts) > 0 else None
            mode_freq = value_counts.iloc[0] if len(value_counts) > 0 else 0
            
            # Get top 5 values as a string
            top_5 = value_counts.head(5)
            top_5_str = ", ".join([f"{val} ({count})" for val, count in top_5.items()])
            
            stats = {
                'Column': col,
                'Count': len(col_data),
                'Unique Values': col_data.nunique(),
                'Mode': str(mode_value),
                'Mode Frequency': mode_freq,
                'Mode %': (mode_freq / len(col_data) * 100) if len(col_data) > 0 else 0,
                'Top 5 Values': top_5_str
            }
            
            stats_list.append(stats)
        except Exception as e:
            print(f"Error processing column {col}: {str(e)}")
            continue
    
    if not stats_list:
        return None
    
    stats_df = pd.DataFrame(stats_list)
    
    # Round percentage
    if 'Mode %' in stats_df.columns:
        stats_df['Mode %'] = stats_df['Mode %'].round(2)
    
    return stats_df


def analyze_missing_values(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Generate comprehensive missing value report.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        DataFrame with missing value statistics for each column
    """
    if df is None or df.empty:
        return None
    
    missing_stats = []
    
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        total_count = len(df)
        missing_pct = (missing_count / total_count * 100) if total_count > 0 else 0
        
        missing_stats.append({
            'Column': col,
            'Data Type': str(df[col].dtype),
            'Missing Count': missing_count,
            'Missing %': missing_pct,
            'Non-Missing Count': total_count - missing_count,
            'Total Count': total_count
        })
    
    missing_df = pd.DataFrame(missing_stats)
    
    # Sort by missing percentage (descending)
    missing_df = missing_df.sort_values('Missing %', ascending=False)
    
    # Round percentage
    missing_df['Missing %'] = missing_df['Missing %'].round(2)
    
    return missing_df


def calculate_correlation_matrix(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Calculate correlation matrix for numerical columns.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Correlation DataFrame
    """
    if df is None or df.empty:
        return None
    
    # Select only numerical columns
    numerical_df = df.select_dtypes(include=[np.number])
    
    if numerical_df.shape[1] < 2:
        return None
    
    try:
        # Calculate correlation matrix
        corr_matrix = numerical_df.corr()
        
        # Round for better display
        corr_matrix = corr_matrix.round(3)
        
        return corr_matrix
    except Exception as e:
        print(f"Error calculating correlation: {str(e)}")
        return None


def create_correlation_heatmap(df: pd.DataFrame) -> Optional[go.Figure]:
    """
    Create an interactive correlation heatmap using Plotly.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Plotly Figure object
    """
    corr_matrix = calculate_correlation_matrix(df)
    
    if corr_matrix is None:
        return None
    
    try:
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu_r',
            zmid=0,
            zmin=-1,
            zmax=1,
            text=corr_matrix.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 10},
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            title='Correlation Heatmap',
            xaxis_title='Features',
            yaxis_title='Features',
            width=800,
            height=800,
            xaxis={'side': 'bottom'},
            yaxis={'side': 'left'}
        )
        
        return fig
    except Exception as e:
        print(f"Error creating heatmap: {str(e)}")
        return None


def create_missing_values_chart(df: pd.DataFrame) -> Optional[go.Figure]:
    """
    Create a bar chart showing missing value percentages.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Plotly Figure object
    """
    missing_df = analyze_missing_values(df)
    
    if missing_df is None:
        return None
    
    try:
        # Filter to only show columns with missing values
        missing_df_filtered = missing_df[missing_df['Missing %'] > 0].copy()
        
        if missing_df_filtered.empty:
            # No missing values - create a message figure
            fig = go.Figure()
            fig.add_annotation(
                text="✅ No missing values detected in the dataset!",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=20)
            )
            fig.update_layout(
                title='Missing Values Analysis',
                height=400
            )
            return fig
        
        # Sort by missing percentage
        missing_df_filtered = missing_df_filtered.sort_values('Missing %', ascending=True)
        
        # Create bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=missing_df_filtered['Missing %'],
                y=missing_df_filtered['Column'],
                orientation='h',
                marker=dict(
                    color=missing_df_filtered['Missing %'],
                    colorscale='Reds',
                    showscale=True,
                    colorbar=dict(title="Missing %")
                ),
                text=missing_df_filtered['Missing %'].round(2),
                texttemplate='%{text}%',
                textposition='outside'
            )
        ])
        
        fig.update_layout(
            title='Missing Values by Column',
            xaxis_title='Missing Percentage (%)',
            yaxis_title='Column Name',
            height=max(400, len(missing_df_filtered) * 30),
            showlegend=False
        )
        
        return fig
    except Exception as e:
        print(f"Error creating missing values chart: {str(e)}")
        return None


def get_data_summary(df: pd.DataFrame) -> Dict:
    """
    Generate comprehensive data summary combining all profiling.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Dictionary with all profiling results
    """
    if df is None or df.empty:
        return {}
    
    summary = {
        'numerical': profile_numerical_columns(df),
        'categorical': profile_categorical_columns(df),
        'missing': analyze_missing_values(df),
        'correlation': calculate_correlation_matrix(df),
        'correlation_heatmap': create_correlation_heatmap(df),
        'missing_chart': create_missing_values_chart(df)
    }
    
    return summary


def format_statistics_summary(df: pd.DataFrame) -> str:
    """
    Create a formatted text summary of key statistics.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Formatted markdown string
    """
    if df is None or df.empty:
        return "No data available for analysis."
    
    # Use consistent column type detection
    column_types = detect_column_types(df)
    numerical_cols = column_types['numerical']
    categorical_cols = column_types['categorical']
    datetime_cols = column_types['datetime']
    
    total_missing = df.isnull().sum().sum()
    total_cells = df.shape[0] * df.shape[1]
    missing_pct = (total_missing / total_cells * 100) if total_cells > 0 else 0
    
    summary = f"""
### 📊 Quick Statistics Summary

**Column Types:**
- Numerical Columns: {len(numerical_cols)}
- Categorical Columns: {len(categorical_cols)}
- Datetime Columns: {len(datetime_cols)}
- Total Columns: {df.shape[1]}

**Data Quality:**
- Total Cells: {total_cells:,}
- Missing Values: {total_missing:,} ({missing_pct:.2f}%)
- Complete Rows: {df.dropna().shape[0]:,} ({(df.dropna().shape[0]/df.shape[0]*100):.2f}%)

**Dataset Shape:**
- Rows: {df.shape[0]:,}
- Columns: {df.shape[1]}
- Memory Usage: {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB
"""
    
    return summary
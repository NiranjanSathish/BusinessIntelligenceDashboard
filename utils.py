"""
Utility Functions Module
Helper functions for exports, formatting, and common operations.
"""

import pandas as pd
import tempfile
import os
from typing import List
import plotly.graph_objects as go
# import kaleido  # Removed top-level import to save memory


def export_dataframe_to_csv(df: pd.DataFrame, filename: str = "filtered_data.csv") -> str:
    """
    Export DataFrame to CSV in temp directory.
    
    Args:
        df: DataFrame to export
        filename: Name for the CSV file
        
    Returns:
        File path for Gradio download
    """
    try:
        if df is None or df.empty:
            return None
        
        # Create temp file
        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, filename)
        
        # Export to CSV
        df.to_csv(file_path, index=False)
        
        return file_path
    except Exception as e:
        print(f"Error exporting CSV: {str(e)}")
        return None


def export_figure_to_png(fig, filename: str = "chart.png") -> str:
    """
    Export plotly figure to PNG.
    
    Args:
        fig: Plotly figure object
        filename: Name for the PNG file
        
    Returns:
        File path for Gradio download
    """
    try:
        if fig is None:
            return None
        
        # Create temp file
        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, filename)
        
        # Export figure (may fail in containerized environments without Kaleido)
        try:
            import kaleido  # Lazy import to save memory until needed
            fig.write_image(file_path, width=1200, height=800, scale=2)
        except ImportError:
            print("Kaleido not installed.")
            raise Exception("Server-side export failed: Kaleido library not found. Please use the camera icon in the chart toolbar.")
        except Exception as kaleido_error:
            print(f"Kaleido export failed: {str(kaleido_error)}")
            # Raise exception with helpful message for the UI
            raise Exception("Server-side export failed. Please use the camera icon in the chart toolbar to download the image directly.")
        
        return file_path
    except Exception as e:
        print(f"Error exporting PNG: {str(e)}")
        # If it's the exception we just raised, re-raise it
        if "Server-side export failed" in str(e):
            raise e
        return None


def export_text_to_file(text: str, filename: str = "insights_report.txt") -> str:
    """
    Export text content to file.
    
    Args:
        text: Text content to export
        filename: Name for the text file
        
    Returns:
        File path for Gradio download
    """
    try:
        if not text:
            return None
        
        # Create temp file
        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, filename)
        
        # Write text to file (strip markdown for plain text)
        clean_text = text.replace('###', '').replace('**', '').replace('*', '')
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(clean_text)
        
        return file_path
    except Exception as e:
        print(f"Error exporting text: {str(e)}")
        return None


def format_number(value: float, decimals: int = 2) -> str:
    """
    Format numbers for display with K, M, B suffixes.
    
    Args:
        value: Number to format
        decimals: Number of decimal places
        
    Returns:
        Formatted string
    """
    try:
        if pd.isna(value):
            return "N/A"
        
        abs_value = abs(value)
        
        if abs_value >= 1_000_000_000:
            return f"{value/1_000_000_000:.{decimals}f}B"
        elif abs_value >= 1_000_000:
            return f"{value/1_000_000:.{decimals}f}M"
        elif abs_value >= 1_000:
            return f"{value/1_000:.{decimals}f}K"
        else:
            return f"{value:.{decimals}f}"
    except:
        return str(value)


def safe_convert_to_datetime(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Safely convert column to datetime with error handling.
    
    Args:
        df: DataFrame
        column: Column name to convert
        
    Returns:
        DataFrame with converted column
    """
    try:
        if df is None or column not in df.columns:
            return df
        
        df_copy = df.copy()
        df_copy[column] = pd.to_datetime(df_copy[column], errors='coerce')
        return df_copy
    except Exception as e:
        print(f"Error converting to datetime: {str(e)}")
        return df


def get_aggregation_methods() -> List[str]:
    """
    Return list of available aggregation methods.
    
    Returns:
        List of aggregation method names
    """
    return ['sum', 'mean', 'count', 'median', 'min', 'max']


def create_sample_sales_data() -> pd.DataFrame:
    """
    Generate sample sales dataset for demonstration.
    
    Returns:
        Sample DataFrame
    """
    import numpy as np
    from datetime import datetime, timedelta
    
    np.random.seed(42)
    
    # Generate dates
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=x) for x in range(365)]
    
    n_records = 1000
    
    data = {
        'Date': np.random.choice(dates, n_records),
        'Product': np.random.choice(['Laptop', 'Phone', 'Tablet', 'Monitor', 'Keyboard', 'Mouse'], n_records),
        'Category': np.random.choice(['Electronics', 'Accessories'], n_records, p=[0.7, 0.3]),
        'Region': np.random.choice(['North', 'South', 'East', 'West'], n_records),
        'Sales': np.random.uniform(100, 2000, n_records).round(2),
        'Quantity': np.random.randint(1, 20, n_records),
        'Customer_ID': np.random.randint(1000, 9999, n_records)
    }
    
    df = pd.DataFrame(data)
    df = df.sort_values('Date').reset_index(drop=True)
    
    return df


def create_sample_hr_data() -> pd.DataFrame:
    """
    Generate sample HR analytics dataset for demonstration.
    
    Returns:
        Sample DataFrame
    """
    import numpy as np
    
    np.random.seed(42)
    
    n_records = 500
    
    data = {
        'Employee_ID': range(1001, 1001 + n_records),
        'Department': np.random.choice(['Sales', 'Engineering', 'Marketing', 'HR', 'Finance'], n_records),
        'Salary': np.random.uniform(40000, 150000, n_records).round(2),
        'Years_Experience': np.random.randint(0, 25, n_records),
        'Age': np.random.randint(22, 65, n_records),
        'Performance_Rating': np.random.choice([1, 2, 3, 4, 5], n_records, p=[0.05, 0.15, 0.4, 0.3, 0.1]),
        'Attrition': np.random.choice(['Yes', 'No'], n_records, p=[0.15, 0.85]),
        'Job_Satisfaction': np.random.randint(1, 6, n_records)
    }
    
    df = pd.DataFrame(data)
    
    return df


def save_sample_datasets(data_dir: str = "data") -> None:
    """
    Create and save sample datasets to the data directory.
    
    Args:
        data_dir: Directory to save the sample files
    """
    try:
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Create and save sales data
        sales_df = create_sample_sales_data()
        sales_path = os.path.join(data_dir, "sales_data.csv")
        sales_df.to_csv(sales_path, index=False)
        print(f"✅ Created {sales_path}")
        
        # Create and save HR data
        hr_df = create_sample_hr_data()
        hr_path = os.path.join(data_dir, "hr_analytics.csv")
        hr_df.to_csv(hr_path, index=False)
        print(f"✅ Created {hr_path}")
        
    except Exception as e:
        print(f"❌ Error creating sample datasets: {str(e)}")


def validate_chart_columns(df: pd.DataFrame, columns: List[str]) -> tuple:
    """
    Validate that columns exist in DataFrame.
    
    Args:
        df: DataFrame to check
        columns: List of column names
        
    Returns:
        Tuple of (is_valid, message)
    """
    if df is None or df.empty:
        return False, "DataFrame is empty"
    
    missing_cols = [col for col in columns if col not in df.columns]
    
    if missing_cols:
        return False, f"Columns not found: {', '.join(missing_cols)}"
    
    return True, "All columns valid"


# Run this once to create sample datasets
if __name__ == "__main__":
    save_sample_datasets()
    print("\n✅ Sample datasets created successfully!")
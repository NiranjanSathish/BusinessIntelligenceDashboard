"""
Dashboard Module
Multi-chart dashboard view for comprehensive data analysis.
"""

import pandas as pd
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Optional
import os
import tempfile
import zipfile
from data_processor import detect_column_types
from visualizations import (
    create_time_series_plot,
    create_distribution_plot,
    create_category_chart,
    create_scatter_plot,
    create_correlation_heatmap,
    create_empty_chart
)
from utils import export_figure_to_png, get_aggregation_methods


def get_smart_chart_config(df: pd.DataFrame, position: int) -> Tuple[str, Dict]:
    """
    Intelligently suggest chart type and configuration based on data.
    
    Args:
        df: DataFrame to analyze
        position: Chart position (1-4)
        
    Returns:
        Tuple of (chart_type, config_dict)
    """
    if df is None or df.empty:
        return "Correlation Heatmap", {}
    
    col_types = detect_column_types(df)
    numerical = col_types.get('numerical', [])
    categorical = col_types.get('categorical', [])
    datetime = col_types.get('datetime', [])
    
    # Smart defaults for each position
    if position == 1:
        # Position 1: Time series if available, else bar chart
        if datetime and numerical:
            return "Time Series", {
                'date_col': datetime[0],
                'value_col': numerical[0],
                'agg': 'sum'
            }
        elif categorical and numerical:
            return "Bar Chart", {
                'cat_col': categorical[0],
                'value_col': numerical[0],
                'agg': 'sum'
            }
        else:
            return "Correlation Heatmap", {}
    
    elif position == 2:
        # Position 2: Category breakdown
        if categorical and numerical:
            return "Pie Chart", {
                'cat_col': categorical[0],
                'value_col': numerical[0] if len(numerical) > 0 else None,
                'agg': 'count'
            }
        elif numerical:
            return "Histogram", {'col': numerical[0]}
        else:
            return "Correlation Heatmap", {}
    
    elif position == 3:
        # Position 3: Distribution
        if numerical:
            col_idx = min(1, len(numerical) - 1)
            return "Box Plot", {'col': numerical[col_idx]}
        else:
            return "Correlation Heatmap", {}
    
    elif position == 4:
        # Position 4: Relationships
        if len(numerical) >= 2:
            return "Scatter Plot", {
                'x_col': numerical[0],
                'y_col': numerical[1],
                'color_col': categorical[0] if categorical else None
            }
        elif categorical and numerical:
            cat_idx = min(1, len(categorical) - 1)
            return "Bar Chart", {
                'cat_col': categorical[cat_idx],
                'value_col': numerical[0],
                'agg': 'mean'
            }
        else:
            return "Correlation Heatmap", {}
    
    return "Correlation Heatmap", {}


def generate_dashboard_chart(df: pd.DataFrame, chart_type: str, config: Dict) -> go.Figure:
    """
    Generate a single chart for the dashboard.
    
    Args:
        df: DataFrame to visualize
        chart_type: Type of chart
        config: Configuration dictionary with column selections
        
    Returns:
        Plotly Figure
    """
    if df is None or df.empty:
        return create_empty_chart("No data available")
    
    try:
        if chart_type == "Time Series":
            return create_time_series_plot(
                df,
                config.get('date_col', ''),
                config.get('value_col', ''),
                config.get('agg', 'sum')
            )
        
        elif chart_type == "Histogram":
            return create_distribution_plot(
                df,
                config.get('col', ''),
                'histogram'
            )
        
        elif chart_type == "Box Plot":
            return create_distribution_plot(
                df,
                config.get('col', ''),
                'box'
            )
        
        elif chart_type == "Bar Chart":
            return create_category_chart(
                df,
                config.get('cat_col', ''),
                'bar',
                config.get('value_col'),
                config.get('agg', 'count'),
                top_n=10
            )
        
        elif chart_type == "Pie Chart":
            return create_category_chart(
                df,
                config.get('cat_col', ''),
                'pie',
                config.get('value_col'),
                config.get('agg', 'count'),
                top_n=10
            )
        
        elif chart_type == "Scatter Plot":
            return create_scatter_plot(
                df,
                config.get('x_col', ''),
                config.get('y_col', ''),
                config.get('color_col')
            )
        
        elif chart_type == "Correlation Heatmap":
            return create_correlation_heatmap(df)
        
        else:
            return create_empty_chart(f"Unknown chart type: {chart_type}")
    
    except Exception as e:
        return create_empty_chart(f"Error generating chart: {str(e)}")

def generate_dashboard_chart_manual(
    df: pd.DataFrame,
    chart_type: str,
    col1: str,
    col2: str,
    col3: str
) -> go.Figure:
    """
    Generate a single chart for the dashboard with manual column selection.
    
    Args:
        df: DataFrame to visualize
        chart_type: Type of chart
        col1: First column selection
        col2: Second column selection
        col3: Third column selection (agg method or color)
        
    Returns:
        Plotly Figure
    """
    if df is None or df.empty:
        return create_empty_chart("No data available")
    
    try:
        if chart_type == "Time Series":
            if not col1 or col1 == "None" or not col2 or col2 == "None":
                return create_empty_chart("Select date and value columns")
            return create_time_series_plot(df, col1, col2, col3 if col3 else 'sum')
        
        elif chart_type == "Histogram":
            if not col1 or col1 == "None":
                return create_empty_chart("Select a column")
            return create_distribution_plot(df, col1, 'histogram')
        
        elif chart_type == "Box Plot":
            if not col1 or col1 == "None":
                return create_empty_chart("Select a column")
            return create_distribution_plot(df, col1, 'box')
        
        elif chart_type == "Bar Chart":
            if not col1 or col1 == "None":
                return create_empty_chart("Select a category column")
            val_col = col2 if col2 and col2 != "None" else None
            agg = col3 if col3 else 'count'
            return create_category_chart(df, col1, 'bar', val_col, agg, top_n=10)
        
        elif chart_type == "Pie Chart":
            if not col1 or col1 == "None":
                return create_empty_chart("Select a category column")
            val_col = col2 if col2 and col2 != "None" else None
            agg = col3 if col3 else 'count'
            return create_category_chart(df, col1, 'pie', val_col, agg, top_n=10)
        
        elif chart_type == "Scatter Plot":
            if not col1 or col1 == "None" or not col2 or col2 == "None":
                return create_empty_chart("Select X and Y columns")
            color = col3 if col3 and col3 != "None" else None
            return create_scatter_plot(df, col1, col2, color)
        
        elif chart_type == "Correlation Heatmap":
            return create_correlation_heatmap(df)
        
        else:
            return create_empty_chart(f"Unknown chart type: {chart_type}")
    
    except Exception as e:
        return create_empty_chart(f"Error: {str(e)}")

def generate_all_dashboard_charts(
    df: pd.DataFrame,
    chart1_type: str,
    chart2_type: str,
    chart3_type: str,
    chart4_type: str
) -> Tuple[go.Figure, go.Figure, go.Figure, go.Figure]:
    """
    Generate all 4 dashboard charts at once.
    
    Args:
        df: DataFrame to visualize
        chart1_type: Chart type for position 1
        chart2_type: Chart type for position 2
        chart3_type: Chart type for position 3
        chart4_type: Chart type for position 4
        
    Returns:
        Tuple of 4 Plotly Figures
    """
    if df is None or df.empty:
        empty = create_empty_chart("No data available")
        return empty, empty, empty, empty
    
    # Get smart configurations for each chart
    _, config1 = get_smart_chart_config(df, 1)
    _, config2 = get_smart_chart_config(df, 2)
    _, config3 = get_smart_chart_config(df, 3)
    _, config4 = get_smart_chart_config(df, 4)
    
    # Generate all charts
    chart1 = generate_dashboard_chart(df, chart1_type, config1)
    chart2 = generate_dashboard_chart(df, chart2_type, config2)
    chart3 = generate_dashboard_chart(df, chart3_type, config3)
    chart4 = generate_dashboard_chart(df, chart4_type, config4)
    
    return chart1, chart2, chart3, chart4


def get_dashboard_summary(df: pd.DataFrame) -> str:
    """
    Generate quick summary for dashboard header.
    
    Args:
        df: DataFrame being analyzed
        
    Returns:
        Formatted summary string
    """
    if df is None or df.empty:
        return "No data loaded"
    
    summary = f"""
### 📊 Dashboard Overview
**Dataset**: {len(df):,} rows × {df.shape[1]} columns | **Memory**: {df.memory_usage(deep=True).sum() / (1024**2):.1f} MB
"""
    return summary

def export_all_dashboard_charts(chart1, chart2, chart3, chart4) -> Tuple[Optional[str], str]:
    """
    Export all 4 dashboard charts as PNG files in a ZIP.
    
    Args:
        chart1, chart2, chart3, chart4: Plotly figures
        
    Returns:
        Tuple of (zip_file_path, status_message)
    """
    try:
        # Check if any charts exist
        charts = [chart1, chart2, chart3, chart4]
        valid_charts = [c for c in charts if c is not None]
        
        if len(valid_charts) == 0:
            return None, "⚠️ No charts to export. Generate dashboard first."
        
        temp_dir = tempfile.gettempdir()
        zip_path = os.path.join(temp_dir, "dashboard_charts.zip")
        
        exported_count = 0
        chart_paths = []
        
        # Export each chart individually
        for i, chart in enumerate(charts, 1):
            if chart is not None:
                try:
                    # Verify it's a plotly figure
                    if not hasattr(chart, 'write_image'):
                        print(f"Chart {i} is not a valid Plotly figure, skipping")
                        continue
                    
                    chart_path = export_figure_to_png(chart, f"dashboard_chart_{i}.png")
                    if chart_path and os.path.exists(chart_path):
                        chart_paths.append((chart_path, f"chart_{i}.png"))
                        exported_count += 1
                except Exception as e:
                    print(f"Error exporting chart {i}: {str(e)}")
                    continue
        
        # Create ZIP file
        if exported_count > 0:
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for chart_path, zip_name in chart_paths:
                    zipf.write(chart_path, zip_name)
            
            return zip_path, f"✅ Exported {exported_count} chart(s) as ZIP file"
        else:
            return None, "❌ No charts could be exported. Charts may not be generated yet."
    
    except Exception as e:
        return None, f"❌ Export error: {str(e)}"
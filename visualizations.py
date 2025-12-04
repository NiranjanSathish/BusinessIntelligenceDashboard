"""
Visualizations Module
Create interactive charts and visualizations using Plotly.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Optional, List


def create_time_series_plot(
    df: pd.DataFrame,
    date_column: str,
    value_column: str,
    agg_method: str = 'sum'
) -> Optional[go.Figure]:
    """
    Create time series line chart with aggregation.
    
    Args:
        df: DataFrame to visualize
        date_column: Column name for x-axis (datetime)
        value_column: Column name for y-axis (numerical)
        agg_method: Aggregation method ('sum', 'mean', 'count', 'median')
        
    Returns:
        Plotly Figure object
    """
    if df is None or df.empty:
        return create_empty_chart("No data available")
    
    if date_column not in df.columns or value_column not in df.columns:
        return create_empty_chart(f"Columns not found: {date_column}, {value_column}")
    
    try:
        # Ensure date column is datetime
        df_copy = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df_copy[date_column]):
            df_copy[date_column] = pd.to_datetime(df_copy[date_column], errors='coerce')
        
        # Group by date and aggregate
        if agg_method == 'sum':
            grouped = df_copy.groupby(date_column)[value_column].sum().reset_index()
        elif agg_method == 'mean':
            grouped = df_copy.groupby(date_column)[value_column].mean().reset_index()
        elif agg_method == 'count':
            grouped = df_copy.groupby(date_column)[value_column].count().reset_index()
        elif agg_method == 'median':
            grouped = df_copy.groupby(date_column)[value_column].median().reset_index()
        else:
            grouped = df_copy.groupby(date_column)[value_column].sum().reset_index()
        
        # Create line chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=grouped[date_column],
            y=grouped[value_column],
            mode='lines+markers',
            name=f'{value_column} ({agg_method})',
            line=dict(width=2),
            marker=dict(size=6)
        ))
        
        fig.update_layout(
            title=f'{value_column} Over Time ({agg_method.capitalize()})',
            xaxis_title=date_column,
            yaxis_title=f'{value_column} ({agg_method})',
            hovermode='x unified',
            height=500
        )
        
        return fig
        
    except Exception as e:
        return create_empty_chart(f"Error creating time series: {str(e)}")


def create_distribution_plot(
    df: pd.DataFrame,
    column: str,
    plot_type: str = 'histogram'
) -> Optional[go.Figure]:
    """
    Create histogram or box plot for distribution analysis.
    
    Args:
        df: DataFrame to visualize
        column: Column name to analyze
        plot_type: 'histogram' or 'box'
        
    Returns:
        Plotly Figure object
    """
    if df is None or df.empty:
        return create_empty_chart("No data available")
    
    if column not in df.columns:
        return create_empty_chart(f"Column not found: {column}")
    
    try:
        data = df[column].dropna()
        
        if len(data) == 0:
            return create_empty_chart(f"No data in column: {column}")
        
        if plot_type == 'histogram':
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=data,
                nbinsx=30,
                name=column,
                marker=dict(
                    color='rgb(55, 83, 109)',
                    line=dict(color='white', width=1)
                )
            ))
            
            fig.update_layout(
                title=f'Distribution of {column}',
                xaxis_title=column,
                yaxis_title='Frequency',
                showlegend=False,
                height=500
            )
            
        elif plot_type == 'box':
            fig = go.Figure()
            fig.add_trace(go.Box(
                y=data,
                name=column,
                marker=dict(color='rgb(55, 83, 109)'),
                boxmean='sd'
            ))
            
            fig.update_layout(
                title=f'Box Plot of {column}',
                yaxis_title=column,
                showlegend=False,
                height=500
            )
        
        return fig
        
    except Exception as e:
        return create_empty_chart(f"Error creating distribution plot: {str(e)}")


def create_category_chart(
    df: pd.DataFrame,
    column: str,
    chart_type: str = 'bar',
    value_column: Optional[str] = None,
    agg_method: str = 'count',
    top_n: int = 10
) -> Optional[go.Figure]:
    """
    Create bar chart or pie chart for categorical data.
    
    Args:
        df: DataFrame to visualize
        column: Categorical column name
        chart_type: 'bar' or 'pie'
        value_column: Optional numerical column for aggregation
        agg_method: Aggregation method ('count', 'sum', 'mean', 'median')
        top_n: Show only top N categories
        
    Returns:
        Plotly Figure object
    """
    if df is None or df.empty:
        return create_empty_chart("No data available")
    
    if column not in df.columns:
        return create_empty_chart(f"Column not found: {column}")
    
    try:
        # Aggregate data
        if value_column and value_column in df.columns:
            if agg_method == 'sum':
                grouped = df.groupby(column)[value_column].sum().reset_index()
            elif agg_method == 'mean':
                grouped = df.groupby(column)[value_column].mean().reset_index()
            elif agg_method == 'median':
                grouped = df.groupby(column)[value_column].median().reset_index()
            else:
                grouped = df.groupby(column)[value_column].count().reset_index()
            
            grouped.columns = [column, 'value']
            y_label = f'{value_column} ({agg_method})'
        else:
            # Count occurrences
            grouped = df[column].value_counts().reset_index()
            grouped.columns = [column, 'value']
            y_label = 'Count'
        
        # Get top N categories
        grouped = grouped.nlargest(top_n, 'value')
        
        if chart_type == 'bar':
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=grouped[column],
                y=grouped['value'],
                marker=dict(
                    color=grouped['value'],
                    colorscale='Viridis',
                    showscale=True
                ),
                text=grouped['value'],
                texttemplate='%{text:.2f}',
                textposition='outside'
            ))
            
            fig.update_layout(
                title=f'{column} Analysis (Top {top_n})',
                xaxis_title=column,
                yaxis_title=y_label,
                showlegend=False,
                height=500
            )
            
        elif chart_type == 'pie':
            fig = go.Figure()
            fig.add_trace(go.Pie(
                labels=grouped[column],
                values=grouped['value'],
                hole=0.3,
                textinfo='label+percent',
                textposition='outside'
            ))
            
            fig.update_layout(
                title=f'{column} Distribution (Top {top_n})',
                height=500
            )
        
        return fig
        
    except Exception as e:
        return create_empty_chart(f"Error creating category chart: {str(e)}")


def create_scatter_plot(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    color_column: Optional[str] = None
) -> Optional[go.Figure]:
    """
    Create scatter plot showing relationship between variables.
    
    Args:
        df: DataFrame to visualize
        x_column: Column name for x-axis
        y_column: Column name for y-axis
        color_column: Optional column for color coding
        
    Returns:
        Plotly Figure object
    """
    if df is None or df.empty:
        return create_empty_chart("No data available")
    
    if x_column not in df.columns or y_column not in df.columns:
        return create_empty_chart(f"Columns not found: {x_column}, {y_column}")
    
    try:
        # Create clean dataframe
        if color_column and color_column in df.columns and color_column != "None":
            df_clean = df[[x_column, y_column, color_column]].dropna()
            
            # Sample if too large to prevent crashes
            if len(df_clean) > 5000:
                df_clean = df_clean.sample(n=5000, random_state=42)
            
            fig = px.scatter(
                df_clean,
                x=x_column,
                y=y_column,
                color=color_column,
                opacity=0.6,
                title=f'{y_column} vs {x_column}'
            )
        else:
            df_clean = df[[x_column, y_column]].dropna()
            
            # Sample if too large to prevent crashes
            if len(df_clean) > 5000:
                df_clean = df_clean.sample(n=5000, random_state=42)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_clean[x_column],
                y=df_clean[y_column],
                mode='markers',
                marker=dict(
                    size=8,
                    color='rgb(55, 83, 109)',
                    opacity=0.6
                ),
                name='Data Points'
            ))
            
            fig.update_layout(
                title=f'{y_column} vs {x_column}',
                xaxis_title=x_column,
                yaxis_title=y_column,
                height=500
            )
        
        # Add trendline only if we have enough valid data points
        if len(df_clean) >= 2:
            try:
                # Remove any infinite values
                x_vals = df_clean[x_column].replace([np.inf, -np.inf], np.nan).dropna()
                y_vals = df_clean[y_column].replace([np.inf, -np.inf], np.nan).dropna()
                
                # Ensure we have matching indices
                valid_idx = x_vals.index.intersection(y_vals.index)
                x_vals = x_vals.loc[valid_idx]
                y_vals = y_vals.loc[valid_idx]
                
                # Check if we have enough variance
                if len(x_vals) >= 2 and x_vals.nunique() > 1:
                    z = np.polyfit(x_vals, y_vals, 1)
                    p = np.poly1d(z)
                    x_trend = np.linspace(x_vals.min(), x_vals.max(), 100)
                    
                    fig.add_trace(go.Scatter(
                        x=x_trend,
                        y=p(x_trend),
                        mode='lines',
                        name='Trend',
                        line=dict(color='red', dash='dash', width=2)
                    ))
            except Exception as e:
                # If trend line fails, just skip it
                pass
        
        return fig
        
    except Exception as e:
        return create_empty_chart(f"Error creating scatter plot: {str(e)}")


def create_correlation_heatmap(df: pd.DataFrame) -> Optional[go.Figure]:
    """
    Create correlation heatmap for numerical columns.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Plotly Figure object
    """
    if df is None or df.empty:
        return create_empty_chart("No data available")
    
    try:
        # Select only numerical columns
        numerical_df = df.select_dtypes(include=[np.number])
        
        if numerical_df.shape[1] < 2:
            return create_empty_chart("Need at least 2 numerical columns for correlation")
        
        # Calculate correlation
        corr_matrix = numerical_df.corr()
        
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
            height=600,
            width=700
        )
        
        return fig
        
    except Exception as e:
        return create_empty_chart(f"Error creating correlation heatmap: {str(e)}")


def create_empty_chart(message: str) -> go.Figure:
    """
    Create an empty chart with a message.
    
    Args:
        message: Message to display
        
    Returns:
        Plotly Figure object
    """
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=16)
    )
    fig.update_layout(
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        height=500
    )
    return fig


def get_available_chart_types() -> List[str]:
    """
    Return list of available chart types.
    
    Returns:
        List of chart type names
    """
    return [
        'Time Series',
        'Histogram',
        'Box Plot',
        'Bar Chart',
        'Pie Chart',
        'Scatter Plot',
        'Correlation Heatmap'
    ]


def validate_chart_inputs(df: pd.DataFrame, chart_type: str, columns: dict) -> tuple:
    """
    Validate that selected columns are appropriate for chart type.
    
    Args:
        df: DataFrame
        chart_type: Selected chart type
        columns: Dictionary of selected columns
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if df is None or df.empty:
        return False, "No data available"
    
    if chart_type == 'Time Series':
        date_col = columns.get('date_column')
        value_col = columns.get('value_column')
        
        if not date_col or not value_col:
            return False, "Please select both date and value columns"
        
        if date_col not in df.columns or value_col not in df.columns:
            return False, "Selected columns not found in dataset"
        
        if not pd.api.types.is_numeric_dtype(df[value_col]):
            return False, f"{value_col} must be numerical"
    
    elif chart_type in ['Histogram', 'Box Plot']:
        column = columns.get('column')
        
        if not column:
            return False, "Please select a column"
        
        if column not in df.columns:
            return False, "Selected column not found in dataset"
        
        if not pd.api.types.is_numeric_dtype(df[column]):
            return False, f"{column} must be numerical"
    
    elif chart_type in ['Bar Chart', 'Pie Chart']:
        column = columns.get('column')
        
        if not column:
            return False, "Please select a category column"
        
        if column not in df.columns:
            return False, "Selected column not found in dataset"
    
    elif chart_type == 'Scatter Plot':
        x_col = columns.get('x_column')
        y_col = columns.get('y_column')
        
        if not x_col or not y_col:
            return False, "Please select both X and Y columns"
        
        if x_col not in df.columns or y_col not in df.columns:
            return False, "Selected columns not found in dataset"
        
        if not pd.api.types.is_numeric_dtype(df[x_col]) or not pd.api.types.is_numeric_dtype(df[y_col]):
            return False, "Both X and Y columns must be numerical"
            
    elif chart_type == 'Correlation Heatmap':
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) < 2:
            return False, "Need at least 2 numerical columns for correlation"
    
    return True, ""
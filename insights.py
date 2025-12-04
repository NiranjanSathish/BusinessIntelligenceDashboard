"""
Insights Module
Generate automated insights, identify trends, anomalies, and patterns.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats


def generate_top_bottom_insights(
    df: pd.DataFrame,
    metric_column: str,
    category_column: Optional[str] = None,
    n: int = 5
) -> str:
    """
    Identify top and bottom performers.
    
    Args:
        df: DataFrame to analyze
        metric_column: Numerical column to rank
        category_column: Optional categorical column for grouping
        n: Number of top/bottom items to show
        
    Returns:
        Formatted string with insights
    """
    if df is None or df.empty:
        return "No data available for analysis."
    
    if metric_column not in df.columns:
        return f"Column '{metric_column}' not found."
    
    insights = []
    
    try:
        if category_column and category_column in df.columns:
            # Group by category and aggregate
            grouped = df.groupby(category_column)[metric_column].sum().sort_values(ascending=False)
            
            # Top performers
            top_n = grouped.head(n)
            insights.append(f"### 🏆 Top {n} {category_column} by {metric_column}\n")
            for i, (cat, val) in enumerate(top_n.items(), 1):
                insights.append(f"{i}. **{cat}**: {val:,.2f}")
            
            # Bottom performers
            bottom_n = grouped.tail(n)
            insights.append(f"\n### 📉 Bottom {n} {category_column} by {metric_column}\n")
            for i, (cat, val) in enumerate(bottom_n.items(), 1):
                insights.append(f"{i}. **{cat}**: {val:,.2f}")
        else:
            # Simple top/bottom values
            sorted_values = df[metric_column].dropna().sort_values(ascending=False)
            
            insights.append(f"### 📊 {metric_column} Statistics\n")
            insights.append(f"**Highest Value**: {sorted_values.iloc[0]:,.2f}")
            insights.append(f"**Lowest Value**: {sorted_values.iloc[-1]:,.2f}")
            insights.append(f"**Average**: {sorted_values.mean():,.2f}")
            insights.append(f"**Median**: {sorted_values.median():,.2f}")
    
    except Exception as e:
        return f"Error generating top/bottom insights: {str(e)}"
    
    return "\n".join(insights)


def detect_trends(
    df: pd.DataFrame,
    date_column: str,
    metric_column: str
) -> str:
    """
    Detect growth trends and patterns over time.
    
    Args:
        df: DataFrame to analyze
        date_column: Date column
        metric_column: Numerical metric column
        
    Returns:
        Formatted string with trend insights
    """
    if df is None or df.empty:
        return "No data available for trend analysis."
    
    if date_column not in df.columns or metric_column not in df.columns:
        return "Required columns not found for trend analysis."
    
    insights = []
    
    try:
        # Ensure date column is datetime
        df_copy = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df_copy[date_column]):
            df_copy[date_column] = pd.to_datetime(df_copy[date_column], errors='coerce')
        
        # Sort by date and aggregate
        daily_data = df_copy.groupby(date_column)[metric_column].sum().sort_index()
        
        if len(daily_data) < 2:
            return "Insufficient data for trend analysis (need at least 2 time points)."
        
        # Calculate overall trend
        first_value = daily_data.iloc[0]
        last_value = daily_data.iloc[-1]
        change = last_value - first_value
        pct_change = (change / first_value * 100) if first_value != 0 else 0
        
        insights.append("### 📈 Trend Analysis\n")
        
        # Overall trend
        if pct_change > 0:
            insights.append(f"**Overall Trend**: 📈 **Increasing** ({pct_change:.2f}% growth)")
        elif pct_change < 0:
            insights.append(f"**Overall Trend**: 📉 **Decreasing** ({abs(pct_change):.2f}% decline)")
        else:
            insights.append(f"**Overall Trend**: ➡️ **Stable** (no significant change)")
        
        insights.append(f"**Start Value**: {first_value:,.2f}")
        insights.append(f"**End Value**: {last_value:,.2f}")
        insights.append(f"**Total Change**: {change:,.2f}")
        
        # Peak and trough
        peak_date = daily_data.idxmax()
        peak_value = daily_data.max()
        trough_date = daily_data.idxmin()
        trough_value = daily_data.min()
        
        insights.append(f"\n**Peak**: {peak_value:,.2f} on {peak_date.strftime('%Y-%m-%d')}")
        insights.append(f"**Trough**: {trough_value:,.2f} on {trough_date.strftime('%Y-%m-%d')}")
        
        # Volatility
        std_dev = daily_data.std()
        mean_val = daily_data.mean()
        cv = (std_dev / mean_val * 100) if mean_val != 0 else 0
        
        if cv > 30:
            insights.append(f"\n**Volatility**: High (CV: {cv:.1f}%)")
        elif cv > 15:
            insights.append(f"\n**Volatility**: Moderate (CV: {cv:.1f}%)")
        else:
            insights.append(f"\n**Volatility**: Low (CV: {cv:.1f}%)")
    
    except Exception as e:
        return f"Error detecting trends: {str(e)}"
    
    return "\n".join(insights)


def detect_anomalies(
    df: pd.DataFrame,
    column: str,
    method: str = 'iqr',
    threshold: float = 1.5
) -> str:
    """
    Detect outliers and anomalies in data.
    
    Args:
        df: DataFrame to analyze
        column: Column to analyze
        method: 'iqr' or 'zscore'
        threshold: Threshold for anomaly detection
        
    Returns:
        Formatted string with anomaly insights
    """
    if df is None or df.empty:
        return "No data available for anomaly detection."
    
    if column not in df.columns:
        return f"Column '{column}' not found."
    
    insights = []
    
    try:
        data = df[column].dropna()
        
        if len(data) < 10:
            return "Insufficient data for anomaly detection (need at least 10 values)."
        
        if method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            outliers = data[(data < lower_bound) | (data > upper_bound)]
        
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(data))
            outliers = data[z_scores > threshold]
        
        else:
            return "Invalid method. Use 'iqr' or 'zscore'."
        
        insights.append(f"### 🔍 Anomaly Detection - {column}\n")
        insights.append(f"**Method**: {method.upper()}")
        insights.append(f"**Total Values**: {len(data):,}")
        insights.append(f"**Anomalies Found**: {len(outliers):,} ({len(outliers)/len(data)*100:.1f}%)")
        
        if len(outliers) > 0:
            insights.append(f"\n**Anomaly Range**:")
            insights.append(f"- Lowest Anomaly: {outliers.min():,.2f}")
            insights.append(f"- Highest Anomaly: {outliers.max():,.2f}")
            
            if len(outliers) <= 10:
                insights.append(f"\n**All Anomalies**: {', '.join([f'{x:.2f}' for x in sorted(outliers)])}")
        else:
            insights.append("\n✅ No significant anomalies detected.")
    
    except Exception as e:
        return f"Error detecting anomalies: {str(e)}"
    
    return "\n".join(insights)


def analyze_correlations(df: pd.DataFrame, threshold: float = 0.7) -> str:
    """
    Find strong correlations between variables.
    
    Args:
        df: DataFrame to analyze
        threshold: Correlation threshold (absolute value)
        
    Returns:
        Formatted string with correlation insights
    """
    if df is None or df.empty:
        return "No data available for correlation analysis."
    
    insights = []
    
    try:
        # Select only numerical columns
        numerical_df = df.select_dtypes(include=[np.number])
        
        if numerical_df.shape[1] < 2:
            return "Need at least 2 numerical columns for correlation analysis."
        
        # Calculate correlation matrix
        corr_matrix = numerical_df.corr()
        
        # Find strong correlations
        strong_corr = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                
                if abs(corr_val) >= threshold:
                    col1 = corr_matrix.columns[i]
                    col2 = corr_matrix.columns[j]
                    strong_corr.append((col1, col2, corr_val))
        
        insights.append(f"### 🔗 Correlation Analysis\n")
        insights.append(f"**Threshold**: {threshold} (absolute value)")
        insights.append(f"**Strong Correlations Found**: {len(strong_corr)}")
        
        if strong_corr:
            # Sort by absolute correlation value
            strong_corr.sort(key=lambda x: abs(x[2]), reverse=True)
            
            insights.append("\n**Top Correlations**:\n")
            for col1, col2, corr_val in strong_corr[:10]:  # Show top 10
                direction = "Positive" if corr_val > 0 else "Negative"
                strength = "Very Strong" if abs(corr_val) > 0.9 else "Strong"
                insights.append(f"- **{col1}** ↔ **{col2}**: {corr_val:.3f} ({strength} {direction})")
        else:
            insights.append("\nℹ️ No strong correlations found above threshold.")
    
    except Exception as e:
        return f"Error analyzing correlations: {str(e)}"
    
    return "\n".join(insights)


def analyze_data_quality(df: pd.DataFrame) -> str:
    """
    Analyze data quality issues.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Formatted string with data quality insights
    """
    if df is None or df.empty:
        return "No data available for quality analysis."
    
    insights = []
    
    try:
        insights.append("### ✅ Data Quality Report\n")
        
        # Missing values
        missing_counts = df.isnull().sum()
        cols_with_missing = missing_counts[missing_counts > 0]
        
        if len(cols_with_missing) > 0:
            total_missing = missing_counts.sum()
            total_cells = df.shape[0] * df.shape[1]
            missing_pct = total_missing / total_cells * 100
            
            insights.append(f"**Missing Values**: {total_missing:,} ({missing_pct:.2f}% of all data)")
            insights.append(f"**Columns with Missing Data**: {len(cols_with_missing)}")
            
            # Show top columns with missing data
            top_missing = cols_with_missing.nlargest(5)
            insights.append("\n**Top Columns with Missing Data**:")
            for col, count in top_missing.items():
                pct = count / df.shape[0] * 100
                insights.append(f"- {col}: {count:,} ({pct:.1f}%)")
        else:
            insights.append("✅ **No missing values detected**")
        
        # Duplicates
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            dup_pct = duplicate_count / len(df) * 100
            insights.append(f"\n**Duplicate Rows**: {duplicate_count:,} ({dup_pct:.2f}%)")
        else:
            insights.append("\n✅ **No duplicate rows detected**")
        
        # Data types
        insights.append(f"\n**Column Types**:")
        dtype_counts = df.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            insights.append(f"- {dtype}: {count} columns")
    
    except Exception as e:
        return f"Error analyzing data quality: {str(e)}"
    
    return "\n".join(insights)


def generate_comprehensive_insights(df: pd.DataFrame) -> Dict[str, str]:
    """
    Generate all insights at once.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Dictionary with all insight types
    """
    if df is None or df.empty:
        return {
            'summary': "No data available for analysis.",
            'quality': "",
            'top_bottom': "",
            'trends': "",
            'anomalies': "",
            'correlations': ""
        }
    
    insights = {}
    
    # Data quality (always available)
    insights['quality'] = analyze_data_quality(df)
    
    # Get column types
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    # Top/Bottom (if we have numerical and categorical columns)
    if numerical_cols and categorical_cols:
        insights['top_bottom'] = generate_top_bottom_insights(
            df, 
            numerical_cols[0], 
            categorical_cols[0], 
            n=5
        )
    elif numerical_cols:
        insights['top_bottom'] = generate_top_bottom_insights(df, numerical_cols[0], n=5)
    else:
        insights['top_bottom'] = "No numerical columns available for top/bottom analysis."
    
    # Trends (if we have datetime and numerical columns)
    if datetime_cols and numerical_cols:
        insights['trends'] = detect_trends(df, datetime_cols[0], numerical_cols[0])
    else:
        insights['trends'] = "No datetime columns available for trend analysis."
    
    # Anomalies (if we have numerical columns)
    if numerical_cols:
        insights['anomalies'] = detect_anomalies(df, numerical_cols[0], method='iqr')
    else:
        insights['anomalies'] = "No numerical columns available for anomaly detection."
    
    # Correlations (always try if we have numerical columns)
    if len(numerical_cols) >= 2:
        insights['correlations'] = analyze_correlations(df, threshold=0.7)
    else:
        insights['correlations'] = "Need at least 2 numerical columns for correlation analysis."
    
    # Create summary
    summary_parts = [
        f"**Total Rows**: {len(df):,}",
        f"**Total Columns**: {df.shape[1]}",
        f"**Numerical Columns**: {len(numerical_cols)}",
        f"**Categorical Columns**: {len(categorical_cols)}",
        f"**Datetime Columns**: {len(datetime_cols)}"
    ]
    insights['summary'] = "### 📊 Dataset Summary\n\n" + "\n".join(summary_parts)
    
    return insights


def format_insights_report(insights_dict: Dict[str, str]) -> str:
    """
    Format insights into a readable markdown report.
    
    Args:
        insights_dict: Dictionary with insight sections
        
    Returns:
        Formatted markdown string
    """
    report_sections = []
    
    # Add each section if it has content
    section_order = ['summary', 'quality', 'top_bottom', 'trends', 'anomalies', 'correlations']
    
    for section in section_order:
        if section in insights_dict and insights_dict[section]:
            report_sections.append(insights_dict[section])
    
    return "\n\n---\n\n".join(report_sections)
"""
Business Intelligence Dashboard - Main Application
A professional BI dashboard for data exploration and analysis.
"""

import gradio as gr
import pandas as pd
import warnings
import asyncio
import sys

# Suppress warnings and fix asyncio issues
warnings.filterwarnings('ignore')
if sys.platform == 'linux':
    try:
        import nest_asyncio
        nest_asyncio.apply()
    except ImportError:
        pass

from data_processor import (
    load_data, 
    get_dataset_info, 
    get_data_preview, 
    format_dataframe_info,
    detect_column_types,
    get_unique_values
)
from statistics import (
    get_data_summary,
    format_statistics_summary,
    profile_numerical_columns,
    profile_categorical_columns,
    analyze_missing_values
)
from filters import (
    generate_filter_config,
    apply_numerical_filter,
    apply_categorical_filter,
    apply_date_filter,
    get_filter_summary
)
from visualizations import (
    create_time_series_plot,
    create_distribution_plot,
    create_category_chart,
    create_scatter_plot,
    create_correlation_heatmap,
    get_available_chart_types,
    validate_chart_inputs
)
from insights import (
    generate_comprehensive_insights,
    format_insights_report
)
from utils import (
    get_aggregation_methods,
    export_dataframe_to_csv,
    export_figure_to_png,
    export_text_to_file
)


def handle_file_upload(file):
    """
    Handle file upload and return dataset information.
    Also resets filter state.
    
    Args:
        file: Uploaded file from Gradio
        
    Returns:
        Tuple of (DataFrame, status_message, info_text, head_df, tail_df, 
                 reset filter components and preview)
    """
    # Load the data
    df, status_msg = load_data(file)
    
    if df is None:
        return (
            None, 
            status_msg, 
            "", 
            pd.DataFrame(), 
            pd.DataFrame(),
            None,  # filtered_df_state
            "",  # filter_summary
            pd.DataFrame(),  # filtered_data_preview
            gr.update(value="None"),  # numerical_col_dropdown
            gr.update(value="None"),  # categorical_col_dropdown
            gr.update(value=[], choices=[]),  # categorical_values_dropdown
            gr.update(value=0),  # num_min_slider
            gr.update(value=100),  # num_max_slider
            gr.update(value="None"),  # datetime_col_dropdown
            gr.update(value=""),  # date_start
            gr.update(value=""),  # date_end
            # Visualization dropdowns (reset)
            gr.update(choices=["None"], value="None"),  # viz_input_1
            gr.update(choices=["None"], value="None"),  # viz_input_2
            gr.update(choices=["None"], value="None")   # viz_input_3
        )
    
    # Get dataset information
    info = get_dataset_info(df)
    info_text = format_dataframe_info(info)
    
    # Get data preview
    head_df, tail_df = get_data_preview(df, n=10)
    
    # Reset filter summary
    filter_summary_reset = ""
    
    # Detect column types for visualization dropdowns
    col_types = detect_column_types(df)
    num_cols = ["None"] + col_types.get('numerical', [])
    cat_cols = ["None"] + col_types.get('categorical', [])
    date_cols = ["None"] + col_types.get('datetime', [])
    all_cols = ["None"] + list(df.columns)

    return (
        df,  # df_state
        status_msg,  # status_msg
        info_text,  # dataset_info
        head_df,  # head_table
        tail_df,  # tail_table
        None,  # filtered_df_state (reset)
        filter_summary_reset,  # filter_summary (clear)
        pd.DataFrame(),  # filtered_data_preview (clear)
        gr.update(value="None"),  # numerical_col_dropdown
        gr.update(value="None"),  # categorical_col_dropdown
        gr.update(value=[], choices=[]),  # categorical_values_dropdown
        gr.update(value=0),  # num_min_slider
        gr.update(value=100),  # num_max_slider
        gr.update(value="None"),  # datetime_col_dropdown
        gr.update(value=""),  # date_start
        gr.update(value=""),  # date_end
        # Visualization dropdowns
        gr.update(choices=date_cols, value="None"),  # viz_input_1 (default Time Series Date)
        gr.update(choices=num_cols, value="None"),  # viz_input_2 (default Time Series Value)
        gr.update(choices=get_aggregation_methods(), value="sum")   # viz_input_3 (default Time Series Agg)
    )


def generate_statistics(df):
    """
    Generate comprehensive statistics for the dataset.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Tuple of (summary_text, numerical_stats, categorical_stats, missing_stats, corr_heatmap, missing_chart)
    """
    if df is None or df.empty:
        empty_msg = "⚠️ No data available. Please upload a dataset first."
        return empty_msg, None, None, None, None, None
    
    try:
        # Get comprehensive summary
        summary = get_data_summary(df)
        summary_text = format_statistics_summary(df)
        
        # Extract individual components
        numerical_stats = summary.get('numerical')
        categorical_stats = summary.get('categorical')
        missing_stats = summary.get('missing')
        corr_heatmap = summary.get('correlation_heatmap')
        missing_chart = summary.get('missing_chart')
        
        return summary_text, numerical_stats, categorical_stats, missing_stats, corr_heatmap, missing_chart
        
    except Exception as e:
        error_msg = f"❌ Error generating statistics: {str(e)}"
        return error_msg, None, None, None, None, None


def setup_filters(df):
    """
    Initialize filter components based on dataset.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Tuple of (filter_info_text, column_types)
    """
    if df is None or df.empty:
        return "⚠️ No data available. Please upload a dataset first.", {}
    
    try:
        column_types = detect_column_types(df)
        
        # Get filter config to check for excluded columns
        from filters import generate_filter_config
        filter_config = generate_filter_config(df)
        excluded = filter_config.get('excluded_categorical', [])
        
        info_text = f"""
### 🔍 Available Filters

**Numerical Columns ({len(column_types['numerical'])}):** {', '.join(column_types['numerical']) if column_types['numerical'] else 'None'}

**Categorical Columns ({len(column_types['categorical'])}):** {', '.join(column_types['categorical']) if column_types['categorical'] else 'None'}

**Datetime Columns ({len(column_types['datetime'])}):** {', '.join(column_types['datetime']) if column_types['datetime'] else 'None'}

*Select filters below to refine your data*
"""
        
        # Add message about excluded categorical columns
        if excluded:
            info_text += "\n\n⚠️ **Note:** Some categorical columns are not available for filtering because they have too many unique values (>100):\n"
            for exc in excluded:
                info_text += f"- **{exc['column']}**: {exc['unique_count']:,} unique values\n"
        
        return info_text, column_types
    except Exception as e:
        return f"❌ Error setting up filters: {str(e)}", {}


def update_datetime_range(df, column):
    """
    Update datetime filter range based on selected column.
    
    Args:
        df: DataFrame
        column: Selected column name
        
    Returns:
        Tuple of (start_date_str, end_date_str)
    """
    if df is None or df.empty or not column or column == "None":
        return "", ""
    
    try:
        if column in df.columns and pd.api.types.is_datetime64_any_dtype(df[column]):
            min_date = df[column].min()
            max_date = df[column].max()
            # Convert to string format
            start_str = min_date.strftime('%Y-%m-%d') if pd.notna(min_date) else ""
            end_str = max_date.strftime('%Y-%m-%d') if pd.notna(max_date) else ""
            return start_str, end_str
    except:
        pass
    
    return "", ""


def apply_filters_to_data(df, num_col, num_min, num_max, cat_col, cat_values, date_col, date_start, date_end):
    """
    Apply selected filters to the DataFrame.
    
    Args:
        df: Original DataFrame
        num_col: Selected numerical column
        num_min: Minimum value for numerical filter
        num_max: Maximum value for numerical filter
        cat_col: Selected categorical column
        cat_values: Selected categorical values
        date_col: Selected datetime column
        date_start: Start date string
        date_end: End date string
        
    Returns:
        Tuple of (filtered_df, summary_text, preview_df)
    """
    if df is None or df.empty:
        return None, "⚠️ No data available", pd.DataFrame()
    
    try:
        filtered_df = df.copy()
        
        # Apply numerical filter
        if num_col and num_col != "None" and num_min is not None and num_max is not None:
            filtered_df = apply_numerical_filter(filtered_df, num_col, num_min, num_max)
        
        # Apply categorical filter
        if cat_col and cat_col != "None" and cat_values and len(cat_values) > 0:
            filtered_df = apply_categorical_filter(filtered_df, cat_col, cat_values)
        
        # Apply datetime filter
        if date_col and date_col != "None" and date_start and date_end:
            try:
                start_date = pd.to_datetime(date_start)
                end_date = pd.to_datetime(date_end)
                filtered_df = apply_date_filter(filtered_df, date_col, start_date, end_date)
            except Exception as e:
                print(f"Error parsing dates: {str(e)}")
        
        # Generate summary
        summary = get_filter_summary(df, filtered_df)
        
        # Get preview
        preview_df = filtered_df.head(100)  # Limit preview to 100 rows
        
        return filtered_df, summary, preview_df
        
    except Exception as e:
        error_msg = f"❌ Error applying filters: {str(e)}"
        return df, error_msg, df.head(100)


def reset_all_filters(df):
    """
    Reset filters and return original data with UI component resets.
    
    Args:
        df: Original DataFrame
        
    Returns:
        Tuple of (df, summary, preview, ui_resets...)
    """
    if df is None or df.empty:
        return (
            None, 
            "⚠️ No data available", 
            pd.DataFrame(),
            gr.update(value="None"),
            gr.update(value="None"),
            gr.update(value=[], choices=[]),
            gr.update(value=0),
            gr.update(value=100),
            gr.update(value="None"),
            gr.update(value=""),
            gr.update(value="")
        )
    
    summary = f"""
### 🔄 Filters Reset

**Total Rows:** {len(df):,}  
All filters have been cleared.
"""
    
    preview = df.head(100)
    
    return (
        df,  # filtered_df_state
        summary,  # filter_summary
        preview,  # filtered_data_preview
        gr.update(value="None"),  # numerical_col_dropdown
        gr.update(value="None"),  # categorical_col_dropdown
        gr.update(value=[], choices=[]),  # categorical_values_dropdown
        gr.update(value=0),  # num_min_slider
        gr.update(value=100),  # num_max_slider
        gr.update(value="None"),  # datetime_col_dropdown
        gr.update(value=""),  # date_start
        gr.update(value="")  # date_end
    )


def update_numerical_range(df, column):
    """
    Update numerical filter range based on selected column.
    
    Args:
        df: DataFrame
        column: Selected column name
        
    Returns:
        Tuple of (min_value, max_value, slider_min, slider_max)
    """
    if df is None or df.empty or not column or column == "None":
        return 0, 100, 0, 100
    
    try:
        if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
            min_val = float(df[column].min())
            max_val = float(df[column].max())
            return min_val, max_val, min_val, max_val
    except:
        pass
    
    return 0, 100, 0, 100


def update_categorical_choices(df, column):
    """
    Update categorical filter choices based on selected column.
    
    Args:
        df: DataFrame
        column: Selected column name
        
    Returns:
        List of unique values
    """
    if df is None or df.empty or not column or column == "None":
        return []
    
    try:
        return get_unique_values(df, column, max_values=100)
    except:
        return []


def generate_chart(df, chart_type, input1, input2, input3):
    """
    Generate chart based on selected type and generic parameters.
    
    Args:
        df: DataFrame
        chart_type: Type of chart
        input1: Generic input 1 (Date/Column/Category/X)
        input2: Generic input 2 (Value/None/Value/Y)
        input3: Generic input 3 (Agg/None/Agg/Color)
    """
    if df is None or df.empty:
        from visualizations import create_empty_chart
        return create_empty_chart("⚠️ No data available. Please upload and optionally filter data first.")
    
    try:
        if chart_type == 'Time Series':
            date_col = input1
            value_col = input2
            agg_method = input3
            
            if not date_col or not value_col or date_col == "None" or value_col == "None":
                from visualizations import create_empty_chart
                return create_empty_chart("Please select date and value columns")
            return create_time_series_plot(df, date_col, value_col, agg_method)
        
        elif chart_type in ['Histogram', 'Box Plot']:
            col = input1
            if not col or col == "None":
                from visualizations import create_empty_chart
                return create_empty_chart("Please select a numerical column")
            return create_distribution_plot(df, col, 'histogram' if chart_type == 'Histogram' else 'box')
        
        elif chart_type in ['Bar Chart', 'Pie Chart']:
            cat_col = input1
            val_col = input2 if input2 and input2 != "None" else None
            agg_method = input3
            
            if not cat_col or cat_col == "None":
                from visualizations import create_empty_chart
                return create_empty_chart("Please select a categorical column")
            return create_category_chart(df, cat_col, 'bar' if chart_type == 'Bar Chart' else 'pie', val_col, agg_method, top_n=10)
        
        elif chart_type == 'Scatter Plot':
            x_col = input1
            y_col = input2
            color_col = input3 if input3 and input3 != "None" else None
            
            if not x_col or not y_col or x_col == "None" or y_col == "None":
                from visualizations import create_empty_chart
                return create_empty_chart("Please select X and Y columns")
            return create_scatter_plot(df, x_col, y_col, color_col)
        
        elif chart_type == 'Correlation Heatmap':
            return create_correlation_heatmap(df)
        
        else:
            from visualizations import create_empty_chart
            return create_empty_chart("Please select a chart type")
            
    except Exception as e:
        from visualizations import create_empty_chart
        return create_empty_chart(f"Error creating chart: {str(e)}")


def update_chart_ui(df, chart_type):
    """
    Update generic UI inputs based on chart type.
    Returns updates for: input1, input2, input3
    """
    if df is None or df.empty:
        return (
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False)
        )
    
    col_types = detect_column_types(df)
    num_cols = ["None"] + col_types.get('numerical', [])
    cat_cols = ["None"] + col_types.get('categorical', [])
    date_cols = ["None"] + col_types.get('datetime', [])
    all_cols = ["None"] + list(df.columns)
    agg_methods = get_aggregation_methods()
    
    if chart_type == 'Time Series':
        return (
            gr.update(label="Date Column", choices=date_cols, value="None", visible=True),
            gr.update(label="Value Column", choices=num_cols, value="None", visible=True),
            gr.update(label="Aggregation", choices=agg_methods, value="sum", visible=True)
        )
    
    elif chart_type in ['Histogram', 'Box Plot']:
        return (
            gr.update(label="Column", choices=num_cols, value="None", visible=True),
            gr.update(visible=False),
            gr.update(visible=False)
        )
    
    elif chart_type in ['Bar Chart', 'Pie Chart']:
        return (
            gr.update(label="Category Column", choices=cat_cols, value="None", visible=True),
            gr.update(label="Value Column (Optional)", choices=num_cols, value="None", visible=True),
            gr.update(label="Aggregation", choices=agg_methods, value="count", visible=True)
        )
    
    elif chart_type == 'Scatter Plot':
        return (
            gr.update(label="X-Axis Column", choices=num_cols, value="None", visible=True),
            gr.update(label="Y-Axis Column", choices=num_cols, value="None", visible=True),
            gr.update(label="Color By (Optional)", choices=all_cols, value="None", visible=True)
        )
    
    elif chart_type == 'Correlation Heatmap':
        return (
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False)
        )
    
    return (gr.update(visible=False), gr.update(visible=False), gr.update(visible=False))


def create_dashboard():
    """
    Create the main Gradio dashboard interface.
    
    Returns:
        Gradio Blocks interface
    """
    with gr.Blocks(title="Business Intelligence Dashboard") as demo:
        
        # Main title and description
        gr.Markdown("""
        # 📊 Business Intelligence Dashboard
        ### Upload your dataset to explore, analyze, and gain insights without the need for coding.
        """)
        
        # State to store the DataFrame across tabs
        df_state = gr.State(value=None)
        
        # Create tabs
        with gr.Tabs():
            
            # ==================== TAB 1: DATA UPLOAD ====================
            with gr.Tab("📁 Data Upload"):
                gr.Markdown("""
                ### Upload Your Dataset
                Supported formats: **CSV** and **Excel** (.xlsx, .xls)
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        file_upload = gr.File(
                            label="Upload CSV or Excel File",
                            file_types=[".csv", ".xlsx", ".xls"],
                            type="filepath"
                        )
                        upload_btn = gr.Button("📤 Load Data", variant="primary", size="lg")
                        
                        status_msg = gr.Textbox(
                            label="Status",
                            placeholder="Upload a file to get started...",
                            interactive=False
                        )
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Dataset Information")
                        dataset_info = gr.Markdown()
                
                gr.Markdown("---")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 📋 First 10 Rows")
                        head_table = gr.Dataframe(
                            label="Data Preview (Head)",
                            interactive=False,
                            wrap=True
                        )
                    
                    with gr.Column():
                        gr.Markdown("### 📋 Last 10 Rows")
                        tail_table = gr.Dataframe(
                            label="Data Preview (Tail)",
                            interactive=False,
                            wrap=True
                        )
            
            # ==================== TAB 2: STATISTICS ====================
            with gr.Tab("📈 Statistics"):
                gr.Markdown("""
                ### 📊 Comprehensive Data Profiling
                Explore summary statistics, distributions, and data quality metrics.
                """)
                
                with gr.Row():
                    generate_stats_btn = gr.Button("🔄 Generate Statistics", variant="primary", size="lg")
                
                with gr.Row():
                    stats_summary = gr.Markdown()
                
                gr.Markdown("---")
                
                # Numerical Statistics
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 📊 Numerical Columns Statistics")
                        numerical_stats_table = gr.Dataframe(
                            label="Statistical Measures",
                            interactive=False,
                            wrap=True
                        )
                
                gr.Markdown("---")
                
                # Categorical Statistics
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 🏷️ Categorical Columns Statistics")
                        categorical_stats_table = gr.Dataframe(
                            label="Categorical Analysis",
                            interactive=False,
                            wrap=True
                        )
                
                gr.Markdown("---")
                
                # Missing Values Analysis
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### ⚠️ Missing Values Analysis")
                        missing_stats_table = gr.Dataframe(
                            label="Missing Values Report",
                            interactive=False,
                            wrap=True
                        )
                    
                    with gr.Column():
                        missing_values_chart = gr.Plot(label="Missing Values Visualization")
                
                gr.Markdown("---")
                
                # Correlation Analysis
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 🔗 Correlation Analysis")
                        gr.Markdown("*Shows relationships between numerical variables*")
                        correlation_heatmap = gr.Plot(label="Correlation Heatmap")
                
            # ==================== TAB 3: FILTER & EXPLORE ====================
            with gr.Tab("🔍 Filter & Explore"):
                gr.Markdown("""
                ### 🔍 Interactive Data Filtering
                Apply filters to explore specific subsets of your data.
                """)
                
                with gr.Row():
                    setup_filters_btn = gr.Button("🔧 Setup Filters", variant="primary", size="lg")
                
                filter_info = gr.Markdown()
                
                gr.Markdown("---")
                
                # Filter Controls
                gr.Markdown("### 🎛️ Filter Controls")
                
                with gr.Row():
                    # Numerical Filter
                    with gr.Column():
                        gr.Markdown("#### 📊 Numerical Filter")
                        numerical_col_dropdown = gr.Dropdown(
                            label="Select Numerical Column",
                            choices=["None"],
                            value="None",
                            interactive=True
                        )
                        
                        with gr.Row():
                            num_min_slider = gr.Number(
                                label="Minimum Value",
                                value=0,
                                interactive=True
                            )
                            num_max_slider = gr.Number(
                                label="Maximum Value",
                                value=100,
                                interactive=True
                            )
                    
                    # Categorical Filter
                    with gr.Column():
                        gr.Markdown("#### 🏷️ Categorical Filter")
                        categorical_col_dropdown = gr.Dropdown(
                            label="Select Categorical Column",
                            choices=["None"],
                            value="None",
                            interactive=True
                        )
                        
                        categorical_values_dropdown = gr.Dropdown(
                            label="Select Values (Multiple)",
                            choices=[],
                            multiselect=True,
                            interactive=True
                        )
                
                with gr.Row():
                    # Datetime Filter
                    with gr.Column():
                        gr.Markdown("#### 📅 Datetime Filter")
                        datetime_col_dropdown = gr.Dropdown(
                            label="Select Datetime Column",
                            choices=["None"],
                            value="None",
                            interactive=True
                        )
                        
                        with gr.Row():
                            date_start = gr.Textbox(
                                label="Start Date (YYYY-MM-DD)",
                                placeholder="e.g., 2023-01-01",
                                interactive=True
                            )
                            date_end = gr.Textbox(
                                label="End Date (YYYY-MM-DD)",
                                placeholder="e.g., 2023-12-31",
                                interactive=True
                            )
                
                with gr.Row():
                    apply_filter_btn = gr.Button("✅ Apply Filters", variant="primary", size="lg")
                    reset_filter_btn = gr.Button("🔄 Reset Filters", variant="secondary", size="lg")
                
                gr.Markdown("---")
                
                # Filter Results
                with gr.Row():
                    filter_summary = gr.Markdown()
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 📋 Filtered Data Preview (First 100 Rows)")
                        filtered_data_preview = gr.Dataframe(
                            label="Filtered Data",
                            interactive=False,
                            wrap=True
                        )
                
                gr.Markdown("---")
                
                # Export Section
                with gr.Row():
                    gr.Markdown("### 💾 Export Filtered Data")
                
                with gr.Row():
                    export_csv_btn = gr.Button("📥 Export as CSV", variant="secondary", size="lg")
                    export_status = gr.Textbox(label="Export Status", interactive=False, value="")
                
                with gr.Row():
                    csv_download = gr.File(label="Download CSV", visible=False)
                
                # Store filtered DataFrame
                filtered_df_state = gr.State(value=None)
            
            # ==================== TAB 4: VISUALIZATIONS ====================
            with gr.Tab("📊 Visualizations"):
                gr.Markdown("""
                ### 📊 Interactive Data Visualizations
                Create charts to explore patterns and relationships in your data.
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### Chart Settings")
                        
                        chart_type_dropdown = gr.Dropdown(
                            label="Select Chart Type",
                            choices=get_available_chart_types(),
                            value="Time Series",
                            interactive=True
                        )
                        
                        use_filtered_data = gr.Checkbox(
                            label="Use Filtered Data (from Filter & Explore tab)",
                            value=False
                        )
                        
                        gr.Markdown("---")
                        gr.Markdown("#### Chart Options")
                        
                        # Generic Inputs
                        viz_input_1 = gr.Dropdown(label="Input 1", choices=["None"], value="None", visible=True)
                        viz_input_2 = gr.Dropdown(label="Input 2", choices=["None"], value="None", visible=True)
                        viz_input_3 = gr.Dropdown(label="Input 3", choices=["None"], value="None", visible=True)
                        
                        generate_chart_btn = gr.Button("📈 Generate Chart", variant="primary", size="lg")
                    
                    with gr.Column(scale=2):
                        gr.Markdown("#### Chart Display")
                        chart_display = gr.Plot(label="Visualization")
                        
                        gr.Markdown("---")
                        
                        with gr.Row():
                            export_chart_btn = gr.Button("📥 Export Chart as PNG", variant="secondary", size="lg")
                            chart_export_status = gr.Textbox(label="Export Status", interactive=False, value="")
                        
                        with gr.Row():
                            chart_download = gr.File(label="Download PNG", visible=False)
                
                # Store current chart
                current_chart_state = gr.State(value=None)
            
            # ==================== TAB 5: INSIGHTS ====================
            with gr.Tab("💡 Insights"):
                gr.Markdown("""
                ### 💡 Automated Insights & Analysis
                Get AI-powered insights about patterns, trends, and anomalies in your data.
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### Options")
                        
                        use_filtered_insights = gr.Checkbox(
                            label="Analyze Filtered Data (from Filter & Explore tab)",
                            value=False
                        )
                        
                        generate_insights_btn = gr.Button(
                            "🔍 Generate Insights", 
                            variant="primary", 
                            size="lg"
                        )
                        
                        gr.Markdown("""
                        ---
                        #### About Insights
                        
                        The system automatically analyzes:
                        - **Data Quality**: Missing values, duplicates
                        - **Top/Bottom Performers**: Best and worst categories
                        - **Trends**: Growth patterns over time
                        - **Anomalies**: Outliers and unusual values
                        - **Correlations**: Strong relationships between variables
                        """)
                    
                    with gr.Column(scale=2):
                        gr.Markdown("#### Generated Insights")
                        insights_display = gr.Markdown(
                            value="*Click 'Generate Insights' to analyze your data*"
                        )
                        
                        gr.Markdown("---")
                        
                        with gr.Row():
                            export_insights_btn = gr.Button("📥 Export Insights as Text", variant="secondary", size="lg")
                            insights_export_status = gr.Textbox(label="Export Status", interactive=False, value="")
                        
                        with gr.Row():
                            insights_download = gr.File(label="Download Text File", visible=False)
                
                # Store current insights
                current_insights_state = gr.State(value=None)
        
        # ==================== EVENT HANDLERS ====================
        
        # Handle file upload
        upload_btn.click(
            fn=handle_file_upload,
            inputs=[file_upload],
            outputs=[
                df_state, 
                status_msg, 
                dataset_info, 
                head_table, 
                tail_table,
                filtered_df_state,
                filter_summary,
                filtered_data_preview,
                numerical_col_dropdown,
                categorical_col_dropdown,
                categorical_values_dropdown,
                num_min_slider,
                num_max_slider,
                datetime_col_dropdown,
                date_start,
                date_end,
                # Visualization dropdowns (reset)
                viz_input_1,
                viz_input_2,
                viz_input_3
            ]
        )
        
        # Handle statistics generation
        generate_stats_btn.click(
            fn=generate_statistics,
            inputs=[df_state],
            outputs=[
                stats_summary,
                numerical_stats_table,
                categorical_stats_table,
                missing_stats_table,
                correlation_heatmap,
                missing_values_chart
            ]
        )
        
        # Setup filters
        def setup_filter_ui(df):
            if df is None or df.empty:
                return "⚠️ No data available", gr.update(choices=["None"]), gr.update(choices=["None"]), gr.update(choices=["None"])
            
            info, col_types = setup_filters(df)
            
            num_choices = ["None"] + col_types.get('numerical', [])
            cat_choices = ["None"] + col_types.get('categorical', [])
            date_choices = ["None"] + col_types.get('datetime', [])
            
            return info, gr.update(choices=num_choices, value="None"), gr.update(choices=cat_choices, value="None"), gr.update(choices=date_choices, value="None")
        
        setup_filters_btn.click(
            fn=setup_filter_ui,
            inputs=[df_state],
            outputs=[filter_info, numerical_col_dropdown, categorical_col_dropdown, datetime_col_dropdown]
        )
        
        # Update numerical range when column changes
        numerical_col_dropdown.change(
            fn=update_numerical_range,
            inputs=[df_state, numerical_col_dropdown],
            outputs=[num_min_slider, num_max_slider, num_min_slider, num_max_slider]
        )
        
        # Update categorical choices when column changes
        categorical_col_dropdown.change(
            fn=lambda df, col: gr.update(choices=update_categorical_choices(df, col), value=[]),
            inputs=[df_state, categorical_col_dropdown],
            outputs=[categorical_values_dropdown]
        )
        
        # Update datetime range when column changes
        datetime_col_dropdown.change(
            fn=update_datetime_range,
            inputs=[df_state, datetime_col_dropdown],
            outputs=[date_start, date_end]
        )
        
        # Apply filters
        apply_filter_btn.click(
            fn=apply_filters_to_data,
            inputs=[
                df_state,
                numerical_col_dropdown,
                num_min_slider,
                num_max_slider,
                categorical_col_dropdown,
                categorical_values_dropdown,
                datetime_col_dropdown,
                date_start,
                date_end
            ],
            outputs=[filtered_df_state, filter_summary, filtered_data_preview]
        )
        
        # Reset filters
        reset_filter_btn.click(
            fn=reset_all_filters,
            inputs=[df_state],
            outputs=[
                filtered_df_state, 
                filter_summary, 
                filtered_data_preview,
                numerical_col_dropdown,
                categorical_col_dropdown,
                categorical_values_dropdown,
                num_min_slider,
                num_max_slider,
                datetime_col_dropdown,
                date_start,
                date_end
            ]
        )
        
        # Export filtered data as CSV
        def export_filtered_data(filtered_df):
            """Export filtered DataFrame to CSV."""
            if filtered_df is None or filtered_df.empty:
                return None, "⚠️ No filtered data to export. Apply filters first.", gr.update(visible=False)
            
            try:
                file_path = export_dataframe_to_csv(filtered_df, "filtered_data.csv")
                if file_path:
                    return file_path, f"✅ Exported {len(filtered_df):,} rows to CSV", gr.update(visible=True, value=file_path)
                else:
                    return None, "❌ Export failed", gr.update(visible=False)
            except Exception as e:
                return None, f"❌ Error: {str(e)}", gr.update(visible=False)
        
        export_csv_btn.click(
            fn=export_filtered_data,
            inputs=[filtered_df_state],
            outputs=[csv_download, export_status, csv_download]
        )
        
        # ==================== VISUALIZATION EVENT HANDLERS ====================
        
        # Update column choices when chart type changes
        chart_type_dropdown.change(
            fn=update_chart_ui,
            inputs=[df_state, chart_type_dropdown],
            outputs=[viz_input_1, viz_input_2, viz_input_3]
        )
        
        # Generate chart
        def create_visualization(df, filtered_df, use_filtered, chart_type, in1, in2, in3):
            # Use filtered data if checkbox is selected and filtered data exists
            data_to_use = filtered_df if (use_filtered and filtered_df is not None and not filtered_df.empty) else df
            
            fig = generate_chart(data_to_use, chart_type, in1, in2, in3)
            
            return fig, fig  # Return to both chart_display and current_chart_state
        
        generate_chart_btn.click(
            fn=create_visualization,
            inputs=[
                df_state,
                filtered_df_state,
                use_filtered_data,
                chart_type_dropdown,
                viz_input_1,
                viz_input_2,
                viz_input_3
            ],
            outputs=[chart_display, current_chart_state]
        )
        
        # Export chart as PNG
        def export_chart(chart):
            """Export chart to PNG."""
            if chart is None:
                return None, "⚠️ No chart to export. Generate a chart first.", gr.update(visible=False)
            
            try:
                file_path = export_figure_to_png(chart, "chart.png")
                if file_path:
                    return file_path, "✅ Chart exported successfully", gr.update(visible=True, value=file_path)
                else:
                    return None, "❌ Export failed", gr.update(visible=False)
            except Exception as e:
                return None, f"❌ Error: {str(e)}", gr.update(visible=False)
        
        export_chart_btn.click(
            fn=export_chart,
            inputs=[current_chart_state],
            outputs=[chart_download, chart_export_status, chart_download]
        )
        
        # ==================== INSIGHTS EVENT HANDLERS ====================
        
        def generate_insights_report(df, filtered_df, use_filtered):
            """Generate comprehensive insights report."""
            # Use filtered data if checkbox is selected and filtered data exists
            data_to_use = filtered_df if (use_filtered and filtered_df is not None and not filtered_df.empty) else df
            
            if data_to_use is None or data_to_use.empty:
                return "⚠️ No data available. Please upload a dataset first.", ""
            
            try:
                insights = generate_comprehensive_insights(data_to_use)
                report = format_insights_report(insights)
                return report, report  # Return to both display and state
            except Exception as e:
                return f"❌ Error generating insights: {str(e)}", ""
        
        generate_insights_btn.click(
            fn=generate_insights_report,
            inputs=[df_state, filtered_df_state, use_filtered_insights],
            outputs=[insights_display, current_insights_state]
        )
        
        # Export insights as text file
        def export_insights(insights_text):
            """Export insights to text file."""
            if not insights_text or insights_text.startswith("⚠️") or insights_text.startswith("❌") or insights_text.startswith("*"):
                return None, "⚠️ No insights to export. Generate insights first.", gr.update(visible=False)
            
            try:
                file_path = export_text_to_file(insights_text, "insights_report.txt")
                if file_path:
                    return file_path, "✅ Insights exported successfully", gr.update(visible=True, value=file_path)
                else:
                    return None, "❌ Export failed", gr.update(visible=False)
            except Exception as e:
                return None, f"❌ Error: {str(e)}", gr.update(visible=False)
        
        export_insights_btn.click(
            fn=export_insights,
            inputs=[current_insights_state],
            outputs=[insights_download, insights_export_status, insights_download]
        )
        
        # Footer
        gr.Markdown("""
        ---
        *Business Intelligence Dashboard v1.0 | Built with Gradio & Pandas*
        """)
    
    return demo

#For deployment
if __name__ == "__main__":
    # Create theme
    custom_theme = gr.themes.Glass(primary_hue="teal", secondary_hue="blue")
    
    demo = create_dashboard()
    demo.launch(theme=custom_theme, ssr_mode=False)

#For local testing
# if __name__ == "__main__":
#     demo = create_dashboard()
#     demo.launch(server_port=7866)
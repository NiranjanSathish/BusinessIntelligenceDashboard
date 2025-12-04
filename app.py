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
from multichart import (
    get_smart_chart_config,
    generate_dashboard_chart,
    generate_dashboard_chart_manual,
    export_all_dashboard_charts,
    get_dashboard_summary
)
from transformations import (
    create_calculated_column,
    extract_date_components,
    bin_numerical_column,
    text_transformation,
    fill_missing_values,
    get_transformation_summary
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
            
            # ==================== TAB 5: DASHBOARD VIEW ====================
            with gr.Tab("📊 Multi Chart View"):
                gr.Markdown("""
                ### 📊 Multi-Chart Dashboard
                View and configure multiple charts simultaneously for comprehensive analysis.
                """)
                
                # Dashboard controls
                with gr.Row():
                    with gr.Column(scale=2):
                        dashboard_summary = gr.Markdown()
                    
                    with gr.Column(scale=1):
                        use_filtered_dashboard = gr.Checkbox(
                            label="Use Filtered Data",
                            value=False
                        )
                        
                        auto_generate_btn = gr.Button(
                            "🎯 Smart Dashboard (Auto)", 
                            variant="primary",
                            size="lg"
                        )
                        
                        manual_generate_btn = gr.Button(
                            "⚙️ Generate (Manual Config)",
                            variant="secondary",
                            size="lg"
                        )
                
                gr.Markdown("---")
                
                # Chart 1 Configuration
                gr.Markdown("### Chart 1 Settings")
                with gr.Row():
                    chart1_selector = gr.Dropdown(
                        label="Chart Type",
                        choices=get_available_chart_types(),
                        value="Time Series",
                        scale=1
                    )
                    chart1_col1 = gr.Dropdown(label="Column 1", choices=["None"], value="None", scale=1)
                    chart1_col2 = gr.Dropdown(label="Column 2", choices=["None"], value="None", scale=1)
                    chart1_col3 = gr.Dropdown(label="Column 3 / Agg", choices=["None"], value="None", scale=1)
                
                # Chart 2 Configuration
                gr.Markdown("### Chart 2 Settings")
                with gr.Row():
                    chart2_selector = gr.Dropdown(
                        label="Chart Type",
                        choices=get_available_chart_types(),
                        value="Bar Chart",
                        scale=1
                    )
                    chart2_col1 = gr.Dropdown(label="Column 1", choices=["None"], value="None", scale=1)
                    chart2_col2 = gr.Dropdown(label="Column 2", choices=["None"], value="None", scale=1)
                    chart2_col3 = gr.Dropdown(label="Column 3 / Agg", choices=["None"], value="None", scale=1)
                
                # Chart 3 Configuration
                gr.Markdown("### Chart 3 Settings")
                with gr.Row():
                    chart3_selector = gr.Dropdown(
                        label="Chart Type",
                        choices=get_available_chart_types(),
                        value="Box Plot",
                        scale=1
                    )
                    chart3_col1 = gr.Dropdown(label="Column 1", choices=["None"], value="None", scale=1)
                    chart3_col2 = gr.Dropdown(label="Column 2", choices=["None"], value="None", scale=1)
                    chart3_col3 = gr.Dropdown(label="Column 3 / Agg", choices=["None"], value="None", scale=1)
                
                # Chart 4 Configuration
                gr.Markdown("### Chart 4 Settings")
                with gr.Row():
                    chart4_selector = gr.Dropdown(
                        label="Chart Type",
                        choices=get_available_chart_types(),
                        value="Scatter Plot",
                        scale=1
                    )
                    chart4_col1 = gr.Dropdown(label="Column 1", choices=["None"], value="None", scale=1)
                    chart4_col2 = gr.Dropdown(label="Column 2", choices=["None"], value="None", scale=1)
                    chart4_col3 = gr.Dropdown(label="Column 3 / Agg", choices=["None"], value="None", scale=1)
                
                gr.Markdown("---")
                
                # 2x2 Chart Grid
                gr.Markdown("### Dashboard Charts")
                with gr.Row():
                    with gr.Column():
                        dashboard_chart1 = gr.Plot(label="Chart 1", show_label=True)
                    with gr.Column():
                        dashboard_chart2 = gr.Plot(label="Chart 2", show_label=True)
                
                with gr.Row():
                    with gr.Column():
                        dashboard_chart3 = gr.Plot(label="Chart 3", show_label=True)
                    with gr.Column():
                        dashboard_chart4 = gr.Plot(label="Chart 4", show_label=True)
                
                gr.Markdown("---")
                
                # Export dashboard
                with gr.Row():
                    gr.Markdown("### 💾 Export All Charts")
                
                with gr.Row():
                    export_all_charts_btn = gr.Button(
                        "📥 Export All as ZIP",
                        variant="secondary",
                        size="lg"
                    )
                    export_dashboard_status = gr.Textbox(
                        label="Export Status",
                        interactive=False,
                        value=""
                    )
                
                with gr.Row():
                    dashboard_zip_download = gr.File(label="Download ZIP", visible=False)
                
                # Store dashboard charts
                dashboard_chart1_state = gr.State(value=None)
                dashboard_chart2_state = gr.State(value=None)
                dashboard_chart3_state = gr.State(value=None)
                dashboard_chart4_state = gr.State(value=None)

            # ==================== TAB 6: INSIGHTS ====================
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

            # ==================== TAB 2: DATA TRANSFORMATION ====================
            with gr.Tab("🔧 Transform Data"):
                gr.Markdown("""
                ### 🔧 Data Transformation
                Create new columns, extract date features, and transform your data.
                """)
                
                with gr.Row():
                    transformation_summary = gr.Markdown("*No transformations applied*")
                
                gr.Markdown("---")
                
                # Transformation Type Selector
                with gr.Row():
                    transform_type = gr.Dropdown(
                        label="Select Transformation Type",
                        choices=[
                            "Calculate New Column",
                            "Extract Date Components",
                            "Bin Numerical Data",
                            "Text Operations",
                            "Fill Missing Values"
                        ],
                        value="Calculate New Column",
                        interactive=True
                    )
                
                gr.Markdown("---")
                
                # Calculate New Column Section
                with gr.Group(visible=True) as calc_group:
                    gr.Markdown("#### ➕ Calculate New Column")
                    gr.Markdown("*Create a new column by combining two existing numerical columns*")
                    
                    with gr.Row():
                        calc_new_name = gr.Textbox(label="New Column Name", placeholder="e.g., Total_Revenue")
                        calc_col1 = gr.Dropdown(label="Column 1", choices=["None"], value="None")
                        calc_operation = gr.Dropdown(label="Operation", choices=["+", "-", "*", "/"], value="+")
                        calc_col2 = gr.Dropdown(label="Column 2", choices=["None"], value="None")
                    
                    calc_btn = gr.Button("➕ Create Column", variant="primary")
                
                # Extract Date Components Section
                with gr.Group(visible=False) as date_group:
                    gr.Markdown("#### 📅 Extract Date Components")
                    gr.Markdown("*Extract year, month, day, etc. from datetime columns*")
                    
                    with gr.Row():
                        date_col_select = gr.Dropdown(label="Date Column", choices=["None"], value="None")
                        date_components = gr.CheckboxGroup(
                            label="Components to Extract",
                            choices=["Year", "Month", "Day", "Quarter", "Weekday", "Week"],
                            value=["Year", "Month"]
                        )
                    
                    extract_btn = gr.Button("📅 Extract Components", variant="primary")
                
                # Bin Numerical Data Section
                with gr.Group(visible=False) as bin_group:
                    gr.Markdown("#### 📊 Bin Numerical Data")
                    gr.Markdown("*Group numerical values into ranges (e.g., age groups, price tiers)*")
                    
                    with gr.Row():
                        bin_col_select = gr.Dropdown(label="Column to Bin", choices=["None"], value="None")
                        bin_num_bins = gr.Slider(label="Number of Bins", minimum=2, maximum=10, value=5, step=1)
                    
                    with gr.Row():
                        bin_new_name = gr.Textbox(label="New Column Name", placeholder="e.g., Age_Group")
                        bin_labels = gr.Textbox(
                            label="Bin Labels (comma-separated, or 'auto')",
                            placeholder="e.g., Low,Medium,High or auto",
                            value="auto"
                        )
                    
                    bin_btn = gr.Button("📊 Create Bins", variant="primary")
                
                # Text Operations Section
                with gr.Group(visible=False) as text_group:
                    gr.Markdown("#### 🔤 Text Operations")
                    gr.Markdown("*Transform text data (uppercase, lowercase, etc.)*")
                    
                    with gr.Row():
                        text_col_select = gr.Dropdown(label="Column", choices=["None"], value="None")
                        text_operation = gr.Dropdown(
                            label="Operation",
                            choices=["uppercase", "lowercase", "title", "trim"],
                            value="uppercase"
                        )
                        text_new_name = gr.Textbox(label="New Column Name (optional)", placeholder="Leave empty to modify in-place")
                    
                    text_btn = gr.Button("🔤 Apply Text Transform", variant="primary")
                
                # Fill Missing Values Section
                with gr.Group(visible=False) as fill_group:
                    gr.Markdown("#### 🔧 Fill Missing Values")
                    gr.Markdown("*Replace missing values with calculated or specified values*")
                    
                    with gr.Row():
                        fill_col_select = gr.Dropdown(label="Column", choices=["None"], value="None")
                        fill_method = gr.Dropdown(
                            label="Method",
                            choices=["mean", "median", "mode", "forward_fill", "backward_fill", "constant"],
                            value="mean"
                        )
                        fill_value = gr.Textbox(label="Fill Value (for 'constant' method)", placeholder="e.g., 0")
                    
                    fill_btn = gr.Button("🔧 Fill Missing", variant="primary")
                
                gr.Markdown("---")
                
                # Transformation status and preview
                with gr.Row():
                    transform_status = gr.Textbox(label="Status", interactive=False, value="")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 👁️ Preview (First 10 Rows)")
                        transform_preview = gr.Dataframe(label="Before → After", interactive=False)
                
                gr.Markdown("---")
                
                # Actions
                with gr.Row():
                    apply_transform_btn = gr.Button("✅ Apply Transformation", variant="primary", size="lg")
                    undo_transform_btn = gr.Button("↩️ Undo Last", variant="secondary", size="lg")
                    download_transformed_btn = gr.Button("💾 Download Transformed Data", variant="secondary", size="lg")
                
                with gr.Row():
                    transformed_download = gr.File(label="Download CSV", visible=False)
                
                # States for transformation
                transformed_df_state = gr.State(value=None)
                original_df_backup = gr.State(value=None)
        
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

        # ==================== DASHBOARD VIEW EVENT HANDLERS ====================
        
        # Helper function to update column dropdowns based on chart type
        def update_dashboard_columns(df, chart_type):
            """Update column choices for a dashboard chart."""
            if df is None or df.empty:
                return (
                    gr.update(choices=["None"], value="None", visible=False),
                    gr.update(choices=["None"], value="None", visible=False),
                    gr.update(choices=["None"], value="None", visible=False)
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
                    gr.update(label="Value Column (Opt)", choices=num_cols, value="None", visible=True),
                    gr.update(label="Aggregation", choices=agg_methods, value="count", visible=True)
                )
            elif chart_type == 'Scatter Plot':
                return (
                    gr.update(label="X Column", choices=num_cols, value="None", visible=True),
                    gr.update(label="Y Column", choices=num_cols, value="None", visible=True),
                    gr.update(label="Color By (Opt)", choices=all_cols, value="None", visible=True)
                )
            elif chart_type == 'Correlation Heatmap':
                return (
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False)
                )
            
            return (gr.update(visible=False), gr.update(visible=False), gr.update(visible=False))
        
        # Update column dropdowns when chart type changes for each chart
        chart1_selector.change(
            fn=lambda df, ct: update_dashboard_columns(df, ct),
            inputs=[df_state, chart1_selector],
            outputs=[chart1_col1, chart1_col2, chart1_col3]
        )
        
        chart2_selector.change(
            fn=lambda df, ct: update_dashboard_columns(df, ct),
            inputs=[df_state, chart2_selector],
            outputs=[chart2_col1, chart2_col2, chart2_col3]
        )
        
        chart3_selector.change(
            fn=lambda df, ct: update_dashboard_columns(df, ct),
            inputs=[df_state, chart3_selector],
            outputs=[chart3_col1, chart3_col2, chart3_col3]
        )
        
        chart4_selector.change(
            fn=lambda df, ct: update_dashboard_columns(df, ct),
            inputs=[df_state, chart4_selector],
            outputs=[chart4_col1, chart4_col2, chart4_col3]
        )
        
        # Smart Auto-Generate Dashboard
        def generate_smart_dashboard(df, filtered_df, use_filtered):
            """Generate dashboard with smart chart selection and auto-populated columns."""
            data_to_use = filtered_df if (use_filtered and filtered_df is not None and not filtered_df.empty) else df
    
            if data_to_use is None or data_to_use.empty:
                from visualizations import create_empty_chart
                empty = create_empty_chart("No data available")
                return "", empty, empty, empty, empty, None, None, None, None
        
            from multichart import get_smart_chart_config, generate_dashboard_chart
        
            # Get smart configurations for each position
            chart1_type, config1 = get_smart_chart_config(data_to_use, 1)
            chart2_type, config2 = get_smart_chart_config(data_to_use, 2)
            chart3_type, config3 = get_smart_chart_config(data_to_use, 3)
            chart4_type, config4 = get_smart_chart_config(data_to_use, 4)
        
            # Generate all charts with smart configs
            c1 = generate_dashboard_chart(data_to_use, chart1_type, config1)
            c2 = generate_dashboard_chart(data_to_use, chart2_type, config2)
            c3 = generate_dashboard_chart(data_to_use, chart3_type, config3)
            c4 = generate_dashboard_chart(data_to_use, chart4_type, config4)
        
            summary = get_dashboard_summary(data_to_use)
        
            return summary, c1, c2, c3, c4, c1, c2, c3, c4  # Return charts twice: for display AND state
        
        auto_generate_btn.click(
            fn=generate_smart_dashboard,
            inputs=[df_state, filtered_df_state, use_filtered_dashboard],
            outputs=[
                dashboard_summary,
                dashboard_chart1,
                dashboard_chart2,
                dashboard_chart3,
                dashboard_chart4,
                dashboard_chart1_state,  # ADD THESE
                dashboard_chart2_state,
                dashboard_chart3_state,
                dashboard_chart4_state
            ]
        )
        
        # Manual Generate with Column Selection
        def generate_manual_dashboard(df, filtered_df, use_filtered,
                              c1_type, c1_1, c1_2, c1_3,
                              c2_type, c2_1, c2_2, c2_3,
                              c3_type, c3_1, c3_2, c3_3,
                              c4_type, c4_1, c4_2, c4_3):
            """Generate dashboard with manual column selections."""
            data_to_use = filtered_df if (use_filtered and filtered_df is not None and not filtered_df.empty) else df
    
            if data_to_use is None or data_to_use.empty:
                from visualizations import create_empty_chart
                empty = create_empty_chart("No data available")
                return "", empty, empty, empty, empty, None, None, None, None
            
            from multichart import generate_dashboard_chart_manual, get_dashboard_summary
            
            # Generate each chart with manual selections
            c1 = generate_dashboard_chart_manual(data_to_use, c1_type, c1_1, c1_2, c1_3)
            c2 = generate_dashboard_chart_manual(data_to_use, c2_type, c2_1, c2_2, c2_3)
            c3 = generate_dashboard_chart_manual(data_to_use, c3_type, c3_1, c3_2, c3_3)
            c4 = generate_dashboard_chart_manual(data_to_use, c4_type, c4_1, c4_2, c4_3)
            
            summary = get_dashboard_summary(data_to_use)
            
            return summary, c1, c2, c3, c4, c1, c2, c3, c4  # Return charts twice: for display AND state
        
        manual_generate_btn.click(
            fn=generate_manual_dashboard,
            inputs=[
                df_state, filtered_df_state, use_filtered_dashboard,
                chart1_selector, chart1_col1, chart1_col2, chart1_col3,
                chart2_selector, chart2_col1, chart2_col2, chart2_col3,
                chart3_selector, chart3_col1, chart3_col2, chart3_col3,
                chart4_selector, chart4_col1, chart4_col2, chart4_col3
            ],
            outputs=[
                dashboard_summary,
                dashboard_chart1,
                dashboard_chart2,
                dashboard_chart3,
                dashboard_chart4,
                dashboard_chart1_state,  # ADD THESE
                dashboard_chart2_state,
                dashboard_chart3_state,
                dashboard_chart4_state
            ]
        )
        
        # Export all dashboard charts
        def export_dashboard(c1, c2, c3, c4):
            """Export all 4 charts as ZIP."""
            
            zip_path, status = export_all_dashboard_charts(c1, c2, c3, c4)
            
            if zip_path:
                return zip_path, status, gr.update(visible=True, value=zip_path)
            else:
                return None, status, gr.update(visible=False)
        
        export_all_charts_btn.click(
            fn=export_dashboard,
            inputs=[dashboard_chart1_state, dashboard_chart2_state, dashboard_chart3_state, dashboard_chart4_state],  # ✅ CORRECT - use state
            outputs=[dashboard_zip_download, export_dashboard_status, dashboard_zip_download]
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

        # ==================== TRANSFORMATION EVENT HANDLERS ====================
        

        # Show/hide transformation groups based on selected type
        def update_transform_ui(transform_type):
            """Show only the relevant transformation group."""
            calc_vis = transform_type == "Calculate New Column"
            date_vis = transform_type == "Extract Date Components"
            bin_vis = transform_type == "Bin Numerical Data"
            text_vis = transform_type == "Text Operations"
            fill_vis = transform_type == "Fill Missing Values"
            
            return (
                gr.update(visible=calc_vis),
                gr.update(visible=date_vis),
                gr.update(visible=bin_vis),
                gr.update(visible=text_vis),
                gr.update(visible=fill_vis)
            )
        
        transform_type.change(
            fn=update_transform_ui,
            inputs=[transform_type],
            outputs=[calc_group, date_group, bin_group, text_group, fill_group]
        )
        
        # Populate column dropdowns when data is uploaded
        def populate_transform_dropdowns(df):
            """Populate dropdowns with column names."""
            if df is None or df.empty:
                return (
                    gr.update(choices=["None"]),
                    gr.update(choices=["None"]),
                    gr.update(choices=["None"]),
                    gr.update(choices=["None"]),
                    gr.update(choices=["None"]),
                    gr.update(choices=["None"])  # 6th output
                )
            
            col_types = detect_column_types(df)
            num_cols = ["None"] + col_types.get('numerical', [])
            date_cols = ["None"] + col_types.get('datetime', [])
            all_cols = ["None"] + list(df.columns)
            
            return (
                gr.update(choices=num_cols),   # calc_col1
                gr.update(choices=num_cols),   # calc_col2
                gr.update(choices=date_cols),  # date_col_select
                gr.update(choices=num_cols),   # bin_col_select
                gr.update(choices=all_cols),   # text_col_select
                gr.update(choices=all_cols)    # fill_col_select (6th output)
            )
        
        # Trigger when file is uploaded (add to upload_btn.click outputs)
        # We'll manually call this after upload
        
        # Calculate New Column
        def preview_calculated_column(df, new_name, col1, op, col2):
            """Preview calculated column before applying."""
            if df is None:
                return None, "⚠️ No data available", pd.DataFrame()
            
            temp_df, msg = create_calculated_column(df, new_name, col1, op, col2)
            
            if temp_df is not None:
                # Show preview
                preview = temp_df[[col1, col2, new_name]].head(10)
                return temp_df, msg, preview
            else:
                return None, msg, pd.DataFrame()
        
        calc_btn.click(
            fn=preview_calculated_column,
            inputs=[df_state, calc_new_name, calc_col1, calc_operation, calc_col2],
            outputs=[transformed_df_state, transform_status, transform_preview]
        )
        
        # Extract Date Components
        def preview_date_extraction(df, date_col, components):
            """Preview date component extraction."""
            if df is None:
                return None, "⚠️ No data available", pd.DataFrame()
            
            temp_df, msg = extract_date_components(df, date_col, components)
            
            if temp_df is not None:
                # Show preview of original + new columns
                new_cols = [c for c in temp_df.columns if c not in df.columns]
                preview_cols = [date_col] + new_cols
                preview = temp_df[preview_cols].head(10)
                return temp_df, msg, preview
            else:
                return None, msg, pd.DataFrame()
        
        extract_btn.click(
            fn=preview_date_extraction,
            inputs=[df_state, date_col_select, date_components],
            outputs=[transformed_df_state, transform_status, transform_preview]
        )
        
        # Bin Numerical Data
        def preview_binning(df, col, num_bins, labels, new_name):
            """Preview binning operation."""
            if df is None:
                return None, "⚠️ No data available", pd.DataFrame()
            
            temp_df, msg = bin_numerical_column(df, col, int(num_bins), labels, new_name)
            
            if temp_df is not None:
                # Show preview
                bin_col_name = new_name if new_name else f"{col}_Binned"
                preview = temp_df[[col, bin_col_name]].head(10)
                return temp_df, msg, preview
            else:
                return None, msg, pd.DataFrame()
        
        bin_btn.click(
            fn=preview_binning,
            inputs=[df_state, bin_col_select, bin_num_bins, bin_labels, bin_new_name],
            outputs=[transformed_df_state, transform_status, transform_preview]
        )
        
        # Text Operations
        def preview_text_transform(df, col, operation, new_name):
            """Preview text transformation."""
            if df is None:
                return None, "⚠️ No data available", pd.DataFrame()
            
            temp_df, msg = text_transformation(df, col, operation, new_name if new_name else None)
            
            if temp_df is not None:
                target_col = new_name if new_name else col
                preview = pd.DataFrame({
                    f'{col}_BEFORE': df[col].head(10),
                    f'{target_col}_AFTER': temp_df[target_col].head(10)
                })
                return temp_df, msg, preview
            else:
                return None, msg, pd.DataFrame()
        
        text_btn.click(
            fn=preview_text_transform,
            inputs=[df_state, text_col_select, text_operation, text_new_name],
            outputs=[transformed_df_state, transform_status, transform_preview]
        )
        
        # Fill Missing Values
        def preview_fill_missing(df, col, method, value):
            """Preview missing value fill."""
            if df is None:
                return None, "⚠️ No data available", pd.DataFrame()
            
            temp_df, msg = fill_missing_values(df, col, method, value)
            
            if temp_df is not None:
                # Show rows that had missing values
                preview = pd.DataFrame({
                    f'{col}_BEFORE': df[col].head(10),
                    f'{col}_AFTER': temp_df[col].head(10)
                })
                return temp_df, msg, preview
            else:
                return None, msg, pd.DataFrame()
        
        fill_btn.click(
            fn=preview_fill_missing,
            inputs=[df_state, fill_col_select, fill_method, fill_value],
            outputs=[transformed_df_state, transform_status, transform_preview]
        )
        
        # Apply Transformation
        def apply_transformation(original_df, transformed_df):
            """Apply the previewed transformation to the main dataset."""
            if transformed_df is None or transformed_df.empty:
                return original_df, "⚠️ No transformation to apply. Create a transformation first.", original_df
            
            summary = get_transformation_summary(original_df, transformed_df)
            
            return transformed_df, "✅ Transformation applied successfully!", summary
        
        apply_transform_btn.click(
            fn=apply_transformation,
            inputs=[df_state, transformed_df_state],
            outputs=[df_state, transform_status, transformation_summary]
        )
        
        # Undo Transformation
        def undo_transformation(backup_df):
            """Undo last transformation."""
            if backup_df is None:
                return None, "⚠️ No backup available", ""
            
            return backup_df, "↩️ Transformation undone", "No transformations applied"
        
        undo_transform_btn.click(
            fn=undo_transformation,
            inputs=[original_df_backup],
            outputs=[df_state, transform_status, transformation_summary]
        )
        
        # Download Transformed Data
        def download_transformed(df):
            """Download the transformed dataset."""
            if df is None or df.empty:
                return None, "⚠️ No data to download", gr.update(visible=False)
            
            try:
                file_path = export_dataframe_to_csv(df, "transformed_data.csv")
                if file_path:
                    return file_path, f"✅ Exported {len(df):,} rows with {df.shape[1]} columns", gr.update(visible=True, value=file_path)
                else:
                    return None, "❌ Export failed", gr.update(visible=False)
            except Exception as e:
                return None, f"❌ Error: {str(e)}", gr.update(visible=False)
        
        download_transformed_btn.click(
            fn=download_transformed,
            inputs=[df_state],
            outputs=[transformed_download, transform_status, transformed_download]
        )
        
        # Populate dropdowns when file is uploaded
        upload_btn.click(
            fn=populate_transform_dropdowns,
            inputs=[df_state],
            outputs=[calc_col1, calc_col2, date_col_select, bin_col_select, text_col_select, fill_col_select]
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
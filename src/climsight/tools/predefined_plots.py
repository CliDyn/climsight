"""
Predefined plotting tools for data_analysis_agent.

These tools generate standard climate visualizations using the same style as
the original plotting functions, with added ERA5 climatology overlay.
They save plots to the results/ folder and return paths.
"""

import os
import logging
import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

logger = logging.getLogger(__name__)


# ============================================================================
# Climate Comparison Plot Tool (same style as plot_climate_data + ERA5 overlay)
# ============================================================================

class PlotClimateComparisonArgs(BaseModel):
    """Arguments for climate comparison plot tool."""
    include_era5: bool = Field(
        default=True,
        description="Whether to include ERA5 observations overlay (if available)"
    )


def _plot_climate_comparison(
    results_dir: str,
    df_list: List[Dict],
    era5_data: Optional[Dict],
    include_era5: bool = True
) -> Dict[str, Any]:
    """
    Creates climate comparison plots using the same style as plot_climate_data,
    with optional ERA5 climatology overlay.

    Uses the exact same style as extract_climatedata_functions.plot_climate_data()
    but adds ERA5 observations and saves to files.
    """
    saved_plots = []
    os.makedirs(results_dir, exist_ok=True)

    if not df_list:
        return {
            "status": "error",
            "message": "No climate data available for plotting (empty df_list)",
            "plots": []
        }

    # Check if dataframe exists in first entry
    first_df = df_list[0].get('dataframe')
    if first_df is None:
        return {
            "status": "error",
            "message": "No climate data available for plotting (no dataframe)",
            "plots": []
        }

    # Get the list of parameters to plot (excluding 'Month' and wind component columns)
    # Same logic as original plot_climate_data
    try:
        parameters = first_df.columns.tolist()
        logger.info(f"Climate data columns: {parameters}")
    except (KeyError, AttributeError) as e:
        return {
            "status": "error",
            "message": f"Invalid climate data format: {e}",
            "plots": []
        }

    # Exclude Month and wind component columns (different models use different names)
    exclude_cols = ['Month', 'wind_u', 'wind_v', 'uas', 'vas', 'avg_10u', 'avg_10v']
    parameters_to_plot = [param for param in parameters if param not in exclude_cols]
    logger.info(f"Parameters to plot: {parameters_to_plot}")

    # ERA5 variable mapping (climate model var -> ERA5 var)
    # Supports all climate model variable naming conventions:
    # - nextGEMS/ICCP: mean2t, tp
    # - AWI_CM: tas, pr (also uses descriptive names like "Present Day Temperature")
    # - DestinE: avg_2t, avg_tprate
    era5_var_map = {
        # Temperature variables -> ERA5 t2m
        'mean2t': 't2m',      # nextGEMS, ICCP
        'tas': 't2m',         # AWI_CM, CMIP6
        'avg_2t': 't2m',      # DestinE
        't2m': 't2m',         # Direct match
        # Precipitation variables -> ERA5 tp
        'tp': 'tp',           # nextGEMS, ICCP
        'pr': 'tp',           # AWI_CM, CMIP6
        'avg_tprate': 'tp',   # DestinE
        # Wind speed - computed from ERA5 u10/v10 components
        'wind_speed': 'wind_speed_computed',  # Special marker for computed wind speed
    }

    # Also support descriptive column names (AWI-CM style)
    # Map descriptive names to ERA5 variables
    descriptive_era5_map = {
        'present day temperature': 't2m',
        'future temperature': 't2m',
        'historical temperature': 't2m',
        'present day precipitation': 'tp',
        'future precipitation': 'tp',
        'historical precipitation': 'tp',
        'present day wind speed': 'wind_speed_computed',
        'future wind speed': 'wind_speed_computed',
        'historical wind speed': 'wind_speed_computed',
    }

    # Create mapping from descriptive names to their variable type for multi-dataframe matching
    # This helps match columns across dataframes with different naming
    variable_type_map = {
        # Temperature
        'present day temperature': 'temperature',
        'future temperature': 'temperature',
        'historical temperature': 'temperature',
        'mean2t': 'temperature',
        'tas': 'temperature',
        'avg_2t': 'temperature',
        't2m': 'temperature',
        # Precipitation
        'present day precipitation': 'precipitation',
        'future precipitation': 'precipitation',
        'historical precipitation': 'precipitation',
        'tp': 'precipitation',
        'pr': 'precipitation',
        'avg_tprate': 'precipitation',
        # Wind
        'present day wind speed': 'wind',
        'future wind speed': 'wind',
        'historical wind speed': 'wind',
        'wind_speed': 'wind',
    }

    # Pre-compute ERA5 wind speed from u10/v10 if available
    era5_wind_speed = None
    if include_era5 and era5_data and 'variables' in era5_data:
        u10_data = era5_data['variables'].get('u10', {}).get('monthly_values', {})
        v10_data = era5_data['variables'].get('v10', {}).get('monthly_values', {})
        if u10_data and v10_data:
            import numpy as np
            era5_wind_speed = {}
            for month in u10_data.keys():
                if month in v10_data:
                    u = u10_data[month]
                    v = v10_data[month]
                    era5_wind_speed[month] = round(np.sqrt(u**2 + v**2), 2)
            logger.info(f"Computed ERA5 wind speed from u10/v10: {era5_wind_speed}")

    fs = 18  # Font size - same as original
    source = df_list[0].get('source', 'Climate Model')

    for param in parameters_to_plot:
        try:
            fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))

            # Get parameter name and units from extracted_vars
            var_info = df_list[0].get('extracted_vars', {}).get(param, {})
            param_full_name = var_info.get('full_name', param)
            units = var_info.get('units', '')

            # Determine the variable type for cross-dataframe matching
            param_lower = param.lower()
            var_type = variable_type_map.get(param_lower)

            # Plot ERA5 observations FIRST (as baseline) if available
            # Check both standard map and descriptive map
            era5_var = era5_var_map.get(param) or descriptive_era5_map.get(param_lower)
            era5_plotted = False
            if include_era5 and era5_data and era5_var:
                # Special handling for wind_speed - use pre-computed values
                if era5_var == 'wind_speed_computed' and era5_wind_speed:
                    months = list(era5_wind_speed.keys())
                    values = list(era5_wind_speed.values())
                    plt.plot(months, values, 'ko-', linewidth=2.5, markersize=8,
                            label='ERA5 Obs (2015-2024)', zorder=10)
                    era5_plotted = True
                # Standard ERA5 variable lookup
                elif 'variables' in era5_data and era5_var in era5_data['variables']:
                    era5_monthly = era5_data['variables'][era5_var].get('monthly_values', {})
                    if era5_monthly:
                        # Convert month names to match df format if needed
                        months = list(era5_monthly.keys())
                        values = list(era5_monthly.values())
                        plt.plot(months, values, 'ko-', linewidth=2.5, markersize=8,
                                label='ERA5 Obs (2015-2024)', zorder=10)
                        era5_plotted = True

            # Plot model data - same as original plot_climate_data
            # For each dataframe, find the matching column (may have different names)
            model_plotted = False
            for data in df_list:
                df = data.get('dataframe')
                if df is None:
                    logger.warning(f"Skipping data entry - no dataframe")
                    continue
                if 'Month' not in df.columns:
                    logger.warning(f"'Month' column missing in dataframe")
                    continue

                # Find the column to plot - try exact match first, then match by variable type
                col_to_plot = None
                if param in df.columns:
                    col_to_plot = param
                elif var_type:
                    # Find column with same variable type
                    for col in df.columns:
                        col_lower = col.lower()
                        if variable_type_map.get(col_lower) == var_type:
                            col_to_plot = col
                            break

                if col_to_plot is None:
                    logger.debug(f"No matching column for '{param}' (type: {var_type}) in dataframe columns: {df.columns.tolist()}")
                    continue

                plt.plot(df['Month'], df[col_to_plot], marker='o',
                        label=data.get('years_of_averaging', 'Model'))
                model_plotted = True
                logger.info(f"Plotted model data for {param}: {data.get('years_of_averaging', 'Model')}")

            if not model_plotted:
                logger.warning(f"No model data plotted for parameter: {param}")

            plt.title(f"{param_full_name} ({units})", fontsize=fs)
            plt.xlabel('Month', fontsize=fs)
            plt.ylabel(f"{param_full_name} ({units})", fontsize=fs)
            plt.xticks(rotation=45)
            axes.tick_params(labelsize=fs)
            axes.grid(color='k', alpha=0.5, linestyle='--')
            axes.legend(fontsize=fs)
            fig.tight_layout()

            # Save plot
            output_path = os.path.join(results_dir, f'climate_{param}.png')
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

            saved_plots.append(output_path)
            logger.info(f"Saved climate plot: {output_path}")

        except Exception as e:
            logger.error(f"Error plotting {param}: {e}")
            plt.close('all')
            continue

    return {
        "status": "success",
        "plots": saved_plots,
        "message": f"Generated {len(saved_plots)} climate comparison plots",
        "reference": f"Climate data source: {source}",
        "era5_included": include_era5 and era5_data is not None
    }


def create_plot_climate_comparison_tool(
    results_dir: str,
    df_list: List[Dict],
    era5_data: Optional[Dict] = None
):
    """Factory function to create climate comparison plot tool with bound data."""
    def plot_wrapper(include_era5: bool = True) -> Dict[str, Any]:
        return _plot_climate_comparison(results_dir, df_list, era5_data, include_era5)

    return StructuredTool.from_function(
        func=plot_wrapper,
        name="plot_climate_comparison",
        description=(
            "Generate standard climate comparison plots (temperature, precipitation, wind). "
            "Plots show climate model projections with ERA5 observations overlay. "
            "Saves plots to results/ folder. CALL THIS FIRST - these plots are mandatory."
        ),
        args_schema=PlotClimateComparisonArgs
    )


# ============================================================================
# Disaster Summary Plot Tool (same style as plot_disaster_counts)
# ============================================================================

class PlotDisasterSummaryArgs(BaseModel):
    """Arguments for disaster summary plot tool."""
    dummy: str = Field(default="", description="No arguments needed")


def _plot_disaster_summary(
    results_dir: str,
    filtered_events: Any
) -> Dict[str, Any]:
    """
    Creates disaster counts visualization using the same style as
    environmental_functions.plot_disaster_counts()
    """
    os.makedirs(results_dir, exist_ok=True)

    if filtered_events is None:
        return {
            "status": "skipped",
            "message": "No hazard/disaster data available for this location",
            "plot": None
        }

    # Check if it's a DataFrame and if it's empty
    if hasattr(filtered_events, 'empty') and filtered_events.empty:
        return {
            "status": "skipped",
            "message": "No disasters recorded for this location",
            "plot": None
        }

    try:
        # Same logic as original plot_disaster_counts
        disaster_counts = filtered_events.groupby(['year', 'disastertype']).size().unstack(fill_value=0)
        place = filtered_events['geolocation'].unique()

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plotting the bar chart - same style as original
        disaster_counts.plot(kind='bar', stacked=False, ax=ax, figsize=(10, 6), colormap='viridis')
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        plt.title('Count of different disaster types in ' + place[0] + ' over time')
        plt.xlabel('Year')
        plt.ylabel('Count')
        plt.legend(title='Disaster Type')

        output_path = os.path.join(results_dir, 'disaster_counts.png')
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        logger.info(f"Saved disaster plot: {output_path}")

        return {
            "status": "success",
            "plot": output_path,
            "message": f"Generated disaster summary plot for {place[0]}",
            "reference": "Rosvold, E.L., Buhaug, H.: GDIS, a global dataset of geocoded disaster locations. https://doi.org/10.1038/s41597-021-00846-6"
        }

    except Exception as e:
        logger.error(f"Error plotting disaster summary: {e}")
        plt.close('all')
        return {
            "status": "error",
            "message": f"Failed to generate disaster plot: {str(e)}",
            "plot": None
        }


def create_plot_disaster_summary_tool(
    results_dir: str,
    filtered_events: Any
):
    """Factory function to create disaster summary plot tool with bound data."""
    def plot_wrapper(dummy: str = "") -> Dict[str, Any]:
        return _plot_disaster_summary(results_dir, filtered_events)

    return StructuredTool.from_function(
        func=plot_wrapper,
        name="plot_disaster_summary",
        description=(
            "Generate disaster/hazard summary bar chart showing historical events by type. "
            "Saves to results/disaster_counts.png. Call this for hazard visualization."
        ),
        args_schema=PlotDisasterSummaryArgs
    )


# ============================================================================
# Population Projection Plot Tool (same style as plot_population)
# ============================================================================

class PlotPopulationProjectionArgs(BaseModel):
    """Arguments for population projection plot tool."""
    dummy: str = Field(default="", description="No arguments needed")


def _plot_population_projection(
    results_dir: str,
    pop_path: str,
    country: str
) -> Dict[str, Any]:
    """
    Creates population projection plot using the same style as
    economic_functions.plot_population()
    """
    os.makedirs(results_dir, exist_ok=True)

    if not pop_path or not country:
        return {
            "status": "skipped",
            "message": "Population data path or country not specified",
            "plot": None
        }

    if not os.path.exists(pop_path):
        return {
            "status": "skipped",
            "message": f"Population data file not found: {pop_path}",
            "plot": None
        }

    try:
        import pandas as pd

        # Import and use the original get_population function
        try:
            from economic_functions import get_population
        except ImportError:
            from ..economic_functions import get_population

        reduced_pop_data = get_population(pop_path, country)

        if reduced_pop_data is None or reduced_pop_data.empty:
            return {
                "status": "skipped",
                "message": f"No population data found for country: {country}",
                "plot": None
            }

        current_year = datetime.date.today().year

        # Same style as original plot_population
        fig, ax1 = plt.subplots(figsize=(10, 6))
        plt.grid()

        # Total population data
        ax1.plot(reduced_pop_data['Time'], reduced_pop_data['TPopulation1July'],
                label='Total Population', color='blue')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('People in thousands', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        # Life expectancy
        ax2 = ax1.twinx()
        ax2.spines.right.set_position(('axes', 1.1))
        ax2.bar(reduced_pop_data['Time'], reduced_pop_data['LEx'],
               label='Life Expectancy', color='purple', alpha=0.1)
        ax2.set_ylabel('Life expectancy in years', color='purple')
        ax2.tick_params(axis='y', labelcolor='purple')

        # Population growth data
        ax3 = ax1.twinx()
        ax3.plot(reduced_pop_data['Time'], reduced_pop_data['PopGrowthRate'],
                label='Population Growth Rate', color='green')
        ax3.set_ylabel('Population growth rate in %', color='green')
        ax3.tick_params(axis='y', labelcolor='green')

        # Net Migrations
        ax4 = ax1.twinx()
        ax4.spines.right.set_position(('axes', 1.2))
        ax4.plot(reduced_pop_data['Time'], reduced_pop_data['NetMigrations'],
                label='Net Migrations', color='black', linestyle='dotted')
        ax4.set_ylabel('Net migrations in thousands', color='black')
        ax4.tick_params(axis='y', labelcolor='black')
        ax4.axvline(x=current_year, color='orange', linestyle='--', label=str(current_year))

        # Combine legends
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines3, labels3 = ax3.get_legend_handles_labels()
        lines4, labels4 = ax4.get_legend_handles_labels()
        ax1.legend(lines + lines2 + lines3 + lines4, labels + labels2 + labels3 + labels4,
                  loc='upper left')

        plt.title(f'Population Data: {country}')
        fig.tight_layout()

        output_path = os.path.join(results_dir, 'population_projection.png')
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        logger.info(f"Saved population plot: {output_path}")

        return {
            "status": "success",
            "plot": output_path,
            "message": f"Generated population projection plot for {country}",
            "reference": "United Nations, Department of Economic and Social Affairs, Population Division: World Population Prospects 2024. https://population.un.org/"
        }

    except Exception as e:
        logger.error(f"Error plotting population: {e}")
        plt.close('all')
        return {
            "status": "error",
            "message": f"Failed to generate population plot: {str(e)}",
            "plot": None
        }


def create_plot_population_projection_tool(
    results_dir: str,
    pop_path: str,
    country: str
):
    """Factory function to create population projection plot tool with bound data."""
    def plot_wrapper(dummy: str = "") -> Dict[str, Any]:
        return _plot_population_projection(results_dir, pop_path, country)

    return StructuredTool.from_function(
        func=plot_wrapper,
        name="plot_population_projection",
        description=(
            "Generate population projection chart for the location's country. "
            "Saves to results/population_projection.png. Call this for demographic context."
        ),
        args_schema=PlotPopulationProjectionArgs
    )

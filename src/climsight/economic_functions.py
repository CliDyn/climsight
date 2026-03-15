"""
Set of functions dedicated to economic and demographic data.
This module includes tools for population studies, economic impact
assessments, and demographic trends extraction/analysis.
"""
import logging
import pandas as pd
import datetime
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

NUMERIC_POPULATION_COLUMNS = [
    'TPopulation1July',
    'PopDensity',
    'PopGrowthRate',
    'LEx',
    'NetMigrations',
]


def _normalize_positive_int(value, field_name):
    """Return a validated positive integer for window sizes and similar inputs."""
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a positive integer, got {value!r}.") from exc

    if normalized <= 0:
        raise ValueError(f"{field_name} must be a positive integer, got {value!r}.")
    return normalized


def get_population(pop_path, country):
    """
    Extracts population data (by UN) for a given country.

    Args:
    - pop_path: Path where the population data is stored
    - country: Takes the country which is returned by the geolocator

    Returns:
    - red_pop_data (pandas.DataFrame): reduced DataFrame containing present day and future values for only the following variables:
        - TPopulation1July (as of 1 July, thousands)
        - PopDensity (as of 1 July, persons per square km)
        - PopGrowthRate (percentage)
        - LEx (Life Expactancy at Birth, both sexes, in years)
        - NetMigrations (Number of Migrants, thousands)    
    """
    pop_dat = pd.read_csv(pop_path, low_memory=False)

    unique_locations = pop_dat['Location'].unique()
    my_location = country

    # check if data is available for the country that we are currently investigating
    if my_location in unique_locations:
        country_data = pop_dat[pop_dat['Location'] == country].copy()
        red_pop_data = country_data[['Time', 'TPopulation1July', 'PopDensity', 'PopGrowthRate', 'LEx', 'NetMigrations']].copy()
        red_pop_data['Time'] = pd.to_numeric(red_pop_data['Time'], errors='coerce')
        for column in NUMERIC_POPULATION_COLUMNS:
            red_pop_data[column] = pd.to_numeric(red_pop_data[column], errors='coerce')
        return red_pop_data
    else:
        print(f"No population data available {'for' +  country if country else ''}.")
        return None
    
def plot_population(pop_path, country):
    """
    Plots population data (by UN) for a given country.

    Args:
    - pop_path: Path where the population data is stored
    - country: Takes the country which is returned by the geolocator

    Returns:
    - plot: visual representation of the data distribution    
    """
    reduced_pop_data = get_population(pop_path, country)
    
    today = datetime.date.today()
    current_year = today.year

    if reduced_pop_data is not None and not reduced_pop_data.empty:
        fig, ax1 = plt.subplots(figsize=(10,6))
        plt.grid()

        # Total population data
        ax1.plot(reduced_pop_data['Time'], reduced_pop_data['TPopulation1July'], label='Total Population', color='blue')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('People in thousands', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        # life expectancy
        ax2 = ax1.twinx()
        ax2.spines.right.set_position(('axes', 1.1))
        ax2.bar(reduced_pop_data['Time'], reduced_pop_data['LEx'], label='Life Expectancy', color='purple', alpha=0.1)
        ax2.set_ylabel('Life expectancy in years', color='purple', )
        ax2.tick_params(axis='y', labelcolor='purple')

        # population growth data
        ax3 = ax1.twinx()
        ax3.plot(reduced_pop_data['Time'], reduced_pop_data['PopGrowthRate'], label='Population Growth Rate', color='green')
        ax3.set_ylabel('Population growth rate in %', color='green')
        ax3.tick_params(axis='y', labelcolor='green')

        # Net Migrations
        ax4 = ax1.twinx()
        ax4.spines.right.set_position(('axes', 1.2))
        ax4.plot(reduced_pop_data['Time'], reduced_pop_data['NetMigrations'], label='Net Migrations', color='black', linestyle='dotted')
        ax4.set_ylabel('Net migrations in thousands', color='black')
        ax4.tick_params(axis='y', labelcolor='black')
        ax4.axvline(x=current_year, color='orange', linestyle='--', label=current_year)

        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines3, labels3 = ax3.get_legend_handles_labels()
        lines4, labels4 = ax4.get_legend_handles_labels()
        ax4.legend(lines+lines2+lines3+lines4, labels+labels2+labels3+labels4, loc='center right')

        plt.title(('Population changes in ' + country))
        return fig 
    else:
        return None
    
def calc_mean(years, dataset):
    """
    Calculates the mean of every column of a dataframe over a given time period and returns those means.

    Parameters:
    years (int): The time period that one is interested in to be averaged.
    dataset (pandas data frame): The corresponding data set. It has to have a column called 'Time' in datetime format.

    Returns:
    pandas data frame: A data frame with the means calculated for the given time span.
    """
    years = _normalize_positive_int(years, 'years')

    if dataset is None:
        return None
    if dataset.empty:
        return dataset.copy()
    if 'Time' not in dataset.columns:
        raise KeyError("dataset must contain a 'Time' column.")

    working = dataset.copy()
    working['Time'] = pd.to_datetime(working['Time'], errors='coerce')
    working = working.dropna(subset=['Time']).sort_values('Time').reset_index(drop=True)

    numeric_columns = [
        column for column in working.select_dtypes(include='number').columns
        if column != 'Time'
    ]
    if not numeric_columns:
        return working[['Time']].iloc[0:0].copy()

    # Ignore rows that carry only a year marker but no actual population values.
    working = working.dropna(subset=numeric_columns, how='all')
    if working.empty:
        return pd.DataFrame(columns=['Time', *numeric_columns])

    base_year = working['Time'].dt.year.iloc[0]
    working['_bucket'] = ((working['Time'].dt.year - base_year) // years).astype(int)

    aggregated = (
        working
        .groupby('_bucket', as_index=False)
        .agg({
            'Time': 'min',
            **{column: 'mean' for column in numeric_columns},
        })
    )
    aggregated['Time'] = aggregated['Time'].dt.year.astype(int)

    return aggregated[['Time', *numeric_columns]]
 
def x_year_mean_population(pop_path, country, year_step=1, start_year=None, end_year=None):
    """
    Returns a reduced data set with the means calculated for every column over a given time span

    Parameters:
    pop_path (string): Path where the data is stored.
    country (string): The country which has been clicked on the map by the user.
    year_step (int): How many years shall be aggregated.
    start_year (int): The year from which onward population data is considered.
    end_year (int): The year until which population data is considered.

    Returns:
    pandas data frame: A data frame containing the mean population data values for a given time period.
    """
    year_step = _normalize_positive_int(year_step, 'year_step')

    # Check if start_year and end_year are within the allowed range
    if (start_year is not None and (start_year < 1950 or start_year > 2100)) or \
       (end_year is not None and (end_year < 1950 or end_year > 2100)):
        logger.warning("Start and end years must be between 1950 and 2100.")
        return None
    if start_year is not None and end_year is not None and start_year > end_year:
        logger.warning("start_year must be less than or equal to end_year.")
        return None
    
    population_xY_mean = get_population(pop_path, country)
    if population_xY_mean is None:
        logger.warning(f"No population data available for {country}.")
        return None
    column_to_remove = ['LEx', 'NetMigrations'] # change here if less / other columns are wanted
    

    if not population_xY_mean.empty:
        population_xY_mean = population_xY_mean.drop(columns=column_to_remove).copy()

        population_xY_mean['Time'] = pd.to_datetime(population_xY_mean['Time'], format='%Y', errors='coerce')
        numeric_columns = ['TPopulation1July', 'PopDensity', 'PopGrowthRate']
        for column in numeric_columns:
            population_xY_mean[column] = pd.to_numeric(population_xY_mean[column], errors='coerce')

        # Filter data based on start_year and end_year
        if start_year is not None:
            start_year = max(min(start_year, 2100), 1950)
            population_xY_mean = population_xY_mean[population_xY_mean['Time'].dt.year >= start_year]
        if end_year is not None:
            end_year = max(min(end_year, 2100), 1950)
            population_xY_mean = population_xY_mean[population_xY_mean['Time'].dt.year <= end_year]

        population_xY_mean = population_xY_mean.dropna(subset=['Time']).sort_values('Time').reset_index(drop=True)
        population_xY_mean = population_xY_mean.dropna(subset=numeric_columns, how='all')
        if population_xY_mean.empty:
            logger.warning(f"No valid population rows available for {country} after filtering.")
            return None

        combinedMean = calc_mean(year_step, population_xY_mean)
        if combinedMean is None or combinedMean.empty:
            return None

        new_column_names = {
            'TPopulation1July': 'TotalPopulationAsOf1July',
            'PopDensity': 'PopulationDensity',
            'PopGrowthRate': 'PopulationGrowthRate',
            # 'LEx': 'LifeExpectancy',
            # 'NetMigrations': 'NettoMigrations'  
        }
        combinedMean.rename(columns=new_column_names, inplace=True)

        return combinedMean
    
    else:
        return None
    

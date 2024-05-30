"""
Set of functions dedicated to economic and demographic data.
This module includes tools for population studies, economic impact
assessments, and demographic trends extraction/analysis.
"""
import pandas as pd
import datetime
import matplotlib.pyplot as plt

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
    pop_dat = pd.read_csv(pop_path)

    unique_locations = pop_dat['Location'].unique()
    my_location = country

    # check if data is available for the country that we are currently investigating
    if my_location in unique_locations:
        country_data = pop_dat[pop_dat['Location'] == country]
        red_pop_data = country_data[['Time', 'TPopulation1July', 'PopDensity', 'PopGrowthRate', 'LEx', 'NetMigrations']]
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
    years = str(years) + 'A'
    dataset.set_index('Time', inplace=True) # Set the 'Time' column as the index
    numeric_columns = dataset.select_dtypes(include='number')
    dataset = numeric_columns.resample(years).mean() # Resample the numeric data in x-year intervals and calculate the mean
    dataset.reset_index(inplace=True) # Reset the index to have 'Time' as a regular column
    dataset['Time'] = dataset['Time'].dt.year # and normal year format
    
    return dataset
 
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
    # Check if start_year and end_year are within the allowed range
    if (start_year is not None and (start_year < 1950 or start_year > 2100)) or \
       (end_year is not None and (end_year < 1950 or end_year > 2100)):
        print("Warning: Start and end years must be between 1950 and 2100.")
        return None
    
    population_xY_mean = get_population(pop_path, country)
    if population_xY_mean is None:
        print(f"No population data available for {country}.")
        return None
    column_to_remove = ['LEx', 'NetMigrations'] # change here if less / other columns are wanted
    

    if not population_xY_mean.empty:
        population_xY_mean = population_xY_mean.drop(columns=column_to_remove)

        population_xY_mean['Time'] = pd.to_datetime(population_xY_mean['Time'], format='%Y')

        # Filter data based on start_year and end_year
        if start_year is not None:
            start_year = max(min(start_year, 2100), 1950)
            population_xY_mean = population_xY_mean[population_xY_mean['Time'].dt.year >= start_year]
        if end_year is not None:
            end_year = max(min(end_year, 2100), 1950)
            population_xY_mean = population_xY_mean[population_xY_mean['Time'].dt.year <= end_year]

        # Subdivide data into two data frames. One that contains the last complete x-year period (z-times the year_step) and the rest (modulo). For each data set the mean is calculated.
        modulo_years = len(population_xY_mean['Time']) % year_step 
        lastFullPeriodYear = population_xY_mean['Time'].dt.year.iloc[-1] - modulo_years  
        FullPeriod = population_xY_mean[population_xY_mean['Time'].dt.year <= lastFullPeriodYear]
        RestPeriod = population_xY_mean[population_xY_mean['Time'].dt.year > lastFullPeriodYear]

        # calculate mean for each period
        FullPeriodMean = calc_mean(year_step, FullPeriod)
        RestPeriodMean = calc_mean(modulo_years - 1, RestPeriod)
        RestPeriodMean = RestPeriodMean.iloc[1:] # drop first row as it will be same as last one of FullPeriodMean

        combinedMean  = pd.concat([FullPeriodMean, RestPeriodMean], ignore_index=True) # combine back into one data set

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
    
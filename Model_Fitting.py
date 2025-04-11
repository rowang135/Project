import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
import matplotlib.pyplot as plt

import global_vs_sector_emissions as emissions
from Foam_Model import Build_Model

td_xr = emissions.make_td_xarray('data')
bu_xr = emissions.make_rac_xarray('GasEmissionsData_01.xlsx')

def get_bu(bu, species, start, end): 
    """
    Function to extract the Bottom-Up emissions data for a specific species

    Args:
        bu (xarray.dataset): Contains Bottom-Up emissions data
        species (str): The specific species of HFC
        start (int): The start year to extract data from
        end (int): The end year to extract data up to
    
    returns:
        rac_df (DataFrame): A DataFrame containing the Bottom-Up emissions data within the defined parameters
    """
    bu = emissions.convert_units(bu)
    rac = bu.sum(dim='Region', skipna=True)
    rac = rac.sum(dim='Sector', skipna=True)
    rac = rac.sel(Year=slice(start, end))
    rac_df = pd.DataFrame(rac[species].values.T)
    
    return(rac_df)
    
def get_td(td, species, start, end):
    """
    Function to extract the Top-Down emissions data for a specific species

    Args:
        td (xarray.dataset): Contains Bottom-Up emissions data
        species (str): The specific species of HFC
        start (int): The start year to extract data from
        end (int): The end year to extract data up to
    
    returns:
         tuple: A tuple containing two DataFrames:
            - td_df (pd.DataFrame): of top-down emissions
            - td_err_df (pd.DataFrame): of uncertainties
    """
    
    td_sel = td.sel(Year=slice(start, end))
    td_df = pd.DataFrame(td_sel[species].values.T)
    td_err_df = pd.DataFrame(td_sel[f"u{species}"].values.T)
    
    return (td_df, td_err_df)

def find_RMSE(td, other):
    """
    Function to calculate the Root Mean Squared Error between two data points
    
    Args:
        td (float): The Top-Down emissions data
        other (float): the emissions from the Rac and foam model

    Returns:
        float: The Root Mean Squared Error between the two data points
    """
    td = td.fillna(0)
    other = other.fillna(0)
    
    mse = mean_squared_error(td, other)    
    rmse = np.sqrt(mse)
    
    return rmse

def aim(params, bu, td, start, end, species, Ef=None):
    """
    Function to minimize the Root Mean Squared Error between the Top-Down model and the combination of the Bottom_up data and the foam model

    Args:
        params (array): holds the variable aprameters for the minimize function
        bu (Pandas.DataFrame): BU emissions data
        td (Pandas.DataFrame): TD emissions data
        start (int): Year to start the model
        end (int): Year to end the model
        species (str): Species to model
        Ef (float, optional): Emission Factor to use. Defaults to None.

    Returns:
        float: The Root Mean Squared Error between the two inputs
    """
    consumption, growth = params

    foam = Build_Model(start, end, species, consumption, growth, Ef)
    foam_df = pd.DataFrame(foam['emis'].values)

    all = bu.add(foam_df)
    
    return find_RMSE(td, all)

def Main_Single(species, show_graph=True):
    """
    Function to optimize the consumption and growth rate of a specific species of HFC

    Args:
        species (str): Species of HFC to find outputs for
        show_graph (bool, optional): Dictates wether a graph of outputs is shown or not. Defaults to True.

    Returns:
        fig (matplotlib.pyplot.figure, optional): A matplotlib figure showing the outputs of the model
        opt_cons, opt_growth (array, optional): The optimized consumption and growth rate of the model
    """
    start = 1990
    end = 2024

    bu = get_bu(bu_xr, species, start, end)
    td, td_err = get_td(td_xr, species, start, end)


    guess = [100, 0.05]   
    result = minimize(aim, guess, args=(bu, td, start, end, species), method='BFGS')
    opt_cons, opt_growth = result.x
    low_RMSE = result.fun

    #print(f'Cons: {opt_cons:.2f}, Growth: {opt_growth*100:.3f}%')
    #print(f'Low RMSE: {low_RMSE:.3f}')

    foam = Build_Model(start, end, species, opt_cons, opt_growth)
    foam_df = pd.DataFrame(foam['emis'].values)

    if show_graph:
        
        years = np.arange(start, end+1)
        
        fig = plt.figure(figsize=(10, 6))
        plt.plot(years, td.values.flatten(), linewidth=2, color='black', label='Top-Down Emissions')
        
        plt.fill_between(years,
                        td.values.flatten() - td_err.values.flatten(),
                        td.values.flatten() + td_err.values.flatten(),
                        alpha=0.4, color='black')
        
        plt.stackplot(years, bu.values.flatten(), 
                      foam_df.values.flatten(),
                      colors=['blue', 'crimson'],
                      alpha = 0.8,
                      labels=['Bottom-Up Emissions', 'Foam Model Emissions'])
        
        plt.xlabel('Year')
        plt.ylabel('Emissions (Gg yr-1)')
        plt.title(f'Emissions Comparison for {species}')
        plt.legend()
        plt.grid(True)
        
        return fig
    else:
        return[opt_cons, opt_growth]

def Test_Ef(species, cons_guess, growth_guess):
    """
    Function to test the effect of the Emission Factor on the input parameters of the model

    Args:
        species (array): Species of HFC to test
        cons_guess (float): Initial guess for the consumption
        growth_guess (float): Initial guess for the growth

    Returns:
        fig (matplotlib.pyplot.figure): Graph showing the outputs of the test
    """
    start = 1990
    end = 2024

    bu = get_bu(bu_xr, species, start, end)
    td, td_err = get_td(td_xr, species, start, end)

    results = []
    
    Ef_range = np.arange(5, 10, 0.1)
    
    for vals in Ef_range:
        guess = [cons_guess, growth_guess]  
        result = minimize(aim, guess, args=(bu, td, start, end, species, (vals/100)), method='BFGS')
        opt_cons, opt_growth = result.x
        low_RMSE = result.fun

        #print(f'Ef: {vals:.2f}% - cons: {opt_cons:.2f}, growth: {opt_growth*100:.3f}%')
        results.append({'Ef': vals, 'Cons': opt_cons, 'Growth': opt_growth})
    
    results_df = pd.DataFrame(results)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(results_df['Ef'], results_df['Cons'], linewidth = 3, label='Consumption', color='blue')
    ax1.set_xlabel('Emission factor (%)')
    ax1.set_ylabel('Consumption (Gg y-1)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    ax2 = ax1.twinx()
    ax2.plot(results_df['Ef'], results_df['Growth'], linewidth = 3, label='Growth', color='crimson')
    ax2.set_ylabel('Growth (%)', color='crimson')
    ax2.tick_params(axis='y', labelcolor='crimson')

    plt.title(f'Consumption change of {species} for different values of Ef')
    ax1.grid(True)
    ax1.legend(loc='center left')
    ax2.legend(loc='center right')

    return fig

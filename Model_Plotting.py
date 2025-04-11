import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Foam_Model import Build_Model
from Model_Fitting import Main_Single

def plot_Model(start, end, species, consumption = None, growth = None):
    """
    Function to plot all of the models outputs

    Args:
        start (int): Year to start the model
        end (int): year to end the model
        species (str): HFC species to model and output
        consumption (bool, optional): Initial consumption to use. Defaults to False.
        growth (bool, optional): Annual growth of consumption. Defaults to False.

    Returns:
        fig (matplotlib.pyplot.figure): Graph showing the outputs of the model
    """
    if consumption is None and growth is None:
        params = Main_Single(species, False)
    else:
        params = [consumption, growth]
    
    data = Build_Model(start, end, species, params[0], params[1])
    df = pd.DataFrame(data)
    
    fig, ax = plt.subplots(2, 2, figsize=(10,8))

    # Plotting emmisions
    ax[0, 0].plot(df['year'], df['emis'], label='Emissions', linewidth=2)
    ax[0, 0].set_title('Annual HFC Emissions')
    ax[0, 0].set_xlabel('Year')
    ax[0, 0].set_ylabel('Emissions (Gg y-1)')
    ax[0, 0].grid(True, linestyle='--', alpha=0.7)

    # Plotting bank
    ax[0, 1].plot(df['year'], df['bank'], label='Bank', linewidth=2)
    ax[0, 1].set_title('Annual HFC Bank')
    ax[0, 1].set_xlabel('Year')
    ax[0, 1].set_ylabel('Bank (Gg y-1)')
    ax[0, 1].grid(True, linestyle='--', alpha=0.7)

    # Plotting consumption
    ax[1, 0].plot(df['year'], df['cons'], label='Consumption', linewidth=2)
    ax[1, 0].set_title('Annual HFC Consumption', fontsize=14)
    ax[1, 0].set_xlabel('Year')
    ax[1, 0].set_ylabel('Consumption (Gg y-1)')
    ax[1, 0].grid(True, linestyle='--', alpha=0.7)
    
    # Plotting CO2 equivilance
    ax[1, 1].plot(df['year'], df['equiv'], label='CO2 equivalance', linewidth=2)
    ax[1, 1].set_title('Annual HFC emissions')
    ax[1, 1].set_xlabel('Year')
    ax[1, 1].set_ylabel('Emissions (Tg CO2-eq y-1)')
    ax[1, 1].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    return fig

def plot_ModelC(start, end, species, consumption = None, growth = None):
    """
    Function to plot all of the models cumulative outputs

    Args:
        start (int): Year to start the model
        end (int): year to end the model
        species (str): HFC species to model and output
        consumption (bool, optional): Initial consumption to use. Defaults to False.
        growth (bool, optional): Annual growth of consumption. Defaults to False.

    Returns:
        fig (matplotlib.pyplot.figure): Graph showing the cumulative outputs of the model
    """
    if consumption is None and growth is None:
        params = Main_Single(species, False)
    else:
        params = [consumption, growth]
    
    data = Build_Model(start, end, species, params[0], params[1])
    df = pd.DataFrame(data)
    
    fig, ax = plt.subplots(2, 2, figsize=(10,8))

    # Plotting emmisions
    ax[0, 0].plot(df['year'], df['emis'].cumsum(), label='Emissions', linewidth=2)
    ax[0, 0].set_title('Cumulative HFC Emissions Over Time')
    ax[0, 0].set_xlabel('Year')
    ax[0, 0].set_ylabel('Emissions (Gg y-1)')
    ax[0, 0].grid(True, linestyle='--', alpha=0.7)

    # Plotting bank
    ax[0, 1].plot(df['year'], df['bank'].cumsum(), label='Bank', linewidth=2)
    ax[0, 1].set_title('Cumulative HFC Bank Over Time')
    ax[0, 1].set_xlabel('Year')
    ax[0, 1].set_ylabel('Bank (Gg y-1)')
    ax[0, 1].grid(True, linestyle='--', alpha=0.7)

    # Plotting consumption
    ax[1, 0].plot(df['year'], df['cons'].cumsum(), label='Consumption', linewidth=2)
    ax[1, 0].set_title('Cumulative HFC Consumption Over Time', fontsize=14)
    ax[1, 0].set_xlabel('Year')
    ax[1, 0].set_ylabel('Consumption (Gg y-1)')
    ax[1, 0].grid(True, linestyle='--', alpha=0.7)
    
    # Plotting CO2 equivilance
    ax[1, 1].plot(df['year'], df['equiv'].cumsum(), label='CO2 equivalance', linewidth=2)
    ax[1, 1].set_title('Cumulative HFC emissions ')
    ax[1, 1].set_xlabel('Year')
    ax[1, 1].set_ylabel('Emissions (Tg CO2-eq y-1)')
    ax[1, 1].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    return fig

def plot_equiv(species, type):
    """
    Function to plot the CO2 equivalence of multiple species

    Args:
        species (array): all of the species to be tested
        type (str): Type of CO2 equivalence to be plotted. Either 'equiv' or 'bank_equiv'

    Returns:
        fig (matplotlib.pyplot.figure): Graph showing species CO2 equivalence
    """

    start = 1990
    end = 2025
    years = np.arange(start, end + 1)

    all_equiv = []
    
    # Collect model outputs for each species
    for sp in species:
        params = Main_Single(sp, False)
        data = Build_Model(start, end, sp, params[0], params[1])
        all_equiv.append(data[type].values)

    fig, ax = plt.subplots(figsize=(10, 6))

    cmap = plt.get_cmap('winter')
    colours = cmap(np.linspace(0, 1, len(species)))

    ax.stackplot(years, *all_equiv, colors=colours, labels=species)
    if type == 'equiv':
        ax.set_title('CO2 Equivalent emissions Over Time')
        ax.set_xlabel('Year')
        ax.set_ylabel('Emissions (Tg CO2-eq y-1)')
    elif type == 'bank_equiv':
        ax.set_title('Bank CO2 Equivalence Over Time')
        ax.set_xlabel('Year')
        ax.set_ylabel('Bank (Tg CO2-eq y-1)')
    ax.legend(loc='upper left')

    return fig

# *******************************************************************************************
# global_vs_sector_emissions.py
# Author: Helene De Longueville, Atmospheric Chemistry Research Group, University of Bristol
# 
# *******************************************************************************************
"""
This script, global_vs_sector_emissions.py, provides functions to analyze and visualize emissions data using both bottom-up and top-down sources. 
It includes functions to load, process, and plot emissions data for various compounds (e.g., CFCs, HCFCs) organized by sector.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import os

import Foam_Model as base

def make_rac_xarray(path):
    """
    Reads bottom-up rac data from an Excel file and converts it to an xarray Dataset.

    Args:
        path (str): File path to the Excel file.

    Returns:
        xarray.Dataset: Dataset with species-related emissions indexed by 'Year', 'Region', and 'Sector'.
    """

    def rename_columns(df, mappings):
        """Helper function to rename columns based on specified mappings."""
        for mapping in mappings:
            df.rename(columns=mapping, inplace=True)
        return df

    if 'HFC_Outlook_Global_GasBankAndEmissions_01c' in path:
        # Load data
        df = pd.read_excel(path, sheet_name='Gas_Emissions', skiprows=12, usecols='B:T')
        rename_columns(df, [
            {df.columns[1]: 'RegionCode'},
            {df.columns[2]: 'Region'},
            {df.columns[3]: 'Sector'}
        ])

        # Replace R by Gas type (e.g., R11 --> CFC-11)
        gas_mappings = [
            {col: f'CFC-{col[1:]}' for col in df.columns[4:7]},
            {col: f'HCFC-{col[1:]}' for col in df.columns[7:9]},
            {col: f'HFC-{col[1:]}' for col in df.columns[9:15]},
            {col: f'HFO-{col[1:]}' for col in df.columns[15:19]}
        ]
        df = rename_columns(df, gas_mappings).drop(columns=['RegionCode'])

        # Convert to xarray
        ds = df.set_index(['Year', 'Region', 'Sector']).to_xarray()
    
    elif 'GasEmissionsData_01' in path:
        # Get region definition
        df_region = pd.read_excel(path, sheet_name='Regions', skiprows=22, usecols='B:E')
        df_region['Region'] = df_region['Region'].apply(lambda x: x.split(" (")[0]) + ' (' + df_region['Montreal Protocol Annex'] + ' g' + df_region['Kigali Group'].astype(str) + ')'
        region_def = dict(zip(df_region['Code'], df_region['Region']))

        # Load data
        df = pd.read_excel(path, sheet_name='Data', skiprows=16, usecols='B:BN')

        # Replace R by Gas type (e.g., R11 --> CFC-11)
        df = df[df['Gas Type'] != 'Non-fluorocarbon'] # Only keep CFC, HCFC, HFC, HFO
        df['Gas Type'] = df['Gas Type'] + '-' + df['Component Gas'].str.replace('R', '')
        df = df.rename(columns={'Gas Type': 'Gas'}).drop(columns=['Component Gas'])

        # Reshape and pivot data
        df = df.melt(id_vars=['RegionCode', 'Sector', 'Gas'], var_name='Year', value_name='Emission')
        df = df.pivot_table(index=['Year', 'RegionCode', 'Sector'], columns='Gas', values='Emission').reset_index()

        # Replace RegionCode by the corresponding region definition
        df['RegionCode'] = df['RegionCode'].map(region_def)
        df = df.rename(columns={'RegionCode': 'Region'})

        # Convert to xarray
        ds = df.set_index(['Year', 'Region', 'Sector']).to_xarray()
        ds['Year'] = ds['Year'].astype(int)
    else: 
        print('Warning: Only reads `GasEmissionsData_01.xlsx` or `HFC_Outlook_Global_GasBankAndEmissions_01c.xlsx`.')
        return None

    # Replace NaN values with zero (after discussion with Ray Gluckman, empty excel cells correspond to zeros)
    ds = ds.fillna(0)
    
    # Add attributes
    ds.attrs['description'] = 'This dataset contains RAC emissions data from Gluckman Consulting.'
    ds.attrs['source'] = f'Generated from {path}.'
    ds = set_units_attributes(ds, 'Mg')

    return ds

def make_foam_xarray(start, end, species, consumption, growth):
    # Call the Foam Model and convert the emission outputs to a xarray dataset
    foam_model = base.Build_Model(start, end, species, consumption, growth)

    ds = xr.Dataset(
        {
            species: (["Year"], foam_model['emis'].values), 
        },
        coords = {
            "Year": foam_model['year'].values,
        },
    )

    ds = ds.expand_dims(dim={"Sector": ["Foam"]}, axis=1)

    # Add attributes
    ds.attrs['description'] = 'This dataset contains Foam emissions data from Rowan Sharland.'
    ds.attrs['source'] = f'Generated from Build_Model with parameters: time = , species = {species}, consumption = {consumption}, growth = {growth}.'
    ds = set_units_attributes(ds, 'Mg') # Tonne

    return ds
    
def make_td_xarray(path):
    """
    Reads top-down data for multiple species from a directory and compiles it into an xarray Dataset.

    Args:
        path (str): Directory path containing subdirectories for each species. Each species directory should contain a CSV file with annual global emissions data.

    Returns:
        xarray.Dataset: An xarray Dataset where each variable corresponds to a species, containing annual global emissions and associated uncertainty values.
    """

    ds = []

    # Filter the species list to include only gas containing 'CFC', 'HCFC', 'HFC', or 'HFO'
    species_list = [species for species in os.listdir(path)
                     if any(x in species for x in ['CFC', 'HCFC', 'HFC', 'HFO'])]

    for species in species_list:
        # Construct the path to the emissions CSV file for each species
        species_file = os.path.join(path, species, 'outputs', f'{species}_Global_annual_emissions.csv')

        # Check if the file exists
        if not os.path.exists(species_file):
            print(f"Warning: No outputs found for {species}. Skipping...")
            continue

        # Load data for the species
        df = pd.read_csv(species_file, delimiter=',', comment='#')

        # Set the index to 'Year' and convert to xarray
        df = df.set_index('Year')
        data_array = xr.Dataset(
            {
                species: (['Year'], df['Global_annual_emissions'].values),
                f'u{species}': (['Year'], df['Global_annual_emissions_1-sigma'].values)
            },
            coords={'Year': df.index.values}
        )

        ds.append(data_array)

    ds = xr.merge(ds)

    # Add attributes
    ds.attrs['description'] = "This dataset contains 12-box atmospheric model emissions data from ACRG."
    ds.attrs['source'] = f'Generated from {path}.'
    ds = set_units_attributes(ds, 'Gg/yr')

    return ds

def set_units_attributes(ds_in, new_unit):
    """
    Sets the `units` attribute for all data variables in an xarray Dataset,
    as well as the global `units` attribute of the Dataset itself.

    Args:
        ds_in (xarray.Dataset): Dataset whose variables and global attributes will be updated.
        new_unit (str): New value for the `units` attribute to be applied to all variables and the global attribute.

    Returns:
        xarray.Dataset: Dataset with the `units` attribute modified for all variables and the global Dataset.
    """
    ds = ds_in.copy()
    for var_name in ds.data_vars:
        ds[var_name].attrs["units"] = new_unit

    ds.attrs["units"] = new_unit

    return ds

def convert_units(ds_in, mass_units = 'Gg', CO2eq = False):
    """
    Converts the mass units and optionally the emissions to CO2-equivalent in an xarray.Dataset.

    Args:
        ds_in (xarray.Dataset): Input dataset with emissions data.
        mass_units (str): Target mass units, either 'Gg' or 'Mg'. Default is 'Gg'.
        CO2eq (bool): Whether to convert to CO2-equivalent emissions. Default is False.

    Returns:
        xarray.Dataset: Dataset with converted units, or None if validation fails.
    """
    ds = ds_in.copy()

    # Extract mass units from attributes
    ds_units = ds.attrs['units']
    if len(ds_units) < 2:
        raise ValueError("Dataset units must have at least two characters to indicate mass units.")

    ds_mass = ds_units[:2]
    valid_mass_units = {'Gg', 'Mg'}

    # Validate mass units
    if ds_mass not in valid_mass_units:
        raise ValueError(f"Mass units must be one of {valid_mass_units} and specified in the first two characters of `ds.attrs['units']`.")

    if mass_units not in valid_mass_units:
        raise ValueError(f"'mass_units' must be one of {valid_mass_units}.")

    # Convert mass units if necessary
    if ds_mass != mass_units:
        conversion_factor = 1000 if ds_mass == 'Gg' and mass_units == 'Mg' else 0.001
        ds = ds * conversion_factor

    label = f'{mass_units} yr-1'

    # Handle CO2-equivalence conversion
    gwp = pd.read_csv('GWP.csv', delimiter=',', comment='#')
    gwp_dict = dict(zip(gwp['Species'], gwp['GWP']))

    if CO2eq:
        label = f'{mass_units} CO2-eq yr-1'

        if 'CO2-eq' in ds_units:
            print('Warning: Emission units are already CO2-equivalent.')
        else:
            for var_name in ds.data_vars:
                if var_name.startswith('u'):
                    species = var_name[1:] # handle uncertainties variables
                else:
                    species = var_name
                if species in gwp_dict:
                    ds[var_name] = ds[var_name] * gwp_dict[species]
                else:
                    print(f'No GWP found for {species}. Removed from Dataset.')
                    ds = ds.drop(var_name)

    else:
        if 'CO2-eq' in ds_units:
            for var_name in ds.data_vars:
                if var_name.startswith('u'):
                    species = var_name[1:] # handle uncertainties variables
                else:
                    species = var_name
                if species in gwp_dict:
                    ds[var_name] = ds[var_name] / gwp_dict[species]
                else:
                    print(f'No GWP found for {species}. Removed from Dataset.')
                    ds = ds.drop(var_name)

    # Update units attribute
    ds = set_units_attributes(ds, label)

    return ds

def make_sector_plot(rac_in=None, foam_in=None, td_in=None, CO2eq=False, species=None, start_date=None, end_date=None, foam_end=None, ylim_max=None):
    """
    Generates a time series plot of emissions data from bottom-up (BU) and top-down (TD) sources,
    with options to display by regions, sectors, or total.

    Args:
        rac_in (xarray.Dataset, optional): Bottom-up RAC emissions dataset, generated by make_bu_xarray.
        foam_in (xarray.Dataset, optional): Foam emissions dataset, generated by make_foam_xarray.
        td_in (xarray.Dataset, optional): Top-down emissions dataset, generated by make_td_xarray.
        CO2eq (bool, optional): Whether to convert to CO2-equivalent emissions. Default is False.
        species (str, optional): Species to plot.
        start_date (int, optional): Start year for the plot.
        end_date (int, optional): End year for the plot.
        ylim_max (float, optional): Maximum limit for the y-axis (emissions in Gg/year).

    Returns:
        matplotlib.figure.Figure: A matplotlib figure containing the emissions plot.
    """

    fig, ax = plt.subplots(figsize=(10, 6))

    if isinstance(species, str):
        if rac_in is not None: # Add bottom-up emissions
            rac = rac_in.copy()
            rac = rac.sum(dim='Region', skipna=True, keep_attrs=True)
            rac = rac[species].to_dataset()
            rac.attrs.update(rac_in.attrs)
            rac = convert_units(rac, CO2eq = CO2eq)
            units = rac.attrs['units']
            rac = rac.sel(Year=slice(start_date, end_date))
            
            num = rac[species].values.T.shape[0]
            rac_labels = rac['Sector'].values
            rac_data = rac[species].values.T
            ax.stackplot(rac['Year'], rac_data, labels=rac_labels)

        if foam_in is not None: # Add foam emissions
            foam = foam_in.copy()
            foam = foam[species].to_dataset()
            foam.attrs.update(foam_in.attrs)
            foam = convert_units(foam, CO2eq = CO2eq)
            units = foam.attrs['units']
            foam = foam.sel(Year=slice(start_date, end_date))

            foam_labels = foam['Sector'].values
            foam_data = foam[species].values.T

            if rac_in is None:
                ax.stackplot(foam['Year'], foam_data, labels=foam_labels, color='white', hatch='....', edgecolor='magenta')
            else:
                all_rac = rac.sum(dim='Sector', skipna=True)
                foam_aligned, all_rac_aligned = xr.align(foam, all_rac, join='outer')
                summed_values = (all_rac_aligned + foam_aligned).sum(dim='Sector', skipna=False)
                ax.fill_between(foam_aligned['Year'], all_rac_aligned[species].values, summed_values[species].values, color='white', hatch='....', edgecolor='magenta', label='Foam')

        if td_in is not None: # Add top-down emissions
            td = td_in.copy()
            td = convert_units(td, CO2eq = CO2eq)
            units = td.attrs['units']
            td = td.sel(Year=slice(start_date, end_date))

            ax.plot(td['Year'], td[species].values, label='py12box_agage', color='black')
            ax.fill_between(td['Year'], td[species].values-td[f'u{species}'].values, td[species].values+td[f'u{species}'].values, alpha=0.2, color='black')
        
        # Add labels, title, legend
        ylabel = f'{species} ({units})'
        if 'CO2-eq' in ylabel:
            ylabel = ylabel.replace('CO2', r'CO$_2$')
        if 'yr-1' in ylabel:
            ylabel = ylabel.replace('yr-1', r'yr$^{-1}$')
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Date')
        ax.set_title('Global sector emissions')
        ax.legend(loc='upper left')

    ax.set_xlim(start_date, end_date)
    if ylim_max is not None:
        ax.set_ylim(0, ylim_max)
    else:
        ax.set_ylim(bottom=0)

    return fig

def save_fig(fig, dir, species=None):
    """
    Saves a matplotlib figure to a specified directory with a structured filename based on
    the species and figure title.

    Args:
        fig (matplotlib.figure.Figure): Figure to be saved.
        dir (str): Directory path where the figure will be saved.
        species (str): Species name.
    """

    species = species.replace('-', '')
    fig_title = fig._suptitle.get_text().lower().replace(' ', '_')

    output_path = f'{dir}/{species}/{species}_{fig_title}.png'   
    output_dir = os.path.dirname(output_path)

    # Create the directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    fig.savefig(output_path,bbox_inches='tight',pad_inches=0.2,dpi=300)

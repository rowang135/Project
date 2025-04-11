import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Imports all of the species specif data needed for the code
data = pd.read_csv('HFC_info.csv')
HFC_info = pd.DataFrame(data)

def calc_Bank(bank, cons, emis):
    """
    Function to calculate the bank of HFCs

    Args:
        bank (float): The previous bank of HFCs in tonnes
        cons (float): The consumption of HFCs in tonnes
        emis (float): The emissions of HFCs in tonnes
    
    Returns:
        new_bank (float): The new bank of HFCs in tonnes   
    """
    
    new_bank = bank + (cons - emis)
    
    return(new_bank)
                          
def calc_emis(Ef, bank, cons):
    """
    Function to calculate the HFC emissions

    Args:
        Ef (float): The Emission factor
        bank (float): The bank of HFCs in tonnes
        cons (float): The consumption of HFCs in tonnes
    
    Returns:
        new_emis (float): The new HFC emssions in Tonnes
    """
    new_emis = Ef * (bank + cons)
    return(new_emis)

def Build_Model(start, end, species, consumption, growth, Emission_Factor=None):
    """
    Function to compute the estimated emmisions

    Args:
        start (int): The start year of the model
        end (int): The end year of the model
        species (str): The species of HFCs
        consumption (float): The annual consumption of HFCs in tonnes
        growth (float): The growth rate of consumption annually
        Emission_Factor (float): The species specific emission factor
    
    Returns:
        df (DataFrame): A DataFrame containing the model outputs   
    """
    # Extracting the species specific data
    HFC_data = HFC_info[HFC_info['Species'] == species].iloc[0] 
    GWP = float(HFC_data['GWP'])
    
    # Checking if the emission factor is provided otherwise standard 5% is used
    if Emission_Factor is not None:
        Ef = Emission_Factor
    else:
        Ef = float(HFC_data['Ef'])
    
    # Creating the initial data dictionary
    data = {'year' : [start],
            'bank' : [0],
            'cons' : [consumption],
            'emis' : [calc_emis(Ef, 0, consumption)],
            'bank_equiv' : [0]}
    
    data['equiv'] = [((GWP * data['emis'][0])/1000)]

    # Iterating the calculation functions to find the model outputs
    for i in range(end - start):
        # Extracting the previous years data
        prev_bank = data['bank'][-1]
        prev_emis = data['emis'][-1]
        prev_cons = data['cons'][-1]

        # Calculating the new years data
        new_cons = prev_cons * (1 + growth)
        new_bank = calc_Bank(prev_bank, prev_cons, prev_emis)
        new_emis = calc_emis(Ef, new_bank, new_cons)

        # Appending the new years data to the dictionary
        data['year'].append(data['year'][-1] + 1)
        data['bank'].append(new_bank)
        data['emis'].append(new_emis)
        data['cons'].append(new_cons)
        data['equiv'].append((GWP * new_emis)/1000)
        data['bank_equiv'].append((GWP * new_bank)/1000)
    
    # Creating a DataFrame from the dictionary
    df = pd.DataFrame(data)
    return df

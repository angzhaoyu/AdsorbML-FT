"""
Microkinetic Modeling (MK) Module

This module provides functions for microkinetic modeling of catalytic reactions.
It calculates steady-state surface coverages and reaction rates based on a set of
elementary reactions and their rate constants.

Functions:
- get_ra(R, P, k): Calculate reaction rate for a given reaction
- get_RP(eq_name, dict): Extract reactants and products from a reaction equation
- find_k(df_Eak, eq_1): Find rate constants for a given reaction
- get_ads(eq_name): Identify adsorbates and small molecules in a reaction
- initial_ads(eqs): Initialize adsorbate concentrations
- ad_bound(y, dict_bound): Apply boundary conditions
- MK(dict_ads, eqs, df, dict_bound): Calculate steady-state coverages and reaction rates
"""

import numpy as np
import pandas as pd
import json
from scipy import optimize
import copy

def get_RP(eq_name, dict):
    """
    Extract reactants and products from a reaction equation.
    
    Args:
        eq_name (str): Reaction equation in the format "R1+R2-P1+P2"
        dict (dict): Dictionary of concentrations
        
    Returns:
        tuple: (R, P) dictionaries of reactant and product concentrations
    """
    R_name, P_name = eq_name.split("-")
    R_name = R_name.split("+")
    P_name = P_name.split("+")
    R = {}
    for R_n in R_name:
        R[R_n] = dict[R_n]
    P = {}
    for P_n in P_name:
        P[P_n] = dict[P_n]

    return R, P

def find_k(df_Eak, eq_1):
    """
    Find rate constants for a given reaction.
    
    Args:
        df_Eak (DataFrame): DataFrame containing rate constants
        eq_1 (str): Reaction equation
        
    Returns:
        dict: Dictionary of rate constants
    """
    eq_name = eq_1.split("-")
    R_name = sorted(eq_name[0].split("+"))
    P_name = sorted(eq_name[1].split("+"))
    R_ns = df_Eak["R"].values
    P_ns = df_Eak["P"].values
    number = None
    for i in range(len(df_Eak)):
        R_n = sorted(R_ns[i].split(","))
        P_n = sorted(P_ns[i].split(","))
        if R_n == R_name and P_n == P_name:
            number = i
    if number is None:
        print(f"eq cannot find k in the table,{eq_1}")
    k = df_Eak["k"].values[number]
    k = json.loads(k)
    Ra_name = ""
    if len(R_name) == 1:
        Ra_name = R_name[0]
    else:
        Ra_name = R_name[0] + "+" + R_name[1]
    Pa_name = ""
    if len(P_name) == 1:
        Pa_name = P_name[0]
    else:
        Pa_name = P_name[0] + "+" + P_name[1]
    K = {}
    K[f"{Ra_name}-{Pa_name}"] = k[0]
    K[f"{Pa_name}-{Ra_name}"] = k[1]
    return K

def get_ra(R, P, k):
    """
    Calculate reaction rate for a given reaction.
    
    Args:
        R (dict): Reactant concentrations
        P (dict): Product concentrations
        k (dict): Rate constants
        
    Returns:
        dict: Reaction rate
    """
    R_values = list(R.values())
    P_values = list(P.values())
    if len(R_values) == 1:
        R["1"] = 1
    if len(P_values) == 1:
        P["1"] = 1
    R_keys = list(R.keys()) 
    R_values = list(R.values())
    P_keys = list(P.keys()) 
    P_values = list(P.values())
    k_values = list(k.values())
    k_keys = list(k.keys())
    ri_value = (k_values[0]*R_values[0]*R_values[1])-(k_values[1]*P_values[0]*P_values[1])
    ri_name = f"({k_keys[0]}.{R_keys[0]}.{R_keys[1]}-{k_keys[1]}.{P_keys[0]}.{P_keys[1]})"
    Ri = {ri_name:ri_value}
    return Ri 

def get_ads(eq_name):
    """
    Identify adsorbates and small molecules in a reaction.
    
    Args:
        eq_name (str): Reaction equation
        
    Returns:
        tuple: (ads_name, moles_name) lists of adsorbates and molecules
    """
    R_name, P_name = eq_name.split("-")
    R_name = R_name.split("+")
    P_name = P_name.split("+")
    names = list(tuple(R_name)+tuple(P_name))
    ads_name = []
    moles_name = []
    for name in names:
        if len(name)>1 and "*" in name:
            ads_name.append(name)
        else:
            if name != "*":
                moles_name.append(name)
    return ads_name, moles_name

def initial_ads(eqs):
    """
    Initialize adsorbate concentrations.
    
    Args:
        eqs (list): List of reaction equations
        
    Returns:
        dict: Dictionary of initial adsorbate concentrations
    """
    if len(eqs) == 1:
        print("eqs == 1")
    ads_all = ()
    for i in range(len(eqs)):
        ads_name = get_ads(eqs[i])[0]
        ads_all = ads_all+tuple(ads_name)
    my_list = list(ads_all)
    unique_list = []
    for item in my_list:
        if item not in unique_list:
            unique_list.append(item)
    dict = {}
    for i in range(len(unique_list)):
        dict[unique_list[i]] = 0
    return dict

def ad_bound(y, dict_bound):
    """
    Apply boundary conditions.
    
    Args:
        y (dict): Dictionary of concentrations
        dict_bound (dict): Dictionary of boundary conditions
        
    Returns:
        dict: Updated dictionary with boundary conditions applied
    """
    for key, value in dict_bound.items():
        y[key] = value
    return y

def get_RP_from_ads_moles(eq, ads_value, dict_ads, dict_bound):
    """
    Extract reactants and products from adsorbate and molecule concentrations.
    
    Args:
        eq (str): Reaction equation
        ads_value (list): List of adsorbate concentrations
        dict_ads (dict): Dictionary of adsorbate names
        dict_bound (dict): Dictionary of boundary conditions
        
    Returns:
        tuple: (R, P) dictionaries of reactant and product concentrations
    """
    eq_name = eq.split("-")
    R_name = sorted(eq_name[0].split("+"))
    P_name = sorted(eq_name[1].split("+"))
    star_ = 1 - sum(ads_value)
    ads_key = list(dict_ads.keys())
    mols_key = list(dict_bound.keys())
    R = {}
    for R_n in R_name:
        if R_n == "*":
            R["*"] = star_
        if R_n in ads_key:
            n = ads_key.index(R_n)
            R[R_n] = (ads_value[n])
        if R_n in mols_key:
            R[R_n] = (dict_bound[R_n])
    P = {}
    for P_n in P_name:
        if P_n == "*":
            P["*"] = star_
        if P_n in ads_key:
            n = ads_key.index(P_n)
            P[P_n] = (ads_value[n])
        if P_n in mols_key:
            P[P_n] = (dict_bound[P_n])
    
    return R, P

def R_net_fslove(ads_value, dict_ads, eqs, dict_bound, df_Eak, A):
    """
    Calculate reaction rates for fsolve function.
    
    Args:
        ads_value (list): List of adsorbate concentrations
        dict_ads (dict): Dictionary of adsorbate names
        eqs (list): List of reaction equations
        dict_bound (dict): Dictionary of boundary conditions
        df_Eak (DataFrame): DataFrame containing rate constants
        A (float): Pre-exponential factor
        
    Returns:
        list: List of reaction rates
    """
    r_ads = []
    ads_names = list(dict_ads.keys())
    star_ = 1 - sum(ads_value) 
    dict_bound["*"] = star_
    for ads_name in ads_names:
        Ri = {}
        for eq in eqs:     
            eq_name = eq.split("-")
            R_name = sorted(eq_name[0].split("+"))
            P_name = sorted(eq_name[1].split("+"))
            if ads_name in R_name:
                R, P = get_RP_from_ads_moles(eq, ads_value, dict_ads, dict_bound)
                k = find_k(df_Eak, eq)
                k_keys = list(k.keys())
                k_values = list(k.values())
                k_values = [x * A for x in k_values]
                k = dict(zip(k_keys, k_values))
                ri = get_ra(R, P, k)
                ri_values = list(ri.values())[0]
                ri_keys = "-"+list(ri.keys())[0]
                Ri[ri_keys] = -ri_values
                
            if ads_name in P_name:
                R, P = get_RP_from_ads_moles(eq, ads_value, dict_ads, dict_bound)
                k = find_k(df_Eak, eq)
                k_keys = list(k.keys())
                k_values = list(k.values())
                k_values = [x * A for x in k_values]
                k = dict(zip(k_keys, k_values))
                ri = get_ra(R, P, k)
                ri_values = list(ri.values())[0]
                ri_keys = list(ri.keys())[0]
                Ri[ri_keys] = ri_values
        Ri_values = list(Ri.values())
        Rads_value = sum(Ri_values)
        r_ads.append(Rads_value)
    return r_ads

def MK(dict_ads, eqs, df, dict_bound):
    """
    Calculate steady-state coverages and reaction rates.
    
    Args:
        dict_ads (dict): Dictionary of adsorbate names
        eqs (list): List of reaction equations
        df (DataFrame): DataFrame containing rate constants
        dict_bound (dict): Dictionary of boundary conditions
        
    Returns:
        tuple: (dict_ads_MK, ads_dis, Ras) - coverages, adsorbate values, and reaction rates
    """
    ads_value = list(dict_ads.values())
    ads_value = [0]*len(ads_value)
    A = 1
    ads_dis = optimize.fsolve(R_net_fslove, ads_value, args=(dict_ads, eqs, dict_bound, df, A))
    ads_keys = list(dict_ads.keys())
    dict_ads_2 = dict(zip(ads_keys, ads_dis))
    A = 6212437991620.12
    ads_dis = optimize.fsolve(R_net_fslove, ads_dis, args=(dict_ads_2, eqs, dict_bound, df, A))
    ads_keys = list(dict_ads.keys())
    dict_ads_MK = dict(zip(ads_keys, ads_dis))
    Ras = R_net(ads_dis, dict_ads_MK, eqs, dict_bound, df, A)

    return dict_ads_MK, ads_dis, Ras

def R_net(ads_value, dict_ads, eqs, dict_bound, df_Eak, A):
    """
    Calculate reaction network.
    
    Args:
        ads_value (list): List of adsorbate concentrations
        dict_ads (dict): Dictionary of adsorbate names
        eqs (list): List of reaction equations
        dict_bound (dict): Dictionary of boundary conditions
        df_Eak (DataFrame): DataFrame containing rate constants
        A (float): Pre-exponential factor
        
    Returns:
        list: List of reaction rates
    """
    R_net = []
    dict_ads_moles = ad_bound(copy.deepcopy(dict_ads), dict_bound)
    ads_names = list(dict_ads_moles.keys())
    star_ = 1 - sum(ads_value) 
    dict_bound["*"] = star_
    for ads_name in ads_names:
        Ri = {}
        for eq in eqs:     
            eq_name = eq.split("-")
            R_name = sorted(eq_name[0].split("+"))
            P_name = sorted(eq_name[1].split("+"))
            if ads_name in R_name:
                R, P = get_RP_from_ads_moles(eq, ads_value, dict_ads, dict_bound)
                k = find_k(df_Eak, eq)
                k_keys = list(k.keys())
                k_values = list(k.values())
                k_values = [float(x * A) for x in k_values]
                k = dict(zip(k_keys, k_values))
                ri = get_ra(R, P, k)
                ri_values = list(ri.values())[0]
                ri_keys = "-"+list(ri.keys())[0]
                Ri[ri_keys] = -ri_values
                
            if ads_name in P_name:
                R, P = get_RP_from_ads_moles(eq, ads_value, dict_ads, dict_bound)
                k = find_k(df_Eak, eq)
                k_keys = list(k.keys())
                k_values = list(k.values())
                k_values = [x * A for x in k_values]
                k = dict(zip(k_keys, k_values))
                ri = get_ra(R, P, k)
                ri_values = list(ri.values())[0]
                ri_keys = list(ri.keys())[0]
                Ri[ri_keys] = ri_values           
        try:        
            Ri_keys = list(Ri.keys())
            Ri_values = list(Ri.values())
            Rads_value = sum(Ri_values)
            Rads_key = Ri_keys[0]
            if len(Ri_keys) != 1:
                for Ri_key in Ri_keys[1:]:
                    Rads_key = f"{Rads_key} + {Ri_key}"
            a = {Rads_key:Rads_value}
            b = [ads_name, a]
            R_net.append(b)
        except: 
            print(f"{ads_name} is not in eqs")
        

    return R_net

def cal_coverge(df_1, dict_bound, ads):
    """
    Calculate coverage and reaction rates.
    
    Args:
        df_1 (DataFrame): DataFrame containing reaction data
        dict_bound (dict): Dictionary of boundary conditions
        ads (list): List of adsorbates to analyze
        
    Returns:
        tuple: (coverage, rate) - coverages and reaction rates
    """
    eqs = df_1["Reaction"].tolist()
    df_1["R"] = df_1["Reaction"].apply(lambda x: str(x.split("-")[0].split("+")).replace('[', '').replace(']', '').replace("'", '').replace(" ", '') if isinstance(x, str) else None)
    df_1["P"] = df_1["Reaction"].apply(lambda x: str(x.split("-")[1].split("+")).replace('[', '').replace(']', '').replace("'", '').replace(" ", '') if isinstance(x, str) else None)
    df_1["k"] = df_1['k'].apply(lambda x: f"[{x}]" if isinstance(x, str) else None)
    dict_ads = initial_ads(eqs)
    coverage, list_ads, list_dict_R = MK(dict_ads, eqs, df_1, dict_bound)
    r_a = {}
    for lis in list_dict_R:
        if lis[0] in ads:
            r_a[lis[0]] = list(lis[1].values())[0]
    rate = {x:r_a[x] for x in ads}

    return coverage, rate

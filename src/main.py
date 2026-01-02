# Download data from the EIA 

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as md
import numpy as np
import os
import datetime
import requests
import json
from Function_EIA_API_call_by_Demand_Net_Gen import get_EIA_data_demand
from Function_EIA_API_call_by_Energy_Source import get_EIA_data_energy_source
from Function_dynamic_demand_analysis import get_demand_analysis

# ********************************************    Get the data from EIA *******************************
# String for Hourly Demand, Demand Forecast, Generation and Interchange is /region-date/
# String for Hourly Generation by Energy Source is /fuel-type-data/
# String for Hourly Demand by Subregion  is /region-sub-ba-data/
fname = 'SRP_'                                                                  
api_key = "WHVN3lll1Aj3ntm8iwfnMpbRefyBAx2N4fEC2kzi"                            
base_url = 'https://api.eia.gov/v2/electricity/rto/region-data/data/'          
fname_type = 'Demand'                                                           
file_flag = 1                                                                   
dowload_flag = 0  # set to 0 if reading CSV from local file

offset = 0
length = 5000
# MUST BE IN ASCENDING ORDER
#demand_division = [4, 7]
demand_division = [3, 5, 6.5, 8]

# Setting the start and end data
#time_to_add = pd.Timedelta(weeks=0, days=5, hours=0, minutes=0)
#start_date = pd.Timestamp('2023-10-04 15:30:00')
#end_date = start_date + time_to_add
start_date = '2023-12-30T00'
end_date = '2025-02-01T00'
print("The start and end dates are {} and {}".format(start_date, end_date))

params1 = {
    'frequency': 'hourly',  # hourly gives answer in UTC, local-hourly does local time
    'data[0]': 'value',     # the [0] is important; doesn't work otherwise
    #'facets[respondent][]': 'AZPS',
    'facets[respondent][]': 'SRP',
    'start': start_date, 
    'end': end_date,
    'sort[0][column]': 'period',
    'sort[0][direction]': 'desc',
    'offset': offset,
    'length': length,
    'api_key': api_key
}

# -------------------- Get demand data from either EIA or file --------------------
if dowload_flag == 1:
    total_DF = get_EIA_data_demand(fname, api_key, base_url, fname_type, params1, file_flag)
    print("\n Data downloaded from EIA:")
    #print(total_DF.head())
    # -------------------- Save downselected dataset --------------------
    with open(fname + fname_type + '_downselect.csv', 'w') as file:
        total_DF.to_csv(file, index=False)
        print("\nData written to file:", fname + fname_type + '_downselect.csv')
else:
    total_DF = pd.read_csv(fname + fname_type + '.csv')
    # Standardize column names from your CSV
    #total_DF.rename(columns={
    #    'Timestamp (Hour Ending)': 'period',
    #    'Demand (MWh)': 'value'
    #}, inplace=True)
    # Convert columns to correct types
    #total_DF['period'] = pd.to_datetime(total_DF['period'], errors='coerce')
    #total_DF['value'] = pd.to_numeric(total_DF['value'], errors='coerce')
    #total_DF['type'] = 'D'

# -------------------- Clean dataframe --------------------
total_DF['period'] = pd.to_datetime(total_DF['period'], errors='coerce')
total_DF['hour'] = total_DF['period'].dt.hour       #  keeps hour separately for plotting
total_DF['date'] = total_DF['period'].dt.date       #  daily grouping ready

# Filter only demand rows (type = D)
df1 = total_DF[total_DF['type'] == 'D']

# Break up dataframe by month
df1_m1 = df1[df1['period'].dt.month == 7]

df1_ds = df1_m1.copy()
print('\n Demand data filtered:')
print(df1_ds.head())
df1_test = df1[df1['period'].dt.year == 2024]

get_demand_analysis(df1_test, 2024, demand_division, 0)

# -------------------- Daily min and max calculation --------------------
hourly_stats = df1_ds.groupby('hour')['value'].agg(['min', 'max']).reset_index()
#print(hourly_stats.head())

# -------------------- FIGURE 1: Demand (daily traces + mean + min/max) --------------------
plt.figure(figsize=(13, 5))

# Plot all daily traces faintly
for day, values in df1_ds.groupby('date'):
    plt.plot(values['hour'], values['value'], alpha=0.15, color='blue')

# Plot mean line
mean_demand = df1_ds.groupby('hour')['value'].mean()
plt.plot(range(24), mean_demand, marker='o', label='Mean Demand', color='black')

# Add min/max bubble points
plt.scatter(hourly_stats['hour'], hourly_stats['min'], s=80, color='red', label='Min', edgecolors='black', alpha=0.8)
plt.scatter(hourly_stats['hour'], hourly_stats['max'], s=80, color='green', label='Max', edgecolors='black', alpha=0.8)

plt.title('FIGURE 1: Hourly Electricity Demand (SRP) from {} to {}'.format(start_date, end_date), fontsize=14, pad=10)
plt.ylabel('MWh', fontsize=12)
plt.xlabel('Hour of Day', fontsize=12)
plt.xticks(range(0, 24))
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()

############################################### Retreving data for Solar Generation from EIA #############################################
print("\n\nSOLAR GENERATION ENERGY SOURCE API CALL SECTION:")
fname_type = "SolarGen"

params3 = {
'frequency': 'hourly',
'data[0]': 'value',     # the [0] is important; doesn't work otherwise
#'facets[parent][]': ['CISO'],
'facets[fueltype][]': ['SUN', 'SNB',],
#'facets[fueltype][]': ["SNB","SUN"],
#'facets[respondent][]': 'AZPS',
'facets[respondent][]': 'SRP',
'start': start_date, #have to go back 8 hours due to UTC to PST conversion
'end': end_date,
'sort[0][column]': 'period',
'sort[0][direction]': 'desc',
'offset': offset,
'length': length,
'api_key': api_key
}

if dowload_flag == 1:
    ''' BELOW IS THE CALL OF get_EIA_data_demand FROM IMPORT '''
    total_DF_energy_source = get_EIA_data_energy_source(fname, api_key, base_url, fname_type, params3, file_flag)
elif (dowload_flag == 0):
    with open(fname + fname_type + '.csv', 'r') as file:
        total_DF_energy_source = pd.read_csv(file)
    file.close()
    print("Data Read from file  {}".format(fname + fname_type + '.csv'))

# ---- Clean solar dataframe (match demand’s timezone & numeric) ----
total_DF_energy_source['period'] = pd.to_datetime(total_DF_energy_source['period'], errors='coerce')
#total_DF_energy_source['value']  = pd.to_numeric(total_DF_energy_source['value'], errors='coerce')
total_DF_energy_source['hour'] = total_DF_energy_source['period'].dt.hour
total_DF_energy_source['date'] = total_DF_energy_source['period'].dt.date

# _________________________ Downselecting data from larger dataset and data cleaning ____________________________________
utility = "SRP"
down_select = "SUN" 
down_select = "SNB"
print("\nUnique fuel types in generation dataset:")
print(total_DF_energy_source['fueltype'].unique())

# Separate Solar (SUN) and Battery (SNB/BAT)
solar_df1 = total_DF_energy_source[total_DF_energy_source['fueltype'] == "SUN"]
batt_df  = total_DF_energy_source[total_DF_energy_source['fueltype'].isin(["SNB","BAT"])]
solar_df_copy = solar_df1.copy()
solar_df = solar_df1.copy()
solar_factor = 2
solar_df['value']=solar_factor*solar_df_copy['value']


# Group by hour-of-day and average across all days
solar_profile = solar_df.groupby(solar_df['period'].dt.hour)['value'].mean()
solar_profile = solar_profile.reindex(range(24))   # enforce 0–23 indexing
solar_0_23 = solar_profile.to_numpy()

# ---- Plot Solar Profile (clean) ----
plt.figure(figsize=(10,5))
plt.plot(range(24), solar_0_23, label="Mean Solar", marker='o', color="black")
plt.xticks(range(0,24))
plt.xlabel("Hour of Day")
plt.ylabel("MWh")
plt.title("Solar Profiles (0–23 hours)")
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()

print("\nSample Solar (SUN) data:")
print(solar_df[['period','fueltype','value']].head(24))

print("\nSample Battery (SNB/BAT) data:")
print ("snb")
print(batt_df[['period','fueltype','value']].head(24))

# Save downselect (write the solar set, not df1)
with open(fname + fname_type + 'down_selct'+ '.csv', 'w') as file: 
    total_DF_energy_source.to_csv(file, index=True)
    file.close()
    print("\nData written to file {}".format(fname + fname_type + '.csv'))

# ---------------- FIGURE 2: Solar (daily traces + mean, single legend) ----------------
top = 1
if top == 1:
    plt.figure(figsize=(13, 5))
    for day, values in solar_df.groupby('date'):
     plt.plot(values['hour'], values['value'], alpha=0.12, color='orange')
  # no per-day label
    # overlay mean solar
    mean_solar = solar_df.groupby('hour')['value'].mean()
    plt.plot(range(24), mean_solar, marker='o', label='Mean Solar', color='black')
    plt.title('FIGURE 2: Electricity {} from {} (Source: EIA API)'.format(fname_type, utility), fontsize=14, pad=10)
    plt.ylabel('MWh', fontsize=12)
    plt.xlabel('Hour of Day', fontsize=12)
    plt.xticks(range(0, 24))
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()

# _____________________________________________________________________________________________________________________________
# Plot demand and generation on one plot
top = 1
if top == 1:
    plt.figure(figsize=(13, 5))
    for day, values in df1.groupby('date'):
        plt.plot(values['hour'], values['value'], alpha=0.10, color='tab:blue')
    for day, values in solar_df.groupby('date'):
        plt.plot(values['hour'], values['value'], alpha=0.10, color='tab:orange')

    mean_demand = df1.groupby('hour')['value'].mean()
    mean_solar  = solar_df.groupby('hour')['value'].mean()

    plt.plot(range(24), mean_demand, marker='o', label='Mean Demand', color='black', linewidth=2)
    plt.plot(range(24), mean_solar,  marker='o', label='Mean Solar',  color='black', linewidth=2)
    plt.title(f'FIGURE 3: Electricity {fname_type} from {utility} (Source: EIA API)', fontsize=14, pad=10)
    plt.ylabel('MWh')
    plt.xlabel('Hour of Day')
    plt.xticks(range(24))
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()


# *****************************************************************
# Calculate electric vehicle load
# ****************************************************************************
CAGR = 0.18              # Compound Annual Growth Rate ~18% (Arizona EV market growth)
num_cars = 75000         # Approx. number of EVs in Arizona in 2024
time = 10                # Projection horizon (years from 2024)

# Projected EV count in 10 years
num_cars_t = num_cars * (1 + CAGR)**time
print('\nProjected number of EVs in Arizona after', time, 'years:', round(num_cars_t))

# Usage assumptions
miles_car = 30           # Average daily miles driven per EV in Arizona
elec_car = 0.32          # Energy consumption per mile (kWh/mile)

# Daily energy demand from EVs
elec_car_daily = num_cars_t * miles_car * elec_car  # in kWh/day
print("\nElectric car total daily demand (kWh/day):", round(elec_car_daily))

# Convert to MWh/day
elec_car_daily_MWh = 1e-3 * elec_car_daily
print("Electric car total daily demand (MWh/day):", round(elec_car_daily_MWh, 2))

# Electricity Use Evenly Distributed over 24 hours
dist1 = elec_car_daily_MWh / 24                     # In MWh/day
ev_demand = df1.copy() 
total_demand = df1.copy() 
ev_demand['value'].values[:] = 0  
ev_demand['value'] = ev_demand['value'] + dist1
total_demand['value'] = df1['value'] + ev_demand['value']

# ---------------- FIGURE 4: EV demand (uniform) ----------------
top = 1
if top == 1:
    ev_demand['hour'] = ev_demand['period'].dt.hour
    plt.figure(figsize=(15, 7))
    plt.plot(ev_demand['hour'], ev_demand['value'], alpha=0.8, color='black', linewidth=2, label='EV (uniform)')
    plt.title(f'FIGURE 4: Daily EV Demand after {time} years (uniform charging)', fontsize=14, pad=10)
    plt.xlabel('Hour of Day')
    plt.ylabel('MWh')
    plt.xticks(range(24))
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()

# ---------------- FIGURE 5: Solar Generation + Total Demand (means overlay) ----------------
top = 1
if top == 1:
    total_demand['hour'] = total_demand['period'].dt.hour
    total_demand['date'] = total_demand['period'].dt.date

    plt.figure(figsize=(15, 7))
    for day, values in solar_df.groupby('date'):
        plt.plot(values['hour'], values['value'], alpha=0.10, color='green')
    for day, values in total_demand.groupby('date'):
        plt.plot(values['hour'], values['value'], alpha=0.10, color='orange')

    mean_solar = solar_df.groupby('hour')['value'].mean()
    mean_total = total_demand.groupby('hour')['value'].mean()
    plt.plot(range(24), mean_solar, marker='o', color='black', linewidth=2, label='Mean Solar')
    plt.plot(range(24), mean_total, marker='o', color='black', linewidth=2, label='Mean Total Demand')
    plt.title('FIGURE 5: Solar Generation & Total Demand (means)', fontsize=14, pad=10)
    plt.xlabel('Hour of Day')
    plt.ylabel('MWh')
    plt.xticks(range(24))
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()




# ----------------- Figure 5.5 ---------------------
top = 1
if top == 1:

    monthly_total_demand = []
    # loop through all 12 months
    for month in range(1, 13):
            # grab the data for the current month in the loop from total_demand
            month_df = total_demand[total_demand['period'].dt.month == month]
            # append the month of data to the monthly total demand array
            monthly_total_demand.append(month_df)

    monthly_solar_df = []
    # loop through all 12 months
    for month in range(1, 13):
        # grab the data for the current month in the loop from solar_df
        month_df = solar_df[solar_df['period'].dt.month == month]
        # append the month of data to the monthly solar df array
        monthly_solar_df.append(month_df)

    plt.figure(figsize=(15, 7))

    #if True:
    if False:
        print("\n\nMONTHLY DATA PRINT CHECK:")
        print(len(monthly_total_demand))
        print(monthly_total_demand)
        print(len(monthly_solar_df))
        print(monthly_solar_df)

    # MONTH PLOTTING SECTION
    # For loops plot one line per day so Matplotlib doesn’t connect hour=23 to hour=0 across days.
    # One Line2D per day = no cross-day wrap segments.
    # Need first _ in front of plot_value because groupby returns (key, group), but only need group for plotting
    # January
    if True:
    #if False:
        set_legend = True
        for _, plot_value in monthly_total_demand[0].groupby(monthly_total_demand[0]['period'].dt.date):
            plt.plot(plot_value['period'].dt.hour, plot_value['value'], alpha=0.2, color='blue', label='January' if set_legend else None)
            set_legend = False
        for _, plot_value in monthly_solar_df[0].groupby(monthly_solar_df[0]['period'].dt.date):
            plt.plot(plot_value['period'].dt.hour, plot_value['value'], alpha=0.2, color='blue')

    # February
    if True:
    #if False:
        set_legend = True
        for _, plot_value in monthly_total_demand[1].groupby(monthly_total_demand[1]['period'].dt.date):
            plt.plot(plot_value['period'].dt.hour, plot_value['value'], alpha=0.2, color='royalblue', label='February' if set_legend else None)
            set_legend = False
        for _, plot_value in monthly_solar_df[1].groupby(monthly_solar_df[1]['period'].dt.date):
            plt.plot(plot_value['period'].dt.hour, plot_value['value'], alpha=0.2, color='royalblue')

    # March
    if True:
    #if False:
        set_legend = True
        for _, plot_value in monthly_total_demand[2].groupby(monthly_total_demand[2]['period'].dt.date):
            plt.plot(plot_value['period'].dt.hour, plot_value['value'], alpha=0.2, color='cyan', label='March' if set_legend else None)
            set_legend = False
        for _, plot_value in monthly_solar_df[2].groupby(monthly_solar_df[2]['period'].dt.date):
            plt.plot(plot_value['period'].dt.hour, plot_value['value'], alpha=0.2, color='cyan')

    # April
    if True:
    #if False:
        set_legend = True
        for _, plot_value in monthly_total_demand[3].groupby(monthly_total_demand[3]['period'].dt.date):
            plt.plot(plot_value['period'].dt.hour, plot_value['value'], alpha=0.2, color='mediumspringgreen', label='April' if set_legend else None)
            set_legend = False
        for _, plot_value in monthly_solar_df[3].groupby(monthly_solar_df[3]['period'].dt.date):
            plt.plot(plot_value['period'].dt.hour, plot_value['value'], alpha=0.2, color='mediumspringgreen')

    # May
    if True:
    # if False:
        set_legend = True
        for _, plot_value in monthly_total_demand[4].groupby(monthly_total_demand[4]['period'].dt.date):
            plt.plot(plot_value['period'].dt.hour, plot_value['value'], alpha=0.2, color='lime', label='May' if set_legend else None)
            set_legend = False
        for _, plot_value in monthly_solar_df[4].groupby(monthly_solar_df[4]['period'].dt.date):
            plt.plot(plot_value['period'].dt.hour, plot_value['value'], alpha=0.2, color='lime')

    # June
    if True:
    #if False:
        set_legend = True
        for _, plot_value in monthly_total_demand[5].groupby(monthly_total_demand[5]['period'].dt.date):
            plt.plot(plot_value['period'].dt.hour, plot_value['value'], alpha=0.2, color='greenyellow', label='June' if set_legend else None)
            set_legend = False
        for _, plot_value in monthly_solar_df[5].groupby(monthly_solar_df[5]['period'].dt.date):
            plt.plot(plot_value['period'].dt.hour, plot_value['value'], alpha=0.2, color='greenyellow')

    # July
    if True:
    #if False:
        set_legend = True
        for _, plot_value in monthly_total_demand[6].groupby(monthly_total_demand[6]['period'].dt.date):
            plt.plot(plot_value['period'].dt.hour, plot_value['value'], alpha=0.2, color='yellow', label='July' if set_legend else None)
            set_legend = False
        for _, plot_value in monthly_solar_df[6].groupby(monthly_solar_df[6]['period'].dt.date):
            plt.plot(plot_value['period'].dt.hour, plot_value['value'], alpha=0.2, color='yellow')

    # August
    if True:
    #if False:
        set_legend = True
        for _, plot_value in monthly_total_demand[7].groupby(monthly_total_demand[7]['period'].dt.date):
            plt.plot(plot_value['period'].dt.hour, plot_value['value'], alpha=0.2, color='orange', label='August' if set_legend else None)
            set_legend = False
        for _, plot_value in monthly_solar_df[7].groupby(monthly_solar_df[7]['period'].dt.date):
            plt.plot(plot_value['period'].dt.hour, plot_value['value'], alpha=0.2, color='orange')

    # September
    if True:
    #if False:
        set_legend = True
        for _, plot_value in monthly_total_demand[8].groupby(monthly_total_demand[8]['period'].dt.date):
            plt.plot(plot_value['period'].dt.hour, plot_value['value'], alpha=0.2, color='red', label='September' if set_legend else None)
            set_legend = False
        for _, plot_value in monthly_solar_df[8].groupby(monthly_solar_df[8]['period'].dt.date):
            plt.plot(plot_value['period'].dt.hour, plot_value['value'], alpha=0.2, color='red')

    # October
    if True:
    # if False:
        set_legend = True
        for _, plot_value in monthly_total_demand[9].groupby(monthly_total_demand[9]['period'].dt.date):
            plt.plot(plot_value['period'].dt.hour, plot_value['value'], alpha=0.2, color='deeppink', label='October' if set_legend else None)
            set_legend = False
        for _, plot_value in monthly_solar_df[9].groupby(monthly_solar_df[9]['period'].dt.date):
            plt.plot(plot_value['period'].dt.hour, plot_value['value'], alpha=0.2, color='deeppink')

    # November
    if True:
    #if False:
        set_legend = True
        for _, plot_value in monthly_total_demand[10].groupby(monthly_total_demand[10]['period'].dt.date):
            plt.plot(plot_value['period'].dt.hour, plot_value['value'], alpha=0.2, color='fuchsia', label='November' if set_legend else None)
            set_legend = False
        for _, plot_value in monthly_solar_df[10].groupby(monthly_solar_df[10]['period'].dt.date):
            plt.plot(plot_value['period'].dt.hour, plot_value['value'], alpha=0.2, color='fuchsia')

    # December
    if True:
    #if False:
        set_legend = True
        for _, plot_value in monthly_total_demand[11].groupby(monthly_total_demand[11]['period'].dt.date):
            plt.plot(plot_value['period'].dt.hour, plot_value['value'], alpha=0.2, color='darkviolet', label='December' if set_legend else None)
            set_legend = False
        for _, plot_value in monthly_solar_df[11].groupby(monthly_solar_df[11]['period'].dt.date):
            plt.plot(plot_value['period'].dt.hour, plot_value['value'], alpha=0.2, color='darkviolet')

    mean_solar = solar_df.groupby('hour')['value'].mean()
    mean_total = total_demand.groupby('hour')['value'].mean()
    plt.plot(range(24), mean_solar, marker='o', color='black', linewidth=2, label='Mean Solar')
    plt.plot(range(24), mean_total, marker='x', color='black', linewidth=2, label='Mean Total Demand')
    plt.title('FIGURE 5.5: Monthly Solar Generation & Total Demand (means)', fontsize=14, pad=10)
    plt.xlabel('Hour of Day')
    plt.ylabel('MWh')
    plt.xticks(range(24))
    plt.grid(True, linestyle='--', alpha=0.4)
    leg = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=2, fontsize=9)
    for legend_color in leg.legend_handles:
        legend_color.set_alpha(1)
    plt.tight_layout()
    plt.show()



# #####################  Electric Cars demand during certain times
charging_time_start = 8
charging_time_stop = 14
dist2 = elec_car_daily_MWh/(charging_time_stop - charging_time_start)  # kept as-is
print("\n\nDIST2 VALUE:")
print(dist2)

df1_r= df1.reset_index()
ev_demand_2 = df1_r.copy()
total_demand_2 = df1_r.copy()
net_demand_2 = df1_r.copy() 
net_EVdemand_2 = df1_r.copy() 
ev_demand_2['value'].values[:] = 0
net_demand_2['value'].values[:] = 0

ev_demand_2['value'] = np.where(
   (ev_demand_2['period'].dt.hour >=  charging_time_start) & (ev_demand_2['period'].dt.hour <=  charging_time_stop),
   ev_demand_2['value'] + dist2,
   ev_demand_2['value']
)

print("\n\nEV Demand with timed charging:")
print(ev_demand_2)

total_demand_2['value'] = df1_r['value'] + ev_demand_2['value']
print("\n\nTotal Demand (regular + EV) with timed charging:")
print(total_demand_2)

total_DF_energy_source_r= solar_df.reset_index()
net_demand_2['value'] = df1_r['value'].subtract(total_DF_energy_source_r['value'])
net_EVdemand_2['value'] = total_demand_2['value'].subtract(total_DF_energy_source_r['value'])
print("total_DF_energy_source",total_DF_energy_source)
print("df1_r",df1_r)
print("total_DF_energy_source_r",total_DF_energy_source_r)
print("net_demand_2",net_demand_2)

get_demand_analysis(net_demand_2, 2024, demand_division, 1)
get_demand_analysis(net_EVdemand_2, 2024, demand_division, 1)

# 0–23 hourly EV profile from data
ev_profile = (
    ev_demand
      .set_index('period')
      .loc[:, 'value']
      .groupby(ev_demand['period'].dt.hour)
      .mean()
      .reindex(range(24), fill_value=0)
)

hours = range(24)
ev_0_23 = ev_profile.values

ev_profile = ev_profile.reindex(range(24), fill_value=0)
ev_0_23 = ev_profile.to_numpy()

print("EV 0–23 hourly profile:")
print(ev_profile)
df1_r['hour'] = df1_r['period'].dt.hour
df1_r['date'] = df1_r['period'].dt.date

total_DF_energy_source_r['hour'] = total_DF_energy_source_r['period'].dt.hour
total_DF_energy_source_r['date'] = total_DF_energy_source_r['period'].dt.date

# ---- Plot EV demand profile ----
plt.figure(figsize=(8,4))
plt.plot(range(24), ev_0_23, marker="o", color="black", label="EV Demand")
plt.xticks(range(24))
plt.xlabel("Hour of Day")
plt.ylabel("MWh")
plt.title("Average EV Demand Profile (0–23 hours)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# ---------------- FIGURE 6: EV timed window ----------------
top = 1
if top == 1:
    ev_demand_2['hour'] = ev_demand_2['period'].dt.hour
    plt.figure(figsize=(15, 7))
    plt.plot(ev_demand_2['hour'], ev_demand_2['value'], alpha=0.8, color='black',
             label=f'EV ({charging_time_start}–{charging_time_stop} h)')
    plt.title(f'FIGURE 6: EV Demand (timed charging {charging_time_start}–{charging_time_stop})', fontsize=14, pad=10)
    plt.xlabel('Hour of Day')
    plt.ylabel('MWh')
    plt.xticks(range(24))
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()

# ---------------- FIGURE 7: Demand and Net Demand including Solar ----------------
top = 1
if top == 1:
    net_demand_2['hour'] = net_demand_2['period'].dt.hour
    df1['hour'] = df1['period'].dt.hour
    plt.figure(figsize=(15, 7))
    plt.plot(net_demand_2['period'].dt.hour, net_demand_2['value'], alpha=0.8, color='blue', linewidth=2, label="Net Demand")
    plt.plot(df1['period'].dt.hour, df1['value'], alpha=0.8, color='orange', linewidth=2, label="Demand")
    plt.title('FIGURE 7: Demand and Net Demand including Solar', fontsize=14, pad=10)
    plt.xlabel('Hour of Day')
    plt.ylabel('MWh')
    plt.xticks(range(24))
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()

# ---------------- FIGURE 8: Demand + EV vs Net + EV (by Hour) ----------------
top = 1
if top == 1:
    total_demand_2['hour'] = total_demand_2['period'].dt.hour
    total_demand_2['date'] = total_demand_2['period'].dt.date
    net_EVdemand_2['hour'] = net_EVdemand_2['period'].dt.hour
    net_EVdemand_2['date'] = net_EVdemand_2['period'].dt.date

    plt.figure(figsize=(15, 7))
    for day, values in total_demand_2.groupby('date'):
        plt.plot(values['hour'], values['value'], color='blue', alpha=0.10)
    for day, values in net_EVdemand_2.groupby('date'):
        plt.plot(values['hour'], values['value'], color='orange', alpha=0.10)

    mean_total_ev = total_demand_2.groupby('hour')['value'].mean()
    mean_net_ev   = net_EVdemand_2.groupby('hour')['value'].mean()
    plt.plot(range(24), mean_total_ev, marker='o', color='black', linewidth=2, label='Mean (Demand + EV)')
    plt.plot(range(24), mean_net_ev,   marker='o', color='black', linewidth=2, label='Mean (Net + EV)')
    plt.title('FIGURE 8: Demand + EV and Net + EV (by Hour)', fontsize=14, pad=10)
    plt.xlabel('Hour of Day')
    plt.ylabel('Demand (MWh)')
    plt.xticks(range(24))
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()

# ---------------- FIGURE 8.5: ----------------
top = 1
if top == 1:
    # build monthly buckets (1..12), keeping originals intact
    monthly_total_demand_2 = []
    for month in range(1, 13):
        monthly_total_demand_2.append(
            total_demand_2[total_demand_2['period'].dt.month == month]
        )

    monthly_net_EVdemand_2 = []
    for month in range(1, 13):
        monthly_net_EVdemand_2.append(
            net_EVdemand_2[net_EVdemand_2['period'].dt.month == month]
        )

    plt.figure(figsize=(15, 7))

    # January
    if True:
    # if False:
        set_legend = True
        for _, plot_value in monthly_total_demand_2[0].groupby(monthly_total_demand_2[0]['period'].dt.date):
            plt.plot(plot_value['period'].dt.hour, plot_value['value'], alpha=0.2, color='blue', label='January' if set_legend else None)
            set_legend = False
        for _, plot_value in monthly_net_EVdemand_2[0].groupby(monthly_net_EVdemand_2[0]['period'].dt.date):
            plt.plot(plot_value['period'].dt.hour, plot_value['value'], alpha=0.2, color='blue')

    # February
    #if True:
    if False:
        set_legend = True
        for _, plot_value in monthly_total_demand_2[1].groupby(monthly_total_demand_2[1]['period'].dt.date):
            plt.plot(plot_value['period'].dt.hour, plot_value['value'], alpha=0.2, color='royalblue', label='February' if set_legend else None)
            set_legend = False
        for _, plot_value in monthly_net_EVdemand_2[1].groupby(monthly_net_EVdemand_2[1]['period'].dt.date):
            plt.plot(plot_value['period'].dt.hour, plot_value['value'], alpha=0.2, color='royalblue')

    # March
    #if True:
    if False:
        set_legend = True
        for _, plot_value in monthly_total_demand_2[2].groupby(monthly_total_demand_2[2]['period'].dt.date):
            plt.plot(plot_value['period'].dt.hour, plot_value['value'], alpha=0.2, color='cyan', label='March' if set_legend else None)
            set_legend = False
        for _, plot_value in monthly_net_EVdemand_2[2].groupby(monthly_net_EVdemand_2[2]['period'].dt.date):
            plt.plot(plot_value['period'].dt.hour, plot_value['value'], alpha=0.2, color='cyan')

    # April
    if True:
    #if False:
        set_legend = True
        for _, plot_value in monthly_total_demand_2[3].groupby(monthly_total_demand_2[3]['period'].dt.date):
            plt.plot(plot_value['period'].dt.hour, plot_value['value'], alpha=0.2, color='mediumspringgreen', label='April' if set_legend else None)
            set_legend = False
        for _, plot_value in monthly_net_EVdemand_2[3].groupby(monthly_net_EVdemand_2[3]['period'].dt.date):
            plt.plot(plot_value['period'].dt.hour, plot_value['value'], alpha=0.2, color='mediumspringgreen')

    # May
    #if True:
    if False:
        set_legend = True
        for _, plot_value in monthly_total_demand_2[4].groupby(monthly_total_demand_2[4]['period'].dt.date):
            plt.plot(plot_value['period'].dt.hour, plot_value['value'], alpha=0.2, color='lime', label='May' if set_legend else None)
            set_legend = False
        for _, plot_value in monthly_net_EVdemand_2[4].groupby(monthly_net_EVdemand_2[4]['period'].dt.date):
            plt.plot(plot_value['period'].dt.hour, plot_value['value'], alpha=0.2, color='lime')

    # June
    #if True:
    if False:
        set_legend = True
        for _, plot_value in monthly_total_demand_2[5].groupby(monthly_total_demand_2[5]['period'].dt.date):
            plt.plot(plot_value['period'].dt.hour, plot_value['value'], alpha=0.2, color='greenyellow', label='June' if set_legend else None)
            set_legend = False
        for _, plot_value in monthly_net_EVdemand_2[5].groupby(monthly_net_EVdemand_2[5]['period'].dt.date):
            plt.plot(plot_value['period'].dt.hour, plot_value['value'], alpha=0.2, color='greenyellow')

    # July
    if True:
    #if False:
        set_legend = True
        for _, plot_value in monthly_total_demand_2[6].groupby(monthly_total_demand_2[6]['period'].dt.date):
            plt.plot(plot_value['period'].dt.hour, plot_value['value'], alpha=0.2, color='yellow', label='July' if set_legend else None)
            set_legend = False
        for _, plot_value in monthly_net_EVdemand_2[6].groupby(monthly_net_EVdemand_2[6]['period'].dt.date):
            plt.plot(plot_value['period'].dt.hour, plot_value['value'], alpha=0.2, color='yellow')

    # August
    #if True:
    if False:
        set_legend = True
        for _, plot_value in monthly_total_demand_2[7].groupby(monthly_total_demand_2[7]['period'].dt.date):
            plt.plot(plot_value['period'].dt.hour, plot_value['value'], alpha=0.2, color='orange', label='August' if set_legend else None)
            set_legend = False
        for _, plot_value in monthly_net_EVdemand_2[7].groupby(monthly_net_EVdemand_2[7]['period'].dt.date):
            plt.plot(plot_value['period'].dt.hour, plot_value['value'], alpha=0.2, color='orange')

    # September
    #if True:
    if False:
        set_legend = True
        for _, plot_value in monthly_total_demand_2[8].groupby(monthly_total_demand_2[8]['period'].dt.date):
            plt.plot(plot_value['period'].dt.hour, plot_value['value'], alpha=0.2, color='red', label='September' if set_legend else None)
            set_legend = False
        for _, plot_value in monthly_net_EVdemand_2[8].groupby(monthly_net_EVdemand_2[8]['period'].dt.date):
            plt.plot(plot_value['period'].dt.hour, plot_value['value'], alpha=0.2, color='red')

    # October
    if True:
    # if False:
        set_legend = True
        for _, plot_value in monthly_total_demand_2[9].groupby(monthly_total_demand_2[9]['period'].dt.date):
            plt.plot(plot_value['period'].dt.hour, plot_value['value'], alpha=0.2, color='deeppink', label='October' if set_legend else None)
            set_legend = False
        for _, plot_value in monthly_net_EVdemand_2[9].groupby(monthly_net_EVdemand_2[9]['period'].dt.date):
            plt.plot(plot_value['period'].dt.hour, plot_value['value'], alpha=0.2, color='deeppink')

    # November
    #if True:
    if False:
        set_legend = True
        for _, plot_value in monthly_total_demand_2[10].groupby(monthly_total_demand_2[10]['period'].dt.date):
            plt.plot(plot_value['period'].dt.hour, plot_value['value'], alpha=0.2, color='fuchsia', label='November' if set_legend else None)
            set_legend = False
        for _, plot_value in monthly_net_EVdemand_2[10].groupby(monthly_net_EVdemand_2[10]['period'].dt.date):
            plt.plot(plot_value['period'].dt.hour, plot_value['value'], alpha=0.2, color='fuchsia')

    # December
    #if True:
    if False:
        set_legend = True
        for _, plot_value in monthly_total_demand_2[11].groupby(monthly_total_demand_2[11]['period'].dt.date):
            plt.plot(plot_value['period'].dt.hour, plot_value['value'], alpha=0.2, color='darkviolet', label='December' if set_legend else None)
            set_legend = False
        for _, plot_value in monthly_net_EVdemand_2[11].groupby(monthly_net_EVdemand_2[11]['period'].dt.date):
            plt.plot(plot_value['period'].dt.hour, plot_value['value'], alpha=0.2, color='darkviolet')

    mean_total_ev = total_demand_2.groupby('hour')['value'].mean()
    mean_net_ev = net_EVdemand_2.groupby('hour')['value'].mean()
    plt.plot(range(24), mean_total_ev, marker='o', color='black', linewidth=2, label='Mean (Demand + EV)')
    plt.plot(range(24), mean_net_ev, marker='x', color='black', linewidth=2, label='Mean (Net + EV)')
    plt.title('FIGURE 8.5: Monthly Demand + EV and Net + EV (by Hour)', fontsize=14, pad=10)
    plt.xlabel('Hour of Day')
    plt.ylabel('Demand (MWh)')
    plt.xticks(range(24))
    plt.grid(True, linestyle='--', alpha=0.4)
    leg = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=2, fontsize=9)
    for legend_color in leg.legend_handles:
        legend_color.set_alpha(1)
    plt.tight_layout()
    plt.show()


get_demand_analysis(df1_test, 2024, demand_division, 1)

# ---------------- Final simple plot (title matched to data) ----------------
dist1 = elec_car_daily / 24
dfs1 = df1.copy()
dfs1['value'] = dfs1['value'] + dist1

plt.figure(figsize=(15, 7))
plt.plot(df1['period'], df1['value'], alpha = 0.8, color ='black')
plt.title('Demand (Source: EIA API)', fontsize=14, pad=10)   # title matches content
plt.ylabel('MWh', fontsize=12)
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()

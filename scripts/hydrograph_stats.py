import pandas as pd
import numpy as np
from glob import glob
import os
from datetime import datetime

from hgs_post_processing.utils.hgsio import read_dat


def nash_sutcliffe(obs, sim):
    # https://en.wikipedia.org/wiki/Nash-Sutcliffe_model_efficiency_coefficient
    # same implementation as R2, the coefficient of determination for statistical models
    # see https://stats.stackexchange.com/questions/185898/difference-between-nash-sutcliffe-efficiency-and-coefficient-of-determination
    obs_mean = np.nanmean(obs)
    num = np.nansum((sim - obs)**2)
    den = np.nansum((obs - obs_mean)**2)
    return 1 - num/den


def perc_bias(obs, sim):
    return 100 * np.nansum(sim-obs)/np.nansum(obs)


def rmse(obs, sim):
    return np.sqrt(np.nanmean((sim-obs)**2))


obs_start_date = ""
sim_start_date = ""

prefix = ""
run_dir = ""
stations = []
sim_data_fps = [os.path.join(run_dir, f"{prefix}o.hydrograph.{station}.dat")
                for station in stations]

obs_data_fps = [os.path.join(run_dir, f"{station}_observed.dat")
                for station in stations]

sim_df = pd.DataFrame(columns=stations, index=read_dat(sim_data_fps[0], header_lines=3).index.values)
for sim_data_fp, station in zip(sim_data_fps, stations):
    df = read_dat(sim_data_fp, header_lines=3)
    sim_df[station] = df['Channel'] + df['Surface']

obs_df = pd.DataFrame(columns=stations, index=read_dat(obs_data_fps[0], header_lines=3).index.values)
for obs_data_fp, station in zip(obs_data_fps, stations):
    df = read_dat(obs_data_fp, header_lines=3)
    obs_df[station] = df['Channel'] + df['Surface']

    
# Resample to daily time steps (obs data already daily)
obs_df.index = pd.to_timedelta(obs_df.index, unit='s')
obs_df.index += datetime.strptime(obs_start_date, '%Y-%m-%d')

sim_df.index = pd.to_timedelta(sim_df.index, unit='s')
sim_df.index += datetime.strptime(sim_start_date, '%Y-%m-%d')

obs_df = obs_df.resample("D").mean()
sim_df = sim_df.resample("D").mean()

# Trim excess data
sim_df = sim_df[max(sim_start_date, obs_start_date):]
obs_df = obs_df[max(sim_start_date, obs_start_date):]


# Generate yearly stats
num_years = len(sim_df) // 365  # days
idx = pd.MultiIndex.from_product([
    ['Nash-Sutcliffe', 'RMSE', 'Percent Bias'],
    ['All Years'] + list(range(1,num_years+1))
], names=['Statistic', 'Year'])

stats_df = pd.DataFrame(columns=stations, index=idx)
for station in stations:
    nse_list = []
    root_mse_list = []
    percent_bias_list = []

    # "All Years" stats
    nse_list.append(nash_sutcliffe(obs_df[station].values, sim_df[station].values))
    root_mse_list.append(rmse(obs_df[station].values, sim_df[station].values))
    percent_bias_list.append(perc_bias(obs_df[station].values, sim_df[station].values))
    
    # Yearly stats
    for year in range(num_years):
        sim_df_year = sim_df[365*year:365*(year+1)]
        obs_df_year = obs_df[365*year:365*(year+1)]
        
        nse_list.append(nash_sutcliffe(obs_df_year[station].values, sim_df_year[station].values))
        root_mse_list.append(rmse(obs_df_year[station].values, sim_df_year[station].values))
        percent_bias_list.append(perc_bias(obs_df_year[station].values, sim_df_year[station].values))
    stats_df[station] = nse_list + root_mse_list + percent_bias_list

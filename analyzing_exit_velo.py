#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  4 10:25:04 2025

@author: lucasben
"""
import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import arviz as az
from scipy.special import gammaln
from fitter import Fitter

df = pd.read_csv('exit_velo_project_data.csv') # raw data set
df2 = pd.read_csv('exit_velo_validate_data.csv') # validation data set

mlb_players = df[df['level_abbr'] == 'mlb']['batter_id'].unique() # identifying players who have played in MLB
minor_players = df[df['level_abbr'] != 'mlb']['batter_id'].unique() # identifying players who played in the minor leagues
progressed_ids = np.intersect1d(mlb_players, minor_players) # finding players who progressed from the minors the majors
players_progressed_to_mlb = df[df['batter_id'].isin(progressed_ids)] # filtering the data set to only include players who progressed from the minors to the majors
players_progressed_to_mlb['batter_id'].nunique() # calculating the number of players who progressed from the minors to the majors
players_progressed_to_mlb.isna().sum() # checking for missing values
players_progressed_to_mlb = players_progressed_to_mlb.dropna(subset=['exit_velo']) # dropping observations with missing exit velocity because it is the target variable
avg_exit_velo_players_progressed_to_mlb = players_progressed_to_mlb.groupby('batter_id', as_index=False)['exit_velo'].mean() # calculating the average exit velocity of each player who progressed from the minors to the majors

sns.histplot(avg_exit_velo_players_progressed_to_mlb['exit_velo'], bins=20, kde=True, color='red')
plt.title('Distribution of Average Exit Velocity for Players Who Progressed to MLB')
plt.xlabel('Average Exit Velocity (MPH)')
plt.ylabel('Count of Players')
plt.grid(True)
plt.show() # plotting the distribution of average exit velocity for players who progressed to the MLB

# understanding the correlations of batted ball metrics
numeric_df = players_progressed_to_mlb.drop(columns=['season', 'level_abbr', 'batter_id', 'pitcher_id', 'hit_type', 'batter_hand', 'pitcher_hand', 'pitch_group', 'outcome']) # creating quantitative dataframe 
corr_matrix = numeric_df.corr() # creating correlation matrix of quantitative variables
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True)
plt.title('Heatmap of Batted Ball Metrics') 
plt.show() # heatmap of quantitative variables

data = avg_exit_velo_players_progressed_to_mlb['exit_velo'] # assessing the best distribution for the average exit velocity of each player who progressed from the minors to the majors
f = Fitter(data, distributions=[
    'beta', 'gamma', 'lognorm', 'weibull_min', 'weibull_max', 
    'norm', 'skewnorm', 'triang'
])
f.fit() # fitting multiple distributions
f.summary()
f.plot_pdf()
plt.show() # plotting best distribution fits

# keeping only 'batter_id' that are in both data sets
matched_data = pd.merge(
    df2[['batter_id']],  
    df[['batter_id', 'exit_velo']],  
    on='batter_id',
    how='inner'  
)
mean_velo = df.groupby('batter_id')['exit_velo'].mean().reset_index() # calculating average exit velocity for each unique 'batter_id'
matched_data = pd.merge(
    df2[['batter_id']],
    mean_velo,
    on='batter_id',
    how='inner'
) # merging with validation set
# preparing data for analysis
validate_data = matched_data
max_ev = validate_data['exit_velo'].max() + 1
mirrored_ev = max_ev - validate_data['exit_velo'].values
# creating indices for 'batter_id'
player_ids = validate_data['batter_id'].unique()
n_players = len(player_ids)
player_idx = validate_data['batter_id'].factorize()[0]
# Bayesian model
with pm.Model() as hierarchical_weibull:
    # priors
    alpha_mu = pm.Gamma('alpha_mu', alpha=2, beta=0.5)  
    alpha_sigma = pm.HalfNormal('alpha_sigma', sigma=0.1) 
    
    beta_mu = pm.Gamma('beta_mu', alpha=50, beta=0.5)  
    beta_sigma = pm.HalfNormal('beta_sigma', sigma=2)  
    # non-centered parameterization for better sampling
    alpha_offset = pm.Normal('alpha_offset', mu=0, sigma=1, shape=n_players)
    beta_offset = pm.Normal('beta_offset', mu=0, sigma=1, shape=n_players)
    # player-specific parameters
    alpha = pm.Deterministic('alpha', alpha_mu + alpha_offset * alpha_sigma)
    beta = pm.Deterministic('beta', beta_mu + beta_offset * beta_sigma)
    # likelihood
    likelihood = pm.Weibull('likelihood',
                          alpha=alpha[player_idx],
                          beta=beta[player_idx],
                          observed=mirrored_ev)
    trace = pm.sample(
        3000, tune=1500, chains=4,
        target_accept=0.99,  
        nuts_kwargs={'max_treedepth': 20},  
        init='adapt_diag',  
        random_seed=42
    )
# diagnostics
print(az.summary(trace, var_names=['alpha_mu', 'beta_mu']))
az.plot_trace(trace, var_names=['alpha_mu', 'beta_mu'])
pm.plot_energy(trace)
# proceeding if model converges well
if all(az.rhat(trace) < 1.05) and az.ess(trace)['mean'].min() > 400:
    # player-specific parameter estimates
    alpha_post = trace.posterior['alpha'].mean(dim=('chain', 'draw')).values
    beta_post = trace.posterior['beta'].mean(dim=('chain', 'draw')).values
    # estimates
    true_ability = max_ev - (beta_post * np.exp(gammaln(1 + 1/alpha_post)))
    # saving estimates
    results = pd.DataFrame({
        'player_id': player_ids,
        'true_ability_estimate': true_ability,
        'alpha': alpha_post,
        'beta': beta_post
    })
    # 2024 projections
    n_sim = 1000
    projections = []
    for i in range(n_players):
        alpha_samples = trace.posterior['alpha'][:, :, i].values.flatten()
        beta_samples = trace.posterior['beta'][:, :, i].values.flatten()
        idx = np.random.choice(len(alpha_samples), n_sim)
        sim_ev = max_ev - np.random.weibull(a=alpha_samples[idx]) * beta_samples[idx]
        projections.append({
            'player_id': player_ids[i],
            'projected_ev_mean': np.mean(sim_ev),
            'projected_ev_sd': np.std(sim_ev),
            'projected_ev_5': np.percentile(sim_ev, 5),
            'projected_ev_95': np.percentile(sim_ev, 95)
        })
    projections_df = pd.DataFrame(projections)
    final_results = pd.merge(results, projections_df, on='player_id')
    final_results.to_csv('exit_velocity_predictions_2024.csv', index=False)
else:
    print("sampling diagnostics failed")
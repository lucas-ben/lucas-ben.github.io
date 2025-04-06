#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm


# In[3]:


indices = pd.read_csv('indices_csv_data.csv')
indices['Unnamed: 0'] = pd.to_datetime(indices['Unnamed: 0'])  # Convert timestamp to datetime if necessary
indices.set_index('Unnamed: 0', inplace=True)
experiment = indices[['SPX Index', 'VIX Index']]
experiment.index = pd.to_datetime(experiment.index)
experiment.columns


# In[5]:


experiment.describe()


# In[6]:


plt.figure(figsize = (12, 8 ))

ax_spx = experiment['SPX Index'].plot()
ax_vix = experiment['VIX Index'].plot(secondary_y=True)

ax_spx.legend(loc=1)
ax_vix.legend(loc=2)

plt.show()


# In[7]:


experiment.diff().hist(
    figsize=(10,7),
    color='blue',
    bins=40)


# In[8]:


experiment.pct_change().hist(
    figsize=(10,7),
    color='blue',
    bins=40)


# In[11]:


log_returns = np.log(experiment / experiment.shift(1)).dropna()
log_returns.plot(
    subplots=True,
    figsize=(10, 8),
    color='blue',
    grid=True
    );
for ax in plt.gcf().axes:
    ax.legend(loc='upper left')


# In[12]:


log_returns.corr()


# In[15]:


log_returns.plot(
    figsize=(10,8),
    x="SPX Index",
    y="VIX Index",
    kind='scatter')

ols_fit = sm.OLS(log_returns['VIX Index'].values,
    log_returns['SPX Index'].values).fit()

plt.plot(log_returns['SPX Index'], ols_fit.fittedvalues, 'r')


# In[16]:


df_sma = pd.DataFrame(index=experiment.index)
df_sma['short'] = experiment['SPX Index'].rolling(window=5, min_periods=5).mean()
df_sma['long'] = experiment['SPX Index'].rolling(window=200, min_periods=30).mean()

# Plot the SMAs
plt.figure(figsize=(12, 8))
plt.plot(df_sma.index, df_sma['short'], label='5-day SMA', color='blue')
plt.plot(df_sma.index, df_sma['long'], label='200-day SMA', color='red')
plt.title("Short and Long-term SMAs for SPX Index")
plt.xlabel("Date")
plt.ylabel("Index Value")
plt.legend()
plt.show()


# In[17]:


df_sma = pd.DataFrame(index=experiment.index)

df_sma['short'] = experiment['VIX Index'].rolling(window=5, min_periods=5).mean()
df_sma['long'] = experiment['VIX Index'].rolling(window=200, min_periods=30).mean()

plt.figure(figsize=(12, 8))
plt.plot(df_sma.index, df_sma['short'], label='5-day SMA', color='green')
plt.plot(df_sma.index, df_sma['long'], label='200-day SMA', color='orange')
plt.title("Short and Long-term SMAs for VIX Index")
plt.xlabel("Date")
plt.ylabel("Index Value")
plt.legend()
plt.show()


# In[18]:


df_sma['short'] = experiment['SPX Index'].ewm(span=5, adjust=False).mean()
df_sma['long'] = experiment['SPX Index'].ewm(span=200, adjust=False).mean()

# Plot the EMAs
plt.figure(figsize=(12, 8))
plt.plot(df_sma.index, df_sma['short'], label='5-day EMA', color='blue')
plt.plot(df_sma.index, df_sma['long'], label='200-day EMA', color='red')
plt.title("Short and Long-term EMAs for SPX Index")
plt.xlabel("Date")
plt.ylabel("Index Value")
plt.legend()
plt.show()


# In[19]:


df_sma['short'] = experiment['VIX Index'].ewm(span=5, adjust=False).mean()
df_sma['long'] = experiment['VIX Index'].ewm(span=200, adjust=False).mean()

plt.figure(figsize=(12, 8))
plt.plot(df_sma.index, df_sma['short'], label='5-day EMA', color='green')
plt.plot(df_sma.index, df_sma['long'], label='200-day EMA', color='orange')
plt.title("Short and Long-term EMAs for VIX Index")
plt.xlabel("Date")
plt.ylabel("Index Value")
plt.legend()
plt.show()


# In[20]:


df_sma = pd.DataFrame(index=experiment.index)
df_sma['200-day EMA'] = experiment['SPX Index'].ewm(span=200, adjust=False).mean()

# Calculate the percentage deviation from the 200-day EMA
df_sma['deviation'] = (experiment['SPX Index'] - df_sma['200-day EMA']) / df_sma['200-day EMA'] * 100

# Set thresholds for buy and sell signals (e.g., 5% deviation)
buy_threshold = -10   # Buy when SPX is 5% below the 200-day EMA
sell_threshold = 10   # Sell when SPX is 5% above the 200-day EMA

# Buy signal: When SPX is 5% below the EMA
df_sma['buy_signal'] = df_sma['deviation'] <= buy_threshold

# Sell signal: When SPX is 5% above the EMA
df_sma['sell_signal'] = df_sma['deviation'] >= sell_threshold

# Plot the SPX Index, EMA, and Buy/Sell signals
plt.figure(figsize=(12, 8))
plt.plot(experiment.index, experiment['SPX Index'], label='SPX Index', color='black', alpha=0.6)
plt.plot(df_sma.index, df_sma['200-day EMA'], label='200-day EMA', color='blue', linestyle='--')

# Plot Buy signals
plt.scatter(df_sma.index[df_sma['buy_signal']], experiment['SPX Index'][df_sma['buy_signal']], label='Buy Signal', marker='^', color='green', alpha=1)

# Plot Sell signals
plt.scatter(df_sma.index[df_sma['sell_signal']], experiment['SPX Index'][df_sma['sell_signal']], label='Sell Signal', marker='v', color='red', alpha=1)

plt.title("SPX Index with Buy and Sell Signals")
plt.xlabel("Date")
plt.ylabel("SPX Index Value")
plt.legend()
plt.show()


# In[21]:


num_buy_signals = df_sma['buy_signal'].sum()
num_sell_signals = df_sma['sell_signal'].sum()

# Output the counts
print(f"Number of Buy Signals: {num_buy_signals}")
print(f"Number of Sell Signals: {num_sell_signals}")


# In[22]:


# Filter the DataFrame for Buy and Sell signals
buy_signals_table = df_sma[df_sma['buy_signal'] == True][['200-day EMA', 'deviation']]
sell_signals_table = df_sma[df_sma['sell_signal'] == True][['200-day EMA', 'deviation']]

# Output the tables of buy and sell signal observations
print("\nBuy Signals Observations:")
print(buy_signals_table)


# In[23]:


df_sma = pd.DataFrame(index=experiment.index)
df_sma['200-day EMA'] = experiment['VIX Index'].ewm(span=200, adjust=False).mean()

# Calculate the percentage deviation from the 200-day EMA
df_sma['deviation'] = (experiment['VIX Index'] - df_sma['200-day EMA']) / df_sma['200-day EMA'] * 100

# Set thresholds for buy and sell signals (e.g., 5% deviation)
buy_threshold = -25   # Buy when SPX is 5% below the 200-day EMA
sell_threshold = 25   # Sell when SPX is 5% above the 200-day EMA

# Buy signal: When SPX is 5% below the EMA
df_sma['buy_signal'] = df_sma['deviation'] <= buy_threshold

# Sell signal: When SPX is 5% above the EMA
df_sma['sell_signal'] = df_sma['deviation'] >= sell_threshold

# Plot the SPX Index, EMA, and Buy/Sell signals
plt.figure(figsize=(12, 8))
plt.plot(experiment.index, experiment['VIX Index'], label='VIX Index', color='black', alpha=0.6)
plt.plot(df_sma.index, df_sma['200-day EMA'], label='200-day EMA', color='blue', linestyle='--')

# Plot Buy signals
plt.scatter(df_sma.index[df_sma['buy_signal']], experiment['VIX Index'][df_sma['buy_signal']], label='Buy Signal', marker='^', color='green', alpha=1)

# Plot Sell signals
plt.scatter(df_sma.index[df_sma['sell_signal']], experiment['VIX Index'][df_sma['sell_signal']], label='Sell Signal', marker='v', color='red', alpha=1)

plt.title("VIX Index with Buy and Sell Signals")
plt.xlabel("Date")
plt.ylabel("VIX Index Value")
plt.legend()
plt.show()


# In[24]:


num_buy_signals = df_sma['buy_signal'].sum()
num_sell_signals = df_sma['sell_signal'].sum()

# Output the counts
print(f"Number of Buy Signals: {num_buy_signals}")
print(f"Number of Sell Signals: {num_sell_signals}")


# In[25]:


buy_signals_table = df_sma[df_sma['buy_signal'] == True][['200-day EMA', 'deviation']]
sell_signals_table = df_sma[df_sma['sell_signal'] == True][['200-day EMA', 'deviation']]


print("\nSell Signals Observations:")
print(sell_signals_table)


# In[ ]:





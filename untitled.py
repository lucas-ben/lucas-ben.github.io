from pybaseball import statcast_pitcher, playerid_lookup
from scipy.stats import beta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pymc as pm
import arviz as az

# Apply dark aesthetic
sns.set_theme(
    context='notebook',
    style='darkgrid',
    palette='dark',
    font='sans-serif',
    font_scale=1.2,  # Slightly larger font
    color_codes=True,
)

# Custom bright palette
custom_palette = [ "#FFD700",
    "#6FC3DF", 
    "#C94C4C",]  # Gold, Blue, Red
sns.set_palette(custom_palette)

# Matplotlib global settings
plt.rcParams.update({
    'axes.facecolor': '#2f2f2f',      # Dark axes background
    'figure.facecolor': '#1f1f1f',    # Dark figure background
    'axes.edgecolor': 'white',        # White axes borders
    'axes.labelcolor': 'white',       # White axis labels
    'xtick.color': 'white',           # White x-ticks
    'ytick.color': 'white',           # White y-ticks
    'text.color': 'white',            # White text (for titles, annotations)
    'grid.color': '#4a4a4a',          # Gray gridlines (softer than white)
    'grid.linestyle': '--',
    'axes.titlesize': 16,             # Title size
    'axes.labelsize': 14,             # Axis label size
})

# calculating league average whiff % 2022-2024
df = pd.read_csv('datasets/stats.csv')
mean_whiff = df['whiff_percent'].mean()
std_whiff_pop = df['whiff_percent'].std(ddof=0)

print("Mean of whiff_percent:", mean_whiff)
print("Population Standard Deviation:", std_whiff_pop)

# 1. Data Collection from Statcast
def get_swing_data(player_id, start_date, end_date):
    """
    Returns total swings and whiffs for a pitcher
    Args:
        player_id: MLBAM ID
        start_date: YYYY-MM-DD
        end_date: YYYY-MM-DD
    Returns:
        n: total swings
        y: total whiffs
    """
    df = statcast_pitcher(start_date, end_date, player_id)
    

    if df.empty:
        raise ValueError(f"No data found for player {player_id} from {start_date} to {end_date}")
    # Count swings and whiffs
    is_swing = df['description'].isin(['swinging_strike', 'foul', 'hit_into_play'])
    is_whiff = df['description'] == 'swinging_strike'
    
    return sum(is_swing), sum(is_whiff)

playerid_lookup('veneziano', 'anthony')

player_id = 685107  # Anthony Veneziano's MLB ID
start_date = '2025-03-29'
end_date = '2025-04-07'

prior_mean = 0.24930448717948714  # μ = 23% league average whiff rate
prior_std = 0.039529122407748423   # σ = 3% standard deviation

prior_var = prior_std**2
α = prior_mean * (prior_mean*(1-prior_mean)/prior_var - 1)
β = (1-prior_mean) * (prior_mean*(1-prior_mean)/prior_var - 1)

try:
    n, y = get_swing_data(player_id, start_date, end_date)
    print(f"Observed: {y} whiffs in {n} swings ({y/n:.1%})")
except ValueError as e:
    print(e)
    exit()
α_post = α + y
β_post = β + n - y
posterior_mean = α_post / (α_post + β_post)
posterior_std = np.sqrt((α_post*β_post)/((α_post+β_post)**2*(α_post+β_post+1)))

ci_95 = beta.ppf([0.025, 0.975], α_post, β_post)

print(f"\nBayesian Estimation Results:")
print(f"Prior: Beta(α={α:.1f}, β={β:.1f})")
print(f"Posterior: Beta(α={α_post:.1f}, β={β_post:.1f})")
print(f"Posterior mean: {posterior_mean:.3f}")
print(f"Posterior std: {posterior_std:.3f}")
print(f"95% Credible Interval: [{ci_95[0]:.3f}, {ci_95[1]:.3f}]")
x = np.linspace(0, 0.5, 1000)
plt.figure(figsize=(10, 6))
plt.plot(x, beta.pdf(x, α, β), 'r-', label=f'Prior (μ={prior_mean:.0%})')
plt.plot(x, beta.pdf(x, α_post, β_post), 'b-', label='Posterior')
plt.axvline(y/n, color='k', linestyle='--', label='Observed Rate')
plt.xlabel('True Whiff Rate (θ)')
plt.ylabel('Probability Density')
plt.title('Anthony Veneziano: Bayesian Whiff Rate Estimation')
plt.legend()
plt.grid(True)
plt.savefig("veneziano_plot.png", dpi=300)  
plt.show()

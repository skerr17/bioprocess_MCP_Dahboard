# the server.py does 3 mian things:
# 1. Loads and prepares my data when it starts up
# 2. Fits the analytics model and keeps it in memory 
# 3. Listens for tool calls from Claude Dasktop and responds to them
# Author: Stephen Kerr
# Date: 2026/31/05


# imports
import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mcp.server import Server
from mcp.server.stdio import stdio_server # a simple server that uses standard input and output to communicate with the client (Claude Desktop)
from mcp import types # defines the data structures MCP uses to communicate between the server and the client (Claude Desktop)


# Constants (potentially could be moved to a config file or revisited)
PHASE_BOUNDARIES = {
    "Lag":         (0,   50),
    "Exponential": (50,  150),
    "Stationary":  (150, 200),
    "Decline":     (200, float("inf")),
}

PCA_VARS = [
    "Aeration rate(Fg:L/h)",
    "Sugar feed rate(Fs:L/h)",
    "PAA flow(Fpaa:PAA flow (L/h))",
    "Dissolved oxygen concentration(DO2:mg/L)",
    "pH(pH:pH)",
    "Temperature(T:K)",
    "Oxygen Uptake Rate(OUR:(g min^{-1}))",
    "Carbon evolution rate(CER:g/h)",
    "carbon dioxide percent in off-gas(CO2outgas:%)",
    "Generated heat(Q:kJ)",
]

PHASE_RANGE_VARS = [
    "pH(pH:pH)",
    "Temperature(T:K)",
    "Dissolved oxygen concentration(DO2:mg/L)",
    "Sugar feed rate(Fs:L/h)",
    "Aeration rate(Fg:L/h)",
    "Base flow rate(Fb:L/h)",
    "Acid flow rate(Fa:L/h)",
    "PAA flow(Fpaa:PAA flow (L/h))",
]

N_PCS = 3
DATA_PATH = "./data/batches-subset-1-10.csv"


# Load and prepare data

# load the data set 
df_batch_1_10 = pd.read_csv(DATA_PATH)
# print(df_batch_1_10.head(5))


# assigning phases to each time point in the data set
def assign_phase(t):
    if t < 50:
        return "Lag"
    elif t < 150:
        return "Exponential"
    elif t < 200:
        return "Stationary"
    else:
        return "Decline"

df_batch_1_10["phase"] = df_batch_1_10["Time (h)"].apply(assign_phase)

#saved_phases = df_batch_1_10["phase"].unique().tolist()
#print("Saved phases:", saved_phases)

# removing unnecessary columns and rows with missing values
# drop unnecessary columns not relevant for analysis
df_batch_1_10 = df_batch_1_10.drop(columns=[
        '0 - Recipe driven 1 - Operator controlled(Control_ref:Control ref)', 
        'Fault reference(Fault_ref:Fault ref)'])


# PCA 
# Standardize the PCA candidate variables
scaler = StandardScaler()
pca_data_scaled = scaler.fit_transform(df_batch_1_10[PCA_VARS])

# Perform PCA
pca = PCA()
scores = pca.fit_transform(pca_data_scaled)



# Hotelling's T^2 Calculation for First 3 Principal Components
scores_t2 = scores[:, :N_PCS]  # Use the scores variable already calculated from PCA

# Calculate T^2 statistic
n_samples = scores_t2.shape[0]
cov_matrix = np.cov(scores_t2.T)
inv_cov = np.linalg.inv(cov_matrix)

t2_values = []
for i in range(n_samples):
    score = scores_t2[i, :]
    t2 = score @ inv_cov @ score.T
    t2_values.append(t2)

df_batch_1_10['T2'] = t2_values

# Calculate 95% and 99% control limits
alpha_95 = 0.05
alpha_99 = 0.01
f_95 = stats.f.ppf(1-alpha_95, N_PCS, n_samples-N_PCS)
f_99 = stats.f.ppf(1-alpha_99, N_PCS, n_samples-N_PCS)

ucl_95 = (N_PCS * (n_samples-1) / (n_samples-N_PCS)) * f_95
ucl_99 = (N_PCS * (n_samples-1) / (n_samples-N_PCS)) * f_99

# phase envelopes calculation
# calculating the 2.5th, 50th, and 97.5th percentiles 
# for each phase for key process parameters

phase_ranges = (
    df_batch_1_10
    .groupby("phase")[PHASE_RANGE_VARS]
    .quantile([0.025, 0.5, 0.975])
    .unstack(level=-1) 
)

'''
# initialise the MCP server
app = Server("pharma-analytics")

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

'''
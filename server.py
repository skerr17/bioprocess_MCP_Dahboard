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
import os
from mcp.server.fastmcp import FastMCP

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
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "batches-subset-1-10.csv")


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


# initialise the MCP server
app = FastMCP("pharma-analytics")


@app.tool()
async def classify_phase(time_h: float) -> str:
    """Classify which fermentation phase a time point falls in."""
    phase = assign_phase(time_h)
    result = {"time_h": time_h, "phase": phase}
    return json.dumps(result)

@app.tool()
async def get_batch_summary(batch_id: int = None) -> str:
    """Descriptive stats for key process variables in one or all batches."""
    key_vars = [
        "pH(pH:pH)",
        "Temperature(T:K)",
        "Dissolved oxygen concentration(DO2:mg/L)",
        "Penicillin concentration(P:g/L)",
        "Oxygen Uptake Rate(OUR:(g min^{-1}))",
        "Carbon evolution rate(CER:g/h)",
        "Sugar feed rate(Fs:L/h)",
    ]
    subset = df_batch_1_10[df_batch_1_10["batch_id"] == batch_id] if batch_id else df_batch_1_10
    if subset.empty:
        return json.dumps({"error": f"Batch {batch_id} not found."})
    summary = subset[key_vars].describe().loc[["mean", "std", "min", "max"]]
    result = {
        "batch_id": batch_id if batch_id else "all",
        "n_observations": len(subset),
        "stats": {
            col: {stat: round(float(summary.loc[stat, col]), 4) for stat in summary.index}
            for col in key_vars
        },
    }
    return json.dumps(result, indent=2)


@app.tool()
async def get_phase_envelope(phase: str, variable: str = None) -> str:
    """Phase-specific operating envelopes — 2.5th, 50th, 97.5th percentiles."""
    valid_phases = list(PHASE_BOUNDARIES.keys())
    if phase not in valid_phases:
        return json.dumps({"error": f"Invalid phase. Choose from: {valid_phases}"})
    row = phase_ranges.loc[phase]
    if variable:
        matching = {k: round(float(v), 4) for k, v in row.items() if variable in k}
        if not matching:
            avail = sorted({str(c[0]) for c in phase_ranges.columns})
            return json.dumps({"error": f"Variable not found. Available: {avail}"})
        out = {}
        for k, v in matching.items():
            quantile = str(k[1])
            out[quantile] = v
        return json.dumps({"phase": phase, "variable": variable, "envelope": out})
    result = {}
    for col, val in row.items():
        var_name = str(col[0])
        quantile = str(col[1])
        result.setdefault(var_name, {})[quantile] = round(float(val), 4)
    return json.dumps({"phase": phase, "envelopes": result}, indent=2)


@app.tool()
async def check_deviation(batch_id: int, time_h: float) -> str:
    """Check Hotelling T² status for a batch at a given time."""
    batch_mask = df_batch_1_10["batch_id"] == batch_id
    if not batch_mask.any():
        return json.dumps({"error": f"Batch {batch_id} not found."})
    batch_df = df_batch_1_10[batch_mask].copy()
    idx = (batch_df["Time (h)"] - time_h).abs().idxmin()
    row = batch_df.loc[idx]
    t2 = float(row["T2"])
    if t2 > ucl_99:
        status = "OUT_OF_CONTROL_99"
        message = "T² exceeds 99% limit — significant deviation from golden batch."
    elif t2 > ucl_95:
        status = "WARNING_95"
        message = "T² exceeds 95% limit — monitor closely."
    else:
        status = "IN_CONTROL"
        message = "Batch is within the golden batch envelope."
    return json.dumps({
        "batch_id": batch_id,
        "requested_time_h": time_h,
        "nearest_time_h": round(float(row["Time (h)"]), 2),
        "phase": str(row["phase"]),
        "t2_value": round(t2, 3),
        "ucl_95": round(float(ucl_95), 3),
        "ucl_99": round(float(ucl_99), 3),
        "status": status,
        "message": message,
    }, indent=2)


@app.tool()
async def get_pca_status(batch_id: int, time_h: float = None) -> str:
    """PC1/PC2/PC3 scores for a batch at a time point or as a phase trajectory summary."""
    batch_mask = df_batch_1_10["batch_id"] == batch_id
    if not batch_mask.any():
        return json.dumps({"error": f"Batch {batch_id} not found."})
    batch_scores = scores[batch_mask.values, :N_PCS]
    batch_times = df_batch_1_10.loc[batch_mask, "Time (h)"].values
    batch_phases = df_batch_1_10.loc[batch_mask, "phase"].values
    ve = {f"PC{i+1}": round(float(pca.explained_variance_ratio_[i]) * 100, 2) for i in range(N_PCS)}
    if time_h is not None:
        nearest_idx = int(np.argmin(np.abs(batch_times - time_h)))
        return json.dumps({
            "batch_id": batch_id,
            "nearest_time_h": round(float(batch_times[nearest_idx]), 2),
            "phase": str(batch_phases[nearest_idx]),
            "pc_scores": {
                "PC1": round(float(batch_scores[nearest_idx, 0]), 4),
                "PC2": round(float(batch_scores[nearest_idx, 1]), 4),
                "PC3": round(float(batch_scores[nearest_idx, 2]), 4),
            },
            "variance_explained": ve,
        }, indent=2)
    phase_summary = {}
    for phase in ["Lag", "Exponential", "Stationary", "Decline"]:
        mask = batch_phases == phase
        if mask.any():
            phase_summary[phase] = {
                "PC1_mean": round(float(batch_scores[mask, 0].mean()), 4),
                "PC2_mean": round(float(batch_scores[mask, 1].mean()), 4),
                "PC3_mean": round(float(batch_scores[mask, 2].mean()), 4),
            }
    return json.dumps({"batch_id": batch_id, "variance_explained": ve, "trajectory_by_phase": phase_summary}, indent=2)


@app.tool()
async def get_correlations(top_n: int = 10) -> str:
    """Correlation matrix for PCA candidate variables with top N pairs highlighted."""
    corr = df_batch_1_10[PCA_VARS].corr()
    pairs = []
    for i, var1 in enumerate(PCA_VARS):
        for j, var2 in enumerate(PCA_VARS):
            if i < j:
                pairs.append((var1, var2, round(float(corr.loc[var1, var2]), 4)))
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    return json.dumps({
        "top_correlated_pairs": [
            {"var1": p[0], "var2": p[1], "correlation": p[2]}
            for p in pairs[:top_n]
        ],
    }, indent=2)


if __name__ == "__main__":
    app.run()
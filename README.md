# Bioprocess MCP Dashboard

A Python MCP (Model Context Protocol) server that exposes pharmaceutical bioprocess analytics as conversational tools for Claude Desktop. Built on the IndPenSim V3 penicillin fermentation dataset, it enables natural language querying of batch data, deviation monitoring, and multivariate process analysis.

## Overview

The server loads and fits a golden batch reference model on startup — standardising process variables, performing PCA, and computing Hotelling T² control limits — then exposes the analytics as callable tools. Claude Desktop connects to the server via stdio transport and can answer questions like:

- *"Is batch 5 within the golden batch envelope at hour 120?"*
- *"What process parameters are most correlated with each other?"*
- *"Give me a summary of batch 3"*
- *"What phase is a batch in at hour 75?"*

The original batch analysis this server is built on is available as [batch_data_analysis.pdf](./batch_data_analysis.pdf).

## Tools

| Tool | Description |
|---|---|
| `classify_phase` | Assigns a fermentation phase (Lag / Exponential / Stationary / Decline) to a given time point |
| `get_batch_summary` | Returns descriptive statistics for key process variables across one or all batches |
| `get_phase_envelope` | Returns 2.5th / 50th / 97.5th percentile operating envelopes per fermentation phase |
| `check_deviation` | Computes Hotelling T² for a batch at a given time and returns status against 95% and 99% UCLs |
| `get_pca_status` | Returns PC1/PC2/PC3 scores for a batch at a time point or as a phase trajectory summary |
| `get_correlations` | Returns a correlation matrix for PCA candidate variables with top-N pairs highlighted |

## Dataset

[IndPenSim V3](https://www.kaggle.com/datasets/danbenzaquen/indpensim-penicillin-simulation) — Industrial Penicillin Simulation (Goldrick et al., 2015). 11,585 time-series observations across 10 recipe-driven batches, 34 process variables.

The CSV is not included in this repository. Download it from Kaggle and place it at:

```
data/batches-subset-1-10.csv
```

## Requirements

- Python 3.12+
- Claude Desktop

## Installation

```bash
git clone https://github.com/skerr17/bioprocess_MCP_Dahboard.git
cd bioprocess_MCP_Dahboard

uv venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS / Linux

uv pip install -r requirements.txt
```

## Claude Desktop Configuration

Add the following to your `claude_desktop_config.json`:

- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "pharma-analytics": {
      "command": "/absolute/path/to/.venv/Scripts/python.exe",
      "args": ["/absolute/path/to/server.py"]
    }
  }
}
```

Use full absolute paths. Restart Claude Desktop after saving.

## Verify the connection

In Claude Desktop, go to **Settings → Developer**. The `pharma-analytics` server should show as **running**.

Test with:
> *"What phase is a batch in at hour 75?"*

Expected response: Exponential.

## Project Structure

```
bioprocess_MCP_Dahboard/
├── server.py                   # MCP server — data loading, model fitting, tool definitions
├── requirements.txt            # Python dependencies
├── README.md
├── batch_data_analysis.pdf     # Original golden batch analysis
├── data/                       # CSV goes here (gitignored)
└── .gitignore
```

## Analytics

The server fits the following on startup using batches 1–10 as the golden batch reference:

- **StandardScaler + PCA** — 10 process variables, 3 components (~71% variance explained)
- **Hotelling T²** — multivariate control chart with F-distribution UCLs at 95% and 99%
- **Phase envelopes** — percentile-based operating ranges per fermentation phase

## Data

The `data/` directory is gitignored. The CSV must be sourced independently from Kaggle and placed locally before running the server.

## Reference

Goldrick, S., Ştefan, A., Lovett, D., Montague, G., & Lennox, B. (2015). The development of an industrial-scale fed-batch fermentation simulation. *Journal of Biotechnology*, 193, 70–82.
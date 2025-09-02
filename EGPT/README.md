# Code for: Electron-Gated Proton Transfer: A Closed-Form Electrostatic De-sign Rule with an External Benchmark

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
<!-- When you have a DOI, uncomment this line and replace the placeholder -->
<!-- [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX) -->

This repository contains the code to reproduce all figures and data presented in our manuscript, *"Electron-Gated Proton Transfer: A Closed-Form Electrostatic De-sign Rule with an External Benchmark"*.

---

## Abstract

[Paste the abstract of your paper here. This provides a quick scientific overview of the project for visitors.]

---

## Generated Figures

Running the main script will generate all figures from the paper. Here are some key examples:

| Figure 1 | Figure 2 |
| :---: | :---: |
| <img src="png/Fig1_alpha_vs_R.png" width="400"> | <img src="png/Fig2_dG_and_rate_vs_ne.png" width="400"> |

| Figure 3 | Figure 4 |
| :---: | :---: |
| <img src="png/Fig3_alpha_heatmap.png" width="400"> | <img src="png/Fig4_orientation.png" width="400"> |

| Figure 5 | Figure 6 |
| :---: | :---: |
| <img src="png/Fig5_multipole.png" width="400"> | <img src="png/Fig6_dynamic_gating.png" width="400"> |

---

## Setup and Usage

### Prerequisites

This project requires Python 3.8 or newer.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
    ```

2.  **(Recommended) Create and activate a virtual environment:**
    ```bash
    # For Unix/macOS
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install dependencies:**
    All required packages are listed in `requirements.txt`. Install them with pip:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: If `requirements.txt` is not provided, you can typically install the necessary libraries manually (e.g., `pip install numpy pandas matplotlib`).*

### Execution

To generate all figures and associated CSV data files, simply run the main script from the project's root directory:

```bash
python simulation.py
```

This command will:
1.  Create two directories if they don't exist: `csv/` and `png/`.
2.  Run all simulations and calculations.
3.  Save raw data as `.csv` files in the `csv/` directory.
4.  Save all figures as `.png` files in the `png/` directory.

---

## Repository Structure

```
.
├── egpc_simulation.py    # Main Python script to run simulations and generate figures
├── requirements.txt      # List of Python dependencies
├── csv/                  # Directory for generated CSV data (created by script)
├── png/                  # Directory for generated figures (created by script)
├── LICENSE               # Project license file (e.g., MIT License)
└── README.md             # This file
```

---



---

## License

This project is licensed under the MIT License.

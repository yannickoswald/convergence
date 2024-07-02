# Global Convergence of Incomes in a Climate-Constrained World

## Repository Overview

This repository stores all code related to the paper **Oswald, 2024, Global Convergence of Incomes in a Climate-Constrained World** (in preparation/review).

All relevant output figures for this paper are produced in Jupyter notebooks. These notebooks employ the classes that represent the actual model, which are Python files.

### Model Python Files

1. **country_class.py** - Defines the country-specific attributes and behaviors.
2. **scenario_class.py** - Handles the different scenarios for income convergence modeling.

### Supporting Modeling Files for Plots

1. **scenariosweeper_class.py** - Necessary for 2-D trade-offs plots.
2. **plots_class.py** - Contains methods for generating various plots.

### Notebooks for Figure Outputs

1. **first_data_explorations.ipynb** - Initial data checks and analysis, includes FIGURE 1.
2. **run_figure2.ipynb** - Generates FIGURE 2.
3. **run_figure3.ipynb** - Generates FIGURE 3.
4. **run_figure4.ipynb** - Generates FIGURE 4.
5. **run_figure5.ipynb** - Generates FIGURE 5.

### Pre-Modelling Data Processing Notebooks

1. **clean_extend_pip_data.ipynb** - Processes and extends initial data for modeling.

## Getting Started

To get started with the code, ensure you have the necessary dependencies installed. You can set up your environment using the provided `requirements.txt` file.

```sh
pip install -r requirements.txt

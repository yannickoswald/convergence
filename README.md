# Global Convergence of Incomes in a Climate-Constrained World

## Repository Overview

This repository contains the data, model code, and execution scripts for the paper:
**Oswald & Millward-Hopkins, 2025, The Carbon Emissions of Global Income Convergence Scenarios**  
[Read the pre-print on SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5333077) *(in preparation/review)*.

The project utilizes a custom Python-based model to evaluate the trade-offs and carbon budget transgressions associated with global income convergence. All output figures are fully reproducible via the provided Jupyter notebooks.

## Repository Structure

The repository is modularized into three main directories to separate raw inputs, model architecture, and executable scripts:

*   `data/`
    Contains all necessary empirical datasets processed by the model. The raw sources include freely available data from the World Bank, the UN, and processed data from the IIASA SSP explorer.
    *   *References:* [World Bank PIP](https://pip.worldbank.org/), [IIASA SSP Database](https://tntcat.iiasa.ac.at/SspDb/), [UN World Population Prospects](https://population.un.org/wpp/).
*   `model_code/`
    Contains the core Python classes that define the simulation logic.
    *   `country_class.py`: Defines country-specific attributes, emission trajectories, and growth behaviors.
    *   `scenario_class.py`: Handles global scenarios, parameter sweeps, and convergence logic.
    *   `scenariosweeper_class.py`: Facilitates multi-parameter trade-off analysis.
    *   `plots_class.py`: Contains custom methods for generating the paper's figures.
*   `jupyter_scripts/`
    Contains the sequential notebooks used to process data, explore baselines, and generate the final publication figures.
    *   `clean_extend_pip_data.ipynb`: Pre-processing of the initial datasets.
    *   `first_data_explorations.ipynb`: Initial data checks and generation of **Figure 1**.
    *   `run_figure2.ipynb` to `run_figure5.ipynb`: Scripts executing the model to generate **Figures 2 through 5**.
    *   `run_country_specifics.ipynb`: Supplementary country-level analysis not featured in the main text.

## Getting Started

To replicate the findings, clone this repository and set up a virtual environment with the required dependencies.

1. **Clone the repository:**
   ```sh
   git clone [https://github.com/yannickoswald/global-convergence-incomes.git](https://github.com/yannickoswald/global-convergence-incomes.git)
   cd global-convergence-incomes


2. **Install dependencies:**
It is recommended to use a virtual environment. Install the required packages via:

    Bash
    pip install -r requirements.txt

    Usage
    Each Jupyter notebook in the jupyter_scripts/ directory is designed to be run independently. The notebooks are configured to automatically locate the model architecture in the model_code/ directory, so no files need to be moved.

3. **Launch the Jupyter environment from the root of the repository:**

    Bash
    jupyter lab
    Navigate to jupyter_scripts/ within the interface and execute the desired notebook to reproduce the corresponding figure.

Reporting Issues
If you encounter any issues reproducing the results or have questions regarding the methodology:

Navigate to the repository's Issues page.

Click New Issue.

Provide a descriptive title and detailed information regarding the error or question.

License
This project is licensed under the MIT License - see the LICENSE.txt file for details.

Contact
For further academic inquiries, please contact: y-oswald@web.de
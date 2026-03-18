# Nuclear_Fuel_Cycle_Simulation
Models of spent nuclear fuel strategies using Monte Carlo simulations.  It assesses economic outcomes, recovered actinide fractions, waste mass, and generates CDF, Pareto, tornado, and violin plots for open, partial, and advanced recycling.   The framework integrates UQ, sensitivity analysis, and risk metrics like CVaR.
## Features
- Monte Carlo simulation of spent nuclear fuel mass, isotopic composition, and recycling efficiency.
- Evaluation of three fuel cycle scenarios:
  - Once-through (OTC)
  - Partial recycling (Mono-recycle)
  - Advanced recycling (Multi-recycle)
- Generation of journal-quality plots:
  - Cumulative Distribution Function (CDF)
  - Pareto fronts
  - Tornado and violin plots
  - Stacked economic breakdown
- Risk and uncertainty analysis:
  - Conditional Value at Risk (CVaR)
  - Expected regret
  - Sobol sensitivity analysis
  - ## Installation

1. Create a virtual environment:
git clone https://github.com/<USERNAME>/<REPO>.git
cd <REPO>
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows

2. Install required packages:

pip install -r requirements.txt

3. Run the main simulation script:

python src/main.py

## References

1. Zhou C, Liu X, Gu Z, Wang Y. Economic analysis of two nuclear fuel cycle options. Ann Nucl Energy. 2014;72:77–85.
2. Taylor R, Mathers G, Banford A. The development of future options for aqueous recycling of spent nuclear fuels. Prog Nucl Energy. 2023;164:104837.
3. Carvalho KA, et al. Closing the nuclear fuel cycle: Strategic approaches for NuScale-like reactor. Nucl Eng Des. 2024;430:113672.
...

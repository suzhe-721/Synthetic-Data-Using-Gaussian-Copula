# GaussianCopulaSynthesizer
## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [Updates](#updates)
- [Experiment Results](#experiment-results)

  
## Overview
`GaussianCopulaSynthesizer` is a Python class for synthesizing data using Gaussian Copula models. This tool is essential in scenarios where data sharing is restricted due to privacy concerns. The model is constructed based on the Synthetic Data Vault (SDV) framework. What sets this implementation apart is its use of Gaussian Mixture Models (GMM) to fit distributions to each column, as opposed to traditional distribution methods.

## Features
- **Data Type Handling:** Handles numerical, categorical, and datetime data types.
- **Missing Value Management:** Efficiently manages missing values in datasets.
- **Customizable Synthesis:** Generates synthetic data with various customization options.
- **Performance Metrics:** Records execution times for different processes.

## Requirements
- Python 3.8
- Pandas
- NumPy
- SciPy
- scikit-learn

## Installation
Install the required packages using pip:
```bash
pip install pandas numpy scipy scikit-learn
```
## Usage
To use `GaussianCopulaSynthesizer`, follow these steps:

### Initialization:
Initialize the synthesizer with a file path to your dataset.
```python
from gaussian_copula_synthesizer import GaussianCopulaSynthesizer
synthesizer = GaussianCopulaSynthesizer(filepath='your_dataset.csv')
```

### Preprocessing:
Preprocess the data to handle different data types and missing values.
```python
synthesizer._identify_columns()
synthesizer.convert_datetime_to_numerical()
synthesizer.handle_missing_values()
synthesizer.assign_intervals()
synthesizer.preprocess_data()
```
### Fitting Distribution:
Fit Gaussian Mixture Models to the data.
```python
synthesizer.fit_distributions()
synthesizer.compute_gmm_cdf()
synthesizer.standard_gaussian_all()
```
### Synthetic Data Generation:
Generate the synthetic dataset.
```python
synthetic_data = synthesizer.generate_synthetic_data(num_rows=1000)
```
### Post-Processing:
Convert the numerical columns back to their original format.
```python
final_data = synthesizer.post_process()
```
### Execution Time:
Get the execution time in seconds for each process and return it as a data frame for the user to inspect
```python
synthesizer.get_execution_times_df()
```
## Contributing
Contributions to GaussianCopulaSynthesizer are welcome! Please read our [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute.

## Updates

### New Features
- **Enhanced Time Conversion:** Implemented functionality to convert time data into numerical values for improved processing efficiency.
- **Optimized Handling of Patterned Columns:** For columns with specific patterns like `customerID`, which are less meaningful in synthetic data generation, the updated model now bypasses traditional distribution fitting. Instead, it captures the column format and randomly generates numbers adhering to this pattern, enhancing computational efficiency.

## Experiment Results
The `GaussianCopulaSynthesizer` has been rigorously tested on several datasets available on Kaggle. Comparisons between the performance of the original SDV model and this enhanced version have been conducted. Detailed results of these experiments can be found in a separate document: [Experiment-Result.md](Experiment-Result.md).

## Future Work
Further optimization of the `GaussianCopulaSynthesizer` is ongoing, particularly focusing on reducing the runtime. Future iterations aim to facilitate application to SQL datasets, enabling comprehensive synthetic data generation for extensive datasets.



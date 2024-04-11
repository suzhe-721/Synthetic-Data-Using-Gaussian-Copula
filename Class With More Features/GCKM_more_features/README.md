# GaussianCopulaKmeansSynthesizer

## Overview
`GaussianCopulaKmeansSynthesizer` is a Python class designed for synthesizing data using Gaussian Copula models integrated with K-means clustering. This tool excels in generating synthetic datasets that closely resemble the statistical properties of the original data, making it ideal for scenarios where data sharing is limited due to privacy concerns.

## Key Features
- **Advanced Gaussian Copula Model:** Employs Gaussian Copula models for accurate data synthesis.
- **Innovative K-means Initialization:** Utilizes a unique approach to initialize K-means clustering, enhancing the efficiency and accuracy of the model.
- **Versatile Dataset Compatibility:** Designed to be applicable across a variety of datasets including home insurance, travel insurance, and fraud insurance.
- **Performance Excellence:** Demonstrates superior performance in comparison to the SDV (Synthetic Data Vault), albeit with a little bit longer computation times.

## Requirements
- Python 3.8
- Pandas
- NumPy
- SciPy
- scikit-learn
- tqdm

## Installation
Install the required packages using pip:
```bash
pip install pandas numpy scipy scikit-learn tqdm
```
## Innovative K-means Initialization
Unlike traditional methods that often rely on the elbow method to determine the optimal number of clusters, GaussianCopulaKmeansSynthesizer introduces an innovative quantile-based approach. This method dynamically determines the best number of clusters for K-means, significantly reducing the need for repetitive pre-runs and enhancing overall model efficiency. The detailed explanation could be found in file named **KMeans Initialization**
## Usage
To use GaussianCopulaKmeansSynthesizer, follow these steps:
- Initialization:
Initialize the synthesizer with your dataset's file path.
```python
from gaussian_copula_kmeans_synthesizer import GaussianCopulaKmeansSynthesizer
synthesizer = GaussianCopulaKmeansSynthesizer(filepath='your_dataset.csv')
```
- Data Preprocessing and Model Fitting:
Preprocess the data and fit the model.
```python
synthesizer._identify_columns()
synthesizer.convert_datetime_to_numerical()
synthesizer.handle_missing_values()
synthesizer.assign_intervals()
synthesizer.preprocess_data()
synthesizer.get_distribution()
synthesizer.calculate_cdfs()
synthesizer.standard_gaussian_all()
synthesizer.optimal_clusters_dynamic()
synthesizer.get_Kmeans()
```
- Generate Synthetic Data:
Create synthetic data based on the fitted model.
```python
synthesizer.generate_data()
synthesizer.generate_synthetic_data(num_of_rows)
```
### Execution Time:
Get the execution time in seconds for each process and return it as a data frame for the user to inspect
```python
synthesizer.get_execution_times_df()
```
## Experimental Results
Comprehensive evaluation results have been conducted on various datasets and are documented in the Experiment-Result.md file in the GitHub repository. These results showcase the model's effectiveness and improvements over the SDV. The detailed experimental result could be found ar [GCKM-Experiment-Result.md](GCKM-Experiment-Result.md).

# Synthetic Data Repository Structure

This section provides a detailed overview of the repository's directory structure and elaborates on the contents and purpose of each folder.

## Directory Explanation

### `SDV_Evaluation/`
**Description**: This folder contains comprehensive evaluations of the Synthetic Data Vault (SDV) package.

### `Gaussian Copula/`
**Description**: Validates the Gaussian Mixture Model's (GMM) performance in fitting the distribution of the American Income Dataset. It also generates data based on Gaussian Copula for two numerical columns and compares the results with the SDV.

### `GMCM_KMeans/`
**Description**: Introduces a new model using KMeans to initialize the Gaussian Mixture Model (GMM).
- **Contents**:
  - `GMCM.ipynb`: Validates the performance of GMCM_KMeans for two numerical columns.
  - `GMCM_Improved.ipynb`: Utilizes GMM to initially fit the distribution, followed by GMCM_KMeans for generating synthetic data. This is then validated and compared with the SDV.

### `Full size synthetic data/`
**Description**: Applies the models to the entire dataset, encompassing both numerical and categorical data.
- **Contents**:
  - `GMM_all_columns_GMM.ipynb`: Applies GMM solely to the entire dataset.
  - `Kmeans_syn.ipynb`: Employs a model involving KMeans to initialize the GMM for the entire dataset.
  - `GMM_all_columns_GMM_improved.ipynb`: Uses GMM to fit the distribution first, followed by GMCM_KMeans for the entire dataset.

### `class with more features/`
**Description**: Enhances both models with additional features such as datetime conversion and includes experimental results for comparison with the SDV.
- **Contents**:
  - It contains the evaluation of GMCM and GCKM by using the American Income Dataset, Travel Insurance and Fraud Insurance dataset 

### `Experiment Stage/`
**Description**: Assesses the efficacy of the Gaussian Mixture Copula Model (GMCM) and the Gaussian Copula KMeans Model (GCKM).
- **Contents**:
  - `experiment_evaluation for the gckm.ipynb`: Compares correlation matrices from the GCKM's synthetic data
  - `All_Model_learning_curve_crab.ipynb`: Performs a Learning Curve Evaluation with runtime efficiency analysis for the GMCM, GCKM, SDV and Y-data on the crab age dataset.
  - `All_Model_learning_curve_fetal.ipynb`: Performs a Learning Curve Evaluation for the GMCM, GCKM, SDV and Y-data on the fetal health dataset.
  - `eigen_analysis.ipynb`: Analyzes the effectiveness of eigenvalue analysis in determining optimal cluster numbers using metrics like the Silhouette Score, Davies-Bouldin Score, and Gap statistic.
  - `All_Model_Aug_data_compr_fetal.ipynb`: Evaluates the quality of GMCM, GCKM, SDV and Y-data generated synthetic data by augmenting the original fetal health dataset with machine learning predictions.
  - `All_Model_Aug_data_compr_crab.ipynb`: Evaluates the quality of GMCM, GCKM, SDV and Y-data generated synthetic data by augmenting the original crab age dataset with machine learning predictions.

## Additional Information

For in-depth details about each component, refer to the README files in each directory. All datasets used in this repository can be accessed here: [Dataset File](https://drive.google.com/drive/folders/1wjKrpA6wsNDpXngOs0OKa-xy3G2MhTGb?usp=sharing)


## FULL-SIZE AMERICAN INCOMES DATASET SYNTHESIZE DATA
1. The ipynb file named **GMM_all_columns_GMM.ipynb** used GMM to capture the distribution  of each column
then used the Gaussian Copula to capture the column relationship, it result in the score of the synthetic data
improved from **71.81%** to **87.67%**

2. The ipynb file named **Kmeans_Syn.ipynb** used KMeans to cluster the dataset and use the GMM to capture the distribution  of each cluster
then used the Gaussian Copula to capture the column relationship, it result in the score of the synthetic data
improved from **71.81%** to **81.65%**

3. The ipynb file named **GMM_all_columns_GMM_improve.ipynb** used GMM to model the distribution of each column and apply KMeans to cluster and use the GMM to capture the distribution  of each cluster
then used the Gaussian Copula to capture the column relationship, it result in the score of the synthetic data
improved from **71.81%** to **87.7%**


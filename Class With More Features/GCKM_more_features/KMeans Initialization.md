## Optimal Cluster Determination for KMeans Clustering

### Overview
This implementation replaces the traditional, subjective elbow method for determining the optimal number of KMeans clusters with an analytical and dynamic approach, thereby automating the process and enhancing objectivity.

### Methodology

#### 1. **Covariance Matrix Calculation**
The data is transformed to a standard normal distribution, and the covariance matrix is calculated. This matrix is foundational for the eigenvalue decomposition, a key component in our cluster determination process.

#### 2. **Eigenvalue Decomposition**
Eigenvalue decomposition is performed on the covariance matrix to understand the variance in the data attributable to each principal component.

#### 3. **Eigenvalue Analysis**
The eigenvalues are sorted in descending order. We then observe the percentage change between consecutive eigenvalues, with significant changes indicating potential cluster boundaries.

#### 4. **Percentile-Based Categorization**
The 25th and 75th percentiles of the delta eigenvalues are calculated. These serve as thresholds for identifying significant shifts in eigenvalues, which suggest potential cluster boundaries.

#### 5. **Identifying Cluster Boundaries**
We iterate through the percentage changes in eigenvalues. Values exceeding the upper percentile or falling between the lower and upper percentiles are marked as potential cluster boundaries.

#### 6. **Determining the Number of Clusters**
The optimal number of clusters is determined as the count of identified cluster boundaries plus one. This number is then used to initialize and fit the KMeans and Gaussian Mixture Model (GMM) clustering.

### Conclusion
This approach provides a data-driven, automated method to determine the optimal number of clusters, enhancing the robustness and efficiency of the KMeans clustering process. It moves away from subjective methods like the elbow method, adapting dynamically to the inherent structure of the dataset.

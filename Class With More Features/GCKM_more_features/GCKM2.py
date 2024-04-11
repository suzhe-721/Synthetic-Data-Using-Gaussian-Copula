import numpy as np
import pandas as pd
from scipy.stats import kstest, norm, uniform, beta, expon, truncnorm, anderson
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import warnings
from joblib import Parallel, delayed
import scipy.stats as stats
from tqdm import tqdm
from scipy.optimize import brentq
from itertools import combinations
import time
import random
import logging
warnings.filterwarnings(action='ignore')


class GaussianCopulaKmeansSynthesizer:

    def __init__(self, filepath):
        self.filepath = filepath
        self.data = pd.read_csv(self.filepath)
        self.execution_times = {}
        self.distributions = {}
        self.match_column = []
        self.num_column = []
        self.category_column = []
        self.special_numeric_series_columns = []
        self.datetime_column = []
        self.special_column_prefixes = []
        self.hyphenated_numeric_columns = []
        self.binary_columns_for_missing = []
        self.digit_counts = {}  # Store digit counts for each column
        self.hyphenated_format = {}
        self.column_min = {}  # Store minimum values for each column
        self.column_max = {}
        self.columns_with_missing_values = []
        self.category_intervals = {}
        self.num_of_clusters = 0
        self.gmms = {}
        self.mixture_gaussian = {}
        self.column_cdfs = {}
        self.parameters_collections = {}
        self.column_inverse_cdfs = {}
        self.processed_data = None
        self.synthetic_data_raw = pd.DataFrame()
        self.synthetic_data = pd.DataFrame()
        self.cdf_results = pd.DataFrame()
        self.standard_gaussian_df = pd.DataFrame()
    
    def _start_timer(self):
        """
        Start the timer for a process.
        """
        self._timer_start = time.time()

    def _stop_timer(self, process_name):
        """
        Stop the timer and store the elapsed time for the process.

        Parameters:
        - process_name (str): The name of the process for which the timer was started.
        """
        elapsed_time = time.time() - self._timer_start
        self.execution_times[process_name] = elapsed_time


    # def detect_match(self):
    #     column_pairs = list(combinations(self.data.columns, 2))
    #     for column_pair in column_pairs:
    #         temp = self.data[[column_pair[0], column_pair[1]]].apply(lambda x: x[column_pair[0]] == x[column_pair[1]], axis=1)
    #         count_true = temp[temp == True].count()
    #         ratio = count_true/len(temp)
    #         if ratio > 0.95:
    #             self.match_column.append(column_pair)



    def _identify_columns(self):
        """
        Identify and classify the columns of the dataset.

        Determines the type of each column (numerical, categorical, datetime, unique identifier, etc.)
        and classifies them into appropriate attributes of the class.
        """

        self._start_timer()
        self.unique_identifier_columns = []
        temp_special_columns = []
        temp_special_num_pattern = []

        data_info = self.data.dtypes.to_dict()
        for key, dtype in data_info.items():
            unique_values = self.data[key].nunique()
            total_values = len(self.data[key])

            # Check for uniqueness
            if unique_values == total_values:
                self.unique_identifier_columns.append(key)
                continue  # Skip further checks for this column
            
            if dtype == 'object' and pd.to_datetime(self.data[key], errors='coerce').notna().any():
                self.datetime_column.append(key)
            elif dtype == 'object':
                self.category_column.append(key)
            elif dtype in ['int64', 'float64']:
                self.num_column.append(key)

            if self.data[key].isnull().any():
                self.columns_with_missing_values.append(key)
        
        temp_column = self.unique_identifier_columns

        for column in temp_column:      
            try:
                if self.data[column].str.match(r'[A-Za-z]+\d+').all():
                    extracted, prefix_lengths, number_lengths = self.extract_number_part(self.data[column])
                    self.special_numeric_series_columns.append(column)
                    # Store extracted prefixes and their lengths
                    self.special_column_prefixes[column] = extracted[0].iloc[0]
                    self.digit_counts[column] = number_lengths.iloc[0]

                elif self.data[column].str.match(r'\d+-\d+-\d+').all():
                    # Handling hyphenated numeric columns
                    self.hyphenated_numeric_columns.append(column)
                    combined_numbers, digit_counts = self.extract_numbers(self.data[column])
                    self.hyphenated_format[column] = digit_counts.iloc[0].tolist()  # Store digit count format
            except AttributeError:
                self.unique_identifier_columns.remove(column)
                self.category_column.append(column)
        
        if self.hyphenated_numeric_columns == [] and self.special_numeric_series_columns == []:
            self.category_column.extend(self.unique_identifier_columns)
            self.unique_identifier_columns.clear()

        
               
        #self.unique_identifier_columns = [col for col in self.unique_identifier_columns if col not in temp_special_columns]
        self.special_numeric_series_columns.extend(temp_special_columns)

        #self.unique_identifier_columns = [col for col in self.unique_identifier_columns if col not in temp_special_num_pattern]
        self.hyphenated_numeric_columns.extend(temp_special_num_pattern)

        self._stop_timer("_identify_columns")
    
    def extract_number_part(self, series):
        """
        Extract alphanumeric prefix and numeric part from a series.

        Parameters:
        - series (pd.Series): A pandas Series from which to extract the alphanumeric prefix and numeric part.

        Returns:
        - tuple: A tuple containing the extracted prefix, prefix lengths, and number lengths.
        """
        
        # Extract both alphanumeric prefix and numeric part
        regex_pattern = r'([A-Za-z]+)(\d+)'
        extracted = series.str.extract(regex_pattern)
        # Calculate the length of each part (prefix and number)
        prefix_lengths = extracted[0].apply(lambda x: len(x) if pd.notnull(x) else 0)
        number_lengths = extracted[1].apply(lambda x: len(x) if pd.notnull(x) else 0)
        
        return extracted, prefix_lengths, number_lengths
    
    def extract_numbers(self, series):
        
        regex_pattern = r'(\d+)-(\d+)-(\d+)'
        extracted = series.str.extract(regex_pattern)

        # Calculate the number of digits in each part
        digit_counts = extracted.applymap(lambda x: len(str(x)) if pd.notnull(x) else 0)

        # Combine the numbers into a single string
        combined_numbers = extracted.apply(lambda row: '-'.join(row.dropna()), axis=1)

        return combined_numbers.astype(str), digit_counts

    def generate_random_hyphenated_number(self, format_pattern):
        """
        Generate a random number based on the observed format pattern.
        format_pattern: List of integers representing the length of each numeric segment.
        """
        random_number_parts = [str(random.randint(0, 10**length - 1)).zfill(length) for length in format_pattern]
        return '-'.join(random_number_parts)
    
    def generate_random_number(self, length):
        """
        Generate a random number of a specified length.
        """
        return str(random.randint(0, 10**length - 1)).zfill(length)
    
    def convert_datetime_to_numerical(self):
        ref_dt = pd.Timestamp('1970-01-01')

        for column in self.datetime_column:
            # Convert to datetime, coerce errors to NaT (missing values)
            self.data[column] = pd.to_datetime(self.data[column], errors='coerce')

            # Convert datetime to numerical value (e.g., days since reference date)
            self.data[column] = (self.data[column] - ref_dt) / np.timedelta64(1, 'D')

            # Reclassify as a numerical column
            self.num_column.append(column)
            if column in self.category_column:
                self.category_column.remove(column)
    
    def transform_to_original_format(self, column, synthetic_series):
        format_pattern = self.hyphenated_format[column]
        formatted_series = synthetic_series.apply(lambda x: '-'.join(part.zfill(length) for part, length in zip(x.split('-'), format_pattern)))
        return formatted_series

    def handle_missing_values(self):
        self._start_timer()
        for column in self.data.columns:
            if self.data[column].isnull().any():
                # Create a binary column to mark missing values
                binary_column_name = column + '_missing'
                self.data[binary_column_name] = self.data[column].isnull().astype(int)
                self.binary_columns_for_missing.append(binary_column_name)
                # Fill missing values in the original column
                self.data[column] = self.data[column].fillna(method='ffill').fillna(method='bfill')
        self._stop_timer("handle_missing_value")
    
    def convert_special_numerical(self):
        for column in self.special_numeric_series_columns:
            numerical_part = self.extract_number_part(self.data[column])
            self.data[column] = numerical_part


    def assign_intervals(self):
        self.category_intervals = {}
        for column_name in self.category_column:
            column_data = self.data[column_name]
            freq = column_data.value_counts(normalize=True)
            intervals = freq.cumsum()
            category_intervals = {}
            a = 0
            for category, cum_freq in intervals.items():
                b = cum_freq
                category_intervals[category] = (a, b)
                a = b
            self.category_intervals[column_name] = category_intervals

    def sample_from_category(self, category_value, column_name):
        try:
            a, b = self.category_intervals[column_name][category_value]
            mean = (a + b) / 2
            sd = (b - a) / 6
            dist = truncnorm((0 - mean) / sd, (1 - mean) / sd, loc=mean, scale=sd)
            return dist.rvs()
        except KeyError:
            # Debugging information
            print(f"KeyError encountered in sample_from_category:")
            print(f"Column: {column_name}, Category Value: {category_value}")
            print(f"Available Categories in '{column_name}': {self.category_intervals[column_name]}")
            raise

    def preprocess_data(self):
        """
        Function: Convert all the categorical column into numerical column
        Result: Make the dataset have the same data type and prepare for the CDF
        """
        # self._identify_columns()
        # self.assign_intervals()
        self._start_timer()
        for column_name in tqdm(self.category_column):
            self.data[column_name] = self.data[column_name].apply(lambda x: self.sample_from_category(x, column_name))
        self._stop_timer("preprocess_data")

    def optimal_clusters_dynamic(self):
        self.relevant_columns = (set(self.num_column) | set(self.category_column)) - set(self.unique_identifier_columns) - set(self.binary_columns_for_missing)
        relevant_columns_list = list(self.relevant_columns)
        self.clean_data = self.data[relevant_columns_list]
        cov_matrix = np.cov(self.clean_data, rowvar=False)
        
        # Perform eigenvalue decomposition
        eigenvalues, _ = np.linalg.eig(cov_matrix)
        
        # Sort the eigenvalues in descending order
        sorted_eigenvalues = np.sort(eigenvalues)[::-1]
        
        # Calculate the percentage change between consecutive eigenvalues
        delta_eigenvalues = np.diff(sorted_eigenvalues) / sorted_eigenvalues[:-1]

        # Calculate percentiles for categorizing delta eigenvalues
        lower_percentile = np.percentile(delta_eigenvalues, 25)
        upper_percentile = np.percentile(delta_eigenvalues, 75)

        # Identifying potential cluster boundaries
        cluster_boundaries = []
        for i, delta in enumerate(delta_eigenvalues):
            if delta > upper_percentile or (lower_percentile < delta < upper_percentile):
                cluster_boundaries.append(i)

        num_clusters = len(cluster_boundaries) + 1 if cluster_boundaries else 1
        
        self.num_of_clusters = num_clusters
    
    def best_fit_distribution(self, column_data, column_name):

        if column_data.nunique() == 1:
            logging.info(f"Column '{column_name}' contains a single unique value: {column_data.iloc[0]}")
            # Return a flag or specific indicator that this column has a single value
            return 'single_value', None
        distributions = ['norm', 'uniform', 'beta', 'expon', 'truncnorm']
        best_fit = None
        best_p_value = -1
        truncation_threshold = 0.07  # Threshold for considering truncation (20% of data at bounds)

        if column_data.min() < 0 or column_data.skew() > 1:
            distributions.remove('beta')

        # Check for potential truncation
        min_count = np.sum(column_data == column_data.min())
        max_count = np.sum(column_data == column_data.max())

        if min_count + max_count >= truncation_threshold * len(column_data):
            # If significant data at bounds, consider it as truncated
            try:
                mean, std_dev = norm.fit(column_data)
                lower_bound = (column_data.min() - mean) / std_dev
                upper_bound = (column_data.max() - mean) / std_dev
                params = stats.truncnorm.fit(column_data, lower_bound, upper_bound)
                _, best_p_value = stats.kstest(column_data, 'truncnorm', args=params)
                best_fit = 'truncnorm'
            except Exception as e:
                logging.warning(f"FitError for truncnorm in column '{column_name}': {e}")
                logging.info(f"Column '{column_name}' data statistics: {column_data.describe()}")
                logging.info(f"Column '{column_name}' data values (sample): {column_data.sample(10)}")
                # Fallback to normal distribution if truncnorm fitting fails
                params = stats.norm.fit(column_data)
                _, best_p_value = stats.kstest(column_data, 'norm', args=params)
                best_fit = 'norm'
        else:
            # If not truncated, proceed with other distributions
            for distribution in distributions:
                try:
                    if distribution == 'norm':
                        params = stats.norm.fit(column_data)
                        _, p_value = stats.kstest(column_data, 'norm', args=params)
                    elif distribution == 'uniform':
                        params = stats.uniform.fit(column_data)
                        _, p_value = stats.kstest(column_data, 'uniform', args=params)
                    elif distribution == 'beta':
                        epsilon = 1e-10
                        scaled_data = (column_data - column_data.min() + epsilon) / (column_data.max() - column_data.min() + 2 * epsilon)
                        params = stats.beta.fit(scaled_data, floc=0, fscale=1)
                        _, p_value = stats.kstest(scaled_data, 'beta', args=params)
                    elif distribution == 'expon':
                        params = stats.expon.fit(column_data)
                        _, p_value = stats.kstest(column_data, 'expon', args=params)
                    elif distribution == 'truncnorm':
                        mean, std_dev = stats.norm.fit(column_data)
                        lower_bound = (column_data.min() - mean) / std_dev
                        upper_bound = (column_data.max() - mean) / std_dev
                        params = stats.truncnorm.fit(column_data, lower_bound, upper_bound)
                        _, best_p_value = stats.kstest(column_data, 'truncnorm', args=params)

                    if p_value > best_p_value:
                        best_p_value = p_value
                        best_fit = distribution
                except Exception as e:
                    # logging.error(f"Error in fitting {distribution} distribution for column '{column_name}': {e}")
                    logging.info(f"Column '{column_name}' data statistics: {column_data.describe()}")
                    logging.info(f"Column '{column_name}' data values (sample): {column_data.sample(10)}")
                    

        return best_fit, best_p_value
    
    def intial_KMean(self):
        self.kmeans = KMeans(n_clusters=self.num_of_clusters, random_state=0)
        self.kmeans.fit(self.clean_data)
        self.kmeans_labels = self.kmeans.labels_
    
    def get_distribution(self):
        self.cluster_distributions = {}
        unique_clusters = np.unique(self.kmeans_labels)

        for cluster_idx in unique_clusters:
            # Filter data for the current cluster
            cluster_data = self.clean_data[self.kmeans_labels == cluster_idx]
            self.cluster_distributions[cluster_idx] = {}

            for column in cluster_data.columns:
                # Apply best_fit_distribution for each column in the cluster
                best_fit, _ = self.best_fit_distribution(cluster_data[column], column)
                self.cluster_distributions[cluster_idx][column] = best_fit
    
    def calculate_cdf(self, column_data, distribution):
        # Fit the distribution and calculate the CDF based on the column_data
        if distribution == 'single_value':
            # Return a constant value or an indicator for single-value columns
            return np.zeros_like(column_data), {'distribution': 'single_value'}
        elif distribution == 'norm':
            mean, std = norm.fit(column_data)
            return norm.cdf(column_data, loc=mean, scale=std), {'distribution': 'norm', 'mean': mean, 'std': std}
        elif distribution == 'beta':
            original_min = column_data.min()
            original_max = column_data.max()
            data_normalized = (column_data - original_min) / (original_max - original_min)
            a, b, loc, scale = beta.fit(data_normalized)
            return beta.cdf(data_normalized, a, b, loc, scale), {'distribution': 'beta', 'a': a, 'b': b, 'loc': loc, 'scale': scale, 'original_min': original_min, 'original_max': original_max}
        elif distribution == 'truncnorm':
            mean, std = norm.fit(column_data)
            low, upp = column_data.min(), column_data.max()
            low_std, upp_std = (low - mean) / std, (upp - mean) / std
            return truncnorm.cdf(column_data, low_std, upp_std, loc=mean, scale=std), {'distribution': 'truncnorm', 'mean': mean, 'std': std, 'low_std': low_std, 'upp_std': upp_std}
        elif distribution == 'uniform':
            min_value, max_value = column_data.min(), column_data.max()
            scale = max_value - min_value
            return uniform.cdf(column_data, loc=min_value, scale=scale), {'distribution': 'uniform', 'min_value': min_value, 'max_value': max_value}
        elif distribution == 'expon':
            scale = 1 / column_data.mean()
            return expon.cdf(column_data, scale=scale), {'distribution': 'expon', 'scale': scale}
        else:
            # Default or error handling if distribution is unrecognized
            return None, {}

    def calculate_cdfs(self):
        self.cluster_cdf_results = {}
        self.cluster_parameters_collections = {}
        epsilon = 1e-10

        unique_clusters = np.unique(self.kmeans_labels)
        for cluster_idx in unique_clusters:
            cluster_data = self.clean_data[self.kmeans_labels == cluster_idx]
            self.cluster_cdf_results[cluster_idx] = {}
            self.cluster_parameters_collections[cluster_idx] = {}

            for column in cluster_data.columns:
                distribution = self.cluster_distributions[cluster_idx].get(column)
                if distribution:
                    cdf_values, parameters = self.calculate_cdf(cluster_data[column], distribution)
                    if distribution != 'single_value':
                        cdf_values = np.where(cdf_values == 0, epsilon, cdf_values)
                        cdf_values = np.where(cdf_values == 1, 1 - epsilon, cdf_values)
                    self.cluster_cdf_results[cluster_idx][column] = cdf_values
                    self.cluster_parameters_collections[cluster_idx][column] = parameters
        
    def standard_gaussian(self, p_value):
        return norm.ppf(p_value)
    
    def standard_gaussian_all(self):
        self.cluster_standard_normal = {}

        unique_clusters = np.unique(self.kmeans_labels)
        for cluster_idx in unique_clusters:
            cdf_results = self.cluster_cdf_results[cluster_idx]
            standard_normal_cluster_data = {}

            for column, cdf_values in cdf_results.items():
                # Skip transformation for 'single_value' columns
                if self.cluster_distributions[cluster_idx][column] == 'single_value':
                    standard_normal_cluster_data[column] = np.zeros_like(cdf_values)
                    continue

                if isinstance(cdf_values, np.ndarray):
                    cdf_values = pd.Series(cdf_values)

                # Transform to standard normal
                standard_normal_data = cdf_values.apply(self.standard_gaussian)

                # Handle NaN values
                if standard_normal_data.isna().any():
                    print(f"NaN values created during transformation in cluster {cluster_idx}, column {column}")
                    mean_value = standard_normal_data.mean()
                    standard_normal_data.fillna(mean_value, inplace=True)

                standard_normal_cluster_data[column] = standard_normal_data

            self.cluster_standard_normal[cluster_idx] = pd.DataFrame(standard_normal_cluster_data)
            
    def check_for_nan_and_inf(self):
        for cluster_idx, standard_normal_data in self.cluster_standard_normal.items():
            # Check for NaN values
            if standard_normal_data.isna().any().any():
                print(f"Cluster {cluster_idx} contains NaN values.")

            # Check for infinite values
            if np.isinf(standard_normal_data.values).any():
                print(f"Cluster {cluster_idx} contains infinite values.")

    def initialize_gmm_for_clusters(self, max_components=15):
        self.cluster_gmms = {}

        unique_clusters = np.unique(self.kmeans_labels)
        for cluster_idx in unique_clusters:
            standard_normal_data = self.cluster_standard_normal[cluster_idx]

            # Check if the cluster has enough variance
            if standard_normal_data.var().min() == 0:
                print(f"Cluster {cluster_idx} has no variance in one or more dimensions. Skipping GMM.")
                self.cluster_gmms[cluster_idx] = None
                continue

            cluster_mean = standard_normal_data.mean().values
            cluster_covariance = standard_normal_data.cov().values
            regularization_value = 1e-6
            np.fill_diagonal(cluster_covariance, cluster_covariance.diagonal() + regularization_value)
            best_gmm, lowest_bic = None, np.inf

            # Iterate over a range of component numbers to find the best one based on BIC
            for num_components in range(1, max_components + 1):
                gmm = GaussianMixture(n_components=num_components, covariance_type='full', random_state=0, means_init=[cluster_mean] * num_components, precisions_init=[np.linalg.inv(cluster_covariance)] * num_components)
                gmm.fit(standard_normal_data)
                bic = gmm.bic(standard_normal_data)

                if bic < lowest_bic:
                    lowest_bic = bic
                    best_gmm = gmm

            self.cluster_gmms[cluster_idx] = best_gmm
    def generate_data_for_cluster(self, cluster_idx, num_samples):
        cluster_covariance = self.cluster_standard_normal[cluster_idx].cov().values
        regularization_value = 1e-6
        np.fill_diagonal(cluster_covariance, cluster_covariance.diagonal() + regularization_value)
        # Cholesky decomposition
        L = np.linalg.cholesky(cluster_covariance)
        # Generate samples
        samples = np.dot(np.random.randn(num_samples, L.shape[0]), L.T)

        return pd.DataFrame(samples, columns=self.cluster_standard_normal[cluster_idx].columns)

            
    # def generate_data_for_cluster(self, cluster_idx, num_samples):
    #     if self.cluster_gmms[cluster_idx] is None:
    #         cluster_means = self.cluster_standard_normal[cluster_idx].mean()
    #         constant_data = pd.DataFrame({col: [cluster_means[col]] * num_samples for col in cluster_means.index})
    #         return constant_data
    #     else:
    #         gmm = self.cluster_gmms[cluster_idx]
    #         samples, _ = gmm.sample(num_samples)
    #         return pd.DataFrame(samples, columns=self.cluster_standard_normal[cluster_idx].columns)

    def generate_proportional_data_from_gmm(self, num_rows):
        self.num_rows = num_rows
        self.generated_cluster_data = {}
        total_original_data_points = len(self.kmeans_labels)
        unique_clusters = np.unique(self.kmeans_labels)

        total_generated = 0
        for cluster_idx in unique_clusters[:-1]:
            cluster_size = sum(self.kmeans_labels == cluster_idx)
            cluster_proportion = cluster_size / total_original_data_points
            num_samples = int(round(num_rows * cluster_proportion))
            total_generated += num_samples

            self.generated_cluster_data[cluster_idx] = self.generate_data_for_cluster(cluster_idx, num_samples)

        last_cluster_idx = unique_clusters[-1]
        num_samples_last_cluster = num_rows - total_generated
        self.generated_cluster_data[last_cluster_idx] = self.generate_data_for_cluster(last_cluster_idx, num_samples_last_cluster)


    
    def inverse_cdf(self,cluster_idx, p, column, distribution):
    
        if distribution == 'norm':
            mean, std = self.cluster_parameters_collections[cluster_idx][column]['mean'], self.cluster_parameters_collections[cluster_idx][column]['std']
            p = np.clip(p, 1e-15, 1 - 1e-15)
            return norm.ppf(p, loc=mean, scale=std)
        
        if distribution == 'beta':
            a, b, loc, scale = self.cluster_parameters_collections[cluster_idx][column]['a'], self.cluster_parameters_collections[cluster_idx][column]['b'], self.cluster_parameters_collections[cluster_idx][column]['loc'], self.cluster_parameters_collections[cluster_idx][column]['scale']  
            p = np.clip(p, 1e-15, 1 - 1e-15)
            normalized_values = beta.ppf(p, a, b, loc, scale)
            original_min = self.cluster_parameters_collections[cluster_idx][column]['original_min']
            original_max = self.cluster_parameters_collections[cluster_idx][column]['original_max']
            return normalized_values * (original_max - original_min) + original_min
        
        if distribution == 'truncnorm':
            mean = self.cluster_parameters_collections[cluster_idx][column]['mean']
            std = self.cluster_parameters_collections[cluster_idx][column]['std']
            low_std, upp_std = self.cluster_parameters_collections[cluster_idx][column]['low_std'], self.cluster_parameters_collections[cluster_idx][column]['upp_std']
            p = np.clip(p, 1e-15, 1 - 1e-15)
            return truncnorm.ppf(p, low_std, upp_std, loc=mean, scale=std)
        
        if distribution == 'uniform':
            min_value = self.cluster_parameters_collections[cluster_idx][column]['min_value']
            max_value = self.cluster_parameters_collections[cluster_idx][column]['max_value']
            scale = max_value - min_value
            p = np.clip(p, 1e-15, 1 - 1e-15)
            return uniform.ppf(p, loc=min_value, scale=scale)
        
        if distribution == 'expon':
            scale = self.cluster_parameters_collections[cluster_idx][column]['scale']
            p = np.clip(p, 1e-15, 1 - 1e-15)
            return expon.ppf(p, scale=scale)
        
        # Add logic for other distributions if needed
        return None
    
    def generate_synthetic_data(self):
        # Generate synthetic data using GMMs
        synthetic_cluster_data = self.generated_cluster_data

        # Initialize an empty DataFrame for the synthetic data
        synthetic_df = pd.DataFrame()

        # Loop over each cluster and apply inverse CDF transformations
        for cluster_idx, cluster_data in synthetic_cluster_data.items():
            transformed_cluster_data = cluster_data.copy()
            for column in transformed_cluster_data.columns:
                distribution = self.cluster_distributions[cluster_idx][column]
                if distribution != 'single_value':
                    inverse_transformed_data = self.inverse_cdf(cluster_idx, transformed_cluster_data[column], column, distribution)
                    transformed_cluster_data[column] = inverse_transformed_data
                else:
                    # Handle single value columns
                    transformed_cluster_data[column] = self.cluster_standard_normal[cluster_idx][column].mean()

            # Append transformed cluster data to the synthetic dataset
            synthetic_df = pd.concat([synthetic_df, transformed_cluster_data], ignore_index=True)

        # Reorder columns to match the original data's order
        self.synthetic_df = synthetic_df[self.data.columns]
    
    def numerical_to_category(self, num_value, column_name):
        """ Convert a numerical value back to its corresponding category. """
        for category, (a, b) in self.category_intervals[column_name].items():
            if a <= num_value < b:
                return category
            elif num_value > 1 and round(b) == 1:
                return category
            elif num_value < 0 and round(a) == 0:
                return category
        return None  # Return None or some default value if no category matches
    
    def numerical_to_datetime(self, num_value):
        """
        Convert a numerical value back to its corresponding datetime.
        """
        return pd.Timestamp("1970-01-01") + pd.to_timedelta(num_value, unit='s')

    def post_process(self):
        """
        Convert all numerical columns back to their original categorical form.

        Parameters:
        - synthetic_df: DataFrame containing synthetic data with numerical values for categorical columns.

        Returns:
        - DataFrame with categorical columns converted back to their original categories.
        """
        self._start_timer()

        for column_name in set(self.category_column) - set(self.unique_identifier_columns) - set(self.binary_columns_for_missing):
            self.synthetic_df[column_name] = self.synthetic_df[column_name].apply(lambda x: self.numerical_to_category(x, column_name))

        for column_name in self.datetime_column:
            self.synthetic_df[column_name] = self.synthetic_df[column_name].round().astype(int)
        # Ensure the column is numeric and represents days since the epoch
            self.synthetic_df[column_name] = pd.to_timedelta(self.synthetic_df[column_name], unit='d') + pd.Timestamp("1970-01-01")

        for column in self.columns_with_missing_values:
            original_column = column.replace('_missing', '')
            self.synthetic_df.loc[self.data[column + '_missing'] == 1, original_column] = np.nan
        self._stop_timer("post_process")
        return self.synthetic_df
    
    def get_execution_times_df(self):
        """
        Convert the execution times dictionary to a DataFrame.

        Returns:
        - pd.DataFrame: A DataFrame with process names and their corresponding execution times.
        """
        return pd.DataFrame(list(self.execution_times.items()), columns=['Process', 'Time (seconds)'])


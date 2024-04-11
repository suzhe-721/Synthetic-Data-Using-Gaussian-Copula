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
from sklearn.metrics import silhouette_score
warnings.filterwarnings(action='ignore')



class GaussianCopulaKmeansSynthesizer:

    def __init__(self, data):
        # self.filepath = filepath
        self.data = data
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
        self._start_timer()
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
            self._stop_timer("assign_intervals")

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
    
    def best_fit_distribution(self, column_data,column_name):
        distributions = ['norm', 'uniform', 'beta', 'expon', 'truncnorm']
        best_fit = None
        best_p_value = -1
        truncation_threshold = 0.05  # Threshold for considering truncation (5% of data at bounds)
        if column_data.min() < 0 or column_data.skew() > 1:
            distributions.remove('beta')
        # Check for potential truncation
        min_count = np.sum(column_data == column_data.min())
        max_count = np.sum(column_data == column_data.max())
        if min_count + max_count >= truncation_threshold * len(column_data):
            # If significant data at bounds, consider it as truncated
            best_fit = 'truncnorm'
            mean, std_dev = norm.fit(column_data)
            lower_bound = (column_data.min() - mean) / std_dev
            upper_bound = (column_data.max() - mean) / std_dev
            params = truncnorm.fit(column_data, lower_bound, upper_bound)
            _, best_p_value = kstest(column_data, 'truncnorm', args=params)
            return best_fit, best_p_value

        # If not truncated, proceed with other distributions
        for distribution in distributions:
            try:
                if distribution == 'norm':
                    params = norm.fit(column_data)
                    _, p_value = kstest(column_data, 'norm', args=params)
                elif distribution == 'uniform':
                    params = uniform.fit(column_data)
                    _, p_value = kstest(column_data, 'uniform', args=params)
                elif distribution == 'beta':
                    epsilon = 1e-10
                    scaled_data = (column_data - column_data.min() + epsilon) / (column_data.max() - column_data.min() + 2 * epsilon)
                    params = beta.fit(scaled_data, floc=0, fscale=1)
                    _, p_value = kstest(scaled_data, 'beta', args=params)
                elif distribution == 'expon':
                    params = expon.fit(column_data)
                    _, p_value = kstest(column_data, 'expon', args=params)

                if p_value > best_p_value:
                    best_p_value = p_value
                    best_fit = distribution
            except Exception as e:
                logging.error(f"Error in fitting {distribution} distribution for column '{column_name}': {e}")
                logging.info(f"Column '{column_name}' data statistics: {column_data.describe()}")
                logging.info(f"Column '{column_name}' data values (sample): {column_data.sample(10)}")



        return best_fit, best_p_value
    
    def get_distribution(self):

        self._start_timer()

        self.relevant_columns = (set(self.num_column) | set(self.category_column)) - set(self.unique_identifier_columns) - set(self.binary_columns_for_missing)
        for column in self.relevant_columns:
            best_fit, best_p_value = self.best_fit_distribution(self.data[column],column)
            self.distributions[column] = best_fit
        
        for column in self.relevant_columns:
            self.column_min[column] = self.data[column].min()
            self.column_max[column] = self.data[column].max()
        self._stop_timer('get_distribution')

    def calculate_cdf(self, column, distribution):
        if distribution == 'norm':
            mean, std = norm.fit(self.data[column])
            self.parameters_collections[column] = {'distribution':'norm', 'mean': mean, 'std': std}
            return norm.cdf(self.data[column], loc=mean, scale=std)
        
        if distribution == 'beta':
            data_normalized = (self.data[column] - self.data[column].min()) / (self.data[column].max() - self.data[column].min())

            # Estimate the parameters of the beta distribution
            a, b, loc, scale = beta.fit(data_normalized)

            # Calculating the CDF values using the beta distribution
            cdf_values = beta.cdf(data_normalized, a, b, loc, scale)

            self.parameters_collections[column] = {'distribution':'beta', 'a': a, 'b': b, 'loc': loc, 'scale':scale}
            return cdf_values
            
        if distribution == 'truncnorm':
            mean = self.data[column].mean()
            std = self.data[column].std()
            low = self.data[column].min()
            upp = self.data[column].max()
            low_std, upp_std = (low - mean) / std, (upp - mean) / std
            self.parameters_collections[column] = {'distribution':'truncnorm', 'mean': mean, 'std': std, 'low_std':low_std, 'upp_std': upp_std}
            return truncnorm.cdf(self.data[column], low_std, upp_std, loc=mean, scale=std)
        
        if distribution == 'uniform':
            min_value = self.data[column].min()
            max_value = self.data[column].max()
            scale = max_value - min_value
            cdf_values = uniform.cdf(self.data[column], loc=min_value, scale=scale)
            self.parameters_collections[column] = {'distribution':'uniform', 'min_value': min_value, 'max_value': max_value}
            return cdf_values
        
        if distribution == 'expon':
            # The scale parameter for the exponential distribution is the inverse of the mean
            scale = 1 / self.data[column].mean()
            cdf_values = expon.cdf(self.data[column], scale=scale)
            self.parameters_collections[column] = {'distribution': 'expon', 'scale': scale}
            return cdf_values

    def calculate_cdfs(self):
        self._start_timer()
        for column in self.relevant_columns:
            distribution = self.distributions.get(column)
            if distribution:
                self.cdf_results[column] = self.calculate_cdf(column, distribution)
        epsilon = 1e-10  # A small epsilon value
        self.cdf_results = self.cdf_results.mask(self.cdf_results == 0, epsilon)
        self.cdf_results = self.cdf_results.mask(self.cdf_results == 1, 1 - epsilon)
        self._stop_timer('calculate_cdfs')
        
    def standard_gaussian(self, p_value):

        return norm.ppf(p_value)
    
    def standard_gaussian_all(self):
        self._start_timer()
        for column in self.cdf_results:
            self.column_inverse_cdfs[column] = self.cdf_results[column].apply(lambda x: self.standard_gaussian(x))
        self._stop_timer('standard_gaussian_all')
    
    def optimal_clusters_dynamic(self):
        self._start_timer()
        # Calculate the covariance matrix
        standard_normal = self.column_inverse_cdfs
        standard_normal_df = pd.DataFrame(data = standard_normal)
        range_n_clusters = range(2, 11)
        silhouette_avg_scores = []
        for n_clusters in range_n_clusters:
            # Initialize the KMeans object with n_clusters and fit it to the data
            kmeans = KMeans(n_clusters=n_clusters, random_state=10)
            cluster_labels = kmeans.fit_predict(standard_normal_df)
            
            # Calculate the silhouette score for the current number of clusters
            silhouette_avg = silhouette_score(standard_normal_df, cluster_labels)
            
            # Print the silhouette score and append it to the list
            silhouette_avg_scores.append(silhouette_avg)
        self.num_of_clusters = range_n_clusters[np.argmax(silhouette_avg_scores)]
        self._stop_timer('optimal_clusters_dynamic')
    
    def get_Kmeans(self):
        self._start_timer()
        self.standard_gaussian_df = pd.DataFrame(data=self.column_inverse_cdfs)
        self.kmeans = KMeans(n_clusters=self.num_of_clusters, random_state=0)
        self.kmeans.fit(self.standard_gaussian_df)
        self.kmeans_labels = self.kmeans.labels_

      # Store the covariance matrices for each cluster
        self.cluster_covariances = {}
        for cluster_idx in range(self.num_of_clusters):
            cluster_data = self.standard_gaussian_df[self.kmeans_labels == cluster_idx]
            self.cluster_covariances[cluster_idx] = np.cov(cluster_data, rowvar=False)
        
        self._stop_timer('get_kmeans')
    
    def inverse_cdf(self, p, column, distribution):
    
        if distribution == 'norm':
            mean, std = self.parameters_collections[column]['mean'], self.parameters_collections[column]['std']
            return norm.ppf(p, loc=mean, scale=std)
        
        if distribution == 'beta':
            a, b, loc, scale = self.parameters_collections[column]['a'], self.parameters_collections[column]['b'], self.parameters_collections[column]['loc'], self.parameters_collections[column]['scale']  
            normalized_values = beta.ppf(p, a, b, loc, scale)
            return normalized_values * (self.data[column].max() - self.data[column].min()) + self.data[column].min()
        
        if distribution == 'truncnorm':
            p = np.clip(p, 1e-25, 1-1e-25)
            mean = self.parameters_collections[column]['mean']
            std = self.parameters_collections[column]['std']
            low_std, upp_std = self.parameters_collections[column]['low_std'], self.parameters_collections[column]['upp_std']
            return truncnorm.ppf(p, low_std, upp_std, loc=mean, scale=std)
        
        if distribution == 'uniform':
            min_value = self.parameters_collections[column]['min_value']
            max_value = self.parameters_collections[column]['max_value']
            scale = max_value - min_value
            return uniform.ppf(p, loc=min_value, scale=scale)
        
        if distribution == 'expon':
            scale = self.parameters_collections[column]['scale']
            return expon.ppf(p, scale=scale)
        
        # Add logic for other distributions if needed
        return None
    
    def sample(self, F_inv, Sigma):
        """
        Sample numerical values from the distribution and covariances of the columns.
        
        Parameters:
        - F_inv: A list of inverse CDF functions for the marginals.
        - Sigma: The covariance matrix.
        
        Returns:
        - A sample vector x in the original space.
        """
        n = Sigma.shape[0]
        regularization_value = 1e-6
        np.fill_diagonal(Sigma, Sigma.diagonal() + regularization_value)
        v = np.random.randn(n)
        L = np.linalg.cholesky(Sigma)
        u = L.dot(v)
        x = [F_inv_i(norm.cdf(u_i)) for F_inv_i, u_i in zip(F_inv, u)]
        return x
    
    def generate_synthetic_data(self, num_rows):
        """
        Generate synthetic data based on fitted GMMs and a covariance matrix.

        Parameters:
        - num_rows: Number of rows to generate.
        - covariance_matrix: Covariance matrix for the Gaussian Copula.

        Returns:
        - Synthetic dataset.
        """
        self._start_timer()
        F_inv = []
        synthetic_datasets = []
        cluster_counts = np.bincount(self.kmeans_labels)
        samples_per_cluster = [int(num_rows * (count / len(self.data))) for count in cluster_counts]
        # inverse_df = pd.DataFrame(self.column_inverse_cdfs)
        relevant_columns = list((set(self.data.columns) - set(self.binary_columns_for_missing)) - set(self.unique_identifier_columns))
        for cluster_idx in tqdm(range(len(cluster_counts))):
          F_inv = []
          cluster_data = self.data[self.kmeans_labels == cluster_idx]
          for column in cluster_data.columns:
              
              distribution = self.distributions.get(column)
              if distribution:
                  F_inv.append(lambda p, column=column, dist=distribution: self.inverse_cdf(p, column, dist))
          cluster_covariance = self.cluster_covariances[cluster_idx]
          for _ in range(samples_per_cluster[cluster_idx]):
              synthetic_point = self.sample(F_inv, cluster_covariance)
              synthetic_datasets.append(synthetic_point)

        synthetic_df = pd.DataFrame(synthetic_datasets, columns=self.data.columns)
        column_means = synthetic_df.mean()
        
        # Fill missing values with the mean of each column
        self.synthetic_data = synthetic_df.fillna(column_means)

        for column in cluster_data.columns:
          min_val = self.column_min.get(column)
          max_val = self.column_max.get(column)
          if min_val and max_val:  # Ensure that min and max values are available
              self.synthetic_data[column] = self.synthetic_data[column].clip(lower=min_val, upper=max_val)


        for column in self.special_numeric_series_columns:
            prefix = self.special_column_prefixes[column]
            number_length = self.digit_counts[column]
            # Generate synthetic data based on extracted structure
            self.synthetic_data[column] = [
                prefix + self.generate_random_number(number_length)
                for _ in range(num_rows)
            ]

        for column in self.hyphenated_numeric_columns:
            format_pattern = self.hyphenated_format.get(column, [])
            self.synthetic_data[column] = [
                self.generate_random_hyphenated_number(format_pattern) for _ in range(num_rows)
            ]
        self._stop_timer("generate_syn_data")
        return self.synthetic_data
    
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
            self.synthetic_data[column_name] = self.synthetic_data[column_name].apply(lambda x: self.numerical_to_category(x, column_name))
            synthetic_data = self.synthetic_data

        for column_name in self.datetime_column:
            self.synthetic_data[column_name] = self.synthetic_data[column_name].round().astype(int)
        # Ensure the column is numeric and represents days since the epoch
            self.synthetic_data[column_name] = pd.to_timedelta(self.synthetic_data[column_name], unit='d') + pd.Timestamp("1970-01-01")

        for column in self.columns_with_missing_values:
            original_column = column.replace('_missing', '')
            self.synthetic_data.loc[self.data[column + '_missing'] == 1, original_column] = np.nan
        self._stop_timer("post_process")
        return self.synthetic_data
    
    def get_execution_times_df(self):
        """
        Convert the execution times dictionary to a DataFrame and calculate
        fitting time, generation time, and post-process time.

        Returns:
        - pd.DataFrame: A DataFrame with the detailed process names and their corresponding execution times.
        - pd.DataFrame: A DataFrame with summarized fitting time, generation time, and post-process time.
        """
        # Create the DataFrame from the execution times dictionary
        execution_times_df = pd.DataFrame(list(self.execution_times.items()), columns=['Process', 'Time (seconds)'])

        # Calculate the 'Fitting Time' as the sum of times for indices 0 to 6
        fitting_time = execution_times_df.iloc[0:9]['Time (seconds)'].sum() if len(execution_times_df) > 6 else 0

        # 'Generation Time' is the time for index 7
        generation_time = execution_times_df.iloc[9]['Time (seconds)'] if len(execution_times_df) > 7 else 0

        # 'Post-Process Time' is the time for index 8
        post_process_time = execution_times_df.iloc[10]['Time (seconds)'] if len(execution_times_df) > 8 else 0

        # Create a summary DataFrame
        summary_df = pd.DataFrame({
            'Process': ['Fitting Time', 'Generation Time', 'Post-Process Time'],
            'Time (seconds)': [fitting_time, generation_time, post_process_time]
        })

        # Return both the detailed and summary DataFrames
        return summary_df
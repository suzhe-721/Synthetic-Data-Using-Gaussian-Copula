import numpy as np
import pandas as pd
from scipy.stats import kstest, norm, uniform, beta, expon, truncnorm
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import warnings
from joblib import Parallel, delayed
import scipy.stats as stats
from tqdm import tqdm
from scipy.optimize import brentq
import random
import time
warnings.filterwarnings(action='ignore')

class GaussianCopulaSynthesizer:

    """
    A class for synthesizing data using Gaussian Copula models.
    """
    def __init__(self, data):

        """
        Initialize the synthesizer with the provided file path.

        Parameters:
        - filepath (str): The file path to the input dataset.
        """

        # self.filepath = filepath
        self.data = data
        self.execution_times = {}
        self.num_column = []
        self.category_column = []
        self.datetime_column = []
        self.unique_identifier_columns = []
        self.special_numeric_series_columns = []
        self.special_column_prefixes = {}
        self.hyphenated_numeric_columns = []
        self.hyphenated_format = {}
        self.columns_with_missing_values = []
        self.binary_columns_for_missing = []
        self.category_intervals = {}
        self.gmms = {}
        self.column_cdfs = {}
        self.column_inverse_cdfs = {}
        self.processed_data = None
        self.synthetic_data = pd.DataFrame()
        self.digit_counts = {}  # Store digit counts for each column
        self.column_min = {}  # Store minimum values for each column
        self.column_max = {}

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
        for column_name in (set(self.category_column)) - set(self.unique_identifier_columns) - set(self.binary_columns_for_missing):
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
        # self._identify_columns()
        # self.convert_datetime_to_numerical()
        # self.handle_missing_values()
        # self.assign_intervals()

        # Process only columns that are not unique identifiers
        self._start_timer()
        for column_name in tqdm(set(self.category_column) - set(self.unique_identifier_columns) - set(self.binary_columns_for_missing)):
            self.data[column_name] = self.data[column_name].apply(lambda x: self.sample_from_category(x, column_name))
        self._stop_timer("preprocess_data")


    
    def _find_best_components(self, data):
        bic = []
        n_components_range = range(1, 20)
        for n_components in n_components_range:
            gmm = GaussianMixture(n_components=n_components)
            gmm.fit(data)
            bic.append(gmm.bic(data))
        return n_components_range[np.argmin(bic)]

    # def _fit_gmm_for_column(self, column):
    #     best_components = self._find_best_components(self.data[column])
    #     gmm = GaussianMixture(n_components=best_components)
    #     gmm.fit(self.data[column].values.reshape(-1, 1))
    #     return column, gmm

    def fit_distributions(self):
        self._start_timer()
        # Include normal numeric and categorical columns (excluding unique identifiers and binary columns for missing values)
        self.relevant_columns = (set(self.num_column) | set(self.category_column)) - set(self.unique_identifier_columns) - set(self.binary_columns_for_missing)

        data_for_gmm = self.data[self.relevant_columns]
        best_components = self._find_best_components(data_for_gmm)

        gmm = GaussianMixture(n_components = best_components)

        gmm.fit(data_for_gmm)

        self.gmm = gmm

        for column in self.relevant_columns:
            self.column_min[column] = self.data[column].min()
            self.column_max[column] = self.data[column].max()

        self._stop_timer('fit_distribution')
    
    def get_gmm(self, column_name):
        return self.gmms.get(column_name, None) #find the gmm for that column, if not found, return None

    def generate_synthetic_data(self, num_rows):

        self._start_timer()
        # Generate synthetic data for regular columns using GMMs
        self.relevant_columns = list((set(self.data.columns) - set(self.binary_columns_for_missing)) - set(self.unique_identifier_columns))

        samples = self.gmm.sample(n_samples=num_rows)[0]  # Get samples from GMM
        synthetic_df = pd.DataFrame(samples, columns=self.relevant_columns)

      
        
        for column in self.relevant_columns:
          min_val, max_val = self.column_min.get(column, None), self.column_max.get(column, None)
          if min_val is not None and max_val is not None:
              synthetic_df[column] = synthetic_df[column].clip(lower=min_val, upper=max_val)
      
        self.synthetic_data = synthetic_df.fillna(synthetic_df.mean())

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
        Convert all numerical columns back to their original categorical or datetime form.

        This method converts columns that were previously transformed into a numerical format
        back to their original categorical or datetime format.

        Returns:
        - pd.DataFrame: A DataFrame with categorical columns converted back to their original categories
                        and datetime columns converted back to their original datetime format.
        """
        self._start_timer()
        for column_name in set(self.category_column) - set(self.unique_identifier_columns) - set(self.binary_columns_for_missing):
            self.synthetic_data[column_name] = self.synthetic_data[column_name].apply(lambda x: self.numerical_to_category(x, column_name))

        for column_name in self.datetime_column:
            self.synthetic_data[column_name] = self.synthetic_data[column_name].round().astype(int)
        # Ensure the column is numeric and represents days since the epoch
            self.synthetic_data[column_name] = pd.to_timedelta(self.synthetic_data[column_name], unit='d') + pd.Timestamp("1970-01-01")

        for column in self.columns_with_missing_values:
            original_column = column.replace('_missing', '')
            self.synthetic_data.loc[self.data[column+'_missing'] == 1, original_column] = np.nan
        
        self._stop_timer("post_process")
        return self.synthetic_data
    
    def get_execution_times_df(self):
        """
        Convert the execution times dictionary to a DataFrame.

        Returns:
        - pd.DataFrame: A DataFrame with process names and their corresponding execution times.
        """
        return pd.DataFrame(list(self.execution_times.items()), columns=['Process', 'Time (seconds)'])
    
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
        fitting_time = execution_times_df.iloc[0:5]['Time (seconds)'].sum() if len(execution_times_df) > 5 else 0

        # 'Generation Time' is the time for index 7
        generation_time = execution_times_df.iloc[5]['Time (seconds)'] if len(execution_times_df) > 5 else 0

        # 'Post-Process Time' is the time for index 8
        post_process_time = execution_times_df.iloc[6]['Time (seconds)'] if len(execution_times_df) > 6 else 0

        # Create a summary DataFrame
        summary_df = pd.DataFrame({
            'Process': ['Fitting Time', 'Generation Time', 'Post-Process Time'],
            'Time (seconds)': [fitting_time, generation_time, post_process_time]
        })

        # Return both the detailed and summary DataFrames
        return summary_df
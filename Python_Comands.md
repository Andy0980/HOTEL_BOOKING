# Data Analysis Template
# Author: [Edwin Mosquera]
# Date: [10/27/2027]
# Purpose: Python template for data analysis steps



#Running a cell with a loop can help prevent the notebook from becoming idle
import time
# Start an infinite loop
while True:
    # Wait for 5 minutes (300 seconds)
    time.sleep(300)




# --- IMPORT LIBRARIES ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# --- LOAD AND INSPECT DATA ---
data = pd.read_csv('path/to/your/data.csv')  # Load data. You need to replace the dataset name.

# Quick inspection
print(data.head())                           # Show first 5 rows
print(data.info())                           # Show column data types and counts
print(data.describe())                       # Summary statistics for numerical columns
print(data.isnull().sum())                   # Check for missingvalues
print(data.columns)                          # print out the list of column names 
print(len(data))                             # Show the rows number inside the column
print(data['Column_Name'])                   # Show the content of any column
print(data['Column_name'].unique())          # Check if there are other unique values in the column


# Check for NaN values across all columns
nan_rows = data[data.isnull().any(axis=1)]
# Display the rows with NaN values
print(nan_rows)



# --- DATA CLEANING ---


#Handle missing values in all data
data.ffill(inplace=True)                      

# Drop missing values (optional)
data = data.dropna()

# Remove duplicates
data = data.drop_duplicates()

# Reset index after cleaning
data.reset_index(drop=True, inplace=True)

# Remove specific column
data = data.drop(columns=['Column_Name'])

# Remove rows with null values in the column
data = data.dropna(subset=['Column_Name'])

// # remove all excess spact6o3es, whether they’re at the beginning, end, or even between items in each cell in the dataframe
for col in data.coll9ñ7{9´9t87t7 umns:
    if data[col].dtype == "object":  # Only apply to string columns
        data[col] = data[col].apply(lambda x: " ".join(x.split()) if isinstance(x, str) else x)


# Remove rows with NaN in specific columns
data.dropna(subset=['column1', 'column2'], inplace=True)


# Remove columns with any NaN values
data.dropna(axis=1, inplace=True)




# Round any column to n decimal places
df['Latitude'] = df['Latitude'].round(n)# n is the number of decimal




# List of columns to standardize
columns_to_standardize = ['Column_Name', 'Column_Name', '.....']
# Standardize each column to n decimal places
for col in columns_to_standardize:
    data[col] = data[col].round(n)# n is the number of decimal
		
		
		


# Assuming `data` is your existing DataFrame with all original columns
# Function to clean and extract duration of tome in minutes
def clean_duration(duration):
    match = re.search(r'\d+', duration)  # Find numeric part
    return f"{match.group()} min" if match else None  # Format result
# Apply the function to the Column_Name
data['Column_Name'] = data['Column_Name'].apply(clean_duration)
# View the updated DataFrame with all columns intact
print(data.head())
		




# List of columns where nulls should trigger row removal
columns_to_check = ['Column1', 'Column2', 'Column3', ..., 'Column30']
data = data.dropna(subset=columns_to_check)
# Drop rows where any of these columns have null values



# If you want to keep as much data as possible in other columns consider filling the null values instead of dropping them
data['Column_Name'] = data['Column_Name'].fillna('Unknown')# replace Unknown for any option

# For categorical data
mode_value = data['Column_Namel'].mode()[0]
data['Column_Name'] = data['Column_Name'].fillna(mode_value)# Fill null values with the mode, mean, or median (for numerical data)



# Identify columns with the 'object' data type
object_columns = df.select_dtypes(include=['object']).columns
# Convert each object column to float (non-numeric values will become NaN)
for col in object_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')
# Display data types after conversion and the data to confirm everything went smoothly
print("\nAfter conversion:")
print(df.dtypes)
print("\nData preview after conversion:")
print(df.head())



# Transfor data from object to numeric values 0 and 1, you could change the column name.

data = {
    'Gender': ['Male', 'Female', 'Female', 'Male'],
    'Internet_Access': ['Yes', 'No', 'Yes', 'No'],
    'School_Type': ['Public', 'Private', 'Public', 'Private']
}
data = pd.DataFrame(data)

# Display data types before conversion
print("Before conversion:")
print(data.dtypes)

# Identify columns with object data type
object_columns = data.select_dtypes(include=['object']).columns

# Loop through each object column and convert unique values to numeric
for col in object_columns:
    # Create a unique mapping for each column
    unique_values = data[col].unique()
    mapping_dict = {value: idx for idx, value in enumerate(unique_values)}
    data[col + '_num'] = data[col].map(mapping_dict)  # Add a new column with numeric values

    # Optional: print the mapping for reference
    print(f"Mapping for {col}: {mapping_dict}")

# Display data after conversion
print("\nAfter conversion:")
print(data)





#opcion number 2 for a large data transfor into 0 and 1

# Display data types before transformation
print("Data types before transformation:")
print(data.dtypes)

# Identify columns with object data type
object_columns = data.select_dtypes(include=['object']).columns

# Loop through each object column and convert unique values to numeric, creating new columns
for col in object_columns:
    unique_values = data[col].unique()
    mapping_dict = {value: idx for idx, value in enumerate(unique_values)}
    data[col + '_num'] = data[col].map(mapping_dict)  # Creates new columns with numeric values

    # Optional: print the mapping for reference
    print(f"Mapping for {col}: {mapping_dict}")

# Verify columns after transformation
print("\nData types after transformation:")
print(data.dtypes)
print("Columns after transformation:", data.columns.tolist())



# delete the blank spaces between
# Create a backup copy
data_backup = data.copy()
# Clean the 'Column_Name' column by removing extra spaces
data['Column_Name'] = data['Column_Name'].apply(lambda x: ', '.join([producer.strip() for producer in x.split(',')]))
# Verify the changes
print(data['Column_Name'].head(10))
# Optionally, save the cleaned DataFrame back to CSV
data.to_csv('your_cleaned_file.csv', index=False)




import pandas as pd
import re
# Example DataFrame setup (replace this with your actual data)
# data = pd.read_csv('your_file.csv')
# Function to clean and format columns with list-like strings
def clean_themes_column(column_value):
    # Use regex to remove brackets and single quotes if present
    cleaned_value = re.sub(r"[\[\]']", "", column_value)  # Remove brackets and single quotes. You can include other charaters
    return cleaned_value.replace(", ", "/")  # Replace comma+space with "/"
# List of columns to clean
columns_to_clean = ['Columns_Name']  # Add any other column names that need cleaning
# Apply the cleaning function to each specified column
for column in columns_to_clean:
    data[column] = data[column].apply(lambda x: clean_themes_column(str(x)))
# Display all columns to check the result
pd.set_option('display.max_columns', None)
print(data.head())





# Function to extract the main rating value in a column
def extract_main_value(value):
    # Split by space and return only the first element
    return value.split(" ")[0]  # Keeps only the first part before any additional text
# Apply this function to the column (replace 'Column_Name' with your actual column name)
data['Column_Name'] = data['Column_Name'].apply(extract_main_value)
# Display unique values in the column to check results (optional)
print(data['Column_Name'].unique())  # To confirm no other unexpected values




# Template to format a column to one decimal without rounding
# Replace 'Column_Name' with your specific column name
data['Column_Name'] = data['Column_Name'].apply(lambda x: f"{str(x)[:n]}") # n is a number of decimal we wants cut off




# Define a function to format the numbers to one decimal place without rounding
def format_one_decimal(x):
    return f"{x:.1f}"
# Apply the function to all numeric columns
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
for col in numeric_columns:
    df[col] = df[col].apply(lambda x: format_one_decimal(x))
# Verify the changes
print(df.head(10))




# Replace NaN values with different values for each column
data = data.fillna({'Column1': 'Value1', 'Column2': 'Value2'})
# Replace NaN values in the entire DataFrame with a specific value
data = data.fillna('YourValue')  # Replace 'YourValue' with the value you want



import pandas as pd

# Replace NaN values in numeric columns with the mean of each column
data = data.fillna(data.mean())

# For a specific date column
data['date_column'] = pd.to_datetime(data['date_column'], format='%Y/%m/%d')  # Convert date string to datetime
mean_date = data['date_column'].mean()  # Calculate the mean date
data['date_column'] = data['date_column'].fillna(mean_date)  # Replace NaN values with the mean date






# --- EXPLORATORY DATA ANALYSIS (EDA) ---

# Value counts of categorical columns
print(data['column_name'].value_counts())


# Correlation matrix
correlation = data.corr()
print(correlation)


# magic command displays a list of all variables currently in the environment 
%whos



# Import necessary library
import pandas as pd
# Load your dataset
data = pd.read_csv('your_dataset.csv')  # Replace 'your_dataset.csv' with your actual file name
# Convert date column to datetime format
data['your_date_column'] = pd.to_datetime(data['your_date_column'])  # Replace 'your_date_column' with your actual date column name
# Extract useful components if needed
data['year'] = data['your_date_column'].dt.year
data['month'] = data['your_date_column'].dt.month
data['day'] = data['your_date_column'].dt.day
# Print the first few rows to verify
print(data.head())



# --- DATA VISUALIZATION ---

# Example: Distribution plot for a specific column
sns.histplot(data['column_name'])
plt.title('Distribution of column_name')
plt.show()



# Example: Correlation heatmap
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()



# Convert all columns to numeric, if possible
data = data.apply(pd.to_numeric, errors='coerce')
# Use the describe method to get the summary statistics
summary_stats = data.describe()
# For median, you can use the median method separately
median_stats = data.median()
# Combine the results for a comprehensive view
combined_stats = summary_stats.append(median_stats.rename('median'))
print(combined_stats)




# --- EXPORT CLEANED DATA ---


# Get the current working directory
import os
cwd = os.getcwd()
print(f"The file is stored in: {cwd}")




#PIPELINE TO CLEAN AND ESTANDARIZED  THE DATASET

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
df = pd.read_csv('path_to_your_dataset.csv')

# Step 1: Inspect the data
print(df.head())  # View the first few rows
print(df.info())  # Get summary information
print(df.describe(include='all'))  # Get descriptive statistics
print(df.isnull().sum())  # Check for missing values

# Step 2: Handle Missing Values
def fill_missing_values(df, columns):
    """
    Fill missing values with mode for specified columns.
    
    :param df: Pandas DataFrame - The DataFrame to be cleaned.
    :param columns: List of column names to clean - The columns to fill missing values in.
    :return: DataFrame with missing values filled - The cleaned DataFrame.
    """
    for column in columns:
        if column in df.columns:
            mode_value = df[column].mode()[0] if not df[column].mode().empty else 'Unknown'
            df[column] = df[column].fillna(mode_value)
    return df

categorical_columns = ['your_categorical_columns']
numerical_columns = ['your_numerical_columns']

df = fill_missing_values(df, categorical_columns)
df.fillna(df.median(), inplace=True)  # Fill missing numerical values with median

# Step 3: Remove Duplicates
df.drop_duplicates(inplace=True)  # Remove duplicate rows

# Step 4: Standardize Data
def standardize_text(df, columns):
    """
    Convert categorical column values to lowercase.
    
    :param df: Pandas DataFrame - The DataFrame to be standardized.
    :param columns: List of categorical column names to standardize - The columns to standardize.
    :return: DataFrame with standardized categorical data - The standardized DataFrame.
    """
    for column in columns:
        if column in df.columns:
            df[column] = df[column].str.lower()
    return df

df = standardize_text(df, categorical_columns)

# Convert date and numeric columns to appropriate data types
df['date_column'] = pd.to_datetime(df['date_column'], format='%Y-%m-%d')
df['numeric_column'] = df['numeric_column'].astype(int)

# Step 5: Normalize Numerical Data
def normalize_numerical_columns(df, columns):
    """
    Normalize numerical columns using MinMaxScaler.
    
    :param df: Pandas DataFrame - The DataFrame to be normalized.
    :param columns: List of numerical column names to normalize - The columns to normalize.
    :return: DataFrame with normalized numerical data - The normalized DataFrame.
    """
    scaler = MinMaxScaler()
    for column in columns:
        if column in df.columns:
            df[column] = scaler.fit_transform(df[[column]])
    return df

df = normalize_numerical_columns(df, numerical_columns)

# Step 6: Outlier Detection and Handling
def handle_outliers(df, columns):
    """
    Detect and handle outliers in specified numerical columns.
    
    :param df: Pandas DataFrame - The DataFrame to be processed.
    :param columns: List of numerical column names to check for outliers.
    :return: DataFrame with outliers handled - The processed DataFrame.
    """
    for column in columns:
        if column in df.columns:
            q1 = df[column].quantile(0.25)
            q3 = df[column].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
    return df

df = handle_outliers(df, numerical_columns)

# Step 7: Feature Engineering
# Create new features and transform existing ones
df['new_feature'] = df['column1'] * df['column2']
df['log_transformed'] = df['column'].apply(lambda x: np.log(x+1))

# Step 8: Data Validation
# Consistency checks to ensure data quality
assert (df['column'] >= 0).all(), "Negative values found!"

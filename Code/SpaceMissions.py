# SPACE MISSION DASHBOARD (ANALYSIS)
# Using numpy, pandas, matplotlib, seaborn — air quality analysis

import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# Load the space mission dataset
file_path = 'space_missions.csv'
data = pd.read_csv(file_path, encoding='latin1')

# Display basic information about the dataset
print("Original Data Preview:")
print(data.head())
print("Data Shape:", data.shape)
print("Missing Values:")
print(data.isnull().sum())
print("Column names and data types:")
print(data.info())

# Check and handle duplicate rows
duplicates = data.duplicated().sum()
print(f"Number of Duplicate Rows: {duplicates}")
if duplicates > 0:
    data = data.drop_duplicates()
    print(f"Dropped {duplicates} duplicate rows.")

# Handle missing values - focusing on key mission attributes
print("Missing Values Before Dropping Key Rows:")
print(data.isnull().sum())
data = data.dropna(subset=['Date', 'MissionStatus', 'Company'])
print(f"Rows with missing key values dropped. Remaining rows: {data.shape[0]}")

# Additional handling for other missing values
data['Price'] = data['Price'].replace(r'[\$,]', '', regex=True).astype(float)
data['Price'].fillna(data['Price'].mean(), inplace=True)

# Feature engineering: Convert date and time columns into usable datetime object
data['LaunchDateTime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], errors='coerce')

# Check for any rows that could not be converted to datetime
if data['LaunchDateTime'].isnull().sum() > 0:
    print(f"Warning: {data['LaunchDateTime'].isnull().sum()} rows could not be converted to datetime.")

# Extract year, month, weekday
data['Year'] = data['LaunchDateTime'].dt.year
data['Month'] = data['LaunchDateTime'].dt.month
data['Weekday'] = data['LaunchDateTime'].dt.day_name()

# Clean price field
data['Price_Clean'] = data['Price'].replace(r'[\$,]', '', regex=True).astype(float)

# Extract country name from location
data['Country'] = data['Location'].apply(lambda x: x.split(',')[-1].strip())

# Display stats for cleaned data
print("Basic Statistics of the Cleaned Data:")
print(data.describe())
print("Cleaned Data Preview:")
print(data.head())

# ***********************
# EXPLORATORY DATA ANALYSIS
# ***********************

# Generate summary statistics grouped by company
print("\nSUMMARY STATISTICS")
company_stats = data.groupby('Company')[['Price_Clean']].describe()
print(company_stats.head())

# Create multi-plot figure for overview of space mission data
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Plot 1: Distribution of launch years
launch_years = data['Year'].value_counts().sort_index()
sns.lineplot(ax=axes[0, 0], x=launch_years.index, y=launch_years.values, marker='o', color='teal')
axes[0, 0].set_title('Launches by Year')
axes[0, 0].set_ylabel('Number of Launches')

# Print details related to Launches by Year
print("\nLaunches by Year Details:")
print(launch_years)

# Plot 2: Top companies by mission count
top_companies = data['Company'].value_counts().head(10)
sns.barplot(ax=axes[0, 1], y=top_companies.index, x=top_companies.values, palette='Blues_r')
axes[0, 1].set_title('Top 10 Launch Companies')
axes[0, 1].set_xlabel('Missions')

# Print details related to Top Companies
print("\nTop 10 Companies by Mission Count:")
print(top_companies)

# Plot 3: Top countries by launch count
top_countries = data['Country'].value_counts().head(10)
sns.barplot(ax=axes[1, 0], y=top_countries.index, x=top_countries.values, palette='Greens')
axes[1, 0].set_title('Top 10 Launching Countries')
axes[1, 0].set_xlabel('Missions')

# Print details related to Top Countries
print("\nTop 10 Countries by Launch Count:")
print(top_countries)

# Plot 4: Mission status count
sns.histplot(ax=axes[1, 1], data=data, x='MissionStatus', hue='MissionStatus', multiple='stack', shrink=0.8, palette='Set2')
axes[1, 1].set_title('Mission Status Distribution')

# Print details related to Mission Status Distribution
mission_status_counts = data['MissionStatus'].value_counts()
print("\nMission Status Distribution:")
print(mission_status_counts)

fig.tight_layout()
plt.savefig('space_mission_overview.png', dpi=300, bbox_inches='tight')
plt.show()

# *****************************
# OBJECTIVE 1: SUCCESS RATE ANALYSIS
# *****************************

# Create hierarchical aggregations for multilevel analysis
country_success = data.groupby(['Country', 'MissionStatus']).size().unstack().fillna(0)
company_success = data.groupby(['Company', 'MissionStatus']).size().unstack().fillna(0)

# Display top performing entities
print("\nTop Countries by Success Count:")
print(country_success.sort_values('Success', ascending=False).head(10))
print("\nTop Companies by Success Count:")
print(company_success.sort_values('Success', ascending=False).head(10))

# Visualization
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Top panel: countries
top_countries_success = country_success.sort_values('Success', ascending=False).head(10)
top_countries_success[['Success', 'Failure']].plot(kind='barh', stacked=True, ax=axes[0], colormap='Set3')
axes[0].set_title('Top Countries: Success vs Failure')
axes[0].set_xlabel('Mission Count')

# Print details related to Top Countries Success
print("\nTop Countries Success vs Failure:")
print(top_countries_success)

# Bottom panel: companies
top_companies_success = company_success.sort_values('Success', ascending=False).head(10)
top_companies_success[['Success', 'Failure']].plot(kind='barh', stacked=True, ax=axes[1], colormap='Paired')
axes[1].set_title('Top Companies: Success vs Failure')
axes[1].set_xlabel('Mission Count')

# Print details related to Top Companies Success
print("\nTop Companies Success vs Failure:")
print(top_companies_success)

plt.tight_layout()
plt.savefig('success_by_country_company.png', dpi=300, bbox_inches='tight')
plt.show()

# *****************************
# OBJECTIVE 2: PRICE ANALYSIS
# *****************************

print("\nPRICE VS OUTCOME ANALYSIS")
filtered_data = data.dropna(subset=['Price_Clean'])

# Plotting average price by mission status
plt.figure(figsize=(10, 6))
sns.barplot(x='MissionStatus', y='Price_Clean', data=filtered_data, palette='deep')
plt.title('Average Launch Price by Mission Outcome')
plt.tight_layout()
plt.savefig('bar_price_vs_status.png', dpi=300, bbox_inches='tight')
plt.show()

# Print details related to Price vs Outcome
price_outcome_stats = filtered_data.groupby('MissionStatus')['Price_Clean'].describe()
print("\nPrice vs Outcome Statistics:")
print(price_outcome_stats)

# *****************************
# OBJECTIVE 3: TEMPORAL TRENDS
# *****************************

print("\nTEMPORAL TRENDS")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Launches per weekday
weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
sns.countplot(ax=axes[0], x='Weekday', data=data, order=weekday_order, palette='Pastel1')
axes[0].set_title('Launches by Weekday')

# Print details related to Launches by Weekday
weekday_counts = data['Weekday'].value_counts()
print("\nLaunches by Weekday Counts:")
print(weekday_counts)

# Launches per month
sns.countplot(ax=axes[1], x='Month', data=data, palette='coolwarm')
axes[1].set_title('Launches by Month')

# Print details related to Launches by Month
month_counts = data['Month'].value_counts()
print("\nLaunches by Month Counts:")
print(month_counts)

plt.tight_layout()
plt.savefig('temporal_trends.png', dpi=300, bbox_inches='tight')
plt.show()

# *****************************
# OBJECTIVE 4: PRICE TRENDS OVER TIME
# *****************************

print("\nPRICE TREND OVER YEARS")
avg_price_by_year = data.groupby('Year')['Price_Clean'].mean().dropna()
plt.figure(figsize=(12, 6))
sns.lineplot(x=avg_price_by_year.index, y=avg_price_by_year.values, marker='o', linewidth=2.5, color='purple')
plt.title('Average Launch Price Over Time')
plt.xlabel('Year')
plt.ylabel('Avg Price (in million USD)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('price_trend.png', dpi=300, bbox_inches='tight')
plt.show()

# Print details related to Price Trend Over Years
print("\nAverage Launch Price Over Years:")
print(avg_price_by_year)

# *****************************
# OBJECTIVE 5: SPATIAL DISTRIBUTION OF LAUNCHES
# *****************************

print("\nSPATIAL DISTRIBUTION OF LAUNCHES")
launch_counts = data['Country'].value_counts()
plt.figure(figsize=(14, 10))
sns.barplot(y=launch_counts.index[:20], x=launch_counts.values[:20], palette='Spectral')
plt.title('Launches by Country (Top 20)')
plt.xlabel('Number of Launches')
plt.ylabel('Country')
plt.tight_layout()
plt.savefig('country_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# Print details related to Launch Distribution by Country
print("\nLaunch Distribution by Country:")
print(launch_counts.head(20))

# *****************************
# OBJECTIVE 6: CORRELATION ANALYSIS
# *****************************

print("\nCORRELATION ANALYSIS")
numeric_df = data[['Price_Clean', 'Year', 'Month']].dropna()
plt.figure(figsize=(8, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap: Price, Year, Month')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# Print details related to Correlation Analysis
print("\nCorrelation Analysis Results:")
print(numeric_df.corr())

# Mission Status Over Time
mission_status_over_time = data.groupby(['Year', 'MissionStatus']).size().unstack().fillna(0)
mission_status_over_time.plot(kind='line', marker='o', figsize=(12, 6))
plt.title('Mission Status Over Time')
plt.xlabel('Year')
plt.ylabel('Number of Missions')
plt.grid(True)
plt.tight_layout()
plt.savefig('mission_status_over_time.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n--- Space Mission Dashboard EDA Completed Successfully ---")

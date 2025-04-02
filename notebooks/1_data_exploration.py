import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set up plotting
plt.style.use('seaborn-v0_8-whitegrid')
sns.set(style="whitegrid")

# Define the data directory
DATA_DIR = './data/synthea/csv'

# Function to load the data
def load_dataset(file_name):
    file_path = os.path.join(DATA_DIR, file_name)
    return pd.read_csv(file_path)

# Load the main datasets
patients_df = load_dataset('patients.csv')
conditions_df = load_dataset('conditions.csv')
medications_df = load_dataset('medications.csv')
encounters_df = load_dataset('encounters.csv')
observations_df = load_dataset('observations.csv')
procedures_df = load_dataset('procedures.csv')

# Display basic information about the datasets
print("=== Dataset Information ===")
print(f"Number of patients: {patients_df.shape[0]}")
print(f"Number of conditions: {conditions_df.shape[0]}")
print(f"Number of medications: {medications_df.shape[0]}")
print(f"Number of encounters: {encounters_df.shape[0]}")
print(f"Number of observations: {observations_df.shape[0]}")
print(f"Number of procedures: {procedures_df.shape[0]}")

# Explore patients dataset
print("\n=== Patients Dataset ===")
print(patients_df.head())
print("\nPatients columns:", patients_df.columns.tolist())

# Explore conditions dataset
print("\n=== Conditions Dataset ===")
print(conditions_df.head())
print("\nConditions columns:", conditions_df.columns.tolist())

# Age distribution of patients
patients_df['BIRTHDATE'] = pd.to_datetime(patients_df['BIRTHDATE'])
current_date = pd.to_datetime('2023-01-01')
patients_df['AGE'] = (current_date - patients_df['BIRTHDATE']).dt.days / 365.25

plt.figure(figsize=(10, 6))
sns.histplot(data=patients_df, x='AGE', bins=20, kde=True)
plt.title('Age Distribution of Patients')
plt.xlabel('Age (years)')
plt.ylabel('Count')
plt.savefig('./plots/age_distribution.png')

# Gender distribution
plt.figure(figsize=(8, 6))
gender_counts = patients_df['GENDER'].value_counts()
gender_counts.plot(kind='bar')
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.savefig('./plots/gender_distribution.png')

# Top 10 conditions
plt.figure(figsize=(12, 8))
top_conditions = conditions_df['DESCRIPTION'].value_counts().head(10)
top_conditions.plot(kind='barh')
plt.title('Top 10 Medical Conditions')
plt.xlabel('Count')
plt.ylabel('Condition')
plt.tight_layout()
plt.savefig('./plots/top_conditions.png')

# Distribution of encounters by type
plt.figure(figsize=(12, 6))
encounters_type = encounters_df['ENCOUNTERCLASS'].value_counts()
encounters_type.plot(kind='bar')
plt.title('Distribution of Encounters by Type')
plt.xlabel('Encounter Type')
plt.ylabel('Count')
plt.savefig('./plots/encounter_types.png')

# Check for missing values in the datasets
print("\n=== Missing Values ===")
for name, df in [('patients', patients_df), 
                 ('conditions', conditions_df), 
                 ('medications', medications_df),
                 ('encounters', encounters_df),
                 ('observations', observations_df),
                 ('procedures', procedures_df)]:
    missing = df.isnull().sum().sum()
    print(f"{name}: {missing} missing values")

print("\nExploration complete. Plots saved to './plots/' directory.") 
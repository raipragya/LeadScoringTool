import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("enriched_leads.csv")

# Select numeric features you want to analyze
numeric_cols = ['Employee count', 'Revenue', 'Year founded', 'Converted']  # Include more if needed

# Drop rows with missing values in those columns
df_cleaned = df[numeric_cols].dropna()

# Convert to proper numeric types
df_cleaned[numeric_cols] = df_cleaned[numeric_cols].apply(pd.to_numeric, errors='coerce')
df_cleaned = df_cleaned.dropna()

# Optional: Bin Converted into categorical buckets if it’s continuous
# For better color separation in plots
df_cleaned['ConvertedCategory'] = pd.cut(df_cleaned['Converted'],
                                         bins=[-0.1, 0.25, 0.75, 1.0],
                                         labels=['Low', 'Medium', 'High'])

# Create the pairplot with hue based on converted status
sns.pairplot(df_cleaned,
             vars=['Employee count', 'Revenue', 'Year founded'],
             hue='ConvertedCategory',
             diag_kind='hist',
             palette='Set1')

plt.suptitle("Pairplot Colored by Converted Status", y=1.02)
plt.tight_layout()
plt.show()

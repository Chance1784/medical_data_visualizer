import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1: Import data
df = pd.read_csv("/workspace/boilerplate-medical-data-visualizer/medical_examination.csv")

# 2: Create 'overweight' column based on BMI
df['BMI'] = df['weight'] / ((df['height'] / 100) ** 2)  # BMI formula
df['overweight'] = df['BMI'].apply(lambda x: 1 if x > 25 else 0)

# 3: Normalize cholesterol and gluc columns
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)  # Normal = 0, Abnormal = 1
df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)  # Normal = 0, Abnormal = 1

# 4: Draw categorical plot
def draw_cat_plot():
    # 5: Melt the data to create a DataFrame for categorical features
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # 6: Group the data by cardio and get the count of each category
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='count')

    # 7: Create a categorical plot
    catplot = sns.catplot(data=df_cat, x='variable', hue='value', col='cardio', kind='count')

    # 8: Set x-axis label to 'variable' and y-axis label to 'total'
    catplot.set_axis_labels("variable", "total")

    # 9: Get the figure for saving
    fig = catplot.fig

    # 10: Save the categorical plot as an image
    fig.savefig('catplot.png')
    return fig


# 10: Draw the heatmap
def draw_heat_map():
    # 11: Clean the data (remove incorrect data)
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &  # Diastolic pressure should be lower than systolic pressure
        (df['height'] >= df['height'].quantile(0.025)) &  # Height within 2.5th percentile
        (df['height'] <= df['height'].quantile(0.975)) &  # Height within 97.5th percentile
        (df['weight'] >= df['weight'].quantile(0.025)) &  # Weight within 2.5th percentile
        (df['weight'] <= df['weight'].quantile(0.975))  # Weight within 97.5th percentile
    ]

    # 12: Drop 'BMI' from the dataframe (as expected in the test)
    df_heat = df_heat.drop(columns=['BMI'])

    # 13: Calculate the correlation matrix
    corr = df_heat.corr()

    # 14: Generate a mask for the upper triangle of the heatmap
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 15: Set up the matplotlib figure for the heatmap
    plt.figure(figsize=(12, 8))

    # 16: Create the heatmap with annotations
    sns.heatmap(corr, mask=mask, annot=True, fmt='.1f', cmap='coolwarm', cbar_kws={'shrink': 0.8})

    # 17: Save the heatmap as an image
    fig = plt.gcf()  # Get the current figure
    fig.savefig('heatmap.png')
    return fig

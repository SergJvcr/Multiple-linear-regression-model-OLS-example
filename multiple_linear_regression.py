import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import variance_inflation_factor

data = pd.read_csv('google_data_analitics\\marketing_sales_data_2.csv')
print(data.head(5))

# DATA EXPLORATION
print(f'This dataset contains {data.shape[0]} columns and {data.shape[1]} rows.')
print(data.info())
print(data.describe(include='all'))

# Create a pairplot to visualize the relationship between the continous variables in data
sns.pairplot(data)
plt.show()
# Radio and Social Media both appear to have linear relationships with Sales. 
# Given this, Radio and Social Media may be useful as independent variables in 
# a multiple linear regression model estimating Sales. 
# TV and Influencer are excluded from the pairplot because they are not numeric.

# Calculate the mean sales for each categorical variable (TV and Influencer)
mean_sales_tv_category = data.groupby('TV')['Sales'].mean()
print(f'The mean Sales for each category in TV column: {mean_sales_tv_category}')
print('----------------------------------------------------')
mean_sales_influencer_category = data.groupby('Influencer')['Sales'].mean()
print(f'The mean Sales for each category in Influencer column: {mean_sales_influencer_category}')

# Remove missing data
# Drop rows that contain missing data and update the DataFrame
missing_values = data.isna().sum() # data before cleaning
print(f'This dataset contains some missing data: {missing_values}')

data = data.dropna(axis=0) #cleaning
missing_values_clean = data.isna().sum() # data after cleaning
print(f'After cleaning this dataset does not contain missing data: {missing_values}')

# Clean column names
# There mustn't be an empty space in variables' names!!!!
data = data.rename(columns={'Social Media': 'Social_Media'}) # rename all columns in data that contain a space
print(data.describe(include='all'))

# MODEL BUILDING
# Take the data which will used in model building
ols_data = data[['Radio', 'TV', 'Sales']]
# Define the OLS formula
ols_formula = 'Sales ~ Radio + C(TV)' # C() - categorical data
# Create an OLS model
OLS = ols(formula=ols_formula, data=ols_data)
# Fit the model
multi_model = OLS.fit()
# Save the results summary
model_results = multi_model.summary()
# Display the model results
print(model_results)

# TV was selected, as the preceding analysis showed a strong relationship 
# between the TV promotional budget and the average Sales.
# Radio was selected because the pairplot showed a strong linear relationship between Radio and Sales.
# Social Media was not selected because it did not increase model performance 
# and it was later determined to be correlated with another independent variable: Radio.
# Influencer was not selected because it did not show a strong relationship to Sales in the preceding analysis.

# CHECK MODEL ASSUMPTIONS
# 1. Linearity assumption
# Create a scatterplot for each independent variable and the dependent variable.
fig, axes = plt.subplots(1,3, figsize=(15, 5))

sns.scatterplot(x=ols_data['Radio'], y=ols_data['Sales'], ax=axes[0], color='green') # for checking
axes[0].set_xlabel('Radio')
axes[0].set_ylabel('Sales')
axes[0].set_title('Radio VS Sales')

sns.scatterplot(x = data['Social_Media'], y = data['Sales'],ax=axes[1]) # just to compaire with Radio and TV data
axes[1].set_title('Social Media and Sales')
axes[1].set_ylabel('Sales')
axes[1].set_xlabel('Social Media')

sns.stripplot(x=ols_data['TV'], y=ols_data['Sales'], ax=axes[2], hue=ols_data['TV'], order=['Low', 'Medium', 'High']) # for checking
axes[2].set_xlabel('TV')
axes[2].set_ylabel('Sales')
axes[2].set_title('TV VS Sales')
plt.tight_layout()
plt.show()

# The regression line for the Radio/Sales scatter plot
plt.figure(figsize=(5.5, 5.5))
sns.regplot(x=ols_data['Radio'], y=ols_data['Sales'], color='lightgreen', line_kws=dict(color="r"))
plt.title('Radio VS Sales')
plt.show()

# The linearity assumption holds for Radio, as there is a clear linear relationship 
# in the scatterplot between Radio and Sales. Social Media was not included in the preceding 
# multiple linear regression model, but it does appear to have a linear relationship with Sales.

# 2. Independence assumption
# The independent observation assumption states that each observation in the dataset is independent. 
# As each marketing promotion (i.e., row) is independent from one another, the independence assumption is not violated.

# 3. Normality assumption
# Plot 1: Histogram of the residuals
# Plot 2: Q-Q plot of the residuals

# Calculate the residuals
residuals = multi_model.resid # get the residuals from the model
# Visualize the distribution of the residuals (here are two ways - two graphs)
# Create a Q-Q plot.
fig, axes = plt.subplots(1,2, figsize=(10, 5))
# Create a histogram with the residuals
sns.histplot(residuals, ax=axes[0], color='lightblue')
# Set the x label of the residual plot
axes[0].set_xlabel('Residual Value')
# Set the title of the residual plot
axes[0].set_title('Histogram of Residuals')
# Create a Q-Q plot of the residuals
sm.qqplot(residuals, line='s', ax=axes[1], color='lightgreen')
# Set the title of the Q-Q plot
axes[1].set_title('Normal Q-Q plot')
# Use matplotlib's tight_layout() function to add space between plots for a cleaner appearance
plt.tight_layout()
plt.show()

# The histogram of the residuals are approximately normally distributed, 
# which supports that the normality assumption is met for this model. 
# The residuals in the Q-Q plot form a straight line, further supporting that this assumption is met.

# 4. Constant variance assumption/ Homoscedasticity
fitted_values = multi_model.fittedvalues # get fitted values
# Create a scatterplot of residuals against fitted values.
fig = sns.scatterplot(x=fitted_values, y=residuals, color='pink')
# Set the x-axis label
fig.set_xlabel('Fitted values from the model')
# Set the y-axis label
fig.set_ylabel('Residuals')
# Set the title
fig.set_title('Fitted values vs Residuals')
# Add a line at y = 0 to visualize the variance of residuals above and below 0
fig.axhline(y=0, color='red')
plt.show()

# The fitted values are in three groups because the categorical variable is dominating in this model, 
# meaning that TV is the biggest factor that decides the sales.
# However, the variance where there are fitted values is similarly distributed, validating that the assumption is met.

# 5. No multicollinearity assumption
# The no multicollinearity assumption states that no two independent variables 
# ( 𝑋𝑖 and  𝑋𝑗) can be highly correlated with each other.
# Two common ways to check for multicollinearity are to:
# - Create scatterplots to show the relationship between pairs of independent variables
# - Use the variance inflation factor to detect multicollinearity
sns.pairplot(data)
plt.show()

# *Calculate the variance inflation factor
# Create a subset of the data with the continous independent variables
X = data[['Radio','Social_Media']]
# Calculate the variance inflation factor for each variable
vif = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
# Create a DataFrame with the VIF results for the column names in X
df_vif = pd.DataFrame(vif, index=X.columns, columns = ['VIF'])
# Display the VIF results
print(df_vif)

# The preceding model only has one continous independent variable, meaning there are no multicollinearity issues.
# If a model used both Radio and Social_Media as predictors, there would be a moderate 
# linear relationship between Radio and Social_Media that violates the multicollinearity assumption. 
# Furthermore, the variance inflation factor when both Radio and Social_Media are included in the model 
# is 5.17 for each variable, indicating high multicollinearity.

# RESULTS AND EVALUATION
print(model_results)

# Using TV and Radio as the independent variables results in a multiple linear regression model with 𝑅2=0.904. 
# In other words, the model explains  90.4% of the variation in Sales. 
# This makes the model an excellent predictor of Sales.

# When TV and Radio are used to predict Sales, the model coefficients are:
# 𝛽0=218.5261
# 𝛽𝑇𝑉𝐿𝑜𝑤=−154.2971
# 𝛽𝑇𝑉𝑀𝑒𝑑𝑖𝑢𝑚=−75.3120
# 𝛽𝑅𝑎𝑑𝑖𝑜=2.9669

# The linear equation:
# Sales=𝛽0+𝛽1∗𝑋1+𝛽2∗𝑋2+𝛽3∗𝑋3
# Sales=𝛽0+𝛽𝑇𝑉𝐿𝑜𝑤∗𝑋𝑇𝑉𝐿𝑜𝑤+𝛽𝑇𝑉𝑀𝑒𝑑𝑖𝑢𝑚∗𝑋𝑇𝑉𝑀𝑒𝑑𝑖𝑢𝑚+𝛽𝑅𝑎𝑑𝑖𝑜∗𝑋𝑅𝑎𝑑𝑖𝑜
# Sales=218.5261−154.2971∗𝑋𝑇𝑉𝐿𝑜𝑤−75.3120∗𝑋𝑇𝑉𝑀𝑒𝑑𝑖𝑢𝑚+2.9669∗𝑋𝑅𝑎𝑑𝑖𝑜

# The default TV category for the model is High since there are coefficients 
# for the other two TV categories, Medium and Low. Because the coefficients for 
# the Medium and Low TV categories are negative, that means the average of sales 
# is lower for Medium or Low TV categories compared to the High TV category when Radio is at the same level.
# For example, the model predicts that a Low TV promotion is 154.2971 lower on average compared 
# to a high TV promotion given the same Radio promotion.
# The coefficient for Radio is positive, confirming the positive linear relationship 
# shown earlier during the exploratory data analysis.

# The p-value for all coefficients is  0.000, meaning all coefficients are statistically significant at  𝑝=0.05 (confidence level). 
# The 95% confidence intervals for each coefficient should be reported when presenting results to stakeholders.
# For example, there is a  95% chance that the interval  [−163.979,−144.616] contains the true parameter of the slope of  𝛽𝑇𝑉𝐿𝑜𝑤, 
# which is the estimated difference in promotion sales when a Low TV promotion is chosen instead of a High TV promotion.

# Beta coefficients allow us to estimate the magnitude and direction (positive or negative) of the effect 
# of each independent variable on the dependent variable. 
# The coefficient estimates can be converted to explainable insights, 
# such as the connection between an increase in TV promotional budgets and sales mentioned previously.

# Given how accurate TV was as a predictor, the model could be improved by getting a more granular view 
# of the TV promotions, such as by considering more categories or the actual TV promotional budgets.
# Furthermore, additional variables, such as the location of the marketing campaign 
# or the time of year, could increase model accuracy.

# RECOMENDAATIONS
# According to the model, high TV promotional budgets result in significantly more sales than medium 
# and low TV promotional budgets. For example, 
# the model predicts that a Low TV promotion is 154.2971 lower on average 
# than a high TV promotion given the same Radio promotion.

# The coefficient for radio is positive, confirming the positive linear relationship shown 
# earlier during the exploratory data analysis.
# The p-value for all coefficients is  0.000, meaning all coefficients are statistically significant at  𝑝=0.05. 
# The 95% confidence intervals for each coefficient should be reported when presenting results to stakeholders.

# For example, there is a  95% chance the interval  [−163.979,−144.616] contains the true parameter of the slope of  𝛽𝑇𝑉𝐿𝑜𝑤,
# which is the estimated difference in promotion sales when a low TV promotional budget is chosen instead of a high TV promotion budget.

# High TV promotional budgets have a substantial positive influence on sales. 
# The model estimates that switching from a high to medium TV promotional budget reduces sales by  $75.3120 million 
# (95% CI  [−82.431,−68.193]), and switching from a high to low TV promotional budget reduces sales by  $154.297 million 
# (95% CI  [−163.979,−144.616]). The model also estimates that an increase of  $1 million in the radio promotional budget 
# will yield a  $2.9669 million increase in sales (95% CI  [2.551,3.383]).

# Thus, it is recommended that the business allot a high promotional budget to TV when possible 
# and invest in radio promotions to increase sales.
import pandas as pd
import statsmodels.api as sm

from sklearn.neighbors import NearestNeighbors

# Set your own file path for the file
mb = pd.read_csv(r'C:\Users\cdemasi\OneDrive - Delaware North\Desktop\Sample San Diego Dataset.csv')
# pd.read_csv(r'C:\Users\cdemasi\OneDrive - Delaware North\Desktop\Braves Analysis\mba313 v2.csv')
pivot_mb = mb.pivot_table(index ='id', columns='name', values='quantity', fill_value=0)

pivot_mb.head()

# Turn all values into binary
binary_df=pivot_mb.applymap(lambda x:1 if x>0 else 0)

# Set Model Type
model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=3)
# Run Model
model.fit(binary_df.T)

distances, indices = model.kneighbors(binary_df.T)


item_names = binary_df.columns
def get_item_name(index):
    return item_names[index]
# Cheese Cup Index
item_index = 31

# Pretzel Index
# item_index = 6
print(f"Item: {get_item_name(item_index)}")
for neighbor_index in indices[item_index][1:]:  # Skip the first one (itself)
    print(f"- {get_item_name(neighbor_index)}")

# Now We're going to run a logit model to see what the probability of a cheese cup purchase is, given a pretzel or Nacho purchase'

# Ensure 'CHEESE CUP' is binary (0 or 1)
pivot_mb['CHEESE CUP'] = pivot_mb['CHEESE CUP'].apply(lambda x: 1 if x > 0 else 0)

# Define the independent variables (features) and the dependent variable (target)
X = pivot_mb[['BAV PRETZEL', 'BP NACHOS']]
y = pivot_mb['CHEESE CUP']

# Add a constant term to the independent variables
X = sm.add_constant(X)

# Fit the logistic regression model
logit_model = sm.Logit(y, X)
result = logit_model.fit()

print(result.summary())

# Specify the values for which you want the predicted probability
combinations = pd.DataFrame({
    'const': 1,
    'BAV PRETZEL': [1, 1, 0, 0],
    'BP NACHOS': [1, 0, 1, 0]
})

# Get the predicted probabilities for all combinations
predicted_probs_all = result.predict(combinations)

# Combine the specified values and predicted probabilities into a DataFrame
result_df_all = pd.concat([combinations[['BAV PRETZEL', 'BP NACHOS']], predicted_probs_all], axis=1)
result_df_all.columns = ['BAV PRETZEL', 'BP NACHOS', 'predicted_probability']

# Display the result
print(result_df_all)
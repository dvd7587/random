import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

data = pd.read_csv("ws.csv")

data['Pos'].apply(lambda x: x.strip())

columns = ("PPG", "RPG", "APG", "BPG", "SPG")

numeric_cols = data[list(columns)].copy()

def standardize_data(X):
             
    '''
    This function standardize an array, its substracts mean value, 
    and then divide the standard deviation.
    
    param 1: array 
    return: standardized array
    '''    
    rows, columns = X.shape
    
    standardizedArray = np.zeros(shape=(rows, columns))
    tempArray = np.zeros(rows)
    
    for column in range(columns):
        
        mean = np.mean(X[:,column])
        std = np.std(X[:,column])
        tempArray = np.empty(0)
        
        for element in X[:,column]:
            
            tempArray = np.append(tempArray, ((element - mean) / std))
 
        standardizedArray[:,column] = tempArray
    
    return standardizedArray
    
std_arr = standardize_data(numeric_cols.values)
u, s, vh = np.linalg.svd(std_arr)
cov_mat= np.cov(std_arr.T)
eigen_values, eigen_vectors = np.linalg.eig(cov_mat)
eigen_values
variance_explained = []
for i in eigen_values:
     variance_explained.append((i/sum(eigen_values))*100)
        
print(variance_explained)
cumulative_variance_explained = np.cumsum(variance_explained)
print(cumulative_variance_explained)
projection_matrix = (eigen_vectors.T[:][:2]).T
print(projection_matrix)
X_pca = std_arr.dot(projection_matrix)
print(X_pca)
data[["pca_x", 'pca_y']] = X_pca
data
fig_ = sns.scatterplot(data=data, x = 'pca_x', y='pca_y', hue = 'Pos', style='Pos', size='FPPM', hue_order=['G', 'GF', 'F', 'FC', 'C'])

for line in range(0, len(data)):
    fig_.text(data.pca_x[line]+0.2, data.pca_y[line], data.Player[line], horizontalalignment='center', size='medium', color='black', alpha=0.3)

plt.show()


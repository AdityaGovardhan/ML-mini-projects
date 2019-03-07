##################################
## HEART PATIENT DATA ANALYSIS  ##
## CREATED BY: ADITYA GOVARDHAN ##
##################################

import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

##############################################
## returns most highly correlated variables ##
##############################################
def most_highly_correlated(mydataframe, numtoreport): 
    cormatrix = mydataframe.corr() 
    cormatrix = cormatrix.where(np.triu(np.ones(cormatrix.shape, dtype = bool), k=1))
    cormatrix = cormatrix.stack()
    cormatrix = cormatrix.reindex(cormatrix.abs().sort_values(ascending=False).index).reset_index()
    cormatrix.columns = ["first_variable", "second_variable", "correlation"] 
    return cormatrix.head(numtoreport)

########################################################
## returns most highly correlated variables with a1p2 ##
########################################################
def result_highly_correlated(mydataframe, result, numtoreport):
	vals = mydataframe.drop(columns=result).corrwith(heart[result])
	indices = vals.abs().sort_values(ascending=False).index
	cormatrix = vals[indices]
	return cormatrix.head(numtoreport)

#################################################
## plots correlation heatmap of all parameters ##
#################################################
def corr_heatmap(mydataframe, cmap):
	corr = mydataframe.corr()
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1)
	ax.set_title("Correlation")
	sns.heatmap(corr.where(np.tril(np.ones(corr.shape, dtype = bool), k=-1)).abs(), cmap="YlGnBu", ax=ax)
	plt.show()

################################################################
## plots pairplot of five most correlated variables with a1p2 ##
################################################################
def var_pairplot(mydataframe, numtoreport):
    sns.set(style='whitegrid', context='notebook')
    cols = mydataframe.drop(columns='a1p2').corrwith(heart['a1p2']).sort_values(ascending=False).index[0:numtoreport]
    sns.pairplot(mydataframe[cols])
    plt.show()

################
## START HERE ##
################
heart = pd.read_csv('heart1.csv')

print('Descriptive Statistics of Variables')
print('===================================')
print(heart.describe().round(4))
print()

print('Correlation Matrix of Variables')
print('===============================')
print(heart.corr().round(4))
print()

print('Correlation Heatmap')
print('===================')
corr_heatmap(heart, 'YlGnBu')
print()

print('Most Highly Correlated Variables')
print('================================')
print(most_highly_correlated(heart, 5).round(4))
print()

print('Most Highly Correlated Variables with a1p2')
print('==========================================')
print(result_highly_correlated(heart, 'a1p2', 5).round(4))
print()

print('Most Highly Correlated Variables with a1p2 Pairplot')
print('===================================================')
var_pairplot(heart, 5)
print()
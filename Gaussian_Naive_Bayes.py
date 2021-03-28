#!/usr/bin/env python
# coding: utf-8

# In[110]:


import numpy as np
import pandas as pd
import scipy.stats as s
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report,accuracy_score,precision_score,recall_score


# In[111]:


class GaussianNB:
    """Intantiate a Gaussian Naive Bayes Object with the following parameters:
    
    feaures            : A dataframe consisting of continuous feaures, excluding labels
    labels             : A series consisting of binary labels
    data_split_ratio   : A tuple consisting of data splitting ratio
    apply_pca          : Boolean value spacifying whether to apply PCA or not
    n_components       : Number of Eigen Vectors having Non Zero values to keep
    """
    
    def __init__(xerox_copy,features,labels,data_split_ratio,apply_pca,n_components):
        xerox_copy.binary_labels=np.array(labels).reshape(labels.shape[0],1)
        xerox_copy.split_ratio=data_split_ratio
        xerox_copy.n_principal_components=n_components
        xerox_copy.unique_labels=list(labels.unique())
        if apply_pca == True:
            xerox_copy.X_new=xerox_copy.apply_dim_reduction(features,xerox_copy.n_principal_components)
            
    def apply_dim_reduction(xerox_copy,data,n_components):
        X=np.array(data)
        mu=np.mean(X,axis=0)
        mu=mu.reshape(-1,mu.shape[0])
        X_dash=X-mu
        sigma_hat=(1/data.shape[0])*np.matmul(X_dash.T,X_dash)
        sigma_hat_decompose=np.linalg.svd(sigma_hat)
        Q=sigma_hat_decompose[0]
        Q_tilda=Q[:,0:n_components]
        X_new=np.matmul(X_dash,Q_tilda)
        return X_new
    
    def data_splitting(xerox_copy):
        new_data=pd.DataFrame(data=xerox_copy.X_new)
        new_data['label']=xerox_copy.binary_labels
        training_data_len=int(xerox_copy.split_ratio[0]*new_data.shape[0])
        neg_training_data=new_data[new_data['label']==xerox_copy.unique_labels[0]].iloc[0:training_data_len//2]
        pos_training_data=new_data[new_data['label']==xerox_copy.unique_labels[1]].iloc[0:training_data_len//2]
        training_data=pd.concat([neg_training_data,pos_training_data])
        cv_data_len= int(xerox_copy.split_ratio[1]*new_data.shape[0])
        neg_remaining_data=new_data[new_data['label']==xerox_copy.unique_labels[0]].iloc[training_data_len//2:]
        pos_remaining_data=new_data[new_data['label']==xerox_copy.unique_labels[1]].iloc[training_data_len//2:]
        remaining_data=pd.concat([neg_remaining_data,pos_remaining_data])
        cv_data=remaining_data.iloc[0:cv_data_len]
        testing_data=remaining_data.iloc[cv_data_len:]
        return training_data,cv_data,testing_data
    
    def train_gaussian_nb(xerox_copy,data):
        mu_hat_pos=np.array(data[data['label']==xerox_copy.unique_labels[1]].iloc[:,0:xerox_copy.n_principal_components].mean())
        sigma_hat_pos=np.array(data[data['label']==xerox_copy.unique_labels[1]].iloc[:,0:xerox_copy.n_principal_components].cov())
        mu_hat_neg=np.array(data[data['label']==xerox_copy.unique_labels[0]].iloc[:,0:xerox_copy.n_principal_components].mean())
        sigma_hat_neg=np.array(data[data['label']==xerox_copy.unique_labels[0]].iloc[:,0:xerox_copy.n_principal_components].cov())
        xerox_copy.neg_likelihood_params=(mu_hat_neg,sigma_hat_neg)
        xerox_copy.pos_likelihood_params=(mu_hat_pos,sigma_hat_pos)
        
        
    def evaluate(xerox_copy,data):
        inputs=np.array(data.iloc[:,0:xerox_copy.n_principal_components])
        posterior_pos=s.multivariate_normal.pdf(inputs,xerox_copy.pos_likelihood_params[0],xerox_copy.pos_likelihood_params[1])
        posterior_neg=s.multivariate_normal.pdf(inputs,xerox_copy.neg_likelihood_params[0],xerox_copy.neg_likelihood_params[1])
        boolean_mask=posterior_pos>posterior_neg
        predicted_category=pd.Series(boolean_mask)
        predicted_category.replace(to_replace=[False,True],value=[xerox_copy.unique_labels[0],xerox_copy.unique_labels[1]],inplace=True)
        predicted_results=np.array(predicted_category) 
        actual_results=np.array(data['label'])
        testing_accuracy=np.count_nonzero(predicted_category==actual_results)/actual_results.shape[0]
        print(classification_report(actual_results,predicted_results,testing_accuracy,target_names=xerox_copy.unique_labels))
        
        


# In[112]:


if __name__=="__main__":
    print('goint to run a module as a script')


# In[ ]:





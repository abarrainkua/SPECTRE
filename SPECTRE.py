"""

.. _ex:

Example: Experiment to test the SPECTRE framework
===========================================

Example of employing SPECTRE with many real-world dataset. 
We load the the dataset with the chosen sensitive attribute. 
We use 10 different partitions for train, val and test. On each iteration we 
calculate the classification error for the overall population as well as 
for the different sensitive groups.
We consider different strategies for hyperparameter tuning.

"""

import folktables
from folktables import ACSDataSource
import numpy as np
import pdb
import pandas as pd
import random
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.impute import SimpleImputer
from sklearn.utils import Bunch
from MRCpy import MRC
from itertools import product
import itertools
import sys


def employment_filter(data):
    """
    Filters for the employment prediction task
    """
    df = data
    df = df[df['AGEP'] > 16]
    df = df[df['AGEP'] < 90]
    df = df[df['PWGTP'] >= 1]
    return df

def adult_filter(data):
    """Mimic the filters in place for Adult data.

    Adult documentation notes: Extraction was done by Barry Becker from
    the 1994 Census database. A set of reasonably clean records was extracted
    using the following conditions:
    ((AAGE>16) && (AGI>100) && (AFNLWGT>1)&& (HRSWK>0))
    """
    df = data
    df = df[df['AGEP'] > 16]
    df = df[df['PINCP'] > 100]
    df = df[df['WKHP'] > 0]
    df = df[df['PWGTP'] >= 1]
    return df

def public_coverage_filter(data):
    """
    Filters for the public health insurance prediction task; focus on low income Americans, and those not eligible for Medicare
    """
    df = data
    df = df[df['AGEP'] < 65]
    df = df[df['PINCP'] <= 30000]
    return df


def normalizeLabels(origY):
    """
    Normalize the labels of the instances in the range 0,...r-1 for r classes
    """

    # Map the values of Y from 0 to r-1
    domY = np.unique(origY)
    Y = np.zeros(origY.shape[0], dtype=int)

    for i, y in enumerate(domY):
        Y[origY == y] = i

    return Y

def one_hot_encode_dataframe(input_df, categorical_columns):

    # Perform one-hot encoding
    encoded_df = pd.get_dummies(input_df, columns=categorical_columns)

    return encoded_df

def get_best_config(strategy, res_group_config, res_acc_config, param, tol=0.05, N = 5):
    
    '''
    Returns the best configuration according to strategy: 
    
        - 'WCE'      : Takes the lambda that has the smallest worst class error.
            If 2 different lambda values have the same worst class error then gets
            the lambda value with the highest accuracy in the validation set. 
        - 'WCE+T+A'  : Considers existing worst class error + tolerance and 
            chooses the configuration with the highest overall accuracy.
        - 'TOPN+WCE' : Selects the configurations with topN accuracy values, and 
            among them selects the value that provides the best worst-class accuracy. 
        - 'ACC'      : Selects the hyperparameter value that provides the highest
            accuracy value.
            
            
            
        Parameters
        -----------
        
        - res_group_config  : Dataframe containing the results on group errors for different 
            values of the hyperparameter.
        - res_acc_config    : The overall accuracy values for different values of the hyperparameter. 
        - param             : The hyperparameter under optimization.
        - tol               : The tolerance value for strategy WCE+A
        - N                 : Number of top hyperparameter values with highest accuracy.
    '''
    
    
    if strategy == 'WCE': 
        # Step 1: Find the maximum 'worst_acc' value
        max_worst_acc = res_acc_config['worst_acc'].max()
        # Step 2: Filter rows with the highest 'worst_acc' value
        filtered_df = res_acc_config[res_acc_config['worst_acc'] == max_worst_acc]
        # Step 3: Find the row with the highest 'acc' among the filtered rows
        best_config = filtered_df.loc[filtered_df['acc'].idxmax(), param]
        
    elif strategy == 'WCE+T+A':
        # Step 1: Find the maximum 'worst_acc' value
        max_worst_acc = res_acc_config['worst_acc'].max()
        # Step 2: Step 2: Filter rows where 'worst_acc' is within the tolerance 
        # range from the max_worst_acc
        filtered_df = res_acc_config[res_acc_config['worst_acc'] >= (max_worst_acc - tol)]
        # Step 3: Find the row with the highest 'acc' among the filtered rows
        best_config = filtered_df.loc[filtered_df['acc'].idxmax(), param]

    elif strategy == 'TOPN+WCE':
        # Step 1: Sort by 'acc' in descending order and select the top N rows
        top_N_configs = res_acc_config.nlargest(N, 'acc')
        # Step 2: Among these top N, find the row with the highest 'worst_acc'
        best_config = top_N_configs.loc[top_N_configs['worst_acc'].idxmax(), param]

    elif strategy == 'ACC':
        # Step 1: Find the row with the lmbd value with the  highest 'acc'
        best_config = res_acc_config.loc[res_acc_config['acc'].idxmax(), param]

    return best_config


# PARAMETERS:
feat_map= 'fourier'  # feature mapping
loss = '0-1'  # loss function
dataset = 'ACSIncome'  # dataset
sens_att = 'race'  # sensitive attribute 'race' or 'int'
strategy = 'WCE'  # The strategy for hyperparameter tuning
val = 20  # The size of the validation set
state = 'NE'  # The state if ACS datasets is considered
scale = True  # Whether to scale the data
N_min = 1  # Minimum instance value for a group to be included in the analysis

root_name1 = 'results/SPECTRE_'

# LOAD THE DATASET

if dataset == 'ACSEmployment':
    ACSEmployment = folktables.BasicProblem(
        features=[
        'AGEP', #age; for range of values of features please check Appendix B.4 of Retiring Adult: New Datasets for Fair Machine Learning NeurIPS 2021 paper
        'SCHL', #educational attainment
        'MAR', #marital status
        'RELP', #relationship
        'DIS', #disability recode
        'ESP', #employment status of parents
        'CIT', #citizenship status
        'MIG', #mobility status (lived here 1 year ago)
        'MIL', #military service
        'ANC', #ancestry recode
        'NATIVITY', #nativity
        'DEAR', #hearing difficulty
        'DEYE', #vision difficulty
        'DREM', #cognitive difficulty
        'SEX', #sex
        'RAC1P', #recoded detailed race code
        'GCL', #grandparents living with grandchildren 
        ],
        target='ESR', #employment status recode
        target_transform=lambda x: x == 1,
        group='SEX',
        preprocess=employment_filter, 
        postprocess=lambda x: np.nan_to_num(x, -1),
    )
    
    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    acs_data = data_source.get_data(states=[state], download=True) #data 
    
    # Loading the dataset
    features, label, group = ACSEmployment.df_to_numpy(acs_data)
    group_aux = np.array(features[:,15])  # race
    group_aux = group_aux.astype(int)
    label = label*1
    group = group-1

    # Get all the values of features into integers
    features = features.astype(int)

    # Get one hot encoding
    col_names =[
        'AGEP', #age; for range of values of features please check Appendix B.4 of Retiring Adult: New Datasets for Fair Machine Learning NeurIPS 2021 paper
        'SCHL', #educational attainment
        'MAR', #marital status
        'RELP', #relationship
        'DIS', #disability recode
        'ESP', #employment status of parents
        'CIT', #citizenship status
        'MIG', #mobility status (lived here 1 year ago)
        'MIL', #military service
        'ANC', #ancestry recode
        'NATIVITY', #nativity
        'DEAR', #hearing difficulty
        'DEYE', #vision difficulty
        'DREM', #cognitive difficulty
        'SEX', #sex
        'RAC1P', #recoded detailed race code
        'GCL', #grandparents living with grandchildren
    ]
    input_df = pd.DataFrame(features, columns=col_names)
    categorical_columns = [
        'SCHL', #educational attainment
        'MAR', #marital status
        'RELP', #relationship
        'DIS', #disability recode
        'ESP', #employment status of parents
        'CIT', #citizenship status
        'MIG', #mobility status (lived here 1 year ago)
        'MIL', #military service
        'ANC', #ancestry recode
        'NATIVITY', #nativity
        'DEAR', #hearing difficulty
        'DEYE', #vision difficulty
        'DREM', #cognitive difficulty
        'SEX', #sex
        'RAC1P', #recoded detailed race code
    ]
    one_hot_df = one_hot_encode_dataframe(input_df, categorical_columns)
    one_hot_df = one_hot_df.astype(int)

    features = one_hot_df.to_numpy()

    X = features #[np.arange(0,50000),] 
    Y = label #[np.arange(0,50000),]
    
    if sens_att == 'race':
        S = group_aux #[np.arange(0,50000),]
        S_aux = group
    elif sens_att == 'int':
        S_gend = group #[np.arange(0,50000),]
        S_race = group_aux
        S = S_race * 10 + S_gend
        S_aux = np.copy(S)
    
    
    # Open files to save the results:
        
    # File to save the final results
    filename1 = root_name1 + dataset + '_' + state + '_' + sens_att + '_val' + str(val) + strategy + 'Nmin' + str(N_min) + '.txt'
    file1 = open(filename1, 'w')

    # File to save the group results for different lambda configurations
    filename2 = root_name1 + dataset + '_' + state + '_' + sens_att + '_val' + str(val) + strategy + 'Nmin' + str(N_min) + '_n_group_err.csv'
    file2 = open(filename2, 'w')

    # File to save the worst group results for different lambda configurations
    filename3 = root_name1 + dataset + '_'+ state + '_' + sens_att + '_val' + str(val) +  strategy + 'Nmin' + str(N_min) + '_worst_case.csv'
    file3 = open(filename3, 'w')
    
    
    
elif dataset == 'ACSIncome':
    ACSIncome = folktables.BasicProblem(
        features=[
            'AGEP',
            'COW',
            'SCHL',
            'MAR',
            'OCCP',
            'POBP',
            'RELP',
            'WKHP',
            'SEX',
            'RAC1P',
        ],
        target='PINCP',
        target_transform=lambda x: x > 50000,
        group='RAC1P',
        preprocess=adult_filter,
        postprocess=lambda x: np.nan_to_num(x, -1),
    )
    
    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    acs_data = data_source.get_data(states=[state], download=True) #data 
    
    # Loading the dataset
    features, label, group = ACSIncome.df_to_numpy(acs_data)
    group_aux = np.array(features[:,8])  # gender
    #features[:,15] = group
    group_aux = group_aux.astype(int)
    label = label*1
    group = group.astype(int)

    # Get all the values of features into integers
    features = features.astype(int)


    # Get one hot encoding
    col_names =[
            'AGEP',
            'COW',
            'SCHL',
            'MAR',
            'OCCP',
            'POBP',
            'RELP',
            'WKHP',
            'SEX',
            'RAC1P',
    ]
    input_df = pd.DataFrame(features, columns=col_names)
    categorical_columns = [
            'COW',
            'SCHL',
            'MAR',
            'OCCP',
            'POBP',
            'RELP',
            'SEX',
            'RAC1P',
    ]
    one_hot_df = one_hot_encode_dataframe(input_df, categorical_columns)
    one_hot_df = one_hot_df.astype(int)

    features = one_hot_df.to_numpy()

    X = features #[np.arange(0,50000),] 
    Y = label #[np.arange(0,50000),]
    
    if sens_att == 'race':
        S = group #[np.arange(0,50000),]
        S_aux = group_aux
    elif sens_att == 'int':
        S_gend = group_aux #[np.arange(0,50000),]
        S_race = group
        S = S_race * 10 + S_gend
        S_aux = np.copy(S)
        
    # Open files to save the results:
    
    # File to save the final results
    filename1 = root_name1 + dataset + '_' + state + '_' + sens_att + '_val' + str(val) + strategy + 'Nmin' + str(N_min) + '.txt'
    file1 = open(filename1, 'w')

    # File to save the group results for different lambda configurations
    filename2 = root_name1 + dataset + '_' + state + '_' + sens_att + '_val' + str(val) + strategy + 'Nmin' + str(N_min) + '_n_group_err.csv'
    file2 = open(filename2, 'w')

    # File to save the worst group results for different lambda configurations
    filename3 = root_name1 + dataset + '_'+ state + '_' + sens_att + '_val' + str(val) + strategy + 'Nmin' + str(N_min) + '_worst_case.csv'
    file3 = open(filename3, 'w')

elif dataset == 'ACSPublicCoverage': 
    ACSPublicCoverage = folktables.BasicProblem(
        features=[
            'AGEP',
            'SCHL',
            'MAR',
            'SEX',
            'DIS',
            'ESP',
            'CIT',
            'MIG',
            'MIL',
            'ANC',
            'NATIVITY',
            'DEAR',
            'DEYE',
            'DREM',
            'PINCP',
            'ESR',
            'ST',
            'FER',
            'RAC1P',
        ],
        target='PUBCOV',
        target_transform=lambda x: x == 1,
        group='RAC1P',
        preprocess=public_coverage_filter,
        postprocess=lambda x: np.nan_to_num(x, -1),
    )
    
    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    acs_data = data_source.get_data(states=[state], download=True) #data 
    
    # Loading the dataset
    features, label, group = ACSPublicCoverage.df_to_numpy(acs_data)
    group_aux = np.array(features[:,3])  # gender
    #features[:,15] = group
    group = group.astype(int)
    label = label*1

    # Get all the values of features into integers
    features = features.astype(int)

    # Get one hot encoding
    col_names =[
            'AGEP',
            'SCHL',
            'MAR',
            'SEX',
            'DIS',
            'ESP',
            'CIT',
            'MIG',
            'MIL',
            'ANC',
            'NATIVITY',
            'DEAR',
            'DEYE',
            'DREM',
            'PINCP',
            'ESR',
            'ST',
            'FER',
            'RAC1P',
    ]
    input_df = pd.DataFrame(features, columns=col_names)
    categorical_columns = [
            'SCHL',
            'MAR',
            'SEX',
            'DIS',
            'ESP',
            'CIT',
            'MIG',
            'MIL',
            'ANC',
            'NATIVITY',
            'DEAR',
            'DEYE',
            'DREM',
            'PINCP',
            'ESR',
            'ST',
            'FER',
            'RAC1P',
    ]
    one_hot_df = one_hot_encode_dataframe(input_df, categorical_columns)
    one_hot_df = one_hot_df.astype(int)

    features = one_hot_df.to_numpy()

    X = features #[np.arange(0,50000),] 
    Y = label #[np.arange(0,50000),]
    
    if sens_att == 'race':
        S = group #[np.arange(0,50000),]
        S_aux = group_aux
    elif sens_att == 'int':
        S_gend = group_aux #[np.arange(0,50000),]
        S_race = group
        S = S_race * 10 + S_gend
        S_aux = np.copy(S)
    
    # Open files to save the results:
        
    # File to save the final results
    filename1 = root_name1 + dataset + '_' + state + '_' + sens_att + '_val' + str(val) + strategy + 'Nmin' + str(N_min) + '.txt'
    file1 = open(filename1, 'w')

    # File to save the group results for different lambda configurations
    filename2 = root_name1 + dataset + '_' + state + '_' + sens_att + '_val' + str(val) + strategy + 'Nmin' + str(N_min) + '_n_group_err.csv'
    file2 = open(filename2, 'w')

    # File to save the worst group results for different lambda configurations
    filename3 = root_name1 + dataset + '_'+ state + '_' + sens_att + '_val' + str(val) + strategy + 'Nmin' + str(N_min) + '_worst_case.csv'
    file3 = open(filename3, 'w')

elif dataset == 'COMPAS':
    
    df = pd.read_csv('propublica-recidivism_original.csv', header = 0)
    data = df.copy(deep = True)
    
    features = data[['age', 'age_cat', 'juv_fel_count', 'juv_misd_count', 'priors_count', 'c_charge_degree', 'c_charge_desc']]
    label = data['two_year_recid']
    group = data['race']
    group_aux = data['sex']
    group_int = data['sex-race']  #intersectional group

    cat_features = ['age_cat', 'c_charge_degree', 'c_charge_desc']

    # One hot encoding of categorical variables
    df_all = one_hot_encode_dataframe(features, cat_features)
    X = df_all.to_numpy()

    Y = label.to_numpy() #[np.arange(0,50000),]
    S_int = group_int.to_numpy()
    
    if sens_att == 'race':
        S = group.to_numpy() #[np.arange(0,50000),]
        S_aux = group_aux.to_numpy()
    elif sens_att == 'int':
        S = group_int.to_numpy()
        S_aux = np.copy(S)
    
    # Open files to save the results:
        
    # File to save the final results
    filename1 = root_name1 + dataset + '_' + sens_att + '_val' + str(val) + strategy + 'Nmin' + str(N_min) +'.txt'
    file1 = open(filename1, 'w')

    # File to save the group results for different lambda configurations
    filename2 = root_name1 + dataset + '_' + sens_att + '_val' + str(val) + strategy + 'Nmin' + str(N_min) + '_n_group_err.csv'
    file2 = open(filename2, 'w')

    # File to save the worst group results for different lambda configurations
    filename3 = root_name1 + dataset + '_'+ sens_att + '_val' + str(val) + strategy + 'Nmin' + str(N_min) + '_worst_case.csv'
    file3 = open(filename3, 'w')

else:
    print("dataset is not supported")
    

N_min = N_min

random_seed = 1234
np.random.seed(random_seed)
random_state = np.random.randint(low=0, high=18374, size=5)

# The dataframe to save the results
res_group = pd.DataFrame(columns=['sig', 'lmbd', 'group', 'error'])
res_worst_case = pd.DataFrame(columns=['sig', 'lmbd','worst_acc', 'acc'])

# Filtering X, Y, S and S_aux to contain only instances from groups
# with more than N_min instances
# Convert S to a pandas Series to count instances per group
S_series = pd.Series(S)
# Step 1: Count occurrences of each group in S
group_counts = S_series.value_counts()
# Step 2: Identify groups with at least N_min instances
valid_groups = group_counts[group_counts >= N_min].index
# Step 3: Create a boolean mask for rows in X, Y, S and S_aux belonging to valid groups
mask = S_series.isin(valid_groups)
# Filter X, Y, S and S_aux based on the mask
X = X[mask]
Y = Y[mask]
S = S[mask]
S_aux = S_aux[mask]

r = len(np.unique(Y))
n, d = X.shape

sigma_vals = np.logspace(-2, 2, 10)
lambda_vals = np.linspace(0.05, 0.5, 10) 

cvError = list() # overall error
cvError_worst = list() # worst group error (gender)
cvError_max_diff = list() # maximum disparity (gender)
cvError_aux_worst = list() # worst error (race)
cvError_aux_max_diff = list() # maximum disparity (race)
cvError_worst_tpr = list()
cvError_max_diff_tpr = list()
cvError_worst_ar = list()
cvError_max_diff_ar = list()
best_lmbd_vals = np.array([])


for rs in random_state:
    
    # 1) FIND THE OPTIMAL $\sigma$
    
    lmbd = 0.3  # initialize \lambda
    
    res_group_iter = pd.DataFrame(columns=['sig','group', 'error'])
    res_worst_case_iter = pd.DataFrame(columns=['sig', 'worst_acc', 'acc'])

    X_train, X_test, y_train, y_test, s_train, s_test, s_aux_train, s_aux_test = train_test_split(
        X, Y, S, S_aux, test_size=0.3, random_state=rs
    ) # train/test 70/30
    X_train, X_val, y_train, y_val, s_train, s_val, s_aux_train, s_aux_val = train_test_split(
        X_train, y_train, s_train, s_aux_train, test_size= val/100 , random_state=rs
    ) # train-train / train-val

    if scale:
        std_scale = preprocessing.StandardScaler().fit(X_train, y_train)
        X_train = std_scale.transform(X_train)
        X_test = std_scale.transform(X_test)
        X_val = std_scale.transform(X_val)
        
    d = X_train.shape[1]
    scale_val = np.sqrt((d * X_train.var()) / 2)
    
    for sigma_val in sigma_vals:
    
        phi_kwargs = dict(
                sigma = scale_val * 1 /(sigma_val)
        )
        
        # Train the MRC with the particular value for sigma
        clf = MRC(phi=feat_map, s = lmbd, loss=loss, solver='cvx', **phi_kwargs)
        clf.fit(X_train, y_train)

        # Predict the class for test instances
        y_pred = clf.predict(X_val)

        # Get group error vector
        err_group = list()
        for s in np.unique(s_val):
            y_pred_group = y_pred[s_val == s]
            y_val_group = y_val[s_val == s]
            group_error = np.average(y_pred_group != y_val_group)
            err_group.append(group_error)
            # Save group error
            group_res_aux = pd.DataFrame({'sig':"{:.3f}".format(sigma_val/scale_val),
                                    'lmbd':"{:.2f}".format(lmbd),
                                    'group':s,
                                    'error':"{:.3f}".format(group_error)}, index = [0])

            # Update overall result dataframe
            res_group = pd.concat([res_group, group_res_aux], ignore_index = True)
            res_group.reset_index()

            # Update the results dataframe of this iteration
            group_res_aux = pd.DataFrame({'sig': sigma_val/scale_val,
                                    'group': s,
                                    'error': group_error}, index = [0])
            res_group_iter = pd.concat([res_group_iter, group_res_aux], ignore_index = True)
            res_group_iter.reset_index()

        worst_error = max(err_group)

        # Save worst group error and overall accuracy
        acc_all = 1 - np.average(y_pred != y_val)
        wc_res = pd.DataFrame({'sig':"{:.3f}".format(sigma_val/scale_val),
                            'lmbd':"{:.2f}".format(lmbd),
                            'worst_acc':"{:.3f}".format(1-worst_error),
                           'acc':"{:.3f}".format(acc_all)}, index = [0])

        # Update the results dataframe
        res_worst_case = pd.concat([res_worst_case, wc_res], ignore_index = True)
        res_worst_case.reset_index()

        # Update the results dataframe of this iteration
        wc_res = pd.DataFrame({'sig': sigma_val/scale_val,
                            'worst_acc': (1-worst_error),
                            'acc': acc_all}, index = [0])
        res_worst_case_iter = pd.concat([res_worst_case_iter, wc_res], ignore_index = True)
        res_worst_case_iter.reset_index()


        # Print the result
        print(f" $\sigma$={sigma_val/scale_val} worst_acc={1-worst_error}")
        print(f" $\sigma$={sigma_val/scale_val} acc_all={acc_all}")



    # Get the optimal sigma according to strategy:
    best_sig = get_best_config(strategy=strategy, res_group_config=res_group_iter, res_acc_config=res_worst_case_iter, param = 'sig')
    

    # 2) FIND THE OPTIMAL $\lambda$

    res_group_iter = pd.DataFrame(columns=['lmbd','group', 'error'])
    res_worst_case_iter = pd.DataFrame(columns=['lmbd', 'worst_acc', 'acc'])
    
    phi_kwargs = dict(
           sigma = 1/best_sig
    )
    
    for lmbd in lambda_vals:
                
        # Train the MRC with the particular value for lambda
        clf = MRC(phi=feat_map, s = lmbd, loss=loss, solver='cvx', **phi_kwargs)
        clf.fit(X_train, y_train)

        # Predict the class for test instances
        y_pred = clf.predict(X_val)
        
        # Get group error vector
        err_group = list()
        for s in np.unique(y_val):
            y_pred_group = y_pred[y_val == s]
            y_val_group = y_val[y_val == s]
            group_error = np.average(y_pred_group != y_val_group)
            err_group.append(group_error)
            #Save group error
            group_res_aux = pd.DataFrame({'sig':"{:.3f}".format(best_sig),
                                    'lmbd':"{:.2f}".format(lmbd),
                                    'group':s,
                                    'error':"{:.3f}".format(group_error)}, index = [0])

            # Update overall result dataframe
            res_group = pd.concat([res_group, group_res_aux], ignore_index = True)
            res_group.reset_index()

            # Update the results dataframe of this iteration
            group_res_aux = pd.DataFrame({'lmbd': lmbd,
                                    'group': s,
                                    'error': group_error}, index = [0])
            res_group_iter = pd.concat([res_group_iter, group_res_aux], ignore_index = True)
            res_group_iter.reset_index()

        worst_error = max(err_group)

        # Save worst group error and overall accuracy
        acc_all = 1 - np.average(y_pred != y_val)
        wc_res = pd.DataFrame({'sig':"{:.3f}".format(best_sig),
                            'lmbd':"{:.2f}".format(lmbd),
                            'worst_acc':"{:.3f}".format(1-worst_error),
                            'acc':"{:.3f}".format(acc_all)}, index = [0])

        # Update the results dataframe
        res_worst_case = pd.concat([res_worst_case, wc_res], ignore_index = True)
        res_worst_case.reset_index()

        # Update the results dataframe of this iteration
        wc_res = pd.DataFrame({'lmbd': lmbd,
                            'worst_acc': (1-worst_error),
                            'acc': acc_all}, index = [0])
        res_worst_case_iter = pd.concat([res_worst_case_iter, wc_res], ignore_index = True)
        res_worst_case_iter.reset_index()


        # Print the result
        print(f" $\lambda$={lmbd} worst_acc={1-worst_error}")
        print(f" $\lambda$={lmbd} acc_all={acc_all}")
        
        
    # Get the optimal sigma according to strategy:
    best_lmbd = get_best_config(strategy=strategy, res_group_config=res_group_iter, res_acc_config=res_worst_case_iter, param = 'lmbd')
        
    
    # Train the final MRC with best lmbd
    X_train, X_test, y_train, y_test, s_train, s_test, s_aux_train, s_aux_test = train_test_split(X, Y, S,
                                                                                                        S_aux,
                                                                                                        test_size=0.30,
                                                                                                        random_state=rs)
    if scale:
        std_scale = preprocessing.StandardScaler().fit(X_train, y_train)
        X_train = std_scale.transform(X_train)
        X_test = std_scale.transform(X_test)
        
    phi_kwargs = dict(
            sigma = 1/best_sig
    )

    # Train the MRC with the best \lambda configuration
    clf = MRC(phi=feat_map, s = best_lmbd, loss=loss, solver='cvx', **phi_kwargs)
    clf.fit(X_train, y_train)

    # Prediction of the test instances
    y_pred = clf.predict(X_test)

    # Get best/worst group errors and disparities in TPR and AR
    acc_group = list()
    tpr_group = list()
    ar_group = list()

    for s in np.unique(s_test):
        y_pred_group = y_pred[s_test == s]
        y_test_group = y_test[s_test == s]
        group_error = np.average(y_pred_group != y_test_group)
        acc_group.append(group_error)

        # Compute TPR
        tp = np.sum((y_pred_group == 1) & (y_test_group == 1))
        fn = np.sum((y_pred_group == 0) & (y_test_group == 1))
        if (tp + fn) > 0 : # Avoid division by zero
            tpr = tp / (tp + fn)
            tpr_group.append(tpr)

        # Compute Acceptance Rate (AR)
        if np.sum(y_test_group == 1) > 0:
            ar = np.mean(y_pred_group == 1)
            ar_group.append(ar)


    worst_error = max(acc_group)
    best_error = min(acc_group)

    worst_tpr = min(tpr_group)
    max_tpr_diff = max(tpr_group) - min(tpr_group)

    worst_ar = min(ar_group)
    max_ar_diff = max(ar_group) - min(ar_group)


    # Get best/worst group errors (auxiliary sensitive attribute)
    aux_worst_error = 0
    aux_best_error = 1
    acc_group_aux = list()
    for s_aux in np.unique(s_aux_test):
        y_pred_group = y_pred[s_aux_test == s_aux]
        y_test_group = y_test[s_aux_test == s_aux]
        aux_group_error = np.average(y_pred_group != y_test_group)
        acc_group_aux.append(aux_group_error)

    aux_worst_error = max(acc_group_aux)
    aux_best_error = min(acc_group_aux)

    # Save results for this repetition
    cvError.append(np.average(y_pred != y_test))
    cvError_worst.append(worst_error)
    cvError_max_diff.append((worst_error-best_error))
    cvError_aux_worst.append(aux_worst_error)
    cvError_aux_max_diff.append((aux_worst_error-aux_best_error))
    cvError_worst_tpr.append(worst_tpr)
    cvError_max_diff_tpr.append(max_tpr_diff)
    cvError_worst_ar.append(worst_ar)
    cvError_max_diff_ar.append(max_ar_diff)


# Get average (and std) of the results 

res_mean = np.average(cvError)
res_std = np.std(cvError)

res_mean_worst = np.average(cvError_worst)
res_std_worst = np.std(cvError_worst)

res_mean_max_diff = np.average(cvError_max_diff)
res_std_max_diff = np.std(cvError_max_diff)

res_mean_aux_worst = np.average(cvError_aux_worst)
res_std_aux_worst = np.std(cvError_aux_worst)

res_mean_aux_max_diff = np.average(cvError_aux_max_diff)
res_std_aux_max_diff = np.std(cvError_aux_max_diff)

res_mean_worst_tpr = np.average(cvError_worst_tpr)
res_std_worst_tpr = np.std(cvError_worst_tpr)

res_mean_max_diff_tpr = np.average(cvError_max_diff_tpr)
res_std_max_diff_tpr = np.std(cvError_max_diff_tpr)

res_mean_worst_ar = np.average(cvError_worst_ar)
res_std_worst_ar = np.std(cvError_worst_ar)

res_mean_max_diff_ar = np.average(cvError_max_diff_ar)
res_std_max_diff_ar = np.std(cvError_max_diff_ar)

results = pd.DataFrame(
    {
        "dataset": 'ACS',
        "best_lmbd": str(best_lmbd_vals),
        "acc": "%1.3g" % np.multiply((1-res_mean),100) + " \pm " + "%1.3g" % np.multiply(res_std,100),
        "worst_acc": "%1.3g" % np.multiply((1-res_mean_worst),100) + " \pm " + "%1.3g" % np.multiply(res_std_worst,100),
        "max_disp": "%1.3g" % np.multiply((res_mean_max_diff),100) + " \pm " + "%1.3g" % np.multiply(res_std_max_diff,100),
        "aux_worst_acc": "%1.3g" % np.multiply((1-res_mean_aux_worst),100) + " \pm " + "%1.3g" % np.multiply(res_std_aux_worst,100),
        "aux_max_disp": "%1.3g" % np.multiply((res_mean_aux_max_diff),100) + " \pm " + "%1.3g" % np.multiply(res_std_aux_max_diff,100),
        "worst_tpr": "%1.3g" % np.multiply((res_mean_worst_tpr),100) + " \pm " + "%1.3g" % np.multiply(res_std_worst_tpr,100),
        "max_disp_tpr": "%1.3g" % np.multiply((res_mean_max_diff_tpr),100) + " \pm " + "%1.3g" % np.multiply(res_std_max_diff_tpr,100),
        "worst_ar": "%1.3g" % np.multiply((res_mean_worst_ar),100) + " \pm " + "%1.3g" % np.multiply(res_std_worst_ar,100),
        "max_disp_ar": "%1.3g" % np.multiply((res_mean_max_diff_ar),100) + " \pm " + "%1.3g" % np.multiply(res_std_max_diff_ar,100)
    },
    index=[0],
)

print(results)


file1.write(str(results['acc']))
file1.write('\n')
file1.write(str(results['worst_acc']))
file1.write('\n')
file1.write(str(results['max_disp']))
file1.write('\n')
file1.write(str(results['aux_worst_acc']))
file1.write('\n')
file1.write(str(results['aux_max_disp']))
file1.write('\n')
file1.write(str(results['best_lmbd']))
file1.write('\n')
file1.write(str(results['worst_tpr']))
file1.write('\n')
file1.write(str(results['max_disp_tpr']))
file1.write('\n')
file1.write(str(results['worst_ar']))
file1.write('\n')
file1.write(str(results['max_disp_ar']))
file1.write('\n')
file1.close()

res_group.to_csv(file2)
res_worst_case.to_csv(file3)  

file2.close()
file3.close()

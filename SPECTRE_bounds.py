"""

.. _ex:

Example: Computing the out-of-sample performance guarantees of SPECTRE
===========================================

Example of employing SPECTRE witha toy and many real-world dataset. 
We load the the dataset with the chosen sensitive attribute. 
We use 10 different partitions for train, val and test. On each iteration we 
calculate the classification error for the overall population as well as 
for the different sensitive groups.

"""

import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import check_random_state
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from MRCpy import MRC
import cvxpy as cvx
import random
import sys
import folktables
from folktables import ACSDataSource

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

# PARAMETERS:
feat_map= 'fourier'  # feature mapping
loss = '0-1'  # loss function
dataset = 'toy'  # dataset
sens_att = ''  # sensitive attribute 'race' or 'int'
strategy = 'WCE'  # The strategy for hyperparameter tuning
val = 20  # The size of the validation set
scale = True  # Whether to scale the data
N_min = 1  # Minimum instance value for a group to be included in the analysis
lmbd_vals = np.linspace(0.05, 0.5, 20) # sigma values
sigma_vals = np.logspace(-2, 2, 10)  # lambda values
state = "" 
scale = False  # Whether to scale the data

# Load the dataset dataset

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
    

elif dataset == "toy":
    df = pd.read_csv('toy3_dataset_1000.csv', header = 0)
    data = df.copy(deep = True)

    features = data[['X1', 'X2']]
    label = data['Y']
    group = data['S']

    X = features.to_numpy()
    Y = label.to_numpy() 
    S = group.to_numpy()
    S_aux = group.to_numpy()


else:
    print("dataset is not supported")


# Open the datasets to store the results

# File to save the bounds
filename1 = 'freqbounds_SPECTRE_' + dataset + state +  '.csv'

# File to save the true error
filename2 = 'freqbounds_truerr_SPECTRE_' + dataset + state + '.csv'



random_seed = np.random.randint(low=0, high=1837418, size=10)

# Dataframe for the true errors
true_err = pd.DataFrame(columns=['group', 'err'])

# Dataset for the bounds
res_df = pd.DataFrame(columns=['group', 'lmbd', 'up_bound', 'low_bound'])

for rs in random_seed:
    
    
    ############################### TRAIN SPECTRE ###########################
    
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
    
    for sigma_val in sigma_vals:
        
        phi_kwargs = dict(
                sigma = 1/sigma_val
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

            # Update the results dataframe of this iteration
            group_res_aux = pd.DataFrame({'sig': sigma_val,
                                    'group': s,
                                    'error': group_error}, index = [0])
            res_group_iter = pd.concat([res_group_iter, group_res_aux], ignore_index = True)
            res_group_iter.reset_index()

        worst_error = max(err_group)
        acc_all = 1 - np.average(y_pred != y_val)

        # Update the results dataframe of this iteration
        wc_res = pd.DataFrame({'sig': sigma_val,
                            'worst_acc': (1-worst_error),
                            'acc': acc_all}, index = [0])
        res_worst_case_iter = pd.concat([res_worst_case_iter, wc_res], ignore_index = True)
        res_worst_case_iter.reset_index()

    # Get the optimal sigma according to strategy:
    best_sig = get_best_config(strategy=strategy, res_group_config=res_group_iter, res_acc_config=res_worst_case_iter, param = 'sig')
    

    # 2) FIND THE OPTIMAL $\lambda$

    res_group_iter = pd.DataFrame(columns=['lmbd','group', 'error'])
    res_worst_case_iter = pd.DataFrame(columns=['lmbd', 'worst_acc', 'acc'])
    
    phi_kwargs = dict(
           sigma = 1/best_sig
    )
    
    for lmbd in lmbd_vals:
                
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

            # Update the results dataframe of this iteration
            group_res_aux = pd.DataFrame({'lmbd': lmbd,
                                    'group': s,
                                    'error': group_error}, index = [0])
            res_group_iter = pd.concat([res_group_iter, group_res_aux], ignore_index = True)
            res_group_iter.reset_index()

        worst_error = max(err_group)
        acc_all = 1 - np.average(y_pred != y_val)

        # Update the results dataframe of this iteration
        wc_res = pd.DataFrame({'lmbd': lmbd,
                            'worst_acc': (1-worst_error),
                            'acc': acc_all}, index = [0])
        res_worst_case_iter = pd.concat([res_worst_case_iter, wc_res], ignore_index = True)
        res_worst_case_iter.reset_index()
        
        
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

    err_group = list()
    for s in np.unique(s_test):
        y_pred_group = y_pred[s_test == s]
        y_test_group = y_test[s_test == s]
        group_error = np.average(y_pred_group != y_test_group)
        err_group.append(group_error)
        new_true = pd.DataFrame({'group': s, 'err': group_error},
                                   index=[0])

        true_err = pd.concat([true_err, new_true], ignore_index = True)
        true_err.reset_index()

    print('True error:')
    print(true_err)

    ################################ THE BOUNDS ################################
    
    
    # Reduce train to 30%
    # Train the final MRC with best lmbd
    X_part, _, y_part, _, s_part, _, s_aux_part, _ = train_test_split(X_train, y_train, s_train, s_aux_train,
                                                                        test_size=0.7, random_state=rs)
    
    
    X_feat_part = clf.phi.transform(X_part)
    column_sums = np.sum(X_feat_part, axis=0)
    inst_num = X_feat_part.shape[0]
    tau = column_sums * (1/inst_num)
    column_std = np.std(X_feat_part, axis=0)
    lambda_ = best_lmbd * column_std * (1/np.sqrt(X_feat_part.shape[0]))

    # Get the probability predictions
    y_pred_prob = clf.predict_proba(X_part)


    num_inst = X_feat_part.shape[0]
    num_feat = X_feat_part.shape[1]
    
    # Define the optimization problem

    for j in range(num_inst):

        new_col = []
        new_col = np.append(new_col, X_feat_part[j,:])
        new_col = np.append(new_col, -X_feat_part[j,:])
        a41 = np.zeros(num_inst)
        a41[j] = 1
        new_col = np.append(new_col, a41)
        a42 = np.zeros(num_inst)
        a42[j] = -1
        new_col = np.append(new_col, a42)
        new_col = np.append(new_col, 0)

        if j == 0 :
            A = [new_col]
        else :
            A = np.vstack((A, new_col))

    z_col = []
    z_col = np.append(z_col, -1.0*(lambda_ + tau))
    z_col = np.append(z_col, tau - lambda_ )
    z_col = np.append(z_col, -1.0*np.ones(num_inst))
    z_col = np.append(z_col, np.zeros(num_inst))
    z_col = np.append(z_col, -1.0)

    A = np.vstack((A, z_col))
    A = A.transpose()

    rows = 2*num_feat + 2*num_inst + 1

    # Get vector b
    b = np.zeros(rows)

    # Get vector h
    h = [1,0]


    for g in np.unique(s_part):

        # Create c and e vectors

        c = np.zeros(num_inst+1)
        e = np.zeros(num_inst+1)

        for k in range(num_inst):

            if s_part[k] == g:
                c[k] =  y_pred_prob[k,y_part[k]] - 1   # negative of 1 - h(y|x)
                e[k] = 1.0

        g3 = np.ones(num_inst + 1)
        g3[-1] = -1.0

        G = np.vstack((e, g3))
        
        
        try:
                
            x = cvx.Variable(num_inst+1)
            prob_up = cvx.Problem(cvx.Minimize(c.T@x),
                    [A @ x <= b, G @ x == h])
            prob_up.solve(solver = 'MOSEK')
            #prob_up.solve()

            prob_low = cvx.Problem(cvx.Maximize(c.T@x),
                    [A @ x <= b, G @ x == h])
        
            prob_low.solve(solver = 'MOSEK')
            #prob_low.solve()

            print("the bounds:")
            print(-prob_up.value)
            print(-prob_low.value)

            new_res = pd.DataFrame({'group': g, 'sigma': sigma_val, 'up_bound': -prob_up.value, 'low_bound': -prob_low.value},
                                index=[0])
            
            
            res_df = pd.concat([res_df, new_res], ignore_index = True)
            res_df.reset_index()
        except cvx.error.SolverError:
            continue



print(res_df)


# Save the results
res_df.to_csv(filename1, index=False)
true_err.to_csv(filename2, index=False)





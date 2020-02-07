dummies = ['MIGRATE1', 'VETSTAT', 'EMPSTAT', 'ANCESTR1', 'STATEFIP', 'OCC_BROAD', 'CLASSWKR', 'NCHILD', 'MARRNO', 'NCHLT5',
           'SEX', 'DEGFIELD', 'HISPAN', 'RACE', 'EDUC', 'MORTGAGE']

# Create training and test sets using smaller split to RandomSearch hyperparameters for LogReg and XGBoost
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9,random_state=42, stratify = y)

# One hot encode categorical variables
X_train_enc = pd.get_dummies(data = X_train, columns = dummies, drop_first = True)
X_test_enc = pd.get_dummies(data = X_test, columns = dummies, drop_first = True)

# Ensure same variables in train and test datasets
# Get missing columns in the training test
missing_cols = set(X_train_enc.columns ) - set(X_test_enc.columns )
# Add a missing column in test set with default value equal to 0
for c in missing_cols:
    X_test_enc[c] = 0
# Ensure the order of column in the test set is in the same order than in train set
X_test_enc = X_test_enc[X_train_enc.columns]

# Keep more important features as displayed in graph above
X_train_feat = X_train_enc[['PCT_HHINC', 'ANCESTR1_western european', 'AGE', 'EDUC_2', 'SEX_2', 'VALUEH', 'EDUC_1',
                           'AGEMARR', 'MORTAMT1', 'MARRNO_2', 'NCHILD_2', 'CLASSWKR_2', 'MORTGAGE_1', 'EDUC_3',
                           'ANCESTR1_north american', 'MORTGAGE_2', 'NCHILD_1', 'PCT_MTG_INC', 'UHRSWORK', 'MIGRATE1_2',
                           'RACE_2', 'OCC_BROAD_service', 'EMPSTAT_3', 'OCC_BROAD_mgmt, biz, fin', 'ANCESTR1_hispanic',
                           'NCHLT5_1']]

X_test_feat = X_test_enc[['PCT_HHINC', 'ANCESTR1_western european', 'AGE', 'EDUC_2', 'SEX_2', 'VALUEH', 'EDUC_1',
                           'AGEMARR', 'MORTAMT1', 'MARRNO_2', 'NCHILD_2', 'CLASSWKR_2', 'MORTGAGE_1', 'EDUC_3',
                           'ANCESTR1_north american', 'MORTGAGE_2', 'NCHILD_1', 'PCT_MTG_INC', 'UHRSWORK', 'MIGRATE1_2',
                           'RACE_2', 'OCC_BROAD_service', 'EMPSTAT_3', 'OCC_BROAD_mgmt, biz, fin', 'ANCESTR1_hispanic',
                           'NCHLT5_1']]

# Run SMOTE since divorced outcome target is only 18% of our dataset
print("Before OverSampling, counts of label '1': {}".format(sum(y_train == 1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train == 0)))

sm = SMOTE_imb(random_state=42)
X_train_sm, y_train_sm = sm.fit_sample(X_train_feat, y_train.ravel())

print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))

print("After OverSampling, counts of label '1': {}".format(sum(y_train_res == 1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_train_res == 0)))

# Normalization required for logistic regression
ss = StandardScaler(with_mean=True)
X_train_sm_ss = ss.fit_transform(X_train_sm)
X_test_ss = ss.transform(X_test_feat)

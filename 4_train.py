import numpy as np
import pandas as pd
import  pickle
from sklearn.feature_selection import SelectKBest,f_regression
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder

# Path settings
ft_lb_folder = "FeaturesAndLabels/"
spectro_folder = "Spectrograms/"
segment_folder = "Segments/"
template_folder = "Templates/"

# Load files
pkl_file = open(template_folder + 'TRAIN_SPEC_FEATURES_freq5.pkl', 'rb')
X1_train = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open(template_folder + 'TEST_SPEC_FEATURES_freq5.pkl', 'rb')
X1_test = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open(template_folder + 'TRAIN_SPEC_LABELS_freq5.pkl', 'rb')
Y_train = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open(template_folder + 'TEST_SPEC_LABELS_freq5.pkl', 'rb')
Y_test = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open(template_folder + 'TRAIN_LOG_SPEC_FEATURES_freq5.pkl', 'rb')
X2_train = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open(template_folder + 'TEST_LOG_SPEC_FEATURES_freq5.pkl', 'rb')
X2_test = pickle.load(pkl_file)
pkl_file.close()


enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(Y_train)
trainLabel = enc.transform(Y_train).toarray()
enc.fit(Y_test)
testLabel = enc.transform(Y_test).toarray()

num_species = len(enc.categories_[0])
# num_species

trainLabel = pd.DataFrame(trainLabel)
testLabel = pd.DataFrame(testLabel)

spec_avg = trainLabel.iloc[:, :10].mean()
plt.plot(spec_avg,'go')
plt.plot(-np.log(spec_avg),'bo')
spec_num_features = -np.log(spec_avg)

# Creating the training dataFrame
spec_names = ['tr_spec_'+str(x) for x in range(X1_train.shape[1]) ]
specDF = pd.DataFrame(X1_train, columns = spec_names )
trainDF_1 = pd.merge(left = trainLabel, right = specDF, left_index = True, right_index = True)

log_spec_names = ['tr_log_spec_'+str(x) for x in range(X2_train.shape[1]) ]
Spec_Log_Df = pd.DataFrame(X2_train, columns = log_spec_names )
trainDF = pd.merge(left = trainDF_1, right = Spec_Log_Df, left_index = True, right_index = True)

# Creating the testing dataFrame
spec_names = ['tr_spec_'+str(x) for x in range(X1_test.shape[1]) ]
specDF = pd.DataFrame(X1_test, columns = spec_names )
testDF_1 = pd.merge(left = testLabel, right = specDF, left_index = True, right_index = True)

log_spec_names = ['tr_log_spec_'+str(x) for x in range(X2_test.shape[1]) ]
Spec_Log_Df = pd.DataFrame(X2_test, columns = log_spec_names )
testDF = pd.merge(left = trainDF_1, right = Spec_Log_Df, left_index = True, right_index = True)

#################################
#           TRAINING            #
#                               #
#################################

CV_FOLDS = 15

RESULT = []
rs = 0
for ID in range(1):    
    for NUM_FEATURES in range(40,50,10):
        for N_ESTIMATORS in range(500,501,100):
            for MAX_FEATURES in range(4,5):
                for MIN_SAMPLES_SPLIT in range(2,3):
                    cv =  np.random.randint(0,CV_FOLDS,len(trainDF))
                    trainDF['cv'] = cv
                    labeled_vector = []
                    predicted_vector = []
                    predicted_test_vector = []
                    for bird in range(num_species):
                        predicted_test_vector.append(np.zeros(len(testDF)))
                        
                    for c in range(CV_FOLDS):
                        df_10 = trainDF[trainDF.cv == c]
                        df_90 = trainDF[trainDF.cv != c]

                        X_90 = df_90[spec_names+log_spec_names]
                        X_10 = df_10[spec_names+log_spec_names]
                        T = testDF[spec_names+log_spec_names]
                        
                        for bird in range(num_species):
                            rs = rs+1

                            y_90 = df_90[bird]
                            y_10 = df_10[bird]

                            selector = SelectKBest(f_regression,NUM_FEATURES + 50 -int(spec_num_features[bird]*10))
                            selector.fit(X_90, y_90)

                            df_90_features = selector.transform(X_90)
                            df_10_features = selector.transform(X_10)
                            T_features = selector.transform(T)

                            rfr = RandomForestRegressor(n_estimators = N_ESTIMATORS, max_features = MAX_FEATURES, min_samples_split = MIN_SAMPLES_SPLIT,random_state = rs*100, verbose = 0)
                            rfr.fit(df_90_features,y_90)

                            p_10 = rfr.predict(df_10_features)
                            T_pred = rfr.predict(T_features)
                            
                            
                            
                            predicted_vector = predicted_vector + list(p_10)
                            labeled_vector = labeled_vector + list(y_10)
                            
                            predicted_test_vector[bird] = predicted_test_vector[bird] + T_pred/CV_FOLDS
                    
                    fpr, tpr, thresholds = metrics.roc_curve(labeled_vector, predicted_vector, pos_label=1)
                    auc = metrics.auc(fpr,tpr)
                    
                    RESULT.append([ID,NUM_FEATURES,N_ESTIMATORS,MAX_FEATURES,MIN_SAMPLES_SPLIT,CV_FOLDS,auc])
                    ResultDf = pd.DataFrame(RESULT,columns=['ID','NUM_FEATURES','N_ESTIMATORS','MAX_FEATURES','MIN_SAMPLES_SPLIT','CV_FOLDS','AUC'])
                    ResultDf.to_csv('rfr_auc_result.txt', index = False)
                    
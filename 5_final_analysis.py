
import numpy as np
import pandas as pd
import scipy as sp
import  pickle
from sklearn.feature_selection import SelectKBest,f_regression
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn import metrics
from time import localtime, strftime
import scipy.ndimage 
from sklearn.preprocessing import OneHotEncoder
import json

# Path settings
ft_lb_folder = "FeaturesAndLabels/"
spectro_folder = "Spectrograms/"
segment_folder = "Segments/"
template_folder = "Templates/"

species =  json.load(open(ft_lb_folder + 'labelsDict.json'))
num_species = len(species)
# num_species

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

##########################
# Import the Patterns    #
##########################

pkl_file = open(segment_folder + 'SPEC_SEGMENTS.pkl', 'rb')
SPEC_SEGMENTS = pickle.load(pkl_file)
pkl_file.close()   

pkl_file = open(segment_folder + 'LOG_SPEC_SEGMENTS.pkl', 'rb')
LOG_SPEC_SEGMENTS = pickle.load(pkl_file)
pkl_file.close() 

#######################################################
## PARAMETER OPTIMIZATION & SUBMISSION CREATION      ##
#######################################################

CV_FOLDS = 15
NUM_FEATURES = 40
N_ESTIMATORS  = 500
MAX_FEATURES = 4 
MIN_SAMPLES_SPLIT = 2 
rs =  0 
print(strftime("%a, %d %b %Y %H:%M:%S +0000", localtime()))
rfr_models = []
top_k_features =[] 
important_features =[]
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
        k_col = selector.transform([X_90.columns])
        top_k_features.append(k_col)
        T_features = selector.transform(T)
        rfr = RandomForestRegressor(n_estimators = N_ESTIMATORS, max_features = MAX_FEATURES, min_samples_split = MIN_SAMPLES_SPLIT,random_state = rs*100, verbose = 0)
        rfr.fit(df_90_features,y_90)
        rfr_models.append(rfr)
        imp_df = pd.DataFrame(np.transpose([k_col[0],rfr.feature_importances_]),columns=['Attribute','Importance'])
        imp_df = imp_df.sort_values('Importance',ascending = False)
        important_features.append(imp_df)
        p_10 = rfr.predict(df_10_features)
        T_pred = rfr.predict(T_features)
        predicted_vector = predicted_vector + list(p_10)
        labeled_vector = labeled_vector + list(y_10)
        predicted_test_vector[bird] = predicted_test_vector[bird] + T_pred/CV_FOLDS

fpr, tpr, thresholds = metrics.roc_curve(labeled_vector, predicted_vector, pos_label=1)
auc = metrics.auc(fpr,tpr)
print("AUC is: ", auc)
print(strftime("%a, %d %b %Y %H:%M:%S +0000", localtime()))


spec_avg[spec_avg>0.05]
bird = 10

NR = 4
NC = 8

for bird in list(spec_avg[spec_avg>0.05].index):
    fig = plt.figure(figsize=(20, 10),dpi = 300,facecolor = '#fce8ae')   
    bird_image = sp.ndimage.imread("birdfolder/"+str(bird)+'.jpg')
    imp_df = important_features[bird]
    plt.subplot(NR,NC,1)
    plt.imshow(bird_image)
    plt.axis('off')
    plt.suptitle('Most important features ('+species[str(bird)][0] +')')
    j=1
    k=0
    #for j in range(1,NC*NR):
    while(j < NC*NR and  k <len(imp_df) ):
        #s = imp_df.Attribute.iloc[j]
        s = imp_df.Attribute.iloc[k]
        k=k+1
        if(s[:8] == 'tr_spec_'):
            segment_id = int(s[8:])
            [yr_min, yr_max, xr_min, xr_max] = SPEC_SEGMENTS[segment_id][2]
            minvalue = SPEC_SEGMENTS[segment_id][3].min()
            pic = np.zeros(200*(xr_max - xr_min+1) ) + minvalue
            pic = pic.reshape([200,xr_max - xr_min+1] )
            pic[yr_min:yr_max+1,:] = SPEC_SEGMENTS[segment_id][3]
            pic.shape
            plt.subplot(NR,NC,j+1)
            plt.imshow(pic,cmap=plt.cm.afmhot_r,aspect='auto')
            #ax[1][j].imshow(SPEC_SEGMENTS[segment_id][3],cmap=plt.cm.afmhot_r)
            plt.axis('off')
            
            j= j+1
        
        if(s[:12] == 'tr_log_spec_'):
            segment_id = int(s[12:])
            [yr_min, yr_max, xr_min, xr_max] = LOG_SPEC_SEGMENTS[segment_id][2]
            minvalue = LOG_SPEC_SEGMENTS[segment_id][3].min()
            pic = np.zeros(200*(xr_max - xr_min+1) ) + minvalue
            pic = pic.reshape([200,xr_max - xr_min+1] )
            pic[yr_min:yr_max+1,:] = LOG_SPEC_SEGMENTS[segment_id][3]
            pic.shape
            plt.subplot(NR,NC,j+1)
            plt.imshow(pic,cmap=plt.cm.afmhot_r,aspect='auto')
            plt.axis('off')
            j= j+1
    
    fig.savefig("birdfolder/"+'important_features_'+str(bird),facecolor='#fce8ae',edgecolor='none',dpi =300)





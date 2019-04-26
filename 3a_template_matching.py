import numpy as np
import scipy as sp
import pickle
from scipy import fft
from time import localtime, strftime
import matplotlib.pyplot as plt
from skimage.feature import match_template
from sklearn.model_selection import train_test_split

ft_lb_folder = "FeaturesAndLabels/"
spectro_folder = "Spectrograms/"
segment_folder = "Segments/"
template_folder = "Templates/"

print(strftime("%a, %d %b %Y %H:%M:%S +0000", localtime()))

features = np.load(ft_lb_folder + 'features.npy')
labels = np.load(ft_lb_folder + 'labels.npy')

X_train, X_test, y_train, y_test = train_test_split(
    features, labels, stratify=labels, test_size=0.25, random_state=42)

N = 88200
K = 512
Step = 4
wind = 0.5*(1 - np.cos(np.array(range(K))*2*np.pi/(K-1)))
ffts = []


###############################
# Create the Spectrograms
## Train + Test
###############################

print(strftime("%a, %d %b %Y %H:%M:%S +0000", localtime()))

for index, f in enumerate(features):

    Spectogram = []
    print("Getting spect for: ", index)
    for j in range(int(Step*N/K)-Step):  # 1246
        vec = f[int(j * K/Step): int((j+Step) * K/Step)] * wind
        Spectogram.append(abs(fft(vec, K)[:int(K/2)]))

    ffts.append(np.array(Spectogram))

print(strftime("%a, %d %b %Y %H:%M:%S +0000", localtime()))

# Import the Patterns
pkl_file = open(segment_folder + 'SPEC_SEGMENTS.pkl', 'rb')
SPEC_SEGMENTS = pickle.load(pkl_file)
pkl_file.close()


##################
# TRAIN FEATURES
##################
print(strftime("%a, %d %b %Y %H:%M:%S +0000", localtime()))
TRAIN_SPEC_FEATURES = []
for index, f in enumerate(X_train):

    print("doing for: " + str(index) + " out of " + str(len(X_train)))
    print(strftime("%a, %d %b %Y %H:%M:%S +0000", localtime()))
    mypic = np.transpose(ffts[index])
    mypic_rev = np.zeros_like(mypic)
    for i in range(mypic.shape[0]):
        mypic_rev[i] = mypic[-i - 1]

    # Focus on the Relevant Frequency Domain
    mypic_rev_small = mypic_rev[50:250, :]
    mypic_rev = mypic_rev_small
    mypic_rev_gauss = sp.ndimage.gaussian_filter(
        mypic_rev, sigma=3)  # Gaussian filter
    Segment_Row = []
    for s in range(len(SPEC_SEGMENTS)):
        y_min = SPEC_SEGMENTS[s][2][0]
        y_max = SPEC_SEGMENTS[s][2][1]
        segment = SPEC_SEGMENTS[s][3]
        if(y_min > 5):
            y_min_5 = y_min-5
        else:
            y_min_5 = 0

        if(y_max < 194):
            y_max_5 = y_max+5
        else:
            y_max_5 = 199

        spectrogram_part = mypic_rev_gauss[y_min_5:y_max_5+1, :]
        result = match_template(spectrogram_part, segment)
        Segment_Row.append(np.max(result))
        if(s%500==0):
            print("matching template for: " + str(s) + " out of " + str(len(SPEC_SEGMENTS)))

    TRAIN_SPEC_FEATURES.append(Segment_Row)


TRAIN_SPEC_FEATURES = np.array(TRAIN_SPEC_FEATURES)

# SAVE THE FEATURES for the TRAINING SET
output = open(template_folder + 'TRAIN_SPEC_FEATURES_freq5.pkl', 'wb')
pickle.dump(TRAIN_SPEC_FEATURES, output)
output.close()

TRAIN_SPEC_LABELS = np.array(y_train)

# SAVE THE LABELS for the TRAINING SET
output = open(template_folder + 'TRAIN_SPEC_LABELS_freq5.pkl', 'wb')
pickle.dump(TRAIN_SPEC_LABELS, output)
output.close()

print(strftime("%a, %d %b %Y %H:%M:%S +0000", localtime()))

###############################################################
# TEST FEATURES
# Same Transformations and Template Matching for the Test Set
###############################################################
print(strftime("%a, %d %b %Y %H:%M:%S +0000", localtime()))
TEST_SPEC_FEATURES = []

for index, f in enumerate(X_test):

    print("doing for: " + str(index) + " out of " + str(len(X_test)))
    mypic = np.transpose(ffts[index])
    mypic_rev = np.zeros_like(mypic)
    for i in range(mypic.shape[0]):
        mypic_rev[i] = mypic[-i - 1]

    # Focus on the Relevant Frequency Domain
    mypic_rev_small = mypic_rev[50:250, :]
    mypic_rev = mypic_rev_small
    mypic_rev_gauss = sp.ndimage.gaussian_filter(
        mypic_rev, sigma=3)  # Gaussian filter
    Segment_Row = []
    for s in range(len(SPEC_SEGMENTS)):
        y_min = SPEC_SEGMENTS[s][2][0]
        y_max = SPEC_SEGMENTS[s][2][1]
        segment = SPEC_SEGMENTS[s][3]
        if(y_min > 5):
            y_min_5 = y_min-5
        else:
            y_min_5 = 0

        if(y_max < 194):
            y_max_5 = y_max+5
        else:
            y_max_5 = 199

        spectrogram_part = mypic_rev_gauss[y_min_5:y_max_5+1, :]
        result = match_template(spectrogram_part, segment)
        Segment_Row.append(np.max(result))
        if(s%500==0):
            print("matching template for: " + str(s) + " out of " + str(len(SPEC_SEGMENTS)))

    TEST_SPEC_FEATURES.append(Segment_Row)


TEST_SPEC_FEATURES = np.array(TEST_SPEC_FEATURES)

# SAVE THE FEATURES for the TEST SET
output = open(template_folder + 'TEST_SPEC_FEATURES_freq5.pkl', 'wb')
pickle.dump(TEST_SPEC_FEATURES, output)
output.close()


TEST_SPEC_LABELS = np.array(y_test)

# SAVE THE LABELS for the TESTING SET
output = open(template_folder + 'TEST_SPEC_LABELS_freq5.pkl', 'wb')
pickle.dump(TEST_SPEC_LABELS, output)
output.close()

print(strftime("%a, %d %b %Y %H:%M:%S +0000", localtime()))

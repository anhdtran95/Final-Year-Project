import numpy as np
import scipy as sp
import pickle
from scipy import fft
from time import localtime, strftime
import matplotlib.pyplot as plt
from skimage.morphology import disk, remove_small_objects
from skimage.filters import rank
from skimage.util import img_as_ubyte
import random as rd 
import json

ft_lb_folder = "FeaturesAndLabels/"
spectro_folder = "Spectrograms/"
segment_folder = "Segments/"

print(strftime("%a, %d %b %Y %H:%M:%S +0000", localtime()))

features = np.load(ft_lb_folder + 'features.npy')
labels = np.load(ft_lb_folder + 'labels.npy')
labelsDict = json.load(open(ft_lb_folder + 'labelsDict.json'))
featureList = json.load(open(ft_lb_folder + 'featureList.json'))

# selecting 30 random features for pattern extraction
selectedList = []
for i in range(7):
    birdName = labelsDict[str(i)][0]
    indxList = [a for a, b in enumerate(featureList) if birdName in b]
    selectedList += rd.sample(indxList, 30)


def pic_to_ubyte(pic):
    a = (pic-np.min(pic)) / (np.max(pic - np.min(pic)))
    a = img_as_ubyte(a)
    return a


N = 88200 #2 sec
K = 512
Step = 4
wind = 0.5*(1 - np.cos(np.array(range(K))*2*np.pi/(K-1)))

SPEC_SEGMENTS = []
LOG_SPEC_SEGMENTS = []
MIN_SEGMENT_SIZE = 99
p = 90

fig = plt.figure(figsize=(20, 10))

print(strftime("%a, %d %b %Y %H:%M:%S +0000", localtime()))

for index, f in enumerate(features[selectedList]):
    Spectogram = []
    print("Getting spect for: ", index)
    for j in range(int(Step*N/K)-Step):  # 1246
        vec = f[int(j * K/Step): int((j+Step) * K/Step)] * wind
        Spectogram.append(abs(fft(vec, K)[:int(K/2)]))

    mypic = np.transpose(np.array(Spectogram))
    mypic_rev = np.zeros_like(mypic)
    for i in range(mypic.shape[0]):
        mypic_rev[i] = mypic[-i - 1]

    mypic_rev_small = mypic_rev[50:250, :]
    mypic_rev = mypic_rev_small
    mypic_rev_log = np.log10(mypic_rev + 0.001)
    mypic_rev_gauss = sp.ndimage.gaussian_filter(mypic_rev, sigma=3)
    mypic_rev_log_gauss = sp.ndimage.gaussian_filter(mypic_rev_log, sigma=3)
    mypic_rev_gauss_bin = mypic_rev_gauss > np.percentile(mypic_rev_gauss, p)
    mypic_rev_log_gauss_bin = mypic_rev_log_gauss > np.percentile(
        mypic_rev_log_gauss, p)
    mypic_rev_gauss_bin_close = sp.ndimage.binary_closing(
        sp.ndimage.binary_opening(mypic_rev_gauss_bin))
    mypic_rev_log_gauss_bin_close = sp.ndimage.binary_closing(
        sp.ndimage.binary_opening(mypic_rev_log_gauss_bin))
    mypic_rev_gauss_grad = rank.gradient(
        pic_to_ubyte(mypic_rev_gauss), disk(3))
    mypic_rev_log_gauss_grad = rank.gradient(
        pic_to_ubyte(mypic_rev_log_gauss), disk(3))
    mypic_rev_gauss_grad_bin = mypic_rev_gauss_grad > np.percentile(
        mypic_rev_gauss_grad, p)
    mypic_rev_log_gauss_grad_bin = mypic_rev_log_gauss_grad > np.percentile(
        mypic_rev_log_gauss_grad, p)
    mypic_rev_gauss_grad_bin_close = sp.ndimage.binary_closing(
        sp.ndimage.binary_opening(mypic_rev_gauss_grad_bin))
    mypic_rev_log_gauss_grad_bin_close = sp.ndimage.binary_closing(
        sp.ndimage.binary_opening(mypic_rev_log_gauss_grad_bin))
    bfh = sp.ndimage.binary_fill_holes(mypic_rev_gauss_grad_bin_close)
    bfh_rm = remove_small_objects(bfh, MIN_SEGMENT_SIZE)
    log_bfh = sp.ndimage.binary_fill_holes(mypic_rev_log_gauss_grad_bin_close)
    log_bfh_rm = remove_small_objects(log_bfh, MIN_SEGMENT_SIZE)

    plt.subplot(6, 2, 1)
    plt.imshow(mypic_rev, cmap=plt.cm.afmhot_r)
    plt.axis('off')
    plt.title('Spectrogram')
    plt.subplot(6, 2, 2)
    plt.imshow(mypic_rev_log, cmap=plt.cm.afmhot_r)
    plt.axis('off')
    plt.title('Spectrogram (log)')
    plt.subplot(6, 2, 3)
    plt.imshow(mypic_rev_gauss, cmap=plt.cm.afmhot_r)
    plt.axis('off')
    plt.title('+ Gaussian Filtering')
    plt.subplot(6, 2, 4)
    plt.imshow(mypic_rev_log_gauss, cmap=plt.cm.afmhot_r)
    plt.axis('off')
    plt.title('+ Gaussian Filtering (log)')
    plt.subplot(6, 2, 5)
    plt.imshow(mypic_rev_gauss_grad, cmap=plt.cm.afmhot_r)
    plt.axis('off')
    plt.title('+ Gradient')
    plt.subplot(6, 2, 6)
    plt.imshow(mypic_rev_log_gauss_grad, cmap=plt.cm.afmhot_r)
    plt.axis('off')
    plt.title('+ Gradient (log)')
    plt.subplot(6, 2, 7)
    plt.imshow(mypic_rev_gauss_grad_bin, cmap=plt.cm.gray)
    plt.axis('off')
    plt.title('+ >90%')
    plt.subplot(6, 2, 8)
    plt.imshow(mypic_rev_log_gauss_grad_bin, cmap=plt.cm.gray)
    plt.axis('off')
    plt.title('+ >90% (log)')
    plt.subplot(6, 2, 9)
    plt.imshow(mypic_rev_gauss_grad_bin_close, cmap=plt.cm.gray)
    plt.axis('off')
    plt.title('+ binary_closing + binary_opening')
    plt.subplot(6, 2, 10)
    plt.imshow(mypic_rev_log_gauss_grad_bin_close, cmap=plt.cm.gray)
    plt.axis('off')
    plt.title('+ binary_closing + binary_opening (log)')

    # SEGMENTS
    labeled_segments, num_seg = sp.ndimage.label(bfh_rm)

    plt.subplot(6, 2, 11)
    plt.imshow(labeled_segments)
    plt.axis('off')
    plt.title('+ binary_fill_holes + remove_small_objects')

    for current_segment_id in range(1, num_seg+1):
        current_segment = (labeled_segments == current_segment_id)*1
        xr = current_segment.max(axis=0)
        yr = current_segment.max(axis=1)
        xr_max = np.max(xr*np.arange(len(xr)))
        xr[xr == 0] = xr.shape[0]
        xr_min = np.argmin(xr)
        yr_max = np.max(yr*np.arange(len(yr)))
        yr[yr == 0] = yr.shape[0]
        yr_min = np.argmin(yr)
        segment_frame = [yr_min, yr_max, xr_min, xr_max]
        subpic = mypic_rev_gauss[yr_min:yr_max+1, xr_min:xr_max+1]
        SPEC_SEGMENTS.append(
            [index, current_segment_id, segment_frame, subpic])

    # LOG SEGMENTS
    labeled_segments, num_seg = sp.ndimage.label(log_bfh_rm)

    plt.subplot(6, 2, 12)
    plt.imshow(labeled_segments)
    plt.axis('off')
    plt.title('+ binary_fill_holes + remove_small_objects (log)')

    for current_segment_id in range(1, num_seg+1):
        current_segment = (labeled_segments == current_segment_id)*1
        xr = current_segment.max(axis=0)
        yr = current_segment.max(axis=1)
        xr_max = np.max(xr*np.arange(len(xr)))
        xr[xr == 0] = xr.shape[0]
        xr_min = np.argmin(xr)
        yr_max = np.max(yr*np.arange(len(yr)))
        yr[yr == 0] = yr.shape[0]
        yr_min = np.argmin(yr)
        segment_frame = [yr_min, yr_max, xr_min, xr_max]
        subpic = mypic_rev_log_gauss[yr_min:yr_max+1, xr_min:xr_max+1]
        LOG_SPEC_SEGMENTS.append(
            [index, current_segment_id, segment_frame, subpic])

    fig.savefig(spectro_folder + str(index) +'_patterns.png', dpi=300)
    fig.clear()


print(strftime("%a, %d %b %Y %H:%M:%S +0000", localtime()))

output = open(segment_folder + 'SPEC_SEGMENTS.pkl', 'wb')
pickle.dump(SPEC_SEGMENTS, output)
output.close()

output = open(segment_folder + 'LOG_SPEC_SEGMENTS.pkl', 'wb')
pickle.dump(LOG_SPEC_SEGMENTS, output)
output.close()

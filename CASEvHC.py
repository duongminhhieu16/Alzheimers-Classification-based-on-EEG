#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 16:30:41 2019

@author: tyleryoshihara 
"""

import warnings
warnings.filterwarnings("ignore")
import os
import matplotlib, scipy, sklearn, parfit, scikitplot, pandas, nilearn
from sklearn import preprocessing, model_selection, linear_model
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from nipype.interfaces import matlab
import matplotlib.pyplot as plt
import pandas
import numpy as np
from imblearn.over_sampling import SMOTE, ADASYN
from skrvm import RVC
import time
from scipy import stats

time0 = time.time()

CASEvHC = pandas.read_csv('./data/CASEvsHCFourier.csv')

CASEvHC = np.asarray(CASEvHC)

CASEvHC = CASEvHC[:,1:] #remove header

YCvH = CASEvHC[:,304] #get label

CASEvHC = CASEvHC[:,0:304] #get signal

print("Reading data complete in " + str(time.time()-time0) + " seconds")
print("There are " + str(CASEvHC.shape[0]) + " samples in the dataset")
time0 = time.time()

CASEvHC = preprocessing.scale(CASEvHC, axis=0) #standardize signal

print("Preprocessing data complete in " + str(time.time()-time0) + " seconds")

time0 = time.time()

freq_bands = scipy.signal.welch(CASEvHC, fs=256) #fft

print("Applying Fast Fourier Transform complete in " + str(time.time()-time0) + " seconds")

time0 = time.time()

# All models for CASEvHC
# Set up Train, Val, Test sets for all datasets
[train_inds, test_inds] = next(
        model_selection.ShuffleSplit(
                test_size=0.2,random_state=100).split(
                        CASEvHC,y=YCvH
                        )
        )

XTempTrain = CASEvHC[train_inds,]

# Split Train Set into Train and Validation Sets
[train2_inds, val_inds] = next(
        model_selection.ShuffleSplit(
                test_size=0.4,random_state=100).split(
                        XTempTrain,y=YCvH[train_inds]
                        )
        )

# Form the indices to select for each set
TrainInds = train_inds[train2_inds]
ValInds = train_inds[val_inds]
TestInds = test_inds


# Create sets of X and Y data using indices  for CASEvHC
    
XTrainCvH = CASEvHC[TrainInds,]
YTrainCvH = YCvH[TrainInds]
XValCvH = CASEvHC[ValInds,]
YValCvH = YCvH[ValInds]
XTestCvH = CASEvHC[TestInds,]
YTestCvH = YCvH[TestInds]

print("Splitting data complete in " + str(time.time()-time0) + " seconds")
# Running RVC (Relevance Vector Classifier)- CASEvHC

RXTrainCvH = CASEvHC[train_inds,]
RYTrainCvH = YCvH[train_inds]
RXTestCvH = CASEvHC[test_inds,]
RYTestCvH = YCvH[test_inds]

time0 = time.time()
print("Initializing RLC model...")
RVCMod = RVC(kernel = 'linear',
             verbose = False)
RVCMod.fit(RXTrainCvH,RYTrainCvH)
print("Training RVC model with " + str(RXTrainCvH.shape[0]) + " samples complete in " + str(time.time()-time0) + " seconds")
time0 = time.time()
def RVMFeatImp(RVs):
    NumRVs = RVs.shape[0]
    SumD = 0
    for RVNum in range(1,NumRVs):
        d1 = RVs[RVNum-1,]
        d2 = sum(np.ndarray.flatten(
                RVs[np.int8(
                        np.setdiff1d(np.linspace(0,NumRVs-1,NumRVs),RVNum))]))
        SumD = SumD + (d1/d2)
    SumD = SumD/NumRVs
    return SumD


RVs = RVCMod.relevance_
DVals = RVMFeatImp(RVs)

RVCPred1 = RVCMod.predict_proba(RXTestCvH)
RVCPred2 = RVCMod.predict(RXTestCvH)
accuracy = (RVCPred2 == RYTestCvH).mean() * 100.

print("Testing RVC model with " + str(RXTestCvH.shape[0]) + " samples complete in " + str(time.time()-time0) + " seconds")
print("RVC accuracy: " + str(accuracy))
time0 = time.time()

# Evaluate Performance (DON'T RELY ON ACCURACY!!!)
# Plot Receiver Operating Characteristic (ROC) Curve
scikitplot.metrics.plot_roc(RYTestCvH,RVCPred1, title = 'CASEvHC: RVC')
plt.savefig("./CASEvHC/RVC_ROC.jpg")
plt.close()
# Plot the Confusion Matrix for additional insight
scikitplot.metrics.plot_confusion_matrix(RYTestCvH,RVCPred2)
plt.savefig("./CASEvHC/RVC_confusionmatrix.jpg")
plt.close()
#%%
# Running RLR (Ridge Regularized Linear Regression) - CASEvHC

#Testing for multicollinearity 

coef1 = np.corrcoef(CASEvHC, rowvar = False)
plt.hist(coef1)
plt.savefig("./CASEvHC/histogram.jpg")
plt.close()
print("Initializing RLR model...")
ncores = 2
grid = {
    'C': np.linspace(1e-10,1e5,num = 100), #Inverse lambda
    'solver': ['liblinear'],
    'penalty': ['l1']
}
paramGrid = sklearn.model_selection.ParameterGrid(grid)
RLRMod = sklearn.linear_model.LogisticRegression(tol = 1e-10,
                                                random_state = 100,
                                                n_jobs = ncores,
                                                verbose = 0)
[bestModel,bestScore,allModels,allScores] = parfit.bestFit(RLRMod,
paramGrid = paramGrid,               
X_train = XTrainCvH,
y_train = YTrainCvH,
X_val = XValCvH,
y_val = YValCvH,
metric = sklearn.metrics.roc_auc_score,
n_jobs = ncores,
scoreLabel = 'AUC')
print("Training and validating RLR model with: " + str(XTrainCvH.shape[0]) + " training samples and " + str(XValCvH.shape[0]) + " validating samples complete in " + str(time.time()-time0) + " seconds")
time0 = time.time()

# Test on Test Set
RLRTestPred = bestModel.predict_proba(XTestCvH)
RLRTestPred2 = bestModel.predict(XTestCvH)
accuracy = (RLRTestPred2 == YTestCvH).mean() * 100.
print("Testing RLR model with " + str(XTestCvH.shape[0]) + " samples complete in " + str(time.time()-time0) + " seconds")
print("RLR accuracy: " + str(accuracy))
time0 = time.time()
# Plot Receiver Operating Characteristic (ROC) Curve
scikitplot.metrics.plot_roc(YTestCvH,RLRTestPred,title = 'CASEvHC: RLR')
plt.savefig("./CASEvHC/RLR_ROC.jpg")
plt.close()
# Plot the Confusion Matrix for additional insight
scikitplot.metrics.plot_confusion_matrix(YTestCvH,RLRTestPred2)
plt.savefig("./CASEvHC/RLR_confusionmatrix.jpg")
plt.close()

# %%
# RF (Random Forest) - CASEvHC
print("Initializing RF model...")
# Define the Hyperparameter grid values
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['sqrt']#['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)

# Code for the Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier 

# Declare the Classifier
RFC = RandomForestClassifier(criterion = 'entropy') #can use 'entropy' instead gini

# Raw Classifier

from sklearn.model_selection import RandomizedSearchCV
# n_iter is the number of randomized parameter combinations tried
RFC_RandomSearch = RandomizedSearchCV(estimator = RFC,
                                      param_distributions = random_grid,
                                      n_iter = 100, cv = 3, verbose=0,
                                      random_state=10, n_jobs = 2)
RFC_RandomSearch.fit(XTrainCvH,YTrainCvH)
print("Training and validating RF model with: " + str(XTrainCvH.shape[0]) + " training samples and " + str(XValCvH.shape[0]) + " validating samples complete in " + str(time.time()-time0) + " seconds")
# Look at the Tuned "Best" Parameters
time0 = time.time()
print("Fine-tuning RF model...")
RFC_RandomSearch.best_params_
RFC_RandomSearch.best_score_
RFC_RandomSearch.best_estimator_.feature_importances_

# Fit using the best parameters
# Look at the Feature Importance
FeatImp = RFC_RandomSearch.best_estimator_.feature_importances_
NZInds = np.nonzero(FeatImp)
num_NZInds = len(NZInds[0])
Keep_NZVals = [x for x in FeatImp[NZInds[0]] if 
               (abs(x) >= np.mean(FeatImp[NZInds[0]]) 
               + 4*np.std(FeatImp[NZInds[0]]))]
ThreshVal = np.mean(FeatImp[NZInds[0]]) + 2*np.std(FeatImp[NZInds[0]])
Keep_NZInds = np.nonzero(abs(FeatImp[NZInds[0]]) >= ThreshVal)
Final_NZInds = NZInds[0][Keep_NZInds]

FeatImp_RF_CvH_reshape = np.reshape(FeatImp,[19,16])
FeatImp_RF_mean = np.mean(FeatImp_RF_CvH_reshape, axis=0)
FeatImp_RF_std = np.std(FeatImp_RF_CvH_reshape, axis=0)
conf_int = stats.norm.interval(0.95, loc=FeatImp_RF_mean, scale=FeatImp_RF_std)

Freq_values = np.linspace(0,30,16)
plt.plot(Freq_values,FeatImp_RF_mean, 'o')
plt.title("RF Feature Importance by Channel")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Mean Feature Importance")
plt.savefig("./CASEvHC/RF_featureimportance.jpg")
plt.close()

time0 = time.time()

Pred1_S2 = RFC_RandomSearch.best_estimator_.predict(XTestCvH)
Pred2_S2 = RFC_RandomSearch.best_estimator_.predict_proba(XTestCvH)

scikitplot.metrics.plot_roc(YTestCvH,Pred2_S2, title = 'CASEvHC: RF')
plt.savefig("./CASEvHC/RF_ROC.jpg")
plt.close()
scikitplot.metrics.plot_confusion_matrix(YTestCvH,Pred1_S2)
plt.savefig("./CASEvHC/RF_confusionmatrix.jpg")
plt.close()
accuracy = (Pred1_S2 == YTestCvH).mean() * 100.
print("Testing RF model with " + str(XTestCvH.shape[0]) + " samples complete in " + str(time.time()-time0) + " seconds")
print("RF classification accuracy : %g%%" % accuracy)

# %%
# FDA - Fisher’s discriminant analysis for CASEvHC
time0 = time.time()
print("Resampling data for FDA...")
[XTrainFDCvH,YTrainFDCvH] = SMOTE(random_state = 100,k_neighbors = 2).fit_resample(XTrainCvH, YTrainCvH)
[XTestFDCvH,YTestFDCvH] = SMOTE(random_state = 100,k_neighbors = 2).fit_resample(XTestCvH, YTestCvH)

FDMod = LinearDiscriminantAnalysis(tol = 1e-4,solver = 'svd')
FDFit = FDMod.fit(XTrainFDCvH,YTrainFDCvH)
print("Training and validating FDA model with: " + str(XTrainFDCvH.shape[0]) + " training samples complete in " + str(time.time()-time0) + " seconds")
FIvec_FD_CvH = FDMod.coef_
time0 = time.time()

FDTestPredCvH = FDFit.predict_proba(XTestFDCvH)
FDTestPred2CvH = FDFit.predict(XTestFDCvH)
scikitplot.metrics.plot_roc(YTestFDCvH,FDTestPredCvH, title = 'CASEvHC: FDA')
plt.savefig("./CASEvHC/FDA_ROC.jpg")
plt.close()
scikitplot.metrics.plot_confusion_matrix(YTestFDCvH,FDTestPred2CvH)
plt.savefig("./CASEvHC/FDA_confusionmatrix.jpg")
plt.close()
accuracy = (FDTestPred2CvH == YTestFDCvH).mean() * 100.
print("Testing FDA model with " + str(XTestFDCvH.shape[0]) + " samples complete in " + str(time.time()-time0) + " seconds")
print("FDA classification accuracy : %g%%" % accuracy)
FIvec_FD_CvH = abs(FIvec_FD_CvH)
FIvec_FD_CvH =FIvec_FD_CvH.T
FIvec_FD_CvH = FIvec_FD_CvH[:,0]


FeatMatrix = np.stack([FeatImp, FIvec_FD_CvH], axis = 1)
FeatCorr = np.corrcoef(FeatMatrix.T)


FeatImp_FDA_CASEvHC_reshape = np.reshape(FIvec_FD_CvH,[19,16])
FeatImp_FDA_mean = np.mean(FeatImp_FDA_CASEvHC_reshape, axis=0)

plt.plot(Freq_values,FeatImp_FDA_mean, 'o')
plt.title("FD Feature Importance by Channel")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Mean Feature Importance")
plt.savefig("./CASEvHC/FDA_CASEvHC_featureimportance.jpg")
plt.close()
np.savetxt("./CASEvHC/FeatImp_FDA_CASEvHC.csv", FIvec_FD_CvH)
np.savetxt("./CASEvHC/FeatImp_RF_CASEvHC.csv", FeatImp)
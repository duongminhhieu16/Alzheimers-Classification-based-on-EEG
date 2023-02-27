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

ADvHC = pandas.read_csv('./data/ADvsHCFourier.csv')

ADvHC = np.asarray(ADvHC)

ADvHC = ADvHC[:,1:] #remove header

YAvH = ADvHC[:,304] #get label

ADvHC = ADvHC[:,0:304] #get signal

print("Reading data complete in " + str(time.time()-time0) + " seconds")
print("There are " + str(ADvHC.shape[0]) + " samples in the dataset")
time0 = time.time()

ADvHC = preprocessing.scale(ADvHC, axis=0) #standardize signal

print("Preprocessing data complete in " + str(time.time()-time0) + " seconds")

time0 = time.time()

freq_bands = scipy.signal.welch(ADvHC, fs=256) #fft

print("Applying Fast Fourier Transform complete in " + str(time.time()-time0) + " seconds")

time0 = time.time()

# All models for ADvHC
# Set up Train, Val, Test sets for all datasets
[train_inds, test_inds] = next(
        model_selection.ShuffleSplit(
                test_size=0.2,random_state=100).split(
                        ADvHC,y=YAvH
                        )
        )

XTempTrain = ADvHC[train_inds,]

# Split Train Set into Train and Validation Sets
[train2_inds, val_inds] = next(
        model_selection.ShuffleSplit(
                test_size=0.4,random_state=100).split(
                        XTempTrain,y=YAvH[train_inds]
                        )
        )

# Form the indices to select for each set
TrainInds = train_inds[train2_inds]
ValInds = train_inds[val_inds]
TestInds = test_inds


# Create sets of X and Y data using indices  for ADvHC
    
XTrainAvH = ADvHC[TrainInds,]
YTrainAvH = YAvH[TrainInds]
XValAvH = ADvHC[ValInds,]
YValAvH = YAvH[ValInds]
XTestAvH = ADvHC[TestInds,]
YTestAvH = YAvH[TestInds]

print("Splitting data complete in " + str(time.time()-time0) + " seconds")
# Running RVC (Relevance Vector Classifier)- ADvHC

RXTrainAvH = ADvHC[train_inds,]
RYTrainAvH = YAvH[train_inds]
RXTestAvH = ADvHC[test_inds,]
RYTestAvH = YAvH[test_inds]

time0 = time.time()
print("Initializing RLC model...")
RVCMod = RVC(kernel = 'linear',
             verbose = False)
RVCMod.fit(RXTrainAvH,RYTrainAvH)
print("Training RVC model with " + str(RXTrainAvH.shape[0]) + " samples complete in " + str(time.time()-time0) + " seconds")
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

RVCPred1 = RVCMod.predict_proba(RXTestAvH)
RVCPred2 = RVCMod.predict(RXTestAvH)
accuracy = (RVCPred2 == RYTestAvH).mean() * 100.

print("Testing RVC model with " + str(RXTestAvH.shape[0]) + " samples complete in " + str(time.time()-time0) + " seconds")
print("RVC accuracy: " + str(accuracy))
time0 = time.time()

# Evaluate Performance (DON'T RELY ON ACCURACY!!!)
# Plot Receiver Operating Characteristic (ROC) Curve
scikitplot.metrics.plot_roc(RYTestAvH,RVCPred1, title = 'ADvHC: RVC')
plt.savefig("./ADvHC/RVC_ROC.jpg")
plt.close()
# Plot the Confusion Matrix for additional insight
scikitplot.metrics.plot_confusion_matrix(RYTestAvH,RVCPred2)
plt.savefig("./ADvHC/RVC_confusionmatrix.jpg")
plt.close()
#%%
# Running RLR (Ridge Regularized Linear Regression) - ADvHC

#Testing for multicollinearity 

coef1 = np.corrcoef(ADvHC, rowvar = False)
plt.hist(coef1)
plt.savefig("./ADvHC/histogram.jpg")
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
X_train = XTrainAvH,
y_train = YTrainAvH,
X_val = XValAvH,
y_val = YValAvH,
metric = sklearn.metrics.roc_auc_score,
n_jobs = ncores,
scoreLabel = 'AUC')
print("Training and validating RLR model with: " + str(XTrainAvH.shape[0]) + " training samples and " + str(XValAvH.shape[0]) + " validating samples complete in " + str(time.time()-time0) + " seconds")
time0 = time.time()

# Test on Test Set
RLRTestPred = bestModel.predict_proba(XTestAvH)
RLRTestPred2 = bestModel.predict(XTestAvH)
accuracy = (RLRTestPred2 == YTestAvH).mean() * 100.
print("Testing RLR model with " + str(XTestAvH.shape[0]) + " samples complete in " + str(time.time()-time0) + " seconds")
print("RLR accuracy: " + str(accuracy))
time0 = time.time()
# Plot Receiver Operating Characteristic (ROC) Curve
scikitplot.metrics.plot_roc(YTestAvH,RLRTestPred,title = 'ADvHC: RLR')
plt.savefig("./ADvHC/RLR_ROC.jpg")
plt.close()
# Plot the Confusion Matrix for additional insight
scikitplot.metrics.plot_confusion_matrix(YTestAvH,RLRTestPred2)
plt.savefig("./ADvHC/RLR_confusionmatrix.jpg")
plt.close()

# %%
# RF (Random Forest) - ADvHC
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
RFC_RandomSearch.fit(XTrainAvH,YTrainAvH)
print("Training and validating RF model with: " + str(XTrainAvH.shape[0]) + " training samples and " + str(XValAvH.shape[0]) + " validating samples complete in " + str(time.time()-time0) + " seconds")
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

FeatImp_RF_AvH_reshape = np.reshape(FeatImp,[19,16])
FeatImp_RF_mean = np.mean(FeatImp_RF_AvH_reshape, axis=0)
FeatImp_RF_std = np.std(FeatImp_RF_AvH_reshape, axis=0)
conf_int = stats.norm.interval(0.95, loc=FeatImp_RF_mean, scale=FeatImp_RF_std)

Freq_values = np.linspace(0,30,16)
plt.plot(Freq_values,FeatImp_RF_mean, 'o')
plt.title("RF Feature Importance by Channel")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Mean Feature Importance")
plt.savefig("./ADvHC/RF_featureimportance.jpg")
plt.close()

time0 = time.time()

Pred1_S2 = RFC_RandomSearch.best_estimator_.predict(XTestAvH)
Pred2_S2 = RFC_RandomSearch.best_estimator_.predict_proba(XTestAvH)

scikitplot.metrics.plot_roc(YTestAvH,Pred2_S2, title = 'ADvHC: RF')
plt.savefig("./ADvHC/RF_ROC.jpg")
plt.close()
scikitplot.metrics.plot_confusion_matrix(YTestAvH,Pred1_S2)
plt.savefig("./ADvHC/RF_confusionmatrix.jpg")
plt.close()
accuracy = (Pred1_S2 == YTestAvH).mean() * 100.
print("Testing RF model with " + str(XTestAvH.shape[0]) + " samples complete in " + str(time.time()-time0) + " seconds")
print("RF classification accuracy : %g%%" % accuracy)

# %%
# FDA - Fisher’s discriminant analysis for ADvHC
time0 = time.time()
print("Resampling data for FDA...")
[XTrainFDAvH,YTrainFDAvH] = SMOTE(random_state = 100,k_neighbors = 3).fit_resample(XTrainAvH, YTrainAvH)
[XTestFDAvH,YTestFDAvH] = SMOTE(random_state = 100,k_neighbors = 3).fit_resample(XTestAvH, YTestAvH)

FDMod = LinearDiscriminantAnalysis(tol = 1e-4,solver = 'svd')
FDFit = FDMod.fit(XTrainFDAvH,YTrainFDAvH)
print("Training and validating FDA model with: " + str(XTrainFDAvH.shape[0]) + " training samples complete in " + str(time.time()-time0) + " seconds")
FIvec_FD_AvH = FDMod.coef_
time0 = time.time()

FDTestPredAvH = FDFit.predict_proba(XTestFDAvH)
FDTestPred2AvH = FDFit.predict(XTestFDAvH)
scikitplot.metrics.plot_roc(YTestFDAvH,FDTestPredAvH, title = 'ADvHC: FDA')
plt.savefig("./ADvHC/FDA_ROC.jpg")
plt.close()
scikitplot.metrics.plot_confusion_matrix(YTestFDAvH,FDTestPred2AvH)
plt.savefig("./ADvHC/FDA_confusionmatrix.jpg")
plt.close()
accuracy = (FDTestPred2AvH == YTestFDAvH).mean() * 100.
print("Testing FDA model with " + str(XTestFDAvH.shape[0]) + " samples complete in " + str(time.time()-time0) + " seconds")
print("FDA classification accuracy : %g%%" % accuracy)
FIvec_FD_AvH = abs(FIvec_FD_AvH)
FIvec_FD_AvH =FIvec_FD_AvH.T
FIvec_FD_AvH = FIvec_FD_AvH[:,0]


FeatMatrix = np.stack([FeatImp, FIvec_FD_AvH], axis = 1)
FeatCorr = np.corrcoef(FeatMatrix.T)


FeatImp_FDA_ADvHC_reshape = np.reshape(FIvec_FD_AvH,[19,16])
FeatImp_FDA_mean = np.mean(FeatImp_FDA_ADvHC_reshape, axis=0)

plt.plot(Freq_values,FeatImp_FDA_mean, 'o')
plt.title("FD Feature Importance by Channel")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Mean Feature Importance")
plt.savefig("./ADvHC/FDA_ADvHC_featureimportance.jpg")
plt.close()
np.savetxt("./ADvHC/FeatImp_FDA_ADvHC.csv", FIvec_FD_AvH)
np.savetxt("./ADvHC/FeatImp_RF_ADvHC.csv", FeatImp)
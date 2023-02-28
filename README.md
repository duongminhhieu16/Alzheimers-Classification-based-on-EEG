# Alzheimers-Classification-based-on-EEG

## Abstract
This code was based on the original code from tsyoshihara's github: https://github.com/tsyoshihara/Alzheimer-s-Classification-EEG. Electroencephalogram (EEG) is a non-invasive tool for dementia diagnosis, including dementia caused by Alzheimer's disease. The EEG data was sourced from 2014 paper titled "Alzheimer’s disease patients classification through EEG signals processing" by Fiscon et al. Different classifiers were used: Relevance Vector Classifier (RVC), Ridge Regularized Linear Regression (RLR), Random Forest (RF) and Fisher's Discriminant Analysis (FDA). 

## Dataset
There were 109 subjects including 37 Mild Cognitive Impairments (MCI), 49 Alzheimer's diseases (AD), and 23 Healthy Controls (HC). The EEG data was preprocessed using Fast Fourier Transform (FFT) with 16 coefficients.
The dataset include 4 csv files in folder "data". Each line of the csv file is an array of 304 number (16 coefficents * 19 channels).

## Results
The accuracy is not as high as in the paper. FFT can be replaced by Discrete Wavelet Transform (DWT) to achieve better result. In this code, there is only FFT data. You can follow the idea: Inverse Fourier Transform -> Raw data -> Discrete Wavelet Transform ->...
The rest of folders are the result from 4 problems:
- Mild Cognitive Impairment vs Healthy Control (MCI vs HC)
- Mild Cognitive Impairment vs Alzheimer’s disease (MCI vs AD)
- Alzheimer’s disease vs Healthy Control (AD vs HC)
- CASE vs Healthy Control (CASE vs HC)

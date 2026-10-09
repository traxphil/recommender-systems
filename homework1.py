from collections import defaultdict
from sklearn import linear_model
import numpy
import math

### - - - - - - - - - - - - - 
#   Regression
### - - - - - - - - - - - - - 

### Question 1

# Find the longest review (number of characters)
def getMaxLen(dataset):
    review_text_vector = [d['review_text'] for d in dataset]
    maxLen = max(len(t) for t in review_text_vector)
    return maxLen

# Feature vector for one data point
# datum = one entry of the dataset in json format
# called for each entry in the data set to determine the feature vector
def featureQ1(datum, maxLen):
    review_length = len(datum['review_text'])
    return [1, review_length/maxLen]

def Q1(dataset):
    maxLen = getMaxLen(dataset)

    # setting up the regression problem
    X = [featureQ1(d, maxLen) for d in dataset]
    Y = [d['rating'] for d in dataset]

    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
    # considering fit_intercept=false to remove bias calculation / already represented by 1 in feature vector
    model = linear_model.LinearRegression(fit_intercept=False);
    model.fit(X, Y)
    theta = model.coef_

    predictions = model.predict(X)
    MSE = numpy.mean((numpy.array(predictions) - numpy.array(Y)) ** 2)

    return theta, MSE

### Question 2

def featureQ2(datum, maxLen):
    # Implement (should be 1, length feature, day feature, month feature)
    # One-Hot Encoding
    date = datum['parsed_date']
    weekday = date.weekday()            # 0 = Monday / 6 = Sunday
    month = date.month                  # 1 = January / 12 = December
    length_feature = len(datum['review_text']) / maxLen

    # one-hot weekday (skip monday)
    weekday_feature = [0] * 6
    if (weekday > 0):
        weekday_feature[weekday - 1] = 1

    # one-hot month (skip January)
    month_feature = [0] * 11
    if (month > 1):
        month_feature[month - 2] = 1;

    return [1, length_feature] + weekday_feature + month_feature

def Q2(dataset):
    # Implement (note MSE should be a *number*, not e.g. an array of length 1)

    maxLen = getMaxLen(dataset)
    
    # setting up the regression problem
    X2 = [featureQ2(d, maxLen) for d in dataset]
    Y2 = [d['rating'] for d in dataset]

    model = linear_model.LinearRegression(fit_intercept=False);
    model.fit(X2, Y2)

    predictions = model.predict(X2)
    MSE2 = numpy.mean((numpy.array(predictions) - numpy.array(Y2)) ** 2)

    return X2, Y2, MSE2

### Question 3

def featureQ3(datum, maxLen):
    date = datum['parsed_date']
    weekday_feature = date.weekday()            # 0 = Monday / 6 = Sunday
    month_feature = date.month                  # 1 = January / 12 = December
    length_feature = len(datum['review_text']) / maxLen

    return [1, length_feature, weekday_feature, month_feature]

def Q3(dataset):

    maxLen = getMaxLen(dataset)
    
    # setting up the regression problem
    X3 = [featureQ3(d, maxLen) for d in dataset]
    Y3 = [d['rating'] for d in dataset]

    model = linear_model.LinearRegression(fit_intercept=False);
    model.fit(X3, Y3)

    predictions = model.predict(X3)
    MSE3 = numpy.mean((numpy.array(predictions) - numpy.array(Y3)) ** 2)

    return X3, Y3, MSE3

### Question 4

def Q4(dataset):

    dataset_length = len(dataset)
    training_set = dataset[:dataset_length // 2]
    test_set = dataset[dataset_length // 2:]

    maxLen = getMaxLen(training_set)    # only for the training set

    Y_training = [d['rating'] for d in training_set]
    Y_test = [d['rating'] for d in test_set]

    # - - - - -

    # Q2: one-hot encoding
    X_training2 = [featureQ2(d, maxLen) for d in training_set]
    X_test2 = [featureQ2(d, maxLen) for d in test_set]
    model2 = linear_model.LinearRegression(fit_intercept=False)
    
    model2.fit(X_training2, Y_training)
    predictions2 = model2.predict(X_test2)
    test_mse2 = numpy.mean((numpy.array(predictions2) - numpy.array(Y_test)) ** 2)

    # - - - - -

    #Q3: direct representation
    X_training3 = [featureQ3(d, maxLen) for d in training_set]
    X_test3 = [featureQ3(d, maxLen) for d in test_set]
    model3 = linear_model.LinearRegression(fit_intercept=False)

    model3.fit(X_training3, Y_training)
    predictions3 = model3.predict(X_test3)
    test_mse3 = numpy.mean((numpy.array(predictions3) - numpy.array(Y_test)) ** 2)

    return test_mse2, test_mse3

### - - - - - - - - - - - - - 
#   Classification
### - - - - - - - - - - - - - 

### Question 5

def featureQ5(datum):
    return [1, len(datum['review/text'])]

def Q5(dataset, feat_func):

    X = [feat_func(d) for d in dataset]
    # label (GT)
    Y = [d['review/overall'] >= 4 for d in dataset]

    # set up model
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    # use balanced clas weights because we have to assume our binary classes are unevenly distributed 
    model = linear_model.LogisticRegression(class_weight='balanced')
    model.fit(X, Y)
    predictions = model.predict(X)

    # Evaluation metrics
    Y = numpy.array(Y)
    predictions = numpy.array(predictions)

    TP = numpy.sum(predictions & Y)
    TN = numpy.sum(~predictions & ~Y)
    FP = numpy.sum(predictions & ~Y)
    FN = numpy.sum(~predictions & Y)

    # False Positive Rate / False Negative Rate
    FPR = FP / (FP + TN)
    FNR = FN / (FN + TP)
    BER = 0.5 * (FPR + FNR)

    # True Positives / True Negatives / False Positives / False Negatives / Balanced Error Rate
    return TP, TN, FP, FN, BER

### Question 6

# precision of classifier @K (for the K most confident classifications)
def Q6(dataset):

    X = [featureQ5(d) for d in dataset]
    Y = [d['review/overall'] >= 4 for d in dataset]

    # set up model
    model = linear_model.LogisticRegression(class_weight='balanced')
    model.fit(X, Y)

    scores = model.decision_function(X)
    Y = numpy.array(Y)

    # sort by confidence (most confident needs to be first)
    # argsort sorts ascending but we want it descenting (therefore -scores)
    ranking = numpy.argsort(-scores)
    Y_sorted = Y[ranking]

    precs = []
    for k in [1, 100, 1000, 10000]:
        precs.append(numpy.sum(Y_sorted[:k]) / k)
 
    return precs

### Question 7

# feels a bit like cheating but using the sub-reviews make the most sense here I'd guess
def featureQ7(datum):
    return [1, len(datum['review/text']), datum['review/aroma'], datum['review/appearance'], datum['review/palate'], datum['review/taste']]


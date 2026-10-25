from collections import defaultdict
from sklearn import linear_model
import numpy
import math

#### Personal Notes and Observations
# reducing the threshold for catID inclusion can increase accuracy (e.g. 200)

#### - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Helper Functions
#### - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Balanced Error Rate: mean of false positive rate and false negative rate
def compute_BER(predictions, labels):

    # make numpy boolean array for easier handling
    predictions = numpy.array(predictions, dtype=bool)      
    labels = numpy.array(labels, dtype=bool)                    

    TP = numpy.sum(predictions & labels)
    TN = numpy.sum(~predictions & ~labels)
    FP = numpy.sum(predictions & ~labels)
    FN = numpy.sum(~predictions & labels)

    # false positive rate
    FPR = FP / (FP + TN)
    # false negative rate
    FNR = FN / (FN + TP)
    # BER
    return 0.5 * (FPR + FNR)

# Training, Validation & Testing Pipeline (reg = regularization constant)
def pipeline(reg, catID, dataTrain, dataValid, dataTest, includeCat=True, includeReview=True, includeLength=True):
    mod = linear_model.LogisticRegression(C=reg, class_weight='balanced', max_iter=1000)

    maxLength = max([len(d['review/text']) for d in dataTrain])
    
    Xtrain = [get_features(d, catID, maxLength, includeCat, includeReview, includeLength) for d in dataTrain]
    Xvalid = [get_features(d, catID, maxLength, includeCat, includeReview, includeLength) for d in dataValid]
    Xtest = [get_features(d, catID, maxLength, includeCat, includeReview, includeLength) for d in dataTest]
    
    yTrain = [d['beer/ABV'] > 7 for d in dataTrain]
    yValid = [d['beer/ABV'] > 7 for d in dataValid]
    yTest = [d['beer/ABV'] > 7 for d in dataTest]
    
    # (1) Fit the model on the training set
    mod.fit(Xtrain, yTrain)
    # (2) Compute validation BER
    vBER = compute_BER(mod.predict(Xvalid), yValid)
    # (3) Compute test BER
    tBER = compute_BER(mod.predict(Xtest), yTest)

    return mod, vBER, tBER


#### - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Feature Definitions
#### - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Thoughts for possible features:
# - subscores (review):
#   - appearance (rating)
#   - palate (rating)
#   - taste (rating)
#   - review time (month, weekday, time of day)
#   - review overall (rating)
#   - review text (length, sentiment)
#   - review aroma (rating)
# - beer properties
#   - style (categorical one hot encoding)
#   - name (probably not)
#   - ABV
#   - beerID (idk)
#   - brewerID (possible)
# - user (possibly relevant for identifying taste of users)

# d = datum (entry)

# only a one-hot encoding for the categories
def get_features(d, catID, maxLength, includeCat=True, includeReview=True, includeLength=True):

    # review categories we consider
    REVIEW_KEYS = ['review/aroma', 'review/appearance', 'review/palate', 'review/taste', 'review/overall']

    # features will be added to the feature vector if selected
    feature_vector = []
    
    if (includeCat): # make one-hot feature vector for beer category
        one_hot_category_feat = [0] * len(catID)
        if d['beer/style'] in catID:
            one_hot_category_feat[catID[d['beer/style']]] = 1
        feature_vector += one_hot_category_feat

    if (includeReview): # include review scores in feature vector
        feature_vector += [d[key] for key in REVIEW_KEYS]

    if (includeLength):
        feature_vector += [len(d['review/text']) / maxLength]
    
    return feature_vector + [1]


#### - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Main Homework Functions: Q1, Q2, etc.
#### - - - - - - - - - - - - - - - - - - - - - - - - - - -

def Q1(catID, dataTrain, dataValid, dataTest):
    # No need to modify this if you've implemented the functions above
    mod, validBER, testBER = pipeline(10, catID, dataTrain, dataValid, dataTest, True, False, False)
    return mod, validBER, testBER


def Q2(catID, dataTrain, dataValid, dataTest):
    mod, validBER, testBER = pipeline(10, catID, dataTrain, dataValid, dataTest, True, True, True)
    return mod, validBER, testBER


def Q3(catID, dataTrain, dataValid, dataTest):
    best_mod, best_validBER, best_testBER = None, None, None
    for c in [0.001, 0.01, 0.1, 1, 10]:
        mod, validBER, testBER = pipeline(c, catID, dataTrain, dataValid, dataTest, True, True, True)
        # Model selection uses the validation set only
        if best_validBER is None or validBER < best_validBER:
            best_mod, best_validBER, best_testBER = mod, validBER, testBER
            print(f"Q3 - C = {c}: validBER = {validBER:.5f}  <- new best")
        else:
            print(f"Q3 - C = {c}: validBER = {validBER:.5f}")
    # Final choice; the fitted model stores its own C
    print(f"Q3 - selected C = {best_mod.C}, validBER = {best_validBER:.5f}, testBER = {best_testBER:.5f}")
    return best_mod, best_validBER, best_testBER


def Q4(C, catID, dataTrain, dataValid, dataTest):
    # Full model (nothing removed) - as baseline for the comparison
    _, _, testBER_full = pipeline(C, catID, dataTrain, dataValid, dataTest, True, True, True)
    print(f"Q4 - C = {C}, full model:  testBER = {testBER_full:.5f}")

    # Beer Categories Removed
    mod, validBER, testBER_noCat = pipeline(C, catID, dataTrain, dataValid, dataTest, False, True, True)
    print(f"Q4 - without categories:   testBER = {testBER_noCat:.5f}  ({testBER_noCat - testBER_full:+.5f})")

    mod, validBER, testBER_noReview = pipeline(C, catID, dataTrain, dataValid, dataTest, True, False, True)
    print(f"Q4 - without ratings:      testBER = {testBER_noReview:.5f}  ({testBER_noReview - testBER_full:+.5f})")

    mod, validBER, testBER_noLength = pipeline(C, catID, dataTrain, dataValid, dataTest, True, True, False)
    print(f"Q4 - without length:       testBER = {testBER_noLength:.5f}  ({testBER_noLength - testBER_full:+.5f})")

    return testBER_noCat, testBER_noReview, testBER_noLength
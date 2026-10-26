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

# Jaccard Similarity = |intersection| / |union| for two sets
def Jaccard(s1, s2):
    union_size = len(s1 | s2)
    # avoid division by zero - can happen if both sets are empty
    if union_size == 0:
        return 0
    return len(s1 & s2) / union_size

def MSE(y, ypred):
    differences = numpy.array(y) - numpy.array(ypred)
    return float(numpy.mean(differences ** 2))

# return mean rating across all of the training data
def getMeanRating(dataTrain):
    return sum(d['star_rating'] for d in dataTrain) / len(dataTrain)

# return the average rating each user has given across all training reviews
# dict user ID -> average rating
def getUserAverages(itemsPerUser, ratingDict):
    userAverages = {}
    for user, items in itemsPerUser.items():
        userAverages[user] = sum(ratingDict[(user, item)] for item in items) / len(items)
    return userAverages

# return the average item rating in the training set
# dict item ID -> average rating
def getItemAverages(usersPerItem, ratingDict):
    itemAverages = {}
    for item, users in usersPerItem.items():
        itemAverages[item] = sum(ratingDict[(user, item)] for user in users) / len(users)
    return itemAverages


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


# Q5: i = ID of reference (query) item / N = count of similar items to return /
# usersPerItem = mapping from item ID to user IDs who rated the product in the training data
def mostSimilar(i, N, usersPerItem):
    users_of_query = usersPerItem[i]
    similarities = []

    # compare query item with all ofhter items seen in training - add jaccard score to the similarity vector
    for other_item in usersPerItem:
        if other_item == i:
            continue
        similarities.append((Jaccard(users_of_query, usersPerItem[other_item]), other_item))

    # (similarity, itemID) tuples sorted by similarity first (highest similarity first)
    similarities.sort(reverse=True)
    return similarities[:N]

# Q6: using the similarity function for predictions
# The Formula from the Assignment decomposed:
# r(u,i) = the predicted rating / u = the user we predict for / i = the item for which we want to get the prediction
# R̄_i = average of training rating for item i
# R̄_j = average of training ratings over another item j (not i) what as also ranked by user u / one average for each other item
# R_uj = rating of user u for item j (remember we want to predict R_ui -> written r(u,i))
# I_u = all items user u has a rating for in the training set
# I_u\{i} = all items except i the user has a rating for -> basically the set of possible j
# Sim(i,j) = Jaccard similarity between two items
#
# Function Inputs:
# - user = str - customer_id for which we do the prediction
# - item = str - product_id for which we are trying to predict the rating
# - ratingMean = float - mean star rating across ALL training reviews
# - reviewsPerUser = dict(list) - [customer_id -> list of that users training reviews] 
#       each review is a dict with the columns: ('customer_id', 'product_id', 'product_title', 'star_rating', 'review_body', 'review_date', ...).
#       'star_rating', 'helpful_votes', 'total_votes' are int type / the rest is str 
# - usersPerItem = dict(set) - [product_id -> set of customer_ids]
# - itemsPerUser = dict(set) - [customer_id -> set or rated product_ids]
# - userAverages = dict [customer_id -> float]
# - itemAverages = dict [product_id -> float]
def predictRating(user, item, ratingMean, reviewsPerUser, usersPerItem, itemsPerUser, userAverages, itemAverages):
    # item never seen in training -> no R̄_i exists, use the global mean
    if item not in itemAverages:
        return ratingMean

    target_item_average = itemAverages[item]      # R̄_i
    target_item_users = usersPerItem[item]        # U_i

    # Numerator and denominator of the fraction, built up as running sums
    numerator = 0.0                               # sum (R_uj − R̄_j) · Sim(i,j)
    denominator = 0.0                             # sum Sim(i,j)

    # Loop over every other item j that this user rated in training
    for review in reviewsPerUser[user]:
        other_item = review['product_id']         # j
        if other_item == item:                    # exclude i 
            continue

        user_rating = review['star_rating']                                  # R_uj
        other_item_average = itemAverages[other_item]                        # R̄_j
        similarity = Jaccard(target_item_users, usersPerItem[other_item])    # Sim(i,j)

        # How much more/less u liked j than everyone else, weighted by how related j is to i
        deviation = user_rating - other_item_average
        numerator += deviation * similarity
        denominator += similarity

    # none of the user's items shares a customer with i -> just R̄_i
    if denominator == 0:
        return target_item_average

    return target_item_average + numerator / denominator

# Q7:
def predictRatingQ7(user, item, ratingMean, reviewsPerUser, usersPerItem, itemsPerUser, userAverages, itemAverages):
    reg_lambda = 5
    prediction = ratingMean
    # Regularized item offset b_i (eq. 3.33 with one parameter)
    if item in itemAverages:
        item_count = len(usersPerItem[item])
        prediction += item_count * (itemAverages[item] - ratingMean) / (item_count + reg_lambda)
    # Regularized user offset b_u, same derivation over the user's ratings
    if user in userAverages:
        user_count = len(itemsPerUser[user])
        prediction += user_count * (userAverages[user] - ratingMean) / (user_count + reg_lambda)
    return prediction
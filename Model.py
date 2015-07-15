import os
import numpy as np
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.cross_validation import train_test_split
import pickle
from sklearn.grid_search import GridSearchCV
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from scoring import *
from datareader import *

        

def peek_model(model):
    for i, weight in enumerate(model.estimator_weights_):
        print "Estimator %d weight=%0.02f"%( i + 1, weight)

    for i, f_weight in enumerate(model.feature_importances_):
        print "Feature %d weight=%0.02f"%( i + 1, f_weight)



def search_model_space(x, y):
    gini_scorer = metrics.make_scorer(normalized_gini, greater_is_better = True)

    parameters = { 'gbm__loss':['lad'],\
                   'gbm__n_estimators':[1500,2000],
                   'gbm__max_features':[1.0], \
                   'gbm__subsample':[1.0],  \
                   'gbm__max_depth':[4,6], 'gbm__learning_rate':[0.04,0.05,0.06], \
                   #'pca__n_components':[x.shape[1]]
                  }
    estimators = [('gbm',GradientBoostingRegressor())]
    clf = Pipeline(estimators)
    estimator = GridSearchCV(clf, \
                param_grid=parameters, \
                scoring=gini_scorer, verbose=10, n_jobs=2)

    estimator.fit(x, y)
    print estimator.grid_scores_
    return estimator

def build_model(x, y):
    """
    estimator = GradientBoostingRegressor(learning_rate=0.05,\
        n_estimators=1500, max_leaf_nodes=4, loss='lad')
    
    Training = 0.415, Dev = 0.363
    """
    estimator = GradientBoostingRegressor(learning_rate=0.05,\
        n_estimators=1500, max_leaf_nodes=4, loss='lad')
    
    
    """
    tree = DecisionTreeRegressor()
    estimator = AdaBoostRegressor(tree,n_estimators=100)
    """
    estimator.fit(x, y)
    return estimator

def apply_model(model, x_test, idx):
    predicted_y = model.predict(x_test)
    f = open(submission_file, 'wb')
    f.write("Id,Hazard\n")
    for i, p_y in enumerate(predicted_y):
        idv = str(idx[i])
        hazard = str(p_y)
        line = idv + "," + hazard + "\n"
        f.write(line)
    f.close()
        

if __name__ == "__main__":
    x, y, feature_names, vectorizer= get_data()
    x_t, x_d, y_t, y_d = train_test_split(x, y, test_size=0.2)

    #estimator = build_model(x_t,y_t)
    #poly = PolynomialFeatures(interaction_only=True)
    #x_t = poly.fit_transform(x_t)
    estimator = search_model_space(x_t, y_t)
    predicted_y = estimator.predict(x_t)
    print "Training gini = %0.3f"%(normalized_gini(y_t, predicted_y))
    print predicted_y[0:2], y_t[0:2]
    #peek_model(estimator)
    #x_d = poly.fit_transform(x_d)
    predicted_y = estimator.predict(x_d)
    print "dev gini = %0.3f"%(normalized_gini(y_d, predicted_y))
    print predicted_y[0:2], y_d[0:2]
    x_test, idx = get_test_data(vectorizer)
    #x_test = poly.fit_transform(x_test)
    print x_test[0:2]
    apply_model(estimator, x_test, idx)
    





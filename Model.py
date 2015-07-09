import os
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.cross_validation import train_test_split
import pickle
from sklearn.grid_search import GridSearchCV
from sklearn import metrics

train_data = os.getcwd() +   '\\data\\train.csv'
test_data  = os.getcwd() +   '\\data\\test.csv'
submission_file = os.getcwd() +   '\\data\\submit.csv'
model_path = os.getcwd() + "\\model\\"
model_file = "tree_stump_ada.pkl"



def gini(solution, submission):                                                 
    df = sorted(zip(solution, submission),    
            key=lambda x: x[1], reverse=True)
    random = [float(i+1)/float(len(df)) for i in range(len(df))]                
    totalPos = np.sum([x[0] for x in df])                                       
    cumPosFound = np.cumsum([x[0] for x in df])                                     
    Lorentz = [float(x)/totalPos for x in cumPosFound]                          
    Gini = [l - r for l, r in zip(Lorentz, random)]                             
    return np.sum(Gini)                                                         


def normalized_gini(solution, submission):                                      
    normalized_gini = gini(solution, submission)/gini(solution, solution)       
    return normalized_gini 

def convert_dict(entries):
    heading = entries[0]
    data_dict = []
    for i, entry in enumerate(entries[1:]):
        dict_entry = {}
        contents = entry
        for j, cell in enumerate(contents):
            if cell.isdigit():
                dict_entry[heading[j]] = float(cell)
            else:
                dict_entry[heading[j]] = cell
        data_dict.append(dict_entry)
    return data_dict
        

def peek_model(model):
    for i, weight in enumerate(model.estimator_weights_):
        print "Estimator %d weight=%0.02f"%( i + 1, weight)

    for i, f_weight in enumerate(model.feature_importances_):
        print "Feature %d weight=%0.02f"%( i + 1, f_weight)

def get_test_data(dict_vectorizer):
    ids = []
    entries = []
    with open(test_data) as f:
        line_no = 0
        for line in f:
            line_no+=1
            contents =line.strip().split(",")
            if line_no > 1:
                ids.append(contents[0])
            entries.append(contents[1:])
    entries_dict = convert_dict(entries)
    x_test = dict_vectorizer.transform(entries_dict)
    return x_test.toarray(), ids
            
    
def get_data():
    entries = []
    y = []
    line_no = 1
    with open(train_data) as f:
        for line in f:
            contents = line.strip().split(",")
            if line_no != 1:
                y.append(contents[1])
            line_no+=1
            entries.append(contents[2:])
    return get_np_vectors(convert_dict(entries), y)

    
def get_np_vectors(data_dict, y=None):
    return_y = None
    if y != None:
        return_y = np.asarray(y, dtype=float)
    dictvec = DictVectorizer()
    x = dictvec.fit_transform(data_dict)
    x = x.toarray()
    return x, return_y, dictvec.feature_names_, dictvec


def search_model_space(x, y):
    gini_scorer = metrics.make_scorer(normalized_gini, greater_is_better = True)

    parameters = { 'loss':['ls'],\
                   'n_estimators':[200, 300, 500],
                   'max_features':[0.8, 1.0], \
                   'subsample':[0.8],  \
                   'max_depth':[5], 'learning_rate':[0.05]
                  }
    estimator = GridSearchCV(GradientBoostingRegressor(warm_start=True), \
                param_grid=parameters, \
                scoring=gini_scorer, verbose=10, n_jobs=4)
    estimator.fit(x, y)
    print estimator.grid_scores_
    return estimator

def build_model(x, y):
    estimator = GradientBoostingRegressor(learning_rate=0.01,\
        n_estimators=500, max_leaf_nodes=2, loss='lad')
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
    estimator = search_model_space(x_t, y_t)
    predicted_y = estimator.predict(x_t)
    print "Training gini = %0.3f"%(normalized_gini(y_t, predicted_y))
    print predicted_y[0:2], y_t[0:2]
    #peek_model(estimator)
    predicted_y = estimator.predict(x_d)
    print "dev gini = %0.3f"%(normalized_gini(y_d, predicted_y))
    print predicted_y[0:2], y_d[0:2]
    x_test, idx = get_test_data(vectorizer)
    print x_test[0:2]
    apply_model(estimator, x_test, idx)
    





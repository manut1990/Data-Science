import pandas as pd
import numpy as np

###Load data files
def load_train_data():
    df_train = pd.read_csv('train.csv', encoding = "ISO-8859-1")
    return df_train.iloc[:1000]

def load_test_data():
    df_test = pd.read_csv('test.csv', encoding = "ISO-8859-1")
    return df_test
 
###Selection of labels & separating & data cleaning
def clean_data(df):
    if 'compliance' in df:
        df = df[['agency_name','violation_street_name','city','violation_code','disposition','judgment_amount','compliance']]
        df = df.dropna(subset=['compliance'])
    else:
        df = df[['agency_name','violation_street_name','city','violation_code','disposition','judgment_amount']]
    return df

###Prepare data for machine learning
def create_train_test_dataset(df):
    from sklearn.model_selection import train_test_split
    y_df = df.compliance.copy()
    del df['compliance']
    X_df = df.copy()
    X_df = pd.get_dummies(X_df)
    X_df['judgment_amount'] = df['judgment_amount']
    #X, y = X_df.values, y_df.values   
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.25, random_state=0)
    return X_train, X_test, y_train, y_test
    
###Dummy model
def Dummy_Model(X_train, X_test, y_train, y_test):
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import roc_curve, auc, recall_score, precision_score, f1_score
    dummy = DummyClassifier(strategy = 'stratified').fit(X_train, y_train)
    
    accuracy = dummy.score(X_test, y_test)
    y_pred = dummy.predict(X_test)    
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    y_score = None
    roc_auc = None
    return accuracy, recall, precision, f1, y_score, roc_auc

###Decision Tree
def decision_tree(X_train, X_test, y_train, y_test):
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import roc_curve, auc, recall_score, precision_score, f1_score, accuracy_score
    from sklearn.model_selection import GridSearchCV
    clf_tree = DecisionTreeClassifier()
    parameters = {'max_depth':[5,8]}
    clf_gd_tree = GridSearchCV(clf_tree, parameters, scoring='accuracy')
    clf_gd_tree.fit(X_train, y_train)
    
    y_pred = clf_gd_tree.predict(X_test)    
    accuracy = clf_gd_tree.score(X_test, y_test)
    recall, precision, f1, y_score = None, None, None, None
    
    return  accuracy, recall, precision, f1, y_score


###Random Forest
def random_forest(X_train, X_test, y_train, y_test):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    from sklearn.metrics import roc_curve, auc, recall_score, precision_score, f1_score, accuracy_score
    
    clf_random_forest = RandomForestClassifier(max_depth=3, random_state=0)
    clf_random_forest.fit(X_train, y_train)
    y_pred = clf_random_forest.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    recall, precision, f1, y_score = None, None, None, None
    
    return  accuracy, recall, precision, f1, y_score
    
    
###Logistic Regression
def Log_Classifier(X_train, X_test, y_train, y_test):
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.metrics import roc_curve, auc, recall_score, precision_score, f1_score, accuracy_score
    clf_log = LogisticRegressionCV(n_jobs=4)
    clf_log.fit(X_train, y_train)
    y_pred = clf_log.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    y_score  = clf_log.decision_function(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    auc = auc(fpr, tpr)
    
    return  accuracy, recall, precision, f1, auc, y_score
  
    
###Naive Bayes
def GNB_Classifier(X_train, X_test, y_train, y_test):
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import roc_curve, auc, recall_score, precision_score, f1_score, accuracy_score
    from sklearn.model_selection import GridSearchCV
    clf_nb = GaussianNB()
    clf_nb.fit(X_train, y_train)
    
    y_pred = clf_nb.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    y_score  = None
    auc = None
    
    return  accuracy, recall, precision, f1, auc, y_score

###Calculating all models and returning result overview of models 
def going_throuth_models():
    Decision_Tree_results = list(decision_tree(X_train, X_test, y_train, y_test))
    Random_Forest_results = list(random_forest(X_train, X_test, y_train, y_test))
    Dummy_Model_results = list(Dummy_Model(X_train, X_test, y_train, y_test))
    GNB_Classifier_results = np.array(GNB_Classifier(X_train, X_test, y_train, y_test))
    Log_Classifier_results = list(Log_Classifier(X_train, X_test, y_train, y_test))
    
    return Dummy_Model_results, Log_Classifier_results, Decision_Tree_results, Random_Forest_results, GNB_Classifier_results

###Preparing models
def compare_models():
    #compare auc
    auc_scores = pd.Series([Decision_Tree_results[4], Random_Forest_results[4],GNB_Classifier_results[4],Log_Classifier_results[4]],
                                index = ['Decision Tree', ' Random Forest', 'Naive Bayes', 'Logistic Regression'])    
    
    #compare accuracy
    accuracy_scores = pd.Series([Decision_Tree_results[0], Random_Forest_results[0],GNB_Classifier_results[0],Log_Classifier_results[0]],
                                index = ['Decision Tree', ' Random Forest', 'Naive Bayes', 'Logistic Regression'])
    
    #compare f1
    f1_scores = pd.Series([Decision_Tree_results[3], Random_Forest_results[3],GNB_Classifier_results[3],Log_Classifier_results[3]],
                                index = ['Decision Tree', ' Random Forest', 'Naive Bayes', 'Logistic Regression'])    
    
    #compare recall
    recall_scores = pd.Series([Decision_Tree_results[1], Random_Forest_results[1],GNB_Classifier_results[1],Log_Classifier_results[1]],
                                index = ['Decision Tree', ' Random Forest', 'Naive Bayes', 'Logistic Regression'])    
    
    #compare precision
    precision_scores = pd.Series([Decision_Tree_results[2], Random_Forest_results[2],GNB_Classifier_results[2],Log_Classifier_results[2]],
                                index = ['Decision Tree', ' Random Forest', 'Naive Bayes', 'Logistic Regression'])    
    
    
    df_overview = pd.DataFrame([auc_scores, accuracy_scores, f1_scores, recall_scores, precision_scores], index = ['Auc Scores', 'Accuracy', 'F1 Score', 'Recall', 'Precision'])
    ##plot graphics
    plot_precision_recall_curve(y_test, Log_Classifier_results[5], 'Log Classifier')
    plot_roc_curve(y_test, Log_Classifier_results[5], 'Log Classifier')




Dummy_Model_results, Log_Classifier_results, Decision_Tree_results, Random_Forest_results, GNB_Classifier_results = going_throuth_models()
compare_models()

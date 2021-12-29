import numpy as np 
import pandas as pd
import json 
import pickle
from sklearn.pipeline import Pipeline
from pickle import load,dump
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier

from sklearn.metrics import SCORERS
from sklearn.metrics import plot_confusion_matrix,plot_roc_curve,classification_report,accuracy_score,confusion_matrix,f1_score,precision_score,recall_score



df = pd.read_csv('Converted_data_set')
X = df.drop('target',axis=1)
y = df['target']


def trainTestValSplit(X, y):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)
    X_test, X_validate, y_test, y_validate = train_test_split(X_test, y_test, test_size=0.50, random_state=42)
    
    X_train.to_csv("X_train.csv")
    X_test.to_csv("X_test.csv")
    
    scaler = StandardScaler()
    
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    with open('Model_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    return X_train, X_test, y_train, y_test, X_validate, y_validate



def storeFiles(X_train, X_test, y_train, y_test, X_validate, y_validate):
    try:
        X_validate.to_csv("X_validate.csv")
        y_train.to_csv("y_train.csv")
        y_test.to_csv("y_test.csv")
        y_validate.to_csv("y_validate.csv")
        return 'Success'

    except Exception as e:
        return e.__str__()

    
  
class ModelTrain_Classification:
    def __init__(self, X_train, X_test, y_train, y_test, X_val, y_val, start: bool):
        try:
            self.frame = pd.DataFrame(columns=['Model_Name', 'Accuracy', 'Classes', 'Precision', 'Recall', 'F1_Score'])
            self.X_train = X_train
            self.X_test = X_test
            self.y_train = y_train
            self.y_test = y_test
            self.X_val = X_val
            self.y_val = y_val

            if start:
                self.LogisticRegression_()
                self.SVC_()
                self.KNeighborsClassifier_()
                self.DecisionTreeClassifier_()
                self.RandomForestClassifier_()
                self.GradientBoostingClassifier_()
                self.AdaBoostClassifier_()

        except Exception as e:
            raise Exception(e)
        
    def LogisticRegression_(self):
        model = LogisticRegression()
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average=None)
        recall = recall_score(self.y_test, y_pred, average=None)
        f1_score_ = f1_score(self.y_test, y_pred, average=None)
        self.frame = self.frame.append(
            {'Model_Name': 'LogisticRegression', 'Accuracy': accuracy, 'Classes': self.y_test.unique(),
             'Precision': precision, 'Recall': recall,
             'F1_Score': f1_score_}, ignore_index=True)

    def SVC_(self):
        model = SVC()
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average=None)
        recall = recall_score(self.y_test, y_pred, average=None)
        f1_score_ = f1_score(self.y_test, y_pred, average=None)
        self.frame = self.frame.append(
            {'Model_Name': 'SVC', 'Accuracy': accuracy, 'Classes': self.y_test.unique(), 'Precision': precision,
             'Recall': recall,
             'F1_Score': f1_score_}, ignore_index=True)

    def KNeighborsClassifier_(self):
        model = KNeighborsClassifier()
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average=None)
        recall = recall_score(self.y_test, y_pred, average=None)
        f1_score_ = f1_score(self.y_test, y_pred, average=None)
        self.frame = self.frame.append(
            {'Model_Name': 'KNeighborsClassifier', 'Accuracy': accuracy, 'Classes': self.y_test.unique(),
             'Precision': precision, 'Recall': recall,
             'F1_Score': f1_score_}, ignore_index=True)

    def DecisionTreeClassifier_(self):
        model = DecisionTreeClassifier()
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average=None)
        recall = recall_score(self.y_test, y_pred, average=None)
        f1_score_ = f1_score(self.y_test, y_pred, average=None)
        self.frame = self.frame.append(
            {'Model_Name': 'DecisionTreeClassifier', 'Accuracy': accuracy, 'Classes': self.y_test.unique(),
             'Precision': precision, 'Recall': recall,
             'F1_Score': f1_score_}, ignore_index=True)

    def RandomForestClassifier_(self):
        model = RandomForestClassifier()
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average=None)
        recall = recall_score(self.y_test, y_pred, average=None)
        f1_score_ = f1_score(self.y_test, y_pred, average=None)
        self.frame = self.frame.append(
            {'Model_Name': 'RandomForestClassifier', 'Accuracy': accuracy, 'Classes': self.y_test.unique(),
             'Precision': precision, 'Recall': recall,
             'F1_Score': f1_score_}, ignore_index=True)

    def GradientBoostingClassifier_(self):
        model = GradientBoostingClassifier()
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average=None)
        recall = recall_score(self.y_test, y_pred, average=None)
        f1_score_ = f1_score(self.y_test, y_pred, average=None)
        self.frame = self.frame.append(
            {'Model_Name': 'GradientBoostingClassifier', 'Accuracy': accuracy, 'Classes': self.y_test.unique(),
             'Precision': precision, 'Recall': recall,
             'F1_Score': f1_score_}, ignore_index=True)

    def AdaBoostClassifier_(self):
        model = AdaBoostClassifier()
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average=None)
        recall = recall_score(self.y_test, y_pred, average=None)
        f1_score_ = f1_score(self.y_test, y_pred, average=None)
        self.frame = self.frame.append(
            {'Model_Name': 'AdaBoostClassifier', 'Accuracy': accuracy, 'Classes': self.y_test.unique(),
             'Precision': precision, 'Recall': recall,
             'F1_Score': f1_score_}, ignore_index=True)
        
    def createPipeLine(self,model,scaler):
        pipe = Pipeline([('scaler', scaler), ('model', model)])
        pipe.fit(self.X_train, self.y_train)
        return pipe
        
    def validation(self,pipe):
        self.results = {}

        y_pred = pipe.predict(self.X_val)

        self.results['Confusion_Matrix'] = str(confusion_matrix(self.y_val,y_pred))
        self.results['Precision'] = str(precision_score(self.y_val,y_pred))
        self.results['Recall'] = str(recall_score(self.y_val,y_pred))
        self.results['F1_score'] = str(f1_score(self.y_val,y_pred))
        
        self.saveJson()
        
        return self.results
    
    def saveJson(self):
        json_data = json.dumps(self.results)
        with open('results.json', 'w') as f:
            f.write(json_data)
        
    def results(self):
        return self.frame
    
def main():
    model_name = {'LogisticRegression':LogisticRegression(),
             'SVC':SVC(),
             'KNeighborsClassifier':KNeighborsClassifier(),
             'DecisionTreeClassifier':DecisionTreeClassifier(),
             'AdaBoostClassifier':AdaBoostClassifier(),
             'GradientBoostClassifier':GradientBoostingClassifier(),
             'RandomForestClassifier':RandomForestClassifier()
            }
    X_train, X_test, y_train, y_test, X_validate, y_validate = trainTestValSplit(X, y)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape, X_validate.shape, y_validate.shape)
    
    msg = storeFiles(X_train, X_test, y_train, y_test, X_validate, y_validate)
    
    print(msg)
    
    train = ModelTrain_Classification(X_train, X_test, y_train, y_test, X_validate, y_validate,start=True)
    result = train.results()
    
    model = model_name[result.iloc[result['Accuracy'].argmax()].Model_Name]
    with open('Model_scaler.pkl', 'rb') as handle:
        scaler = pickle.load(handle)    
    pipe = train.createPipeLine(model,scaler)
    
    print(train.validation(pipe))
    

    
    
result = main()
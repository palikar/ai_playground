
#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import pickle

from ai_streaming.model_streaming.ml_stremer import MLStreamer
from ai_streaming.model_streaming.decorators import with_model_builder
import ai_streaming.model_streaming.utils as ut

from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
import joblib

class TestModel(MLStreamer):


    def __init__(self):
         MLStreamer.__init__(self, "scikit_trees")
         self.score = None
        

    def arg_setup(self, argumentar):
        argumentar.add_split_data()
        argumentar.add_custom_files(['alive'])

    def file_loader_setup(self, loader, config):
        loader['alive'] = lambda x: pd.read_csv(x)
        

    @with_model_builder('scikit_tree_classifier', config='tree')
    def model_setup(self, config, tree):
        print('Decision Tree built!')
        return tree

    
    def model_load(self, files):
        print("Not supported!")
        exit(1)
        


    def load_data(self, config, files):
        train, validate, test = files

        print(f'Loading train data from {train}')
        print(f'Loading test data from {test}')


        encoder = LabelEncoder()
        
        train_df = pd.read_csv(train)
        train_df = train_df[config['columns']]
        
        train_df = ut.fill_nan(train_df, 'Age', 'mean')
        train_df.dropna(inplace=True)

        #encoding lables
        train_df = pd.get_dummies(train_df, columns=['Sex'], drop_first=True)
        train_df['Embarked'] = train_df[['Embarked']].apply(encoder.fit_transform)
        

        test_df = pd.read_csv(test)
        test_df = test_df[config['test_columns']]

        test_df = ut.fill_nan(test_df, 'Age', 'mean')
        test_df.fillna(method='ffill', inplace=True)

        #encoding lables
        test_df = pd.get_dummies(test_df, columns=['Sex'], drop_first=True)
        test_df['Embarked'] = test_df[['Embarked']].apply(encoder.transform)

        prediction = config['predict']
        X_train = train_df.loc[:, train_df.columns != prediction].values
        y_train = train_df.loc[:, train_df.columns == prediction].values
        X_test = test_df.loc[:, test_df.columns != 'PassengerId'].values
        index_test = test_df.loc[:, test_df.columns == 'PassengerId'].values

        
        print('X_train shape', X_train.shape)
        print('y_train shpe:', y_train.shape)
        print('X_test shpe:', X_test.shape)

        
        # exit(1)
        return (X_train, y_train, X_test, index_test)

        
            
    def pipeline(self, config, model):
        print('Runnung inner pipeline')
        return dict()

    
    def train_model(self, config, data, model, pipeline):
        X, y, _, _ = data
        
        print('Fitting the tree')
        model.fit(X, y)
                        
    
    def eval_model(self, config, data, model, pipeline):
        X, y, X_test, index_test = data

        self.score = model.score(X, y)
        print("Score on train set:", self.score)

        test_result = model.predict(X_test)

        res_df = pd.DataFrame({"Passengerid": index_test.ravel(), "Survived":test_result})

        self.get_dumper().dump_df(res_df, "ecaluation_results")
        
        

        
    def save_model(self, config, model, output):

        joblib.dump(model, os.path.join(output, "model_weights.joblib"))
        
        pass
        
        


def main():
    TestModel().run()

if __name__ == '__main__':
    main()

#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.model_selection as ms
import sklearn.metrics as mt
import sklearn.preprocessing as pr

from sklearn.model_selection import KFold
from sklearn.mixture import GaussianMixture

from ai_streaming.model_streaming.ml_stremer import MLStreamer
from ai_streaming.model_streaming.decorators import with_model_builder
from ai_streaming.model_streaming.decorators import with_model_loader
import ai_streaming.model_streaming.utils as ut

class TestModel(MLStreamer):


    def __init__(self):
        MLStreamer.__init__(self, "clusterer")
        self.actions = dict()
        self.actions['1']='Bufonidae'
        self.actions['2']='Dendrobatidae'
        self.actions['3']='Hylidae'
        self.actions['4']='Leptodactylidae'

     
    def get_action(self, act):
        if act not in self.actions.keys():
            print(f'There is no action with key {act}')
            exit(1)
        return self.actions[act]            
        

    def arg_setup(self, argumentar):
        # argumentar.add_custom_files(['custom'])
        argumentar.add_common_data()
        
        pass

    def file_loader_setup(self, loader, config):
        # loader['custom'] = lambda x: pd.read_csv(x)
        pass
        
    def model_setup(self, config):
        print('Building model!')

        count = config['counts'][config['predict']]
        
        model = [None]*count
        for i in range(0,count):
            model[i] = GaussianMixture(n_components=4,
                                covariance_type='full',
                                tol=0.001,
                                reg_covar=1e-06,
                                max_iter=100,
                                n_init=1,
                                init_params='kmeans',
                                weights_init=None,
                                means_init=None,
                                precisions_init=None,
                                random_state=config['radnom_state'],
                                warm_start=False,
                                verbose=0,
                                verbose_interval=10)
        return model

    
    def model_load(self, files):
        print("Not supported!")
        exit(1)
        

    def load_data(self, config, files):
        print("Loading data!")

        data_df = pd.read_csv(files, index_col='RecordID')        
        cluster_df = data_df[config['cluster']]

        X = cluster_df.values
        y = data_df[config['predict']].values
        
        self.encoder = pr.LabelEncoder()
        y = self.encoder.fit_transform(y)

        return X,y


    def pipeline(self, config, model):
        print('Runnung inner pipeline')
        return dict()

    
    def train_model(self, config, data, model, pipeline):
        print('Fitting the mixture')

        X,y = data

        folds_number = 20
        folder = KFold(n_splits=folds_number)
        acc = 0
        fold = 0

        for train,test in folder.split(X,y):
            X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
            for index, i  in enumerate(np.unique(y_train)):
                X_action = X_train[y_train==i]
                model[index].fit(X_action)
                
            res = np.array([model[i].score_samples(X_test) for i in range(0,len(model))])
            res = res.argmax(axis=0)
            acc_ = mt.accuracy_score(y_test,res)
            acc += acc_
            print(f'Accuracy for fold {fold+1} : {acc_}')
            fold += 1
        acc /= folds_number

        print(f'Final accuracy: {acc}')
            

    def eval_model(self, config, data, model, pipeline):
        pass
    


    def save_model(self, config, model, output):       
        pass
        
        


def main():
    TestModel().run()

if __name__ == '__main__':
    main()

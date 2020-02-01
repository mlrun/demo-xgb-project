import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import xgboost as xgb
import os
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score
from mlrun.artifacts import TableArtifact, PlotArtifact
import pandas as pd


def iris_generator(context, target=''):
    iris = load_iris()
    iris_dataset = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    iris_labels = pd.DataFrame(data=iris.target, columns=['label'])
    iris_dataset = pd.concat([iris_dataset, iris_labels], axis=1)
    context.logger.info('saving iris dataframe to {}'.format(target))
    context.log_artifact(TableArtifact('iris_dataset', df=iris_dataset, target_path=target))
    

def xgb_train(context, 
              dataset='',
              model_name='model.bst',
              max_depth=6,
              num_class=10,
              eta=0.2,
              gamma=0.1,
              steps=20):

    df = pd.read_csv(dataset)
    X = df.drop(['label'], axis=1)
    y = df['label']
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)
    dtrain = xgb.DMatrix(X_train, label=Y_train)
    dtest = xgb.DMatrix(X_test, label=Y_test)

    param = {"max_depth": max_depth,
             "eta": eta, "nthread": 4,
             "num_class": num_class,
             "gamma": gamma,
             "objective": "multi:softprob"}

    xgb_model = xgb.train(param, dtrain, steps)

    preds = xgb_model.predict(dtest)
    best_preds = np.asarray([np.argmax(line) for line in preds])

    context.log_result('accuracy', float(accuracy_score(Y_test, best_preds)))
    context.log_artifact('model', body=bytes(xgb_model.save_raw()), 
                         target_path=model_name, labels={'framework': 'xgboost'})
    
    
import matplotlib
import matplotlib.pyplot as plt
from io import BytesIO

def plot_iter(context, iterations, col='accuracy', num_bins=10):
    df = pd.read_csv(BytesIO(iterations.get()))
    x = df['output.{}'.format(col)]
    fig, ax = plt.subplots(figsize=(6,6))
    n, bins, patches = ax.hist(x, num_bins, density=1)
    ax.set_xlabel('Accuraccy')
    ax.set_ylabel('Count')
    context.log_artifact(PlotArtifact('myfig', body=fig))

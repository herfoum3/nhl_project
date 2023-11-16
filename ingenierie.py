#import sys
#sys.path.append("lib/python3.10/site-packages/")

from comet_ml import Experiment
from comet_ml.integration.sklearn import log_model

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter

from loader_parser import organiser, parse_data_v2
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc

import warnings
warnings.simplefilter("ignore")
import os
import json

import xgboost as xgb



net_x = {'left': 89,
         'right': -89}
"""
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()
"""
# ON A REMPLACE PAR DES 1 ET 0
def encode_column(df, column_name):

  df = df.copy()

  # POUR LINSTANT ON A JUST FAIT EN SORTE QUE DF_TRAIN OG NAIS PAS DE NONE
  for i in range(df.shape[0]):
    val = df.loc[i,column_name]
    if val == None:
       df.loc[i,column_name] = False

  le = LabelEncoder()
  df[column_name] = le.fit_transform(df[column_name])

  return df

def distance(row, last=False):
    rink_side = row['RINK_SIDE']
    if rink_side == 'SHOOT_OUT':
        return 0
    x = row['COORD_X'] if not last else row['LAST_COORD_X']
    y = row['COORD_Y'] if not last else row['LAST_COORD_Y']
    nx = net_x[rink_side]
    return np.sqrt((x - nx)**2 + y**2)

def angle(row, last=False):
    """
    return angle in degrees
    """
    rink_side = row['RINK_SIDE']
    if rink_side == 'SHOOT_OUT':
        return 0
    y = row['COORD_Y'] if not last else row['LAST_COORD_Y']
    return np.arcsin(y/distance(row,last))*180/np.pi

def split_data(df, label, test_size: float = 0.2, random_state: int = 42):
    X = df.drop(columns=[label])
    y = df[label]
    X_train, X_val, y_train, y_val = train_test_split( X, y, test_size=test_size, random_state=random_state )

    return X_train, X_val, y_train, y_val

def percent_formatter(x,pos):
    return f"{x*100:.0f}%"

def trace_courbe_taux_buts(y_val, goal_probabilities, avec_carac, bp):
    df = pd.DataFrame([])
    df['IS_GOAL'] = y_val
    df['PROBABILITIES'] = goal_probabilities
    df['PERCENTILE'] = df['PROBABILITIES'].rank(pct=True)
    df_sorted = df.sort_values('PERCENTILE')
    cumulative_goals = df_sorted['IS_GOAL'].cumsum()
    goals_shots = np.arange(1, df.shape[0] + 1)
    goal_rate = cumulative_goals / goals_shots

    plt.figure(figsize=(10, 6))

    plt.plot(df_sorted['PERCENTILE'], goal_rate)

    plt.xlabel('Taux de buts')
    plt.ylabel('Buts/(Buts+Tirs)')
    plt.title('Centile de probabilité de tir: ' + avec_carac )
    plt.gca().xaxis.set_major_formatter(FuncFormatter(percent_formatter))
    plt.gca().yaxis.set_major_formatter(FuncFormatter(percent_formatter))
    plt.xlim([1, 0])
    plt.xticks(np.arange(0, 1.01, step=0.1))
    plt.grid()
    plt.savefig('/home/mchelfi/Desktop/PROJET_NHL/blog_website'+bp, dpi=150)
    plt.show()

def tracer_courbe_proportion(y_val, goal_probabilities, avec_carac,bp):
    df = pd.DataFrame([])
    df['IS_GOAL'] = y_val
    df['PROBABILITIES'] = goal_probabilities
    # df = df[df['IS_GOAL'] == 1]
    df['PERCENTILE'] = df['PROBABILITIES'].rank(pct=True)
    df_sorted = df.sort_values('PERCENTILE')
    cumulative_goals = df_sorted['IS_GOAL'].cumsum()
    nb_goals = y_val.sum()
    goal_rate = 1 - cumulative_goals / nb_goals  # tel qu'indique dans les Q&A sur piazza

    plt.figure(figsize=(10, 6))

    plt.plot(df_sorted['PERCENTILE'], goal_rate)

    plt.xlabel('Buts cumulés % ')
    plt.ylabel('Proportion')
    plt.title('Centile de probabilité de tir: ' + avec_carac)
    plt.gca().xaxis.set_major_formatter(FuncFormatter(percent_formatter))
    plt.gca().yaxis.set_major_formatter(FuncFormatter(percent_formatter))
    plt.xlim([1, 0])
    plt.xticks(np.arange(0, 1.01, step=0.1))
    plt.grid()
    plt.savefig('/home/mchelfi/Desktop/PROJET_NHL/blog_website' + bp, dpi=150)
    plt.show()
def tracer_courbe_roc(y_val, goal_probabilities, avec_carac,bp):
    fpr, tpr, _ = roc_curve(y_val, goal_probabilities)
    roc_auc = auc(fpr, tpr)

    #on prepare la reference a un classifieur aleatoire a 50% de chance d'etre un but
    n_samples = y_val.shape[0]
    random_fpr = np.linspace(0, 1, num=n_samples)
    random_tpr = np.linspace(0, 1, num=n_samples)

    # Plotting the ROC curves
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label='courbes ROC du modele (AUC = %0.2f)' % roc_auc)
    plt.plot(random_fpr, random_tpr, color='grey', lw=2, linestyle='--', label='Classifieur aleatoire')

    # Adding labels and title
    plt.xlabel('Taux de Fausses Positives')
    plt.ylabel('Taux de Vraies Positives')
    plt.title('Courbe Receiver Operating Characteristic (ROC): ' + avec_carac)
    plt.legend(loc="lower right")
    plt.savefig('/home/mchelfi/Desktop/PROJET_NHL/blog_website' + bp, dpi=150)
    plt.show()
def last_angle(row):
    x = row['LAST_COORD_Y']
    y = row['LAST_COORD_Y']
    return np.arcsin(y/distance(row,True)*180/np.pi)
def toseconds(priod_time):
    minutes, seconds = map(int, priod_time.split(':'))
    return minutes * 60 + seconds
def partie4(data_path,all=False,experiment=None):

    if os.path.isdir(data_path):
        df_train = organiser(data_path, 2016, 2019)
    else:
        with open(data_path, "r") as file:
            game_json = json.load(file)

        plays = parse_data_v2(game_json)
        df_train = pd.DataFrame(plays)

    df_train = encode_column(df_train, 'IS_EMPTY_NET')
    df_train = encode_column(df_train, 'IS_GOAL')

    df_train['DISTANCE'] = df_train.apply(lambda row: distance(row), axis=1)
    df_train['ANGLE'] = df_train.apply(lambda row: angle(row), axis=1)

    df_s4 = df_train[
        ['PERIOD_TIME', 'LAST_ELAPSED_TIME', 'PERIOD', 'COORD_X', 'COORD_Y', 'LAST_COORD_X', 'LAST_COORD_Y', 'DISTANCE',
         'LAST_DISTANCE', 'ANGLE', 'SHOT_TYPE', 'LAST_EVENT_ID', 'RINK_SIDE','IS_EMPTY_NET', 'IS_GOAL']].copy()

    df_s4['PERIOD_TIME'] = df_s4['PERIOD_TIME'].apply(toseconds)
    df_s4['REBOND'] = df_s4.apply(lambda row: True if row['LAST_EVENT_ID'] == 506 else False, axis=1)
    df_s4['CHANGE_ANGLE'] = df_s4.apply(lambda row: row['ANGLE'] - angle(row, True) if row['REBOND'] else 0,axis=1)

    df_s4['SPEED'] = df_s4.apply(
        lambda row: row['LAST_DISTANCE'] / row['LAST_ELAPSED_TIME'] if row['LAST_ELAPSED_TIME'] != 0 else 0, axis=1)

    df_s4['SPEED_ANGLE'] = df_s4.apply(
        lambda row: row['CHANGE_ANGLE'] / row['LAST_ELAPSED_TIME'] if row['LAST_ELAPSED_TIME'] != 0 else 0, axis=1)

    df_s4 = df_s4.drop('RINK_SIDE', axis=1)

    '''
    experiment.log_dataframe_profile(
        df_s4,
        name='wpg_v_wsh_2017021065',  # keep this name
        dataframe_format='csv'  # ensure you set this flag!
    )
    '''
    return df_s4

def main(experiment):
    df_data = organiser("raw",2016,2019)
    #df_test =  organiser("/home/mchelfi/Desktop/PROJET_NHL/raw",2020,2020)

    #encode IS_EMPTY_NET and IS_GOAL to 0's and 1's
    df_data = encode_column(df_data, 'IS_EMPTY_NET')
    df_data = encode_column(df_data, 'IS_GOAL')
    #df_test  = encode_column(df_test, 'IS_EMPTY_NET')
    #df_test  = encode_column(df_test, 'IS_GOAL')

    X_data_4col = df_data[['IS_GOAL', 'IS_EMPTY_NET']].copy()
    X_data_4col['DISTANCE'] = df_data.apply(lambda x: distance(x), axis=1)
    X_data_4col['ANGLE'] = df_data.apply(lambda x: angle(x), axis=1)

    caracteristiques = ['DISTANCE']
    X_data_4col = X_data_4col.dropna(subset=caracteristiques+['IS_GOAL'])

    X_train, X_val, y_train, y_val = split_data(X_data_4col,'IS_GOAL')

    clf = LogisticRegression()
    clf.fit( X_train[caracteristiques], y_train)

    y_pred = clf.predict( X_val[caracteristiques] )

    report = classification_report(y_val, y_pred)
    accuracy = accuracy_score(y_val, y_pred)
    print(f"La precision est: {accuracy}")
    print(report)

    y_score = clf.predict_proba(X_val[caracteristiques])
    # la deuxieme colomne nous les probabilites des buts esperes
    goal_probabilities = y_score[:, 1]

    fpr, tpr, _ = roc_curve(y_val, goal_probabilities)
    roc_auc = auc(fpr, tpr)

    return df_data
    #experiment.log_parameter("random_state", 42)
    #experiment.log_parameter("test_size", 0.2)
    #experiment.log_metric("accuracy", accuracy)
    #experiment.log_metric("auc", roc_auc)
    #log_model(experiment, "LogisticRegression-distance", clf)
    #experiment.register_model("Distance")
    #experiment.log_model("model_name", "./raw")
    #experiment.log_dataframe_profile(
        #subset_df,
    #    name='wpg_v_wsh_2017021065', # keep this name
    #    dataframe_format='csv' # ensure you set this flag!
    #)

if __name__ == "__main__":
    experiment = None

    experiment = Experiment(
        api_key=os.environ.get('COMET_API_KEY'),
        project_name="milestone_2",
        workspace="kevinchelfi"
    )

    #df_data = main(experiment)
    #df_data = partie4("raw",experiment)
    #print(df_data)
    #print(df_data.shape)

    #df_train = organiser("raw", 2016, 2019)
    df_s5 = partie4("raw")

    le = LabelEncoder()
    df_s5['LAST_EVENT_ID'] = le.fit_transform(df_s5['LAST_EVENT_ID'])
    df_s5['SHOT_TYPE'] = le.fit_transform(df_s5['SHOT_TYPE'])

    #df_s5 = encode_column(df_s5, 'IS_EMPTY_NET')
    #df_s5 = encode_column(df_s5, 'IS_GOAL')

    #df_s5['DISTANCE'] = df_s5.apply(lambda row: distance(row), axis=1)
    #df_s5['ANGLE'] = df_s5.apply(lambda row: angle(row), axis=1)

    #df_q1 = df_s5[['DISTANCE', 'ANGLE', 'IS_GOAL', 'IS_EMPTY_NET']].copy()
    df_s5 = df_s5.dropna(subset=['ANGLE', 'IS_GOAL'])
    X = df_s5[['ANGLE']]
    y = df_s5['IS_GOAL']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    xg_train = xgb.DMatrix(X_train, label=y_train)
    xg_val = xgb.DMatrix(X_val, label=y_val)

    params = {
        'max_depth': 3,
        'eta': 0.3,
        'objective': 'binary:logistic',
        # 'num_class': 2,
        'eval_metric': 'auc'
    }
    num_round = 20

    bst = xgb.train(params, xg_train, num_round)

    goal_probabilities = bst.predict(xg_val)

    xgb_classifier = xgb.XGBClassifier()
    xgb_classifier.fit(X_train, y_train)
    y_pred = xgb_classifier.predict(X_val)

    accuracy =  accuracy_score(y_val, y_pred)
    fpr, tpr, _ = roc_curve(y_val, goal_probabilities)
    roc_auc = auc(fpr, tpr)
    # experiment.log_parameter("random_state", 42)
    # experiment.log_parameter("test_size", 0.2)
    experiment.log_metric("accuracy", accuracy)
    experiment.log_metric("auc", roc_auc)
    #log_model(experiment, "LogisticRegression-distance", clf)
    # experiment.register_model("Distance")
    # experiment.log_model("model_name", "./raw")













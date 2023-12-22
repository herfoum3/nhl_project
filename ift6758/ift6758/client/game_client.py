import json
import requests
import os
import logging
import pandas as pd
from loader_parser import *
from ingenierie import *
from ift6758.client.serving_client import *

logger = logging.getLogger(__name__)


class GameClient:
    def __init__(self, base_url:str, gameid: str='2022030411', serving_client=None):
        self.base_url = base_url
        self.gameid = gameid
        logger.info(f"Initializing client; base URL: {self.base_url}")
        self.directory = './games'

        if serving_client is None:
            serving_client = ServingClient(ip=os.environ.get('SERVING_HOST', '127.0.0.1'), port=8890, features=["DISTANCE"])
        self.serving_client = serving_client

        self.events = pd.DataFrame([],columns=['PLAY_ID'])

    def refreshGame(self):
        game_json = {}
        try:
            rsp = requests.get(f'{self.base_url}/{self.gameid}/play-by-play')
            if rsp.status_code == 200:
               game_json = json.loads(rsp.text)
        except Exception as e:
            logger.info(f"Error downloading game {self.gameid}: {e}")
        return game_json

    def downloadGame(self):
        if not os.path.exists(self.directory):
            os.mkdir(self.directory)

        game_file = f'{self.directory}/{self.gameid}.json'

        if os.path.exists(game_file):
            with open(game_file, 'r') as game_file:
                game_json = json.load(game_file)
        else:
            rsp = requests.get(f'{self.base_url}/{self.gameid}/play-by-play')
            if rsp.status_code == 200:
                with open(game_file, "w") as file:
                     file.write(rsp.text)
                game_json = json.loads(rsp.text)

        return game_json

    def getGameStats(self):
        rsp = requests.get(f'{self.base_url}/{self.gameid}/boxscore')
        boxscore = json.loads(rsp.text)

        return {'period': boxscore.get('period',None),
                'timeleft':boxscore.get('clock',{}).get('timeRemaining',None),
                'awayteam': {
                    'name': boxscore.get('awayTeam',{}).get('name',{}).get('default',None),
                    'score': boxscore.get('awayTeam',{}).get('score',None)
                },
                'hometeam': {
                    'name': boxscore.get('homeTeam', {}).get('name', {}).get('default', None),
                    'score': boxscore.get('homeTeam', {}).get('score', None)
                }
                }

    def parse(self):

        game_json = self.refreshGame()

        game = parse_data_v2(game_json)

        df = pd.DataFrame(game)

        df = encode_column(df, 'IS_EMPTY_NET')
        df = encode_column(df, 'IS_GOAL')
        df['DISTANCE'] = df.apply(lambda row: distance(row), axis=1)
        df['ANGLE'] = df.apply(lambda row: angle(row), axis=1)

        df_s4 = df[
            ['PLAY_ID','TEAM_SHOT','AWAY_TEAM','PERIOD_TIME', 'LAST_ELAPSED_TIME', 'PERIOD', 'COORD_X', 'COORD_Y',
             'LAST_COORD_X', 'LAST_COORD_Y','DISTANCE','LAST_DISTANCE', 'ANGLE', 'SHOT_TYPE', 'LAST_EVENT_ID',
             'RINK_SIDE', 'IS_EMPTY_NET', 'IS_GOAL']].copy()

        df_s4['PERIOD_TIME'] = df_s4['PERIOD_TIME'].apply(toseconds)
        df_s4['REBOND'] = df_s4.apply(lambda row: True if row['LAST_EVENT_ID'] == 506 else False, axis=1)
        df_s4['CHANGE_ANGLE'] = df_s4.apply(lambda row: row['ANGLE'] - angle(row, True) if row['REBOND'] else 0, axis=1)

        df_s4['SPEED'] = df_s4.apply(
            lambda row: row['LAST_DISTANCE'] / row['LAST_ELAPSED_TIME'] if row['LAST_ELAPSED_TIME'] != 0 else 0, axis=1)

        df_s4['SPEED_ANGLE'] = df_s4.apply(
            lambda row: row['CHANGE_ANGLE'] / row['LAST_ELAPSED_TIME'] if row['LAST_ELAPSED_TIME'] != 0 else 0, axis=1)
        df_s4 = df_s4.drop('RINK_SIDE', axis=1)
        return df_s4

    def predict(self):
        df = self.parse()

        #filter events
        eventids = self.events['PLAY_ID']

        df = df[~df['PLAY_ID'].isin(eventids)]

        if df.shape[0] != 0:
            pred = self.serving_client.predict(df)
            df['predictions'] = pred['predictions']

            if eventids.shape[0] == 0:
                self.events = df
            else:
                self.events =  self.events.append(df, ignore_index=False)

        mask = self.events['TEAM_SHOT'] == self.events['AWAY_TEAM']
        away_sum = self.events[mask].sum()['predictions']
        mask = self.events['TEAM_SHOT'] != self.events['AWAY_TEAM']
        home_sum = self.events[mask].sum()['predictions']

        cols = self.serving_client.model_details()['features'] +  ['PLAY_ID', 'predictions']
        return {'df':self.events[cols], 'away_xG':round(away_sum,2), 'home_xG': round(home_sum,2)}



if __name__ == '__main__':
  gc = GameClient(base_url='https://api-web.nhle.com/v1/gamecenter', gameid='2023020498')

  df = gc.parse()

  sc = ServingClient(ip='127.0.0.1', port=8890, features=["DISTANCE"])
  pred = sc.predict(df)

  #mask = df['TEAM_SHOT'] == df['AWAY_TEAM']
  #mask = df['TEAM_SHOT'] != df['AWAY_TEAM']

  print(gc.predict())
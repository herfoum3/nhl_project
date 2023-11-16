import numpy as np
import pandas as pd
import requests
import os
from tqdm import tqdm
import json

ALL_SEASONS = [2016, 2017, 2018, 2019, 2020]


def season_list(start: str, stop: str):
    return [j for j in range(start, stop + 1)]


def pading(n):
    if n < 10:
        return '000' + str(n)
    elif n < 100:
        return '00' + str(n)
    else:
        return '0' + str(n)


def downloadGames():
    if not os.path.exists('raw'):
        RAWDATA = os.mkdir('raw')

        for year in ALL_SEASONS:
            for event in range(1, 5):
                for game in tqdm(range(1, 2000), desc='Processing'):
                    GAME_ID = str(year) + '0' + str(event) + pading(game)
                    rsp = requests.get(f"https://statsapi.web.nhl.com/api/v1/game/{GAME_ID}/feed/live/")
                    if rsp.status_code == 404:
                        break
                    with open(f'raw/{GAME_ID}.json', "w") as file:
                        file.write(rsp.text)


def extract_coordinates(coordinates, period):
    coord_x = coordinates.get('x', None)
    coord_y = coordinates.get('y', None)
    if coord_x == None or coord_y == None:
        if period == 5 and coord_x != None:  # shoutout
            coord_y = 0  # we assume at net level
        else:
            return None, None  # here we ignore goals and shots without coordinates
    return coord_x, coord_y

def toseconds(priod_time):
    minutes, seconds = map(int, priod_time.split(':'))
    return minutes * 60 + seconds
def distance_to(position1, position2):
    return np.sqrt((position1[0] - position2[0])**2 + (position1[1]-position2[1])**2)

def  extract_coordinates_v2(details,period):
    coord_x = details.get('xCoord', None)
    coord_y = details.get('yCoord', None)
    if coord_x == None or coord_y == None:
        if period == 5 and coord_x != None:  # shoutout
            coord_y = 0  # we assume at net level
        else:
            return None, None  # here we ignore goals and shots without coordinates
    return coord_x, coord_y
def parse_data_v2(game_json):
    '''
    This version handles the new NHL api.
    unfortunatley there is no way, in the new json, from the play-by-play to extrat the empty net and force (powerplay)
    info
    '''
    homeTeam = game_json.get("homeTeam", {}).get("abbrev", None)
    awayTeam = game_json.get("awayTeam", {}).get("abbrev", None)

    teams = {
        game_json.get("homeTeam", {}).get("id", None): homeTeam,
        game_json.get("awayTeam", {}).get("id", None): awayTeam
    }

    players_by_id = {}
    for p in game_json.get('rosterSpots',[]):
        players_by_id[p['playerId']] = p['firstName']['default'] + ' ' + p['lastName']['default']

    game_id = int(game_json.get("id", None))

    previous_event = None
    shots = 0
    goals = 0
    tempplays = []  # used to add the goals and shots for every game
    plays = []      # fill up with what we did down there and then DF
    all_plays = game_json.get('plays', [])
    for p in all_plays:
        eventIdx = p['eventId']
        eventTypeId = int(p['typeCode'])

        #GOAL=505, SHOT_ON_GOAL=506
        if not (eventTypeId == 505 or eventTypeId == 506):
            previous_event = p
            continue

        period = int(p['period'])
        period_time = p['timeInPeriod']
        details = p.get('details', {})

        coord_x, coord_y = extract_coordinates_v2(details,period)
        if coord_x == None:  # here we ignore goals and shots without coordinates
            continue

        # info related to previous event
        last_event_id = previous_event['typeCode']
        last_details = previous_event.get('details', {})
        last_period = int(previous_event['period'])
        last_coord_x, last_coord_y = extract_coordinates_v2(last_details, last_period)
        if last_coord_x == None:  # 0 for None previous events positions
            last_coord_x = 0
        if last_coord_y == None:  # 0 for None previous events positions
            last_coord_y = 0

        last_period_time = previous_event['timeInPeriod']

        # keep a count of shots and goals
        if eventTypeId == 506:
            shots += 1
        else:
            goals += 1

        # print(eventIdx,coord_x,coord_y,last_coord_x,last_coord_y,last_event_id  )
        play = {'GAME_ID': game_id,
                'HOME_TEAM': homeTeam,
                'AWAY_TEAM': awayTeam,

                'PERIOD': period,
                'PERIOD_TIME': period_time,
                'COORD_X': coord_x,
                'COORD_Y': coord_y,
                'IS_EMPTY_NET': False,

                'LAST_EVENT_ID': last_event_id,
                'LAST_COORD_X': last_coord_x,
                'LAST_COORD_Y': last_coord_y,
                'LAST_ELAPSED_TIME': toseconds(period_time) - toseconds(last_period_time),
                'LAST_DISTANCE': distance_to((coord_x, coord_y), (last_coord_x, last_coord_y))
                }

        team_shot = details.get('eventOwnerTeamId')
        team_shot = teams.get(team_shot, None)

        #play['HEURE'] = p['about']['dateTime']
        play['PLAY_ID'] = eventIdx
        play['TEAM_SHOT'] = team_shot


        scorer_id  = details.get("scoringPlayerId", None)
        shooter_id = details.get("shootingPlayerId", None)
        goalie_id  = details.get("goalieInNetId", None)

        if goalie_id:
            play['NAME_KEEPER'] = players_by_id[goalie_id]

        play['NAME_SHOOTER'] =  players_by_id[scorer_id] if scorer_id else players_by_id[shooter_id]

        play['IS_GOAL'] = eventTypeId == 505
        play['SHOT_TYPE'] = details.get('shotType', None)

        #play['IS_EMPTY_NET'] = int(p.get('situationCode',0)) != 1560
        #play['IS_EQUAL_FORCE'] = int(p.get('situationCode', 0)) != 1531
        home_defense_side = p.get('homeTeamDefendingSide', None)
        away_defence_side = 'right' if home_defense_side == 'left' else 'left'


        if period == 5:
            rs = 'SHOOT_OUT'
        else:
            rs = away_defence_side if team_shot == homeTeam else home_defense_side

        if rs == None:
            # infere it from the position of the shot
            rs = 'left' if coord_x > 0 else 'right'

        play['RINK_SIDE'] = rs

        plays.append(play)
        tempplays.append(play)  # keep to add the number of shots and goals

        previous_event = p

    for play in tempplays:
        play["NUMBER_SHOTS"] = shots
        play["NUMBER_GOALS"] = goals
    return plays
def parse_data(game_json):
    gameData = game_json["gameData"]
    all_plays = game_json['liveData']['plays']['allPlays']
    periods = game_json["liveData"]["linescore"]["periods"]

    # TODO maybe needed
    rink_side = {}
    for period in periods:
        rink_side[period['num']] = {
            'home': period['home'].get('rinkSide', None),
            'away': period['away'].get('rinkSide', None)
        }

    homeTeam = gameData.get("teams", {}).get("home", {}).get("triCode", None)
    awayTeam = gameData.get("teams", {}).get("away", {}).get("triCode", None)

    game_id = int(gameData.get("game", {}).get("pk", None))
    if game_id == None:
        print('--> Should not be')

    previous_event = None
    shots = 0
    goals = 0
    tempplays = []  # used to add the goals and shots for every game
    plays = []      # fill up with what we did down there and then DF

    for p in all_plays:
        eventIdx = p['about']['eventIdx']
        eventTypeId = p['result']['eventTypeId']

        if not (eventTypeId == "SHOT" or eventTypeId == "GOAL"):
            previous_event = p
            continue

        period = int(p['about']['period'])
        period_time = p['about']['periodTime']
        coordinates = p.get('coordinates', {})

        coord_x, coord_y = extract_coordinates(coordinates, period)
        if coord_x == None:  # here we ignore goals and shots without coordinates
            continue

        # info related to previous event
        last_event_id = previous_event['result']['eventTypeId']
        last_coordinates = previous_event.get('coordinates', {})
        last_period = int(previous_event['about']['period'])
        last_coord_x, last_coord_y = extract_coordinates(last_coordinates, last_period)
        if last_coord_x == None:  # 0 for None previous events positions
            last_coord_x = 0
        if last_coord_y == None:  # 0 for None previous events positions
            last_coord_y = 0

        last_period_time = previous_event['about']['periodTime']

        # keep a count of shots and goals
        if eventTypeId == "SHOT":
            shots += 1
        else:
            goals += 1

        # print(eventIdx,coord_x,coord_y,last_coord_x,last_coord_y,last_event_id  )
        play = {'GAME_ID': game_id,
                'HOME_TEAM': homeTeam,
                'AWAY_TEAM': awayTeam,

                'PERIOD': period,
                'PERIOD_TIME': period_time,
                'COORD_X': coord_x,
                'COORD_Y': coord_y,

                'LAST_EVENT_ID': last_event_id,
                'LAST_COORD_X': last_coord_x,
                'LAST_COORD_Y': last_coord_y,
                'LAST_ELAPSED_TIME': toseconds(period_time) - toseconds(last_period_time),
                'LAST_DISTANCE': distance_to((coord_x, coord_y), (last_coord_x, last_coord_y))
                }

        team_shot = p.get('team', {}).get('triCode')

        play['HEURE'] = p['about']['dateTime']
        play['PLAY_ID'] = eventIdx
        play['TEAM_SHOT'] = team_shot

        players = p.get("players", [])
        for player in players:
            if player.get("playerType") in ["Shooter", "Scorer"]:
                play['NAME_SHOOTER'] = player.get("player", {}).get("fullName")
            if player.get("playerType") == "Goalie":
                play['NAME_KEEPER'] = player.get("player", {}).get("fullName")

        play['IS_GOAL'] = eventTypeId == "GOAL"
        play['SHOT_TYPE'] = p.get('result', {}).get('secondaryType')
        play['IS_EMPTY_NET'] = p.get('result', {}).get('emptyNet')
        play['IS_EQUAL_FORCE'] = p.get('result', {}).get('strength', {}).get('code') == "EVEN"

        if period == 5:
            rs = 'SHOOT_OUT'
        else:
            rs = rink_side.get(period, {}).get('home' if team_shot == homeTeam else 'away', None)
        if rs == None:
            # infere it from the position of the shot
            rs = 'left' if coord_x > 0 else 'right'

        play['RINK_SIDE'] = rs

        plays.append(play)
        tempplays.append(play)  # keep to add the number of shots and goals

        previous_event = p

    for play in tempplays:
        play["NUMBER_SHOTS"] = shots
        play["NUMBER_GOALS"] = goals
    return plays

def organiser(path: str, start: int, stop: int, seasonType: tuple = [2]):
    entries = os.listdir(path)
    seasons = season_list(start, stop)
    plays = []
    for e in entries:
        if int(e[0:4]) not in seasons or int(e[5]) not in seasonType:
            continue
        with open(path + '/' + e, 'r') as game_file:
            game_json = json.load(game_file)
        plays += parse_data(game_json)
        #break
    df = pd.DataFrame(plays)
    return df

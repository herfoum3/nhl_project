import numpy as np
import pandas as pd
import requests
import os
from tqdm import tqdm
import json 

Lyear = ["2016","2017","2018","2019","2020"]


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
    
       for year in Lyear:
           for event in range(1,5):
                for game in tqdm(range(1,2000), desc= 'Processing'):
                    GAME_ID = year + '0' + str(event) + pading(game)
                    rsp = requests.get(f"https://statsapi.web.nhl.com/api/v1/game/{GAME_ID}/feed/live/")
                    if rsp.status_code == 404:
                        break
                    with open(f'raw/{GAME_ID}.json', "w") as file:
                        file.write(rsp.text)
                     
       
def organiser(path: str):
   
    entries = os.listdir(path)
        
    plays = [] # fill up with what we did down there and then DF 
      
    for e in entries: 
        with open(path + '/' + e, 'r') as game_file:               
            game_json = json.load(game_file)
   
        all_plays = game_json['liveData']['plays']['allPlays']
        
        game_id = e[0:-5]
     
        shots = 0
        goals = 0
        tempplays = [] # used to add the goals and shots for every game
   
        for p in all_plays:
           
           eventIdx = p['about']['eventIdx']
           
           eventTypeId= p['result']['eventTypeId']
           
           if not (eventTypeId == "SHOT" or eventTypeId == "GOAL"):
               continue
           
           # keep a count of shots and goals
           if eventTypeId == "SHOT":
              shots += 1
           else:
              goals += 1       
               
           play = {'GAME_ID': game_id}
        
           play['PLAY_ID'] = eventIdx 
        
           play['HEURE_P'] = p['about']['dateTime'] 
        
           play['PERIODE'] =  p['about']['period'] 
           
           play['TEAM_SHOT'] = p.get('team',{}).get('name')
           
           play['COORDINATES'] = p.get('coordinates')
           
            
           players = p.get("players",[])
           for player in players:
               #this 
               if player.get("playerType") in ["Shooter", "Scorer"]:
                   play['NAME_SHOOTER'] = player.get("player",{}).get("fullName")              
               elif player.get("playerType") == "Goalie":
                   play['NAME_KEEPER'] = player.get("player",{}).get("fullName")
                              
           play['IS_GOAL'] = eventTypeId == "GOAL"
       
           play['SHOT_TYPE'] = p.get('result',{}).get('secondaryType')       
          
           play['IS_EMPTY_NET'] = p.get('result',{}).get('emptyNet') 
                 
           play['IS_EQUAL_FORCE'] = p.get('result',{}).get('strength',{}).get('code') == "EVEN"         
        
           plays.append(play)
           tempplays.append(play) # keep to add the number of shots and goals
           
        for play in tempplays:
           play["NUMBER_SHOTS"] = shots
           play["NUMBER_GOALS"] = goals 
           
        
    df = pd.DataFrame(plays)    
        
    return df   
                
   
df = organiser("raw")
print(df)































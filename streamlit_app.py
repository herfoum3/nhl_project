import streamlit as st
import pandas as pd
import numpy as np
from ift6758.client.serving_client import *
from ift6758.client.game_client import *
"""
General template for your streamlit app. 
Feel free to experiment with layout and adding functionality!
Just make sure that the required functionality is included as well
"""

st.title('Hockey Visualization App')

if 'serving_client' not in st.session_state:
    sc = ServingClient(ip=os.environ.get('SERVING_HOST', '0.0.0.0'), port=8890, features=['DISTANCE'])
    st.session_state['serving_client'] = sc



def get_model(workspace, version, model):
    print('called',workspace,version,model  )
    print(type(workspace))
    st.session_state['serving_client'].download_registry_model(workspace,version,model )
    #ensures game_client is reset when model changes
    if 'game_client' in st.session_state:
        del st.session_state['game_client']

def ping_game(gameid):
    print('called',gameid )
    #the url is hardcode cause if changes, everything else changes and a rebuild is required
    if 'game_client' not in st.session_state:
        sc = st.session_state['serving_client']
        gc = GameClient(base_url='https://api-web.nhle.com/v1/gamecenter', gameid=gameid, serving_client=sc)
        st.session_state['game_client'] = gc

    gc = st.session_state['game_client']

    st.session_state['gameStats'] = gc.getGameStats()
    st.session_state['predictions'] = gc.predict()


with st.sidebar:
    # TODO: Add input for the sidebar
    workspace = st.text_input('Workspace', '')

    model = st.text_input('Model', '')
    version = st.text_input('Version', '')
    st.button("Get model", key="model_info", on_click=get_model, args=(workspace,model,version) )


with st.container():
    # TODO: Add Game ID input
    gameid = st.text_input('Game ID', '2021020329')
    st.button("Ping Game", on_click=ping_game, args=(gameid,))


with st.container():
    # TODO: Add Game info and predictions

    if 'gameStats' in st.session_state:
        awayteam_name = st.session_state['gameStats']['awayteam']['name']
        hometeam_name = st.session_state['gameStats']['hometeam']['name']

        st.header(f"Game {gameid}: {awayteam_name} vs {hometeam_name}")
        st.write("Period", st.session_state['gameStats']['period'],'- ',st.session_state['gameStats']['timeleft'], 'left')

        away, home = st.columns(2)
        awayteam_score = st.session_state['gameStats']['awayteam']['score']
        hometeam_score  = st.session_state['gameStats']['hometeam']['score']

        away_xG = st.session_state['predictions']['away_xG']
        home_xG = st.session_state['predictions']['home_xG']
        away.metric(f'{awayteam_name} xG (actual', f' {away_xG} ({awayteam_score})', awayteam_score-away_xG)
        home.metric(f'{hometeam_name} xG (actual', f' {home_xG} ({hometeam_score})', hometeam_score-home_xG)


with st.container():
    # TODO: Add data used for predictions
    st.header(f"Data used for predictions (and predictions)")
    if 'predictions' in st.session_state:
        df=st.session_state['predictions']['df']
        df = df.set_index('PLAY_ID')
        st.dataframe(df)




"""
If you are in the same directory as this file (app.py), you can run run the app using gunicorn:
    
    $ gunicorn --bind 0.0.0.0:<PORT> app:app

gunicorn can be installed via:

    $ pip install gunicorn

"""
import os
from pathlib import Path
import logging
from flask import Flask, jsonify, request, abort
import sklearn
import pandas as pd
import joblib
import json as jsonob
from io import StringIO

#import ift6758
from comet_ml.api import API, APIExperiment

LOG_FILE = os.environ.get("FLASK_LOG", "flask.log")


app = Flask(__name__)


def get_model_file_path(file_path):
    with open(f'{file_path}/CometModel', 'r') as file:
        details = jsonob.load(file)
    format = details.get('format')
    return os.path.join(file_path, details.get('model_metadata', {}).get(format,{}).get('model_path'))

def load_model(model_info):
    global model
    global model_details

    workspace=model_info.get('workspace')
    model_name=model_info.get('model')
    version=model_info.get('version')

    model_details = {
        'workspace': workspace,
        'model': model_name,
        'version': version
    }

    if model_name == 'distance':
        model_details['features']= ['DISTANCE']
    elif model_name == 'angle':
        model_details['features'] = ['ANGLE']
    elif model_name == 'angle_and_distance':
        model_details['features']  = ['ANGLE','DISTANCE']

    if workspace == None or model_name == None or version == None:
         return None, {'status': 'all model details must be specified: workspace, model_name and version'}

    model_full_name = f'{workspace}_{model_name}_{version}'

    MODEL_DIRECTORY = './models'

    # TODO: check to see if the model you are querying for is already downloaded
    file_path = os.path.join(MODEL_DIRECTORY, f'{model_full_name}')

    try:
        if os.path.exists(file_path):
            # TODO: if yes, load that model and write to the log about the model change.
            model = joblib.load(get_model_file_path(file_path))
            app.logger.info(f'Model changed successfully to {file_path}')
            response = {'status': 'Model already downloaded', 'info': model_full_name}
        else:
            # TODO: if no, try downloading the model: if it succeeds, load that model and write to the log
            # about the model change. If it fails, write to the log about the failure and keep the
            # currently loaded model
            # Logic to download the model
            # ...
            api_key = os.environ.get('COMET_API_KEY')
            comet_api = API(api_key=api_key)

            comet_api.download_registry_model(workspace, model_name, version, output_path=file_path)
            app.logger.info(f'Model successfully downloaded to {file_path}')

            model = joblib.load(get_model_file_path(file_path))
            # if the downdload and load fail the model is not changed. The previous model is kept since model var is
            # not assigned here because of the exception branching to the oustside block

            app.logger.info(f'Model changed successfully to {model_full_name}')
            response = {'status': 'Model successfully downloaded', 'info': model_full_name}
    except  Exception as e:
        app.logger.error(f'Error downloading and/or loading model {model_full_name}: {e}')
        response = {'status': 'Error downloading and/or loading model', 'info': f'{model_full_name}-{e}'}

    return model, response

#@app.before_first_request
def before_first_request():
    """
    Hook to handle any initialization before the first request (e.g. load model,
    setup logging handler, etc.)
    """
    print("This runs before the first request.")
    # TODO: setup basic logging configuration
    logging.basicConfig(filename=LOG_FILE, level=logging.INFO)

    # TODO: any other initialization before the first request (e.g. load default model)
    default_model = {
        'workspace': 'kevinchelfi',
        'model': 'distance',
        'version': '1.0.0'
    }
    model, response = load_model(default_model)
    app.logger.info(response)

# before_first_request was deprecated and not recognized in flask version 3
# downgrading to 2.2 was more preblematic with the python version. so I used
# the equivalent below
with app.app_context():
    before_first_request()

@app.route("/logs", methods=["GET"])
def logs():
    """Reads data from the log file and returns them as the response"""
    # TODO: read the log file specified and return the data
    #raise NotImplementedError("TODO: implement this endpoint")

    response = None
    try:
        # Open the log file
        with open(LOG_FILE, 'r') as file:
            response = file.read()
    except Exception as e:
        error = f'Error reading log file: {e}'
        app.logger.error(error)
        return jsonify({'error': error}), 500

    return jsonify(response)  # response must be json serializable!


@app.route("/download_registry_model", methods=["POST"])
def download_registry_model():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/download_registry_model
    The comet API key should be retrieved from the ${COMET_API_KEY} environment variable.
    Recommend (but not required) json with the schema:
        {
            workspace: (required),
            model: (required),
            version: (required),
            ... (other fields if needed) ...
        }
    """
    # Get POST json data
    json = request.get_json()
    app.logger.info(json)

    #using load_model function to implement the different TODO's of current function
    #this way load_model is also reused in before_first_request
    model , response = load_model(json)

    # Tip: you can implement a "CometMLClient" similar to your App client to abstract all of this
    # logic and querying of the CometML servers away to keep it clean here
    #raise NotImplementedError("TODO: implement this endpoint")

    app.logger.info(response)
    return jsonify(response)  # response must be json serializable!


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/predict
    Returns predictions
    """
    # Get POST json data
    json = request.get_json()
    app.logger.info(json)
    # TODO:
    response = None
    df = pd.read_json(StringIO(jsonob.dumps(json)))
    if 'model' in globals():
        print('COLUMNS',df.columns)
        predictions = model.predict_proba(df)
        response = {'predictions': predictions[:,1].tolist()}
    else:
        response = {'error': 'Model not loaded'}
        app.logger.error(response['error'])
    #raise NotImplementedError("TODO: implement this enpdoint")

    app.logger.info(response)
    return jsonify(response)  # response must be json serializable!

@app.route("/model_details", methods=["GET"])
def get_model_details():
    if 'model_details' in globals():
        return jsonify(model_details)
    else:
        return jsonify({})


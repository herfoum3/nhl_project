import json
import requests
import pandas as pd
import logging


logger = logging.getLogger(__name__)


class ServingClient:
    def __init__(self, ip: str = "0.0.0.0", port: int = 5000, features=None):
        self.base_url = f"http://{ip}:{port}"
        logger.info(f"Initializing client; base URL: {self.base_url}")

        if features is None:
            features = ["distance"]
        self.features = features

        # any other potential initialization

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Formats the inputs into an appropriate payload for a POST request, and queries the
        prediction service. Retrieves the response from the server, and processes it back into a
        dataframe that corresponds index-wise to the input dataframe.
        
        Args:
            X (Dataframe): Input dataframe to submit to the prediction service.
        """

        print(self.model_details()['features'])
        X_to_predict = X[self.model_details()['features']]

        json_payload = json.loads(X_to_predict.to_json())

        try:
            # we send the post to the prediction service
            response = requests.post(self.base_url + '/predict', json=json_payload)

            # if code 200 it's successuful
            if response.status_code == 200:
                # we preproces the response to correpond incides and predictions
                pred_response = response.json()
                predictions = pd.DataFrame(pred_response)
                predictions.index = X.index

                return predictions
            else:
                print(f'Request failed: status code = {response.status_code}')
                return pd.DataFrame()
        except Exception as e:
            print(f'Error requesting prediction: {e}')
            return pd.DataFrame()  # Return empty DataFrame or handle as needed



    def logs(self) -> dict:
        """Get server logs"""

        try:
            # we send a get to the prediction service logs endpoint
            response = requests.get(self.base_url + '/logs')

            # if code 200 it's successuful
            if response.status_code == 200:
                return  response.json()
            else:
                print(f'Request failed: status code = {response.status_code}')
                return {'Error': 'Failed to retrieve logs'}
        except Exception as e:
            print(f'Error requesting logs: {e}')
            return {'error': str(e)}

    def download_registry_model(self, workspace: str, model: str, version: str) -> dict:
        """
        Triggers a "model swap" in the service; the workspace, model, and model version are
        specified and the service looks for this model in the model registry and tries to
        download it. 

        See more here:

            https://www.comet.ml/docs/python-sdk/API/#apidownload_registry_model
        
        Args:
            workspace (str): The Comet ML workspace
            model (str): The model in the Comet ML registry to download
            version (str): The model version to download
        """

        model_details = {
            'workspace': workspace,
            'model': model,
            'version': version
        }

        try:
            # Send the POST request to the download_registry_model endpoint
            response = requests.post(self.base_url + '/download_registry_model', json=model_details)

            # if code 200 it's successuful
            if response.status_code == 200:
                # the response from the service
                return response.json()
            else:
                print(f'Request failed: status code = {response.status_code}')
                return {'error': 'Failed to download the model'}
        except Exception as e:
            print(f'Error requesting model change: {e}')
            return {'error': str(e)}

    def model_details(self):
        response = requests.get(self.base_url + '/model_details')
        if response.status_code == 200:
            return response.json()
        else:
            return {}

if __name__ == '__main__':
    from loader_parser import *
    from ingenierie import *


    data_path="/home/mchelfi/Desktop/PROJET_NHL/raw"
    df_data = organiser(data_path,2016,2016)

    #print(df_train)
    #print(df_train.columns)

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

    sc = ServingClient(ip='127.0.0.1', port=8890, features=["DISTANCE"])
    print(sc.predict(X_train))
    print(X_train)
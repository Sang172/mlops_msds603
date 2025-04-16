import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import pickle

from metaflow import FlowSpec, step, Parameter


class ScoringFlow(FlowSpec):

    x_test_path = Parameter('x-test-path',
                            help='Path to the pickled X_test DataFrame.',
                            default='data/X_test.pkl')

    y_test_path = Parameter('y-test-path',
                            help='Path to the pickled y_test Series/DataFrame.',
                            default='data/y_test.pkl')

    best_model_info_path = Parameter('best-model-info-path',
                                      help='Path to the pickled DataFrame containing best model info (run_id, model_name).',
                                      default='data/best_model.pkl')


    @step
    def start(self):
        self.next(self.score_model)

    @step
    def score_model(self):
        # load test data, load best model, get predictions, and compute accuracy
        X_test = pd.read_pickle(self.x_test_path)
        y_test = pd.read_pickle(self.y_test_path)
        best_model = pd.read_pickle(self.best_model_info_path)
        
        best_run_id = best_model.iloc[0]['run_id']
        best_model_name = best_model.iloc[0]['model_name']
        best_model_val_loss = best_model.iloc[0]['validation_loss']
        model_uri = f"runs:/{best_run_id}/{best_model_name}"
        with mlflow.start_run(run_name="run_best_model") as trial_run:
            loaded_model = mlflow.pyfunc.load_model(model_uri)
            y_pred = loaded_model.predict(X_test)
            y_pred_path = 'data/y_pred.pkl'
            with open(y_pred_path, 'wb') as f:
                pickle.dump(y_pred, f)
            print(f"Predictions saved to {y_pred_path}")
            acc = accuracy_score(y_test, y_pred)
            mlflow.log_metric("validation_binary_cross_entropy_loss", best_model_val_loss)
            mlflow.log_metric("test_set_accuracy", acc)
        print(f"Accuracy Score of the best model (Run ID: {best_run_id}) on the test set: {acc}")
        self.next(self.end)


    @step
    def end(self):
        print('workflow finished')

if __name__ == '__main__':
     ScoringFlow()
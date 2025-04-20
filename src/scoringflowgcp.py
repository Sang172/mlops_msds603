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
import gcsfs

from metaflow import FlowSpec, step, Parameter, current, kubernetes
from metaflow import timeout, retry, catch, conda, conda_base
from metaflow import S3

@conda_base(libraries={
    "metaflow": "2.15.7",
    "pandas": "2.2.2",
    "scikit-learn": "1.5.1", 
    "mlflow": "2.15.1",
    "numpy": "1.26.4",
    "gcsfs": "2025.3.2",
    "python": "3.12" 
})

mlflow.set_tracking_uri('https://mlflow-service-265226491381.us-west2.run.app')

class ScoringFlow(FlowSpec):

    gcs_bucket = Parameter(
        'gcs_bucket',
        help='GCS bucket URL for data',
        default='gs://msds603-lab7/data/',
        required=True
    )


    @step
    def start(self):
        self.next(self.score_model)

    @kubernetes(cpu="2", memory="4Gi")
    @timeout(seconds=300)
    @retry(times=3, minutes_between_attempts=2)
    @step
    def score_model(self):
        # load test data, load best model, get predictions, and compute accuracy
        X_test = pd.read_pickle(self.gcs_bucket + 'X_test.pkl')
        y_test = pd.read_pickle(self.gcs_bucket + 'y_test.pkl')
        best_model = pd.read_pickle(self.gcs_bucket + 'best_model.pkl')
        
        best_run_id = best_model.iloc[0]['run_id']
        best_model_name = best_model.iloc[0]['model_name']
        best_model_val_loss = best_model.iloc[0]['validation_loss']
        model_uri = f"runs:/{best_run_id}/{best_model_name}"
        with mlflow.start_run(run_name="run_best_model") as trial_run:
            loaded_model = mlflow.pyfunc.load_model(model_uri)
            y_pred = loaded_model.predict(X_test)
            y_pred_path = self.gcs_bucket + 'y_pred.pkl'
            with open(y_pred_path, 'wb') as f:
                pickle.dump(y_pred, f)
            print(f"Predictions saved to {y_pred_path}")
            acc = accuracy_score(y_test, y_pred)
            mlflow.log_metric("validation_binary_cross_entropy_loss", best_model_val_loss)
            mlflow.log_metric("test_set_accuracy", acc)
        print(f"Accuracy Score of the best model (Run ID: {best_run_id}) on the test set: {acc:.2f}")
        self.next(self.end)


    @step
    def end(self):
        print('workflow finished')

if __name__ == '__main__':
     ScoringFlow()
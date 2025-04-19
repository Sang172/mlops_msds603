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

class TrainingFlow(FlowSpec):

    seed = Parameter('seed',
                        help='Random seed for data splitting and model initialization.',
                        default=5158,
                        type=int)

    gcs_bucket = Parameter(
        'gcs_bucket',
        help='GCS bucket URL for data',
        default='gs://msds603-lab7/data/',
        required=True
    )

    @step
    def start(self):
        self.next(self.data)

    @step
    def data(self):
        # data loading and preprocessing
        # save test set for scoring
        diabetes = pd.read_csv(self.gcs_bucket+'diabetes_binary.csv')
        cols = list(diabetes.columns)
        X = diabetes[cols[1:]]
        y = diabetes[cols[:1]]
        X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=self.seed)
        X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.25, shuffle=True, random_state=self.seed)



        numerical = ['BMI','MentHlth','PhysHlth']
        one_hot = ['GenHlth','Age','Education','Income']

        scaler = MinMaxScaler()

        scaler.fit(X_train[numerical])

        X_train[numerical] = scaler.transform(X_train[numerical])
        X_val[numerical] = scaler.transform(X_val[numerical])
        X_test[numerical] = scaler.transform(X_test[numerical])


        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

        encoder.fit(X_train[one_hot])

        X_train_encoded = pd.DataFrame(encoder.transform(X_train[one_hot]))
        X_val_encoded = pd.DataFrame(encoder.transform(X_val[one_hot]))
        X_test_encoded = pd.DataFrame(encoder.transform(X_test[one_hot]))

        feature_names = encoder.get_feature_names_out(one_hot)
        X_train_encoded.columns = feature_names
        X_val_encoded.columns = feature_names
        X_test_encoded.columns = feature_names

        X_train.reset_index(drop=True, inplace=True)
        X_val.reset_index(drop=True, inplace=True)
        X_test.reset_index(drop=True, inplace=True)
        X_train_encoded.reset_index(drop=True, inplace=True)
        X_val_encoded.reset_index(drop=True, inplace=True)
        X_test_encoded.reset_index(drop=True, inplace=True)

        X_train.drop(one_hot, axis=1, inplace=True)
        X_train = pd.concat([X_train, X_train_encoded], axis=1)

        X_val.drop(one_hot, axis=1, inplace=True)
        X_val = pd.concat([X_val, X_val_encoded], axis=1)

        X_test.drop(one_hot, axis=1, inplace=True)
        X_test = pd.concat([X_test, X_test_encoded], axis=1)
        X_test.to_pickle(self.gcs_bucket+'X_test.pkl')
        y_test.to_pickle(self.gcs_bucket+'y_test.pkl')

        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

        self.next(self.train)

    @kubernetes(cpu="2", memory="4Gi")
    @timeout(seconds=300)
    @retry(times=3, minutes_between_attempts=2)
    @step
    def train(self):
        # training over hyperparameter grid
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [10],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 3]
        }


        for params in ParameterGrid(param_grid):
            with mlflow.start_run(run_name="RF_Trials") as trial_run:
                mlflow.log_params(params)

                rf_model = RandomForestClassifier(**params, random_state=self.seed)
                rf_model.fit(self.X_train, self.y_train.values.ravel())

                mlflow.log_param("X_train_shape", self.X_train.shape)
                mlflow.log_param("y_train_shape", self.y_train.shape)
                mlflow.log_param("X_val_shape", self.X_val.shape)
                mlflow.log_param("y_val_shape", self.y_val.shape)

                y_pred_proba_val = rf_model.predict_proba(self.X_val)
                val_loss = log_loss(self.y_val, y_pred_proba_val)
                mlflow.log_metric("validation_binary_cross_entropy_loss", val_loss)
                model_name = f"rf_model_{'_'.join([f'{k}-{v}' for k, v in params.items()])}"
                mlflow.sklearn.log_model(rf_model, model_name)
        self.next(self.best_model)


    @step
    def best_model(self):
        # organize training logs into a dataframe and sort so that best model is at top row
        # save the dataframe
        run_names = ["RF_Trials"]
        all_runs_data = []

        client = mlflow.tracking.MlflowClient()

        for run_name in run_names:
            runs = mlflow.search_runs(
                filter_string=f"tags.mlflow.runName = '{run_name}'",
                output_format="pandas",
                order_by=["metrics.validation_binary_cross_entropy_loss ASC"]
            )
            if not runs.empty:
                for index, row in runs.iterrows():
                    run_id = row['run_id']
                    loss = row['metrics.validation_binary_cross_entropy_loss']

                    artifacts = client.list_artifacts(run_id)
                    model_name = None
                    for artifact in artifacts:
                        if artifact.path.startswith("rf_model"):
                            model_name = artifact.path
                            break

                    all_runs_data.append({'run_id': run_id, 'run_name': run_name, 'validation_loss': loss, 'model_name': model_name})



        all_runs_df = pd.DataFrame(all_runs_data)
        best_model = all_runs_df.sort_values(by='validation_loss', ascending=True)
        best_model.to_pickle(self.gcs_bucket+'best_model.pkl')
        self.next(self.register_model)


    @retry(times=2)
    @catch(var='registration_exception')
    @step
    def register_model(self):
        # register best model
        best_model = pd.read_pickle(self.gcs_bucket+'best_model.pkl')
        
        best_run_id = best_model.iloc[0]['run_id']
        best_model_name = best_model.iloc[0]['model_name']
        model_uri = f"runs:/{best_run_id}/{best_model_name}"

        registered_model_name = "Best_Model"

        mv = mlflow.register_model(model_uri, registered_model_name)
        print(f"Registered model name: {registered_model_name}")
        print(f"Model version: {mv.version}")

        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=registered_model_name,
            version=mv.version,
            stage="Staging"
        )
        print(f"Model version {mv.version} transitioned to Staging.")
        self.next(self.end)

    @step
    def end(self):
        print('workflow finished')

if __name__ == '__main__':
     TrainingFlow()
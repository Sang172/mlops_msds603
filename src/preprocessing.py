import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import pickle


diabetes = pd.read_csv('data/diabetes_binary.csv')
cols = list(diabetes.columns)
X = diabetes[cols[1:]]
y = diabetes[cols[:1]]
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.25, shuffle=True)

numerical_features = ['BMI', 'MentHlth', 'PhysHlth']
categorical_features = ['GenHlth', 'Age', 'Education', 'Income']

numerical_transformer = Pipeline(steps=[
    ('scaler', MinMaxScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

data_pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

data_pipeline.fit(X_train)

X_train_processed = pd.DataFrame(data_pipeline.transform(X_train), columns=data_pipeline.get_feature_names_out())
X_val_processed = pd.DataFrame(data_pipeline.transform(X_val), columns=data_pipeline.get_feature_names_out())
X_test_processed = pd.DataFrame(data_pipeline.transform(X_test), columns=data_pipeline.get_feature_names_out())

X_train_processed.index = X_train.index
X_val_processed.index = X_val.index
X_test_processed.index = X_test.index

train_processed = pd.concat([y_train.reset_index(drop=True), X_train_processed], axis=1)
val_processed = pd.concat([y_val.reset_index(drop=True), X_val_processed], axis=1)
test_processed = pd.concat([y_test.reset_index(drop=True), X_test_processed], axis=1)

train_processed.to_csv('data/diabetes_train.csv', index=False)
val_processed.to_csv('data/diabetes_val.csv', index=False)
test_processed.to_csv('data/diabetes_test.csv', index=False)

pipeline_filename = 'data/diabetes_preprocessing_pipeline.pkl'
with open(pipeline_filename, 'wb') as file:
    pickle.dump(data_pipeline, file)
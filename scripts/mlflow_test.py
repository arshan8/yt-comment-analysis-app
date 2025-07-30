import mlflow

mlflow.set_tracking_uri("http://13.60.38.114:5000")
mlflow.set_experiment('test-dvc-pipline')

with mlflow.start_run() :
    mlflow.log_param("test_param", "test_value")
    mlflow.log_metric("test_metric", 0.5)   

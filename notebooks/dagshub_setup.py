import dagshub
import mlflow
dagshub.init(repo_owner='onkar-git', repo_name='mlops-emotion-detection-text-classifier', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold, train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from time import time
import numpy as np
import comet_ml
from comet_ml.integration.sklearn import log_model

#create training and testing data
data = load_breast_cancer(as_frame=True)
x_train, x_test, y_train, y_test = train_test_split(
    data["data"], data["target"], test_size=0.2
)

#set-up comet
comet_ml.login()
experiment = comet_ml.start(project_name="ai_pc_demo")
experiment.add_tag('xg')

#set hyper-parameters
hyper_params = {}
hyper_params['learning_rate'] = 0.1
hyper_params['n_estimators'] = 100 
experiment.log_parameters(hyper_params)

#initialize model
model = XGBClassifier(learning_rate=hyper_params['learning_rate'], n_estimators=hyper_params['n_estimators'])

#training
start = time()
model.fit(x_train, y_train)
end = time()
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
score = cross_val_score(model, x_train, y_train, scoring='f1', cv=cv, n_jobs=-1)

#log training metrics 
experiment.log_metric('train_f1', np.mean(score)*100)
experiment.log_metric('train_std', np.std(score))
experiment.log_metric('train_speed', np.round(end - start, 3))

#run and log validation
y_pred = model.predict(x_test)
val_acc = accuracy_score(y_test, y_pred)
experiment.log_metric('val_acc', val_acc)

#log a df to comet for advanced debugging
debug_df = x_test.copy()
debug_df["pred"] = y_pred
debug_df["ground_truth"] = y_test
experiment.log_table("prediction_debug_table.csv", debug_df)

log_model(experiment, "my-xg-model", model)
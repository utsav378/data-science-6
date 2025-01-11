import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import shap

# Load dataset
data = pd.read_csv("healthcare_data.csv")  # Replace with actual path
X = data.drop('outcome', axis=1)
y = data['outcome']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Explain predictions
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Visualize
shap.summary_plot(shap_values, X_test)

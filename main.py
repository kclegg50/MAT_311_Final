import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import math
from sklearn.compose import make_column_transformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay 
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics.pairwise import euclidean_distances
from sklearn import set_config
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
adult = pd.read_csv('adult.data.csv')
adult_income = pd.get_dummies(adult['income'])
adult = pd.concat([adult_income, adult], axis=1)
adult.drop([" <=50K"], axis=1, inplace=True)
sns.countplot(x='income', data=adult)
plt.title('Income Class Distribution')
plt.show()
sns.countplot(x='sex', data=adult, hue='income')
plt.title('Sex Distribution')
sns.countplot(x='race', data=adult, hue='income')
plt.title('Race Distribution')
plt.xticks(rotation=45)
plt.show()
sns.countplot(x='martital_status', data=adult, hue='income')
plt.title('Martital Status Distribution')
plt.xticks(rotation=45)
plt.show()
sns.countplot(x='family_status', data=adult, hue='income')
plt.title('Family Status Distribution')
plt.xticks(rotation=45)
plt.show()
sns.countplot(x='age', data=adult, hue='income')
plt.title('Age Distribution')
plt.xticks(rotation=45)
plt.show()
sns.countplot(x='work_class', data=adult, hue='income')
plt.title('Work Class Distribution')
plt.xticks(rotation=45)
plt.show()
sns.countplot(x='education', data=adult, hue='income')
plt.title('Education Distribution')
plt.xticks(rotation=50)
plt.show()
sns.countplot(x='education_num', data=adult, hue='income')
plt.title('education_num Status Distribution')
plt.xticks(rotation=45)
plt.show()
sns.countplot(data=adult, x='sex', hue=' >50K', palette=['green', 'red'])
plt.title('male and female with >50K')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.legend(title='sex', labels=['Male', 'Female'])
plt.show()
adult.drop(["capital_loss", "capital_gain", "occupation", "fnlwgt", "native_country", "hours_per_week", "work_class"], axis=1, inplace=True)
adult_education = pd.get_dummies(adult['education'])
adult_family_status = pd.get_dummies(adult['family_status'])
adult_race = pd.get_dummies(adult['race'])
adult_sex = pd.get_dummies(adult['sex'])
adult.drop(["education", "family_status", "race", "sex", "martital_status", "income"], axis=1, inplace=True)
adult_complete = pd.concat([adult, adult_education, adult_family_status, adult_race, adult_sex], axis=1)
adult_complete.to_csv('adult_clean.csv', index=False)
adult_complete
X = adult_complete.drop(' >50K', axis=1)
y = adult_complete[' >50K']
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2222, random_state=42)
knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
conf_matrix_knn = confusion_matrix(y_test, y_pred_knn)
sns.heatmap(conf_matrix_knn, annot=True, fmt='d', cmap='Blues', cbar=True)
plt.title('Confusion Matrix - k-NN')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
print(classification_report(y_test, y_pred_knn))
TP1 = conf_matrix_knn[1,1]
TN1 = conf_matrix_knn[0,0]
FP1 = conf_matrix_knn[0,1]
FN1 = conf_matrix_knn[1,0]
specificity1 = (TN1/(FP1+TN1))
specificity1
Bayes = GaussianNB()
Bayes.fit(X_train, y_train)
y_pred_Bayes = Bayes.predict(X_test)
conf_matrix_Bayes = confusion_matrix(y_test, y_pred_Bayes)
sns.heatmap(conf_matrix_Bayes, annot=True, fmt='d', cmap='Reds', cbar=True)
plt.title('Confusion Matrix Bayes')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
print(classification_report(y_test, y_pred_Bayes))
TP2 = conf_matrix_Bayes[1,1]
TN2 = conf_matrix_Bayes[0,0]
FP2 = conf_matrix_Bayes[0,1]
FN2 = conf_matrix_Bayes[1,0]
specificity2 = (TN2/(FP2+TN2))
specificity2
features = ['age', 'education_num', ' 10th', ' 11th', ' 12th', ' 1st-4th',
       ' 5th-6th', ' 7th-8th', ' 9th', ' Assoc-acdm', ' Assoc-voc',
       ' Bachelors', ' Doctorate', ' HS-grad', ' Masters', ' Preschool',
       ' Prof-school', ' Some-college', ' Husband', ' Not-in-family',
       ' Other-relative', ' Own-child', ' Unmarried', ' Wife',
       ' Amer-Indian-Eskimo', ' Asian-Pac-Islander', ' Black', ' Other',
       ' White', ' Female', ' Male']
depth_limit = 10
DT = DecisionTreeClassifier(criterion= 'entropy', max_depth=depth_limit)
DT.fit(X_train[features],y_train)
y_pred_train = DT.predict(X_train[features])
y_pred_test = DT.predict(X_test[features])
plt.figure(figsize=(12,8))
plot_tree(DT, feature_names = features, class_names=['education_num', 'age', ' 10th', ' 11th', ' 12th', ' 1st-4th',
       ' 5th-6th', ' 7th-8th', ' 9th', ' Assoc-acdm', ' Assoc-voc',
       ' Bachelors', ' Doctorate', ' HS-grad', ' Masters', ' Preschool',
       ' Prof-school', ' Some-college', ' Husband', ' Not-in-family',
       ' Other-relative', ' Own-child', ' Unmarried', ' Wife',
       ' Amer-Indian-Eskimo', ' Asian-Pac-Islander', ' Black', ' Other',
       ' White', ' Female', ' Male'], filled=True)
plt.title(f'Decision Tree (Features: {features}, Max Depth: {depth_limit})')
plt.show()
conf_matrix_dt = confusion_matrix(y_test, y_pred_test)
sns.heatmap(conf_matrix_dt, annot=True, fmt='d', cmap='Blues', cbar=True)
plt.title('Confusion Matrix - DT')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
print(classification_report(y_test, y_pred_test))
TP3 = conf_matrix_dt[1,1]
TN3 = conf_matrix_dt[0,0]
FP3 = conf_matrix_dt[0,1]
FN3 = conf_matrix_dt[1,0]
specificity3 = (TN3/(FP3+TN3))
specificity3
y_prob_knn = knn.predict_proba(X_val)[:, 1]  
y_prob_nb = Bayes.predict_proba(X_val)[:, 1]
y_prob_dt = DT.predict_proba(X_val)[:, 1]
roc_curve1 = RocCurveDisplay.from_predictions(y_val, y_prob_dt, name="Tree", color="#00AFCE")
roc_curve2 = RocCurveDisplay.from_predictions(y_val, y_prob_knn, ax=roc_curve1.ax_, name="KNN", color="#522D72")
roc_curve3 = RocCurveDisplay.from_predictions(y_val, y_prob_nb, ax=roc_curve1.ax_, name="NB", color="#E14F3D")
plt.show()

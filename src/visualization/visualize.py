conf_matrix_knn = confusion_matrix(y_test, y_pred_knn)
sns.heatmap(conf_matrix_knn, annot=True, fmt='d', cmap='Blues', cbar=True)
plt.title('Confusion Matrix - k-NN')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

conf_matrix_Bayes = confusion_matrix(y_test, y_pred_Bayes)
sns.heatmap(conf_matrix_Bayes, annot=True, fmt='d', cmap='Reds', cbar=True)
plt.title('Confusion Matrix Bayes')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

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

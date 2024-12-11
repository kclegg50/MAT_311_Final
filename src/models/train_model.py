knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
print(classification_report(y_test, y_pred_knn))

Bayes = GaussianNB()
Bayes.fit(X_train, y_train)
y_pred_Bayes = Bayes.predict(X_test)
print(classification_report(y_test, y_pred_Bayes))

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
print(classification_report(y_test, y_pred_test)

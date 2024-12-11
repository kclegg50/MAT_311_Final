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

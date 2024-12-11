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

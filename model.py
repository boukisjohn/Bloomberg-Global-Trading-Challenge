import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_excel('dataset_clean.xlsx')

last_week = df[df['label'].isna()]

df = df[df['label'].notna()]

feature_cols = ['excess_1w', 'return_4w', 'return_12w',
                'volatility_20d', 'beta_60d', 'cyclicality']

x = df[feature_cols]
y = df['label'].astype(int)

# Split the dataset.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=False)
print(f'x_train shape is: {x_train.shape},\nx_test shape is: {x_test.shape}.')

rf_classifier = RandomForestClassifier(n_estimators=300, # How many trees.
                                       max_depth=3,  # Each tree makes that many questions.
                                       min_samples_leaf=20,  # Every final box should have >= 20 lines
                                       random_state=42)   # Same results each time.

rf_classifier.fit(x_train, y_train)
y_predicted = rf_classifier.predict(x_test)

lr_classifier = LogisticRegression()
lr_classifier.fit(x_train, y_train)
# y_predicted = lr_classifier.predict(x_test)


accuracy = accuracy_score(y_test, y_predicted)
print(f'Model Accuracy: {100*accuracy:.2f}%.')
print(f'Baseline Accuracy: {100*y_test.mean():.2f}%.')
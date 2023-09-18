import pandas as pd
import pickle
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.combine import SMOTEENN
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

df = pd.read_csv('bankdata.csv')
dfEnc = df.copy()

dfEnc["gender"] = LabelEncoder().fit_transform(dfEnc["gender"])
dfEnc["country"] = LabelEncoder().fit_transform(dfEnc["country"])

X = dfEnc.drop(['customer_id', 'churn'], axis=1)
Y = dfEnc['churn']

# X_bal, Y_bal = SMOTEENN().fit_resample(X, Y)
sm = SMOTE(random_state=2)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
X_train, Y_train = sm.fit_resample(X_train, Y_train)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestClassifier()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
print("Classification Report:\n\n", classification_report(Y_test, Y_pred))
print("Confusion Matrix:\n", confusion_matrix(Y_test, Y_pred))

filename = 'model.sav'
pickle.dump(model, open(filename, 'wb'))

# n_estimators=194, min_samples_split=7, min_samples_leaf=2, max_features='sqrt',
# max_depth=50, criterion='gini'

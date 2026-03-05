import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# Load dataset
data = pd.read_csv("creditcard.csv")

# Select features
X = data[['Time','V1','V2','V3','Amount']]
y = data['Class']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

# Train model
model = LogisticRegression()
model.fit(X_train,y_train)

# Save model
pickle.dump(model, open("fraud_model.pkl","wb"))

print("Model saved successfully")
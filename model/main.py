import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle  # Standard pickle module

def get_clean_data():
    data = pd.read_csv("brestdaata.csv")  # Make sure the file is in the correct directory

    data = data.drop(["id"], axis=1)
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})  # Map diagnosis to numeric
    return data

def create_model(data):
    X = data.drop('diagnosis', axis=1)
    y = data['diagnosis']

    # scale it
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # train it
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # test model
    y_pred = model.predict(X_test)
    print('Accuracy of our model: ', accuracy_score(y_test, y_pred))
    print("Classification report: \n", classification_report(y_test, y_pred))

    return model, scaler

def main():
    os.makedirs('model', exist_ok=True)
    data = get_clean_data()  # This will work now as get_clean_data is defined before

    model, scaler = create_model(data)

    # Save model
    with open('model/model.pkl', 'wb') as f:
        pickle.dump(model, f)

    # Save scaler
    with open('model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

if __name__ == '__main__':
    main()

from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv("pima_indians.csv")

inputs = data.drop('Outcome', axis=1)
output = data['Outcome']

X_train, X_test, y_train, y_test = train_test_split(inputs, output, test_size=0.2, random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

def index(request):
    return render(request, 'index.html')

def predict(request):
    if request.method == 'POST':
        features = ['pregnancies', 'glucose', 'bloodpressure', 'skinthickness', 'insulin', 'bmi', 'dpf', 'age']
        data = [float(request.POST[feature]) for feature in features]
        df = pd.DataFrame([data], columns=features)
        
        prediction = model.predict(df)[0]
        result = "Diabetic" if prediction == 1 else "Not Diabetic"
        return HttpResponse(f'The predicted outcome is: {result}')
    else:
        return HttpResponse("Invalid request")

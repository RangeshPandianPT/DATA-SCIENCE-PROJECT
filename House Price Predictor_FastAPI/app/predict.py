import joblib
from app.utils import preprocess

model = joblib.load("model/model.pkl")

def make_prediction(features):
    data = preprocess(features)
    prediction = model.predict([data])
    return prediction[0]

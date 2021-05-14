from fastapi import FastAPI
import uvicorn
import pandas as pd
import joblib
from pythainlp.ulmfit import process_thai
import numpy as np


MODELS_PATH = "joblib_lgr.pkl"
TFIDF = joblib.load("tfidf.pickle")
SCALER = joblib.load("scaler.pickle")


def load_model():
    return joblib.load(MODELS_PATH)

LOADED_MODEL = load_model()

app = FastAPI(title="Thai Sentiment Analysis", description="API to predict sentiment on Thai text")

def make_inference_df(input_text):
    
    # create df
    ui = pd.DataFrame({"texts": str(input_text)}, index=[0])
    # preprocess
    ui["processed"] = "|".join(process_thai(str(input_text)))
    ui["wc"] =  ui.processed.map(lambda x: len(x.split("|")))
    ui["uwc"] = ui.processed.map(lambda x: len(set(x.split("|"))))
    # transform
    tfidf = TFIDF.transform(ui["texts"])
    scaler = SCALER.transform(ui[["wc","uwc"]].astype(float))
    tfidf_user = pd.DataFrame(tfidf.toarray())
    scaler_user = pd.DataFrame(scaler)
    user_texts = pd.concat([scaler_user, tfidf_user], axis=1)
    return user_texts

@app.get("/")
def read_root():

    return {"message": "Welcome to the API"}

@app.post("/predict")
def predict(text: str):

    model_input_df = make_inference_df(text)
    prediction = LOADED_MODEL.predict_proba(model_input_df)

    neg_probability = prediction[0][0]
    neu_probability = prediction[0][1]
    pos_probability = prediction[0][2]
    q_probability = prediction[0][3]
    p = np.argmax(prediction, axis=1)
    
    return {
        
        "Text" : str(text),
        "prediction": 'neg' if p==0 else ('neu' if p==1 else ('pos' if p==2 else 'q')),
        "neg" : round(float(neg_probability),2),
        "neu" : round(float(neu_probability),2),
        "pos" : round(float(pos_probability),2),
        "q" : round(float(q_probability),2)

    }

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

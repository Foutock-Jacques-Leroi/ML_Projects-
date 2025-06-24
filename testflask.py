from flask import Flask, request
from pydantic import BaseModel
import pandas as pd




app = Flask(__name__)

class critics_classifier(BaseModel):
    text: str

@app.route("/", methods=["GET"])
def critics():
    return "hello world"

@app.route('/predict', methods=["POST"])
def prediction():
    data = critics_classifier(**request.json)
    data_dict = pd.DataFrame([data.dict()])
    print(data_dict.model_dump())
    print(data.model_dump())
    return data_dict

if __name__ == "__main__":
    app.run(debug=True)
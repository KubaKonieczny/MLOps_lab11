from tokenizers import Tokenizer
import onnxruntime as ort
from fastapi import FastAPI
import numpy as np
from mangum import Mangum
from .api_model import PredictRequest, PredictResponse

app = FastAPI()

tokenizer = Tokenizer.from_file("./model/tokenizer.json") 
ort_session = ort.InferenceSession("./model/embedding_model.onnx") 
ort_classifier = ort.InferenceSession("./model/classifier.onnx")

SENTIMENT_MAP = {0: "negative", 1: "neutral", 2: "positive"}



@app.post("/predict")
def predict(request: PredictRequest) -> PredictResponse:
    # tokenize input
    cleaned_text = request.text
    encoded = tokenizer.encode(cleaned_text)

    # prepare numpy arrays for ONNX
    input_ids = np.array([encoded.ids])
    attention_mask = np.array([encoded.attention_mask])

    # run embedding inference
    embedding_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
    embeddings = ort_session.run(None, embedding_inputs)[0]

    # run classifier inference
    classifier_input_name = ort_classifier.get_inputs()[0].name
    classifier_inputs = {classifier_input_name: embeddings.astype(np.float32)}
    prediction = ort_classifier.run(None, classifier_inputs)[0]

    label = SENTIMENT_MAP.get(prediction[0], "unknown") # return this label as response

    return PredictResponse(prediction=label)

handler = Mangum(app)
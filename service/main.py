import os

from fastapi import FastAPI
import numpy as np
import uvicorn
from pydantic import BaseModel
import pandas as pd
from onnxruntime import InferenceSession, SessionOptions


def createInferenceSession(
    model_path,
    intra_op_num_threads=1,
    provider='CPUExecutionProvider'
):

    options = SessionOptions()
    options.intra_op_num_threads = intra_op_num_threads

    # load the model as a onnx graph
    session = InferenceSession(model_path, options, providers=[provider])
    session.disable_fallback()
    return session

def getDialectsPrediction(input, session, tokenizer):
    LABEL_COLUMNS = ['IQ', 'BH', 'LY']
    encoding = tokenizer.encode_plus(
        input,
        add_special_tokens=True,
        max_length=35,
        return_token_type_ids=True,
        padding=False,
        return_attention_mask=True,
        truncation=True,
        return_tensors='pt',
    )
    inputs_onnx = {k: v.cpu().detach().numpy() for k, v in encoding.items()}
    del inputs_onnx['token_type_ids']
    ort_outs = session.run(["output"], inputs_onnx)
    torch_output = torch.tensor(ort_outs[0], dtype=torch.float32)
    logits = torch_output.detach().numpy()

    predictions = dict()
    for label, prediction in zip(LABEL_COLUMNS, logits[0]):
        predictions[label] = round(prediction, 4)

    del encoding

    return predictions

app = FastAPI(title="Dialects Classification API", description="API to predict the text Dialects")
dialectsSession = createInferenceSession('model/dialects_model.onnx')


class Data(BaseModel):
    text:str

@app.get("/")
def read_root():
    return {"message": "Welcome from the API"}

@app.post("/predict")
def predict(data:Data):
    dialectsPredictions = getDialectsPrediction(text, dialectsSession, tokenizer)
    dialectspredictions = {k: float("{:.4f}".format(v)) for k, v in sorted(dialectsPredictions.items(), key=lambda item: item[1])}
    return {
        "text" : data.text,
        "prediction": dialectspredictions
    }


if __name__ == '__main__':
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
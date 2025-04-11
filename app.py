
from fastapi import UploadFile, File, Form
from starlette.middleware.cors import CORSMiddleware
from generate_chain import generate_chain
from disease_prediction_leaf import LeafDiseaseDetection
from Model import  ChatRequest
import os
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from Model import PredictionRequest
from WaterPurposeModel import WaterPurposeModel


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
chain = generate_chain()
leafDiseaseDetection = LeafDiseaseDetection()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/predict")
async def predict(data: PredictionRequest):
    try:
        parameters = [
            data.PH,
            data.EC,
            data.ORP,
            data.DO,
            data.TDS,
            data.TSS,
            data.TS,
            data.TOTAL_N,
            data.NH4_N,
            data.TOTAL_P,
            data.PO4_P,
            data.COD,
            data.BOD,
        ]

        if None in parameters:
            raise HTTPException(status_code=400, detail="Missing parameters in input data")

        model = WaterPurposeModel()
        result = model.predict(parameters)
        return {"prediction": result}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing input data: {str(e)}")



@app.post("/chatbot")
async def agri_chatbot(chat_request : ChatRequest):
    question = chat_request.question
    language = chat_request.language
    response = chain.invoke(question)
    result = leafDiseaseDetection.convert_text(response , language=language)
    return JSONResponse(content={
        "answer" : result
        }
    )


@app.post("/disease")
async def detect_disease(image: UploadFile = File(...), lang: str = Form("hi")):
    contents = await image.read()
    disease = leafDiseaseDetection.predict_leaf_disease(contents)
    information = leafDiseaseDetection.find_information(disease, lang)
    return JSONResponse(content={
        "diseaseInfo" : information
        }
    )

if __name__ == "__main__":
    config = uvicorn.Config("app:app", port=5000, log_level="info")
    server = uvicorn.Server(config)
    server.run()
from fastapi import FastAPI, HTTPException
from app import model, mongodb
from app.cpsModel import load_cps_model_if_exists, predict_cps
from app.schemas import Headline, Feature, FlowerSpecies
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        model.load_model_if_exists()
        load_cps_model_if_exists()
        print("Models loaded at startup.")
    except Exception as e:
        print(f"Model not loaded at startup: {e}")
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/")
def root():
    return {"message": "ML Service is running!"}

@app.post("/data")
def add_iris_sample(sample: FlowerSpecies):
    mongodb.insert_sample(sample.model_dump())
    return {"message": "Sample added successfully."}

@app.get("/data")
def get_iris_samples(species: str = None):
    return mongodb.get_samples(species)

@app.post("/train")
def train_model_for_iris():
    samples = mongodb.get_samples()
    if not samples:
        raise HTTPException(status_code=400, detail="No data available to train the model.")
    model.train(samples)
    return {"message": f"Model trained successfully on {len(samples)} samples."}

@app.post("/predict")
def predict_iris(input_data: Feature):
    try:
        features = input_data.model_dump()
        prediction = model.predict(features)
        return {"species": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-cps")
def predict_cps_for_text(input_data: Headline):
    text = input_data.headline
    if not text:
        raise HTTPException(status_code=400, detail="Headline text is required.")
    try:
        prediction = predict_cps(text)
        return {"predicted_category": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

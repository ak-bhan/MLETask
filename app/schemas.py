from pydantic import BaseModel

class Feature(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class FlowerSpecies(Feature):
    species: str

class Headline(BaseModel):
    headline: str

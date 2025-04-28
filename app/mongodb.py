import os
from pymongo import MongoClient

env = os.getenv("ENV", "dev")

if env == "test":
    mongo_uri = "mongodb://localhost:2025"
else:
    mongo_uri = os.getenv("MONGO_URI", "mongodb://mongo:27017")

client = MongoClient(mongo_uri)
db = client.iris_dataset

def insert_sample(sample):
    db.samples.insert_one(sample)

def get_samples(species=None):
    query = {"species": species} if species else {}
    samples = list(db.samples.find(query, {"_id": 0}))
    return samples

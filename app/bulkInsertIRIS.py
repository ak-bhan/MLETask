import csv
from pymongo import MongoClient

client = MongoClient("mongodb://localhost:2025")
db = client.iris_dataset
collection = db.samples

csv_file_path = "data/iris.csv"

def bulk_insert():
    documents = []
    with open(csv_file_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            doc = {
                "sepal_length": float(row["sepal_length"]),
                "sepal_width": float(row["sepal_width"]),
                "petal_length": float(row["petal_length"]),
                "petal_width": float(row["petal_width"]),
                "species": row["species"]
            }
            documents.append(doc)

    if documents:
        collection.delete_many({})
        collection.insert_many(documents)
        print(f"Inserted {len(documents)} records into MongoDB!")
    else:
        print("No documents found to insert.")

if __name__ == "__main__":
    bulk_insert()

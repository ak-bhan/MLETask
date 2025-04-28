from app.cpsModel import train_cps_model_from_big_file

if __name__ == "__main__":
    big_json_path = "data/CPS_use_case_classification/CPS_use_case_classification_training.json"
    train_cps_model_from_big_file(big_json_path)

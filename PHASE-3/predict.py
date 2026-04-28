

import pandas as pd
import pickle

def predict_patient(data_dict):
    import pandas as pd
    import pickle

    scaler = pickle.load(open("model/scaler.pkl", "rb"))
    selector = pickle.load(open("model/selector.pkl", "rb"))
    model = pickle.load(open("model/logistic_model.pkl", "rb"))

    all_features = pickle.load(open("model/all_features.pkl", "rb"))
    selected_features = pickle.load(open("model/selected_features.pkl", "rb"))

    df = pd.DataFrame([data_dict])

    # Step 1: match ALL features (15)
    df = df.reindex(columns=all_features, fill_value=0)

    # Step 2: scale
    scaled = scaler.transform(df)

    # Step 3: select top 10
    selected = selector.transform(scaled)

    # Step 4: predict
    pred = model.predict(selected)[0]
    prob = model.predict_proba(selected)[0][1]

    return pred, prob
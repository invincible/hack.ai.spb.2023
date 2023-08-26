from fastapi import FastAPI

import os
import pandas as pd
from transliterate import translit
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import QuantileTransformer
from scipy.sparse import csr_matrix, hstack
from sklearn.neighbors import NearestNeighbors

import json
import warnings
warnings.filterwarnings('ignore')


add_data = "/code/app/additional_data"

df_building = pd.read_csv(f"{add_data}/building_20230808.csv", low_memory=False)
df_building = df_building.rename(columns={"id": "target_building_id"})

path = "/code/app/datasets/"
files = os.listdir(path)

df = []
for i, file in enumerate(files):
    if i == 0:
        df = pd.read_csv(path + file, low_memory=False)
    else:
        temp = pd.read_csv(path + file, low_memory=False)
        df = pd.concat([df, temp], ignore_index=True)

use = ["address", "target_address"]
df.drop_duplicates(subset=use, inplace=True)
df.drop_duplicates(subset="address", inplace=True)

df = df.merge(df_building, how="left", on=["target_building_id"])

df = df[df["is_actual"] == True]
df["target_building_id"] = df["target_building_id"].astype(int)

df["address"] = df["address"].apply(lambda x: translit(x, "ru"))

count_char = CountVectorizer(analyzer='char_wb', ngram_range=(1, 2), token_pattern='\\w+', max_df=0.9)
count_ngram = CountVectorizer(ngram_range=(1, 1), token_pattern='\\w+', max_df=0.9)

char_csr = count_char.fit_transform(df['target_address'])
ngram_csr = count_ngram.fit_transform(df['target_address'])

csr = hstack([char_csr, ngram_csr])
sca = QuantileTransformer()
csr = sca.fit_transform(csr)

neigh = NearestNeighbors(n_neighbors=10, metric='cosine')
neigh.fit(csr)

app = FastAPI()

def parse_csv(df):
    res = df.to_json(orient="records")
    parsed = json.loads(res)
    return parsed

@app.get("/search/{address}")
async def search_addr(address: str):

    if len(address)<4:
        return "Sorry! I need more letters!"

    char_csr_test = count_char.transform([address])
    ngram_csr_test = count_ngram.transform([address])
    csr_test = hstack([char_csr_test, ngram_csr_test])
    csr_test = sca.transform(csr_test)

    PREDICT_SIZE = 10
    pred = neigh.kneighbors(csr_test[0], PREDICT_SIZE, return_distance=False)

    y_pred = df["target_address"].iloc[pred[0].flatten()].drop_duplicates()

    return parse_csv(y_pred)
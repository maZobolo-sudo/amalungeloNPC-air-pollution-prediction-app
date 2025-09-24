import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
FEATURES = ["PM10","SO2","NO2","O3","CO","Temp","RH","Wind"]
TARGET = "PM25_next"
def train(df: pd.DataFrame):
    X, y = df[FEATURES], df[TARGET]
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42)
    m = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1).fit(Xtr,ytr)
    pred = m.predict(Xte); mae = mean_absolute_error(yte, pred); r2 = r2_score(yte, pred)
    return m, {"mae": float(mae), "r2": float(r2)}
def forecast(m, df: pd.DataFrame):
    return m.predict(df[FEATURES])

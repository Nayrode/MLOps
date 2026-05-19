"""FastAPI inference service — POST /predict {team1, team2, map}"""
import sys
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "01_feature_engineering"))
from feature_engineering import build_feature_state, compute_features_for_match, update_state_with_result

MAPS = ["Cache", "Cobblestone", "Default", "Dust2",
        "Inferno", "Mirage", "Nuke", "Overpass", "Train", "Vertigo"]

app = FastAPI(title="CS:GO Match Predictor")

# Load once at startup
history = pd.read_csv(ROOT / "data/results.csv", parse_dates=["date"])
pipeline = joblib.load(ROOT / "data/model.pkl")

state = build_feature_state(history)
for _, m in history.sort_values("date").iterrows():
    update_state_with_result(m, state)


def latest_rank(df: pd.DataFrame, team: str) -> int:
    rows = df[(df["team_1"] == team) | (df["team_2"] == team)].sort_values("date")
    if rows.empty:
        return 50
    last = rows.iloc[-1]
    return int(last["rank_1"] if last["team_1"] == team else last["rank_2"])


class PredictRequest(BaseModel):
    team1: str
    team2: str
    map: str


class PredictResponse(BaseModel):
    team1: str
    team2: str
    map: str
    winner: str
    loser: str
    confidence: float
    team1_win_probability: float


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if req.map not in MAPS:
        raise HTTPException(status_code=422, detail=f"Unknown map '{req.map}'. Valid: {MAPS}")

    rank1 = latest_rank(history, req.team1)
    rank2 = latest_rank(history, req.team2)

    match = {"team_1": req.team1, "team_2": req.team2,
             "rank_1": rank1, "rank_2": rank2, "_map": req.map}

    features = compute_features_for_match(match, state)
    features["_map"] = req.map
    X = pd.DataFrame([features])

    proba = float(pipeline.predict_proba(X)[0][1])
    team1_wins = proba >= 0.5
    winner = req.team1 if team1_wins else req.team2
    loser  = req.team2 if team1_wins else req.team1
    confidence = proba if team1_wins else 1 - proba

    return PredictResponse(
        team1=req.team1,
        team2=req.team2,
        map=req.map,
        winner=winner,
        loser=loser,
        confidence=round(confidence, 4),
        team1_win_probability=round(proba, 4),
    )


@app.get("/health")
def health():
    return {"status": "ok", "teams_in_history": len(state["elo"])}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

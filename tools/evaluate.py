"""Offline metric inspection of an already frozen prediction CSV.

No model, GPU or raw Test H5 is accessed. This tool evaluates only a supplied prediction CSV.
"""
from pathlib import Path
import argparse, json, sys
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parent))
from metrics import calculate

def main():
    p=argparse.ArgumentParser()
    p.add_argument('csv',type=Path)
    p.add_argument('--label-column',default='agb_true')
    p.add_argument('--prediction-column',required=True)
    p.add_argument('--id-column',default=None)
    a=p.parse_args()
    df=pd.read_csv(a.csv)
    if a.id_column:
        ids=df[a.id_column]
        if ids.isna().any() or not ids.is_unique: raise ValueError('missing/duplicate sample identity')
    result=calculate(df[a.label_column].to_numpy(np.float64),df[a.prediction_column].to_numpy(np.float64))
    print(json.dumps(result,indent=2,allow_nan=False))

if __name__=='__main__': main()

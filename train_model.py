# train_model.py
import os, joblib, pandas as pd, numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score

def load_and_merge(data_dir):
    orders = pd.read_csv(os.path.join(data_dir, "orders.csv"))
    delivery = pd.read_csv(os.path.join(data_dir, "delivery_performance.csv"))
    routes = pd.read_csv(os.path.join(data_dir, "routes_distance.csv"))
    cost = pd.read_csv(os.path.join(data_dir, "cost_breakdown.csv"))
    feedback = pd.read_csv(os.path.join(data_dir, "customer_feedback.csv"))

    df = orders.merge(delivery, on='Order_ID', how='left')
    df = df.merge(routes[['Order_ID','Distance_KM','Traffic_Delay_Minutes','Weather_Impact']], on='Order_ID', how='left')
    df = df.merge(cost, on='Order_ID', how='left')
    df = df.merge(feedback[['Order_ID','Rating']], on='Order_ID', how='left')
    return df

def prepare(df):
    # Target: delayed if Actual > Promised (days)
    df['delayed'] = (pd.to_numeric(df['Actual_Delivery_Days'], errors='coerce') >
                     pd.to_numeric(df['Promised_Delivery_Days'], errors='coerce')).astype(int)

    # numeric and categorical features chosen based on your CSVs
    num_feats = ['Promised_Delivery_Days','Actual_Delivery_Days','Distance_KM','Traffic_Delay_Minutes',
                 'Fuel_Cost','Labor_Cost','Vehicle_Maintenance','Packaging_Cost','Order_Value_INR','Rating']
    num_feats = [c for c in num_feats if c in df.columns]
    cat_feats = ['Priority','Product_Category','Origin','Destination','Customer_Segment','Carrier']
    cat_feats = [c for c in cat_feats if c in df.columns]

    for c in num_feats:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    df_model = df[~df['delayed'].isnull()].dropna(subset=num_feats, how='all').copy()

    X = df_model[num_feats + cat_feats].copy()
    y = df_model['delayed'].astype(int)
    return X, y, num_feats, cat_feats

def train(data_dir="data", model_out="models/delivery_model.joblib"):
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    df = load_and_merge(data_dir)
    X, y, num_feats, cat_feats = prepare(df)

    # split
    if len(y.unique()) > 1 and min(y.value_counts()) >= 2:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    else:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    num_transform = ('num', SimpleImputer(strategy='median'), num_feats)
    # NOTE: use sparse_output=False for newer scikit-learn versions
    cat_transform = ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_feats)
    from sklearn.compose import ColumnTransformer
    pre = ColumnTransformer([num_transform, cat_transform], remainder='drop')

    clf = RandomForestClassifier(n_estimators=300, class_weight='balanced' if len(y.unique())>1 else None, random_state=42, n_jobs=-1)
    pipe = Pipeline([('pre', pre), ('clf', clf)])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_val)
    try:
        y_proba = pipe.predict_proba(X_val)[:,1]
        auc = roc_auc_score(y_val, y_proba) if len(y_val.unique())>1 else None
    except:
        auc = None

    print("Rows used:", X.shape[0])
    print("Class distribution:\n", y.value_counts())
    print("ROC AUC:", auc)
    print("Classification report:\n", classification_report(y_val, y_pred))

    joblib.dump(pipe, model_out)
    print("Saved model to", model_out)

if __name__ == "__main__":
    import sys
    data_dir = "data" if len(sys.argv) < 2 else sys.argv[1]
    train(data_dir)

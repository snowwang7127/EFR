# Fixed Decision Study estimators. Input pipelines and hyperparameters reproduced verbatim.
import numpy as np
from scipy.special import logit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
CLOCK=['approval_hour','approval_weekday','approval_month','approval_hours']
STATIC=['item_value','freight','freight_share','lines','products','sellers','categories','weight_total','weight_max','volume_total','category','customer_state','seller_state']
CAT=['category','customer_state','seller_state']
def fit_model(name,cols,train):
    numeric=[c for c in cols if c not in CAT]
    cats=[c for c in cols if c in CAT]
    stages=[('n',make_pipeline(SimpleImputer(strategy='median'),StandardScaler()) if name=='LR' else 'passthrough',numeric)]
    if cats:stages.append(('c',OneHotEncoder(handle_unknown='ignore',sparse_output=False),cats))
    prep=ColumnTransformer(stages)
    kw=dict(random_state=417)
    if name=='LR': model=LogisticRegression(C=.25,max_iter=600,**kw)
    elif name=='CatBoost':model=CatBoostClassifier(iterations=128,depth=5,learning_rate=.065,thread_count=6,allow_writing_files=False,verbose=False,random_seed=417)
    else:model=LGBMClassifier(n_estimators=128,num_leaves=15,learning_rate=.065,n_jobs=6,verbosity=-1,force_col_wise=True,deterministic=True,**kw)
    x=prep.fit_transform(train[cols]).astype('float32')
    model.fit(x,train.y.astype(int))
    return prep,model
def score(prep,model,frame,cols):
    x=prep.transform(frame[cols]).astype('float32')
    return model.booster_.predict(x,num_threads=6) if isinstance(model,LGBMClassifier) else model.predict_proba(x)[:,1].astype(float)
def lp(p):return logit(np.clip(p,1e-12,1-1e-12)).reshape(-1,1)


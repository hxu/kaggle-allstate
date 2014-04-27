import pandas as pd
from sklearn_pandas import DataFrameMapper
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC,LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer,LabelEncoder
from classes import *
import pdb
class SKLmapper():
    def __init__(self,transformdict):
        self.transforms=transformdict
        return
    def fit(self,X,y=None):
        for TF in self.transforms.keys():
            self.transforms[TF].fit(X[TF])
        #print self.transforms.keys()
        return
    def transform(self,X,y=None):
        TFlist=[]
        for TF in self.transforms.keys():
            print TF
            TFlist.append(self.transforms[TF].transform(X[TF]))
            print self.transforms[TF].transform(X[TF]).shape
        return(np.hstack(TFlist))
    def fit_transform(self,X,y=None):
        TFlist=[]
        for TF in self.transforms.keys():
            self.transforms[TF].fit(X[TF])
            TFlist.append(self.transforms[TF].transform(X[TF]))
        return(np.hstack(TFlist))
class LabelFixerMixin(object):
    """
    Fix signatures on LabelEncoder and LabelBinarizer
    This enables the classes to be used in a Pipeline
    """
    def fit(self, X, y=None):
        return super(LabelFixerMixin, self).fit(X)

    def transform(self, X, y=None):
        return super(LabelFixerMixin, self).transform(X)

    def fit_transform(self, X, y=None):
        return super(LabelFixerMixin, self).fit_transform(X)
class FixLabelEncoder(LabelFixerMixin, LabelEncoder):
    pass

class FixLabelBinarizer(LabelFixerMixin, LabelBinarizer):
    pass

class EncoderBinarizer(Pipeline):
    """
    Transforms the state into binary columns

    Basically the same as the encode_cat, but I think conforms more to the
    Scikit API
    """
    def __init__(self):
        super(EncoderBinarizer, self).__init__([
            ('encode', FixLabelEncoder()),
            ('binarize', FixLabelBinarizer())
        ])
def fill_na(x):
    if x.dtype !='O':
        return(x.fillna(-1))
    else:
        return(x.fillna("missing"))
class returnself():
    def __init__(self):
        return
    def fit(self,X,y=None):
        return
    def transform(self,X):
        return(np.reshape(X,(X.shape[0],1)))
    def fit_transform(self,X,y=None):
        return(X)

data=pd.read_csv("rolled_up2.csv")
print data.keys()
rep_col=[]
#for i in data.columns:
 #   rep_col.append(str.strip().strip(['u']))
#data.columns=rep_col
#pdb.set_trace()
#print data['customer_ID']
merge_actual=pd.read_csv("train.csv")
print "merging relevant data"
merge_actual=get_actual_plan(merge_actual)
#merge_actual=get_actual(merge_actual)
#print merge_actual.keys()                                                                                                                                                                
print "filling nas"
data=data.apply(fill_na)
data=data.merge(merge_actual[['customer_ID','plan']],on='customer_ID',how='left')
#train,test=train_test_split(data,.3)
def mapper_purchases(dic=False):
    if not dic:
        dfm={}
        for i in range(1,13):
            for option in ['A','B','C','D','E','F','state']:
                dfm[option+str(i)]=EncoderBinarizer()
            for option in ['location','car_age','risk_factor','C_previous','married_couple','homeowner']:
                dfm[option+str(i)]=LabelBinarizer()
            for option in ['age_oldest','age_youngest','car_value','car_age','cost']:
                dfm[option+str(i)]=returnself()
        return(SKLmapper(dfm))
    else:
        dfm={}
        for i in range(1,13):
            for option in ['A','B','C','D','E','F']:
                dfm[option+str(i)]=encode_cat()
        return(SKLmapper(dfm))
DFM=mapper_purchases()
data['accurate']=0
data['accurate'][data['plan']==data['last_plan']]=1
print data['accurate'].mean()
def test_algorithms(data,DFM):
    train,test=train_test_split(data,.3)
    #pdb.set_trace()
    DFM.fit(data)
    tr=DFM.transform(train)
    t=DFM.transform(test)
    print "logistic regression"
    for penalty in ['l1','l2']:
        print penalty
        for c in [.01,.5,1,1.5]:
            print c
            CLF=LogisticRegression(penalty=penalty,C=c)
            CLF.fit(tr,train['accurate'])
            test['predict']=CLF.predict(t)
            test['accuracy']=0
            test['accuracy'][test['predict']==test['accurate']]=1
            print test['accuracy'].mean()
    print "GBC"
    for l_rate in [.01,.1,.4]:
        print l_rate
        for max_depth in [3,6]:
            print max_depth
            for min_samples_split in [2,10]:
                print min_samples_split
                CLF=GradientBoostingClassifier(learning_rate=l_rate,max_depth=max_depth,min_samples_split=min_samples_split)
                CLF.fit(tr,train['accurate'])
                test['predict']=CLF.predict(t)
                test['accuracy']=0
                test['accuracy'][test['predict']==test['accurate']]=1
                print test['accuracy'].mean()
    print "Linear_SVC"
    for loss in ['l1','l2']:
        print loss
        for C in [.01,.5,1,1.5]:
            print C
            CLF=LinearSVC(penalty=loss,C=C)
            CLF.fit(tr,train['accurate'])
            test['predict']=CLF.predict(t)
            test['accuracy']=0
            test['accuracy'][test['predict']==test['accurate']]=1
            print test['accuracy'].mean()
    return
test_algorithms(data,DFM)
print "done"

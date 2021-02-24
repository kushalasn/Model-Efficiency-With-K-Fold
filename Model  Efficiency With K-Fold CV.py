# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from  sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.datasets import load_digits
digits=load_digits()


# %%
from sklearn.model_selection import train_test_split


# %%
X_train,X_test,y_train,y_test=train_test_split(digits.data,digits.target,test_size=0.2)


# %%
lr=LogisticRegression()
lr.fit(X_train,y_train)
lr.score(X_test,y_test)


# %%
svm=SVC()
svm.fit(X_train,y_train)
svm.score(X_test,y_test)


# %%
rf=RandomForestClassifier()
rf.fit(X_train,y_train)
rf.score(X_test,y_test)


# %%
from sklearn.model_selection import KFold
kf=KFold(n_splits=3)


# %%
kf


# %%
def get_score(model,X_train,X_test,y_train,y_test):
    model.fit(X_train,y_train)
    return model.score(X_test,y_test)


# %%
score_lr=[]
score_svm=[]
score_rf=[]

for train_index,test_index in kf.split(digits.data):
    X_train,X_test,y_train,y_test=digits.data[train_index],digits.data[test_index],digits.target[train_index],digits.target[test_index]
    score_lr.append(get_score(LogisticRegression(),X_train,X_test,y_train,y_test))
    score_svm.append(get_score(SVC(),X_train,X_test,y_train,y_test))
    score_rf.append(get_score(RandomForestClassifier(),X_train,X_test,y_train,y_test))
    
    


# %%
max(score_lr)


    
    


# %%
score_svm


# %%
score_rf


# %%
from sklearn.model_selection import cross_val_score
max(cross_val_score(LogisticRegression(),digits.data,digits.target))


# %%
max(cross_val_score(SVC(),digits.data,digits.target))


# %%
max(cross_val_score(RandomForestClassifier(),digits.data,digits.target))


# %%
print("SVC Model worked the best!")


# %%




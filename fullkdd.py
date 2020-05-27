import pandas as pd
import numpy
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, zero_one_loss, classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.datasets import fetch_kddcup99
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV


names = ["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate",'target']


data = pd.read_csv('~/Code/ISD/KD99/kddcup.csv', names=names)

print('Transforming Data')
target = data['target']
data.drop(['target'], axis=1, inplace=True)

data = pd.concat([data,pd.get_dummies(data['protocol_type'], prefix='type')],axis=1)
data.drop(['protocol_type'],axis=1, inplace=True)
data = pd.concat([data,pd.get_dummies(data['service'], prefix='service')],axis=1)
data.drop(['service'],axis=1, inplace=True)
data = pd.concat([data,pd.get_dummies(data['flag'], prefix='flag')],axis=1)
data.drop(['flag'],axis=1, inplace=True)

# Change labels to dos, u2r, r2l, probe
print('Categories attacks into 5 categories')
dos = ["back", "land", "neptune", "pod", 'smurf', 'teardrop']
u2r = ['buffer_overflow', 'loadmodule', 'perl', 'rootkit']
r2l = ['ftp_write', 'guess_passwd', 'imap', 'multihop','phf','spy','warezclient','warezmaster']
probe = ['ipsweep','nmap','portsweep','satan']

categorie = []

for label in target:
    if any(x in str(label) for x in dos):
        categorie.append("dos")
    elif any(x in str(label) for x in u2r):
        categorie.append('u2r')
    elif any(x in str(label) for x in r2l):
        categorie.append('r2l')
    elif any(x in str(label) for x in probe):
        categorie.append('probe')
    else:
        categorie.append('normal')

le = LabelEncoder()
le.fit(categorie)
y = le.transform(categorie)

print(le.classes_)

print('Training..')
X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.3)

clf = RandomForestClassifier(verbose=1, n_jobs=-1, class_weight="balanced", n_estimators=500,
                                max_features='auto',
                                criterion='entropy', max_depth=4) # best results so far
# param_grid = { 
#     # 'n_estimators': [200, 500],
#      'n_estimators': [200],
#     # 'max_features': ['auto', 'sqrt', 'log2'],
#     'max_features': ['auto'],
#     # 'max_depth' : [4,5,6,7,8],
#     'max_depth' : [4],
#     'criterion' :['gini', 'entropy']
#     # 'criterion' :['gini']
# }
# CV_rfc = GridSearchCV(estimator=clf, param_grid=param_grid, cv= 5)
# CV_rfc.fit(X_train, y_train)
clf.fit(X_train, y_train)
# print(CV_rfc.best_params_)

y_pred = clf.predict(X_test)
print ("Score: ", clf.score(X_train, y_train))

print ("Computing performance metrics")
results = confusion_matrix(y_test, y_pred)
report =classification_report(y_test, y_pred, target_names=le.classes_)
error = zero_one_loss(y_test, y_pred)
print ("Confusion matrix:\n", results)
print ("Report :\n", report)
print ("Error: ", error)
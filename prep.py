import pandas as pd
#to remove stop words
import nltk
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("punkt_tab")
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download("wordnet")
nltk.download("omw-1.4")
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import TfidfVectorizer
#To divide the dataset into train and test datasets
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack
#Train the model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report)




#Read the csv files
df_fake= pd.read_csv("Fake.csv")
df_true= pd.read_csv("True.csv")

print(" Fake  Dataset")

#Print out the headers
print(df_fake.head())
print("True  Dataset ")
print(df_true.head())

#Add a label of either true or false
df_fake["label"]=0
df_true["label"]=1


#Merge the two files 
df=pd.concat([df_fake,df_true], ignore_index=True)

#Shuffle then so they're in a random order
df= shuffle(df,random_state=42).reset_index(drop=True)

print(df.head())

print("***************")

#combine title and text     
df['combined'] = df['title'] + ' ' + df['text']

#Convert it into lowercase for standardization
df["combined"]=df["combined"].str.lower()

#Drop rows with true NaNs in title or text
cols=['combined','subject','date']
blank=set()
for col in cols:
    for index,text in df[col].items():
         if isinstance(text, str) and text.strip() == "":
            blank.add(index)
df.drop(blank,inplace=True)


#nlp= spacy.load("en_core_web_sm")



#Remove stopwords

stop_words_nltk=set(stopwords.words("english"))

#Lemmitize the data
#Instantiate the Lemmatizer
l=WordNetLemmatizer()


#A function to tokenize the string

def remove_stop_words_lemmatize(text):
    word= word_tokenize(text)
    filtered= []
    for w in word:
        if w not in stop_words_nltk:
            filtered.append(w)
    lemmas=[l.lemmatize(word)for word in filtered]
    return " ".join(lemmas) 
#Use the stop_word function

for col in cols:
    df[col]=df[col].apply(remove_stop_words_lemmatize)

print("******************")
print(df.head())
print("******************")
print(df["combined"].head())
print(df["label"].head())


#Cleaned Dataset
df_clean=df[['combined','subject','date','label']]
df_clean.to_csv("Merged.csv",index=False)


#One code encoding

#To divide the dataset into train and test datasets
#x_train,x_test,y_train

df_train, df_test = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

#TfidVectorize
v_com=TfidfVectorizer(max_features=5)
x_train_comb=v_com.fit_transform(df_train['combined'])
x_test_comb=v_com.transform(df_test['combined'])


#v_sub=TfidfVectorizer()
#x_sub=v_sub.fit_transform(df['subject'])

v_date=TfidfVectorizer(max_features=20)
x_train_date=v_date.fit_transform(df_train['date'])
x_test_date=v_date.transform(df_test['date'])


x_train=hstack([x_train_comb,x_train_date])
x_test=hstack([x_test_comb,x_test_date])

y_train=df_train['label'].values
y_test=df_test['label'].values



print("X_train shape: ", x_train.shape)
print("X_test shape: ", x_test.shape)

#Create an object 
model=LogisticRegression(max_iter=5000,random_state=42)
#Train your model
model.fit(x_train,y_train)

#Predict the values on the test data.
#Returns a 1D array of the same length as x_test. each entry is either 1 or 0
y_pred= model.predict(x_test)


#Compute overall accuracy
#accuracy_score(y_true, y_pred) compares your model’s predictions 
# (y_pred) to the ground-truth  labels (y_test) and calculates:
acc=accuracy_score(y_test, y_pred)
print("******************** CHECK ACCURACY")
print(f"Test-set accuracy: {acc:.3f}")

#Classification Report
print("\n Classification Report: ")
print(classification_report(y_test,y_pred,target_names=["Fake","Real"]))


#Confusion Matrix
print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred))






import pandas as pd

df = pd.read_csv('V:\Desktop\medicine.csv')
df.shape
df.head()
df.info()

df.columns = ['Id','drugName','condition','review','rating','date','usefulCount']    
df.head()
df.info()

df['drugName'].count()
df['condition'].count()  
df['review'].count()
df['rating'].count()
df['usefulCount'].count()

# Drop rows with any empty cells
df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
# OR
df=df.dropna()
df['condition'] = df['condition'].dropna('condition')
df.dropna(how='all')

# drop rows containing '</span>', 'Not Listed' in condition column
df = df[~df['condition'].str.contains('</span>', na=False)]
df = df[~df['condition'].str.contains('Not Listed', na=False)]
# OR
df[~df.condition.str.contains("</span>")]

# check for null
df.isnull().any().any()    

df['condition'].count()
df['drugName'].count()
df.shape
df.info()

df2 = df[['Id','drugName','condition','review','rating']].copy()    

df2.head()
df2.info()
df2.isnull().any().any()    

df2.info()       
df2['Id'].unique()      
df2['Id'].count()      
df2['Id'].nunique()    
df2.review[1]            
df2['condition'].nunique()
df2['drugName'].nunique()

pip install vaderSentiment       

import nltk
nltk.download(['punkt','stopwords'])
from nltk.corpus import stopwords
stopwords = stopwords.words('english')

df2['cleanReview'] = df2['review'].apply(lambda x: ' '.join([item for item in x.split() if item not in stopwords]))    

df2.head()

import vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
df2['vaderReviewScore'] = df2['cleanReview'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
df2.head()

positive_num = len(df2[df2['vaderReviewScore'] >=0.05])
neutral_num = len(df2[(df2['vaderReviewScore'] >-0.05) & (df2['vaderReviewScore']<0.05)])
negative_num = len(df2[df2['vaderReviewScore']<=-0.05])
positive_num,neutral_num, negative_num

df2['vaderSentiment']= df2['vaderReviewScore'].map(lambda x:int(2) if x>=0.05 else int(1) if x<=-0.05 else int(0) )
df2['vaderSentiment'].value_counts()

Total_vaderSentiment = positive_num + neutral_num + negative_num
Total_vaderSentiment

df2.loc[df2['vaderReviewScore'] >=0.05,"vaderSentimentLabel"] ="positive"
df2.loc[(df2['vaderReviewScore'] >-0.05) & (df2['vaderReviewScore']<0.05),"vaderSentimentLabel"]= "neutral"
df2.loc[df2['vaderReviewScore']<=-0.05,"vaderSentimentLabel"] = "negative"
df2.shape
df2.head()

positive_rating = len(df2[df2['rating'] >=5.0])
neutral_rating = len(df2[(df2['rating'] >=4) & (df2['rating']<5)])
negative_rating = len(df2[df2['rating']<=3])
positive_rating,neutral_rating,negative_rating

Total_rating = positive_rating+neutral_rating+negative_rating
Total_rating
df2['ratingSentiment']= df2['rating'].map(lambda x:int(2) if x>=7 else int(1) if x<=3 else int(0) )
df2['ratingSentiment'].value_counts()
df2.head()

df2.loc[df2['rating'] >=5.0,"ratingSentimentLabel"] ="positive"
df2.loc[(df2['rating'] >=4.0) & (df2['rating']<5.0),"ratingSentimentLabel"]= "neutral"
df2.loc[df2['rating']<=3.0,"ratingSentimentLabel"] = "negative"
df2.head()

df2 = df2[['Id','review','cleanReview','rating','ratingSentiment','ratingSentimentLabel','vaderReviewScore','vaderSentiment','vaderSentimentLabel']]
df2.head(10)
df2['vaderSentimentLabel'].count()

df2.info()
df2.head()

df2 = df2.drop(columns=df2.columns[0])
df2.head()

df2.groupby('vaderSentimentLabel').size()

import matplotlib.pyplot as plt

df2.groupby('vaderSentimentLabel').count().plot.bar()
plt.show()

df2.groupby('ratingSentimentLabel').size()
df2.groupby('ratingSentimentLabel').count().plot.bar()
plt.show()

df2.groupby('ratingSentiment').size()

positive_vader_sentiments = df2[df2.ratingSentiment == 2]
positive_string = []
for s in positive_vader_sentiments.cleanReview:
  positive_string.append(s)
positive_string = pd.Series(positive_string).str.cat(sep=' ')

from wordcloud import WordCloud
wordcloud = WordCloud(width=2000,height=1000,max_font_size=200).generate(positive_string)
plt.imshow(wordcloud,interpolation='bilinear')
plt.show()

for s in positive_vader_sentiments.cleanReview[:20]:
  if 'side effect' in s:
    print(s)
    
negative_vader_sentiments = df2[df2.ratingSentiment == 1]
negative_string = []
for s in negative_vader_sentiments.cleanReview:
  negative_string.append(s)
negative_string = pd.Series(negative_string).str.cat(sep=' ')

from wordcloud import WordCloud
wordcloud = WordCloud(width=2000,height=1000,max_font_size=200).generate(negative_string)
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis('off')
plt.show()

for s in negative_vader_sentiments.cleanReview[:20]:
  if 'side effect' in s:
    print(s)
    
neutral_vader_sentiments = df2[df2.ratingSentiment == 0]
neutral_string = []
for s in neutral_vader_sentiments.cleanReview:
  neutral_string.append(s)
neutral_string = pd.Series(neutral_string).str.cat(sep=' ')

from wordcloud import WordCloud
wordcloud = WordCloud(width=2000,height=1000,max_font_size=200).generate(neutral_string)
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis('off')
plt.show()

for s in neutral_vader_sentiments.cleanReview[:20]:
  if 'side effect' in s:
    print(s)
    
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(stop_words='english',ngram_range=(1,2,))
features = tfidf.fit_transform(df2.cleanReview)
labels   = df2.vaderSentiment

features.shape

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
x_train,x_test,y_train,y_test = train_test_split(df2['cleanReview'],df2['ratingSentimentLabel'],random_state=0)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

models = [RandomForestClassifier(n_estimators=200,max_depth=3,random_state=0),LinearSVC(),MultinomialNB(),LogisticRegression(random_state=0,solver='lbfgs',max_iter=2000,multi_class='auto')]
CV = 5
cv_df2 = pd.DataFrame(index=range(CV * len(models)))
entries = []
for model in models:
  model_name = model.__class__.__name__
  accuracies = cross_val_score(model,features,labels,scoring='accuracy',cv=CV)
  for fold_idx,accuracy in enumerate(accuracies):
    entries.append((model_name,fold_idx,accuracy))
cv_df2 = pd.DataFrame(entries,columns=['model_name','fold_idx','accuracy'])

cv_df2

cv_df2.groupby('model_name').accuracy.mean()

from sklearn.preprocessing import Normalizer

model = LinearSVC('l2')
x_train,x_test,y_train,y_test = train_test_split(features,labels,test_size=0.20,random_state=0)
normalize = Normalizer()
x_train = normalize.fit_transform(x_train)
x_test = normalize.transform(x_test)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))

from sklearn.metrics import confusion_matrix
conf_mat = confusion_matrix(y_test,y_pred)
conf_mat

from mlxtend.plotting import plot_confusion_matrix

fig,ax = plot_confusion_matrix(conf_mat=conf_mat,colorbar=True,show_absolute=True,cmap='viridis')

from  sklearn.metrics import classification_report
print(classification_report(y_test,y_pred,target_names= df2['ratingSentimentLabel'].unique()))


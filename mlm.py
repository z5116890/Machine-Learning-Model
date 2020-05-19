#Austin Lam z5116890
import sys
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import pearsonr
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, accuracy_score, recall_score
import numpy as np

def build(df):
    #clean
    df['cast'] = df['cast'].str.findall(r""""character": "(.*?)",""")
    df['crew'] = df['crew'].str.findall(r""""name": "(.*?)"}""")
    df['genres'] = df['genres'].str.findall(r""""name": "(.*?)"}""")
    df['keywords'] = df['keywords'].str.findall(r""""name": "(.*?)"}""")
    df['production_companies'] = df['production_companies'].str.findall(r""""name": "(.*?)",""")
    df['release_date'] = df['release_date'].replace(to_replace=r'^20\d\d', value=20, regex=True)
    df['release_date'] = df['release_date'].replace(to_replace=r'^19\d\d', value=19, regex=True)
    df['spoken_languages'] = df['spoken_languages'].str.findall(r""""name": "(.*?)"}""")
    #count of all things
    df["cast_count"] = df.cast.apply(lambda x: len(x)).astype('int64')
    df["crew_count"] = df.crew.apply(lambda x: len(x)).astype('int64')
    df["genre_count"] = df.genres.apply(lambda x: len(x)).astype('int64')
    df["keywords_count"] = df.keywords.apply(lambda x: len(x)).astype('int64')
    df['production_companies_count'] = df.production_companies.apply(lambda x: len(x)).astype('int64')
    df['spoken_languages_count'] = df.spoken_languages.apply(lambda x: len(x)).astype('int64')
    df['runtime'] = df['runtime'].astype('int64')
    df['homepage'] = df['homepage'].notnull().astype('int64')
    df.status = pd.get_dummies(df.status)
    return df[['budget','runtime', 'status', 'cast_count', 'crew_count']].values

def load_movie(movie_train_path):
    training_df = pd.read_csv(movie_train_path, index_col=0)


    movies_x = build(training_df)
    #actual revenue
    movies_y = training_df['revenue'].values


    return movies_x, movies_y


if __name__ == "__main__":
    if(len(sys.argv) != 3):
        raise ValueError("not enough files")
    training_csv = sys.argv[1]
    validation_csv = sys.argv[2]
    #REGRESSION
    movies_x_train, movie_y_train = load_movie(training_csv)
    model = linear_model.LinearRegression()
    model.fit(movies_x_train, movie_y_train)

    #Applying the model on validation data
    validation_df = pd.read_csv(validation_csv)
    movies_x_validation = build(validation_df)
    movies_y_validation = validation_df['revenue'].values
    validation_pred = model.predict(movies_x_validation)
    validation_pred = [round(x, 2) for x in validation_pred]

    p = pearsonr(movies_y_validation, validation_pred)[0]
    summary = {'zid': ['z5116890'], 'MSR': [round(mean_squared_error(movies_y_validation, validation_pred), 2)], 'correlation': [round(p, 2)]}
    summary_df = pd.DataFrame(data=summary)
    summary_df.to_csv("z5116890.PART1.summary.csv", index=False)
    output = {'movie_id': validation_df['movie_id'].values, 'predicted_revenue': validation_pred}
    output_df = pd.DataFrame(data=output)
    output_df.to_csv("z5116890.PART1.output.csv", index=False)


    #CLASSIFICATION
    df_train = pd.read_csv(training_csv)
    df_test = pd.read_csv(validation_csv)
    df_train['runtime'] = df_train['runtime'].divide(other=60).astype('int64')
    df_train['budget'] = df_train['budget'].divide(other=1000000).astype('int64')
    movie_x_train = build(df_train)
    movie_y_train = df_train['rating'].values

    df_test['runtime'] = df_test['runtime'].divide(other=60).astype('int64')
    df_test['budget'] = df_test['budget'].divide(other=1000000).astype('int64')
    movie_x_test =  build(df_test)
    movie_y_test = df_test['rating'].values

    # train a classifier
    svc = SVC()
    svc.fit(movie_x_train, movie_y_train)

    # predict the test set
    predictions = svc.predict(movie_x_test)

    summary = {'zid': ['z5116890'], 'average_precision': [round(precision_score(movie_y_test, predictions, average='macro', zero_division=1),2)],
               'average_recall': [round(recall_score(movie_y_test, predictions, average='macro'), 2)], 'accuracy': [round(accuracy_score(movie_y_test, predictions), 2)]}

    summary_df = pd.DataFrame(data=summary)
    summary_df.to_csv("z5116890.PART2.summary.csv", index=False)
    output = {'movie_id': df_test['movie_id'].values, 'predicted_rating': predictions}
    output_df = pd.DataFrame(data=output)
    output_df.to_csv("z5116890.PART2.output.csv", index=False)

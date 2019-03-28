
## Building a Movie Recommendation Service 

### Part 1. Matrix factorization algorithm implement by hand

#### load necessary packages


```python
import pandas as pd
import numpy as np
import sys, numpy as np
from numpy import genfromtxt
import codecs
from numpy import linalg as LA
```

#### load movie data and rating data


```python
movies=pd.read_csv("movies.csv")
ratings=pd.read_csv("ratings.csv")
```


```python
movies.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Jumanji (1995)</td>
      <td>Adventure|Children|Fantasy</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Grumpier Old Men (1995)</td>
      <td>Comedy|Romance</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Waiting to Exhale (1995)</td>
      <td>Comedy|Drama|Romance</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Father of the Bride Part II (1995)</td>
      <td>Comedy</td>
    </tr>
  </tbody>
</table>
</div>




```python
ratings.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>5.0</td>
      <td>847117005</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2</td>
      <td>3.0</td>
      <td>847642142</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>10</td>
      <td>3.0</td>
      <td>847641896</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>32</td>
      <td>4.0</td>
      <td>847642008</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>34</td>
      <td>4.0</td>
      <td>847641956</td>
    </tr>
  </tbody>
</table>
</div>



Movie ids are not continuous, build movie dicitionary with line no as numpy movie id ,its actual movie id as the key.


```python
def build_movies_dict(movies_file):
    i = 0
    movie_id_dict = {}
    with codecs.open(movies_file, 'r', 'latin-1') as f:
        for line in f:
            if i == 0:
                i = i+1
            else:
                movieId,title,genres = line.split(',')
                movie_id_dict[int(movieId)] = i-1
                i = i +1
    return movie_id_dict
```

Each line of i/p file represents one tag applied to one movie by one user,and has the following format: userId,movieId,tag,timestamp make sure you know the number of users and items for your dataset return the sparse matrix as a numpy array.


```python
def read_data(input_file,movies_dict):
    #no of users
    users = 718
    #users = 5
    #no of movies
    movies = 8927
    #movies = 135887
    X = np.zeros(shape=(users,movies))
    i = 0
    #X = genfromtxt(input_file, delimiter=",",dtype=str)
    with open(input_file,'r') as f:
        for line in f:
            if i == 0:
                i = i +1
            else:
                #print "i is",i
                user,movie_id,rating,timestamp = line.split(',')
                #get the movie id for the numpy array consrtruction
                id = movies_dict[int(movie_id)]
                #print "user movie rating",user, movie, rating, i
                X[int(user)-1,id] = float(rating)
                i = i+1
    return X
```

#### matrix factorization implementation

X is the user-rate-movie matrix, it's very sparse. P, Q are user-feature matrix and movie-featuree matrix respectively. The objective is to use gradient descend method to find the P,Q where $P \times Q$ approximates X.


```python
def matrix_factorization(X,P,Q,K,steps,alpha,beta):
    Q = Q.T
    for step in xrange(steps):
        print (step)
        #for each user
        for i in xrange(X.shape[0]):
            #for each item
            for j in xrange(X.shape[1]):
                if X[i][j] > 0 :

                    #calculate the error of the element
                    eij = X[i][j] - np.dot(P[i,:],Q[:,j])
                    #second norm of P and Q for regularilization
                    sum_of_norms = 0
                    #for k in xrange(K):
                    #    sum_of_norms += LA.norm(P[:,k]) + LA.norm(Q[k,:])
                    #added regularized term to the error
                    sum_of_norms += LA.norm(P) + LA.norm(Q)
                    #print sum_of_norms
                    eij += ((beta/2) * sum_of_norms)
                    #print eij
                    #compute the gradient from the error
                    for k in xrange(K):
                        P[i][k] = P[i][k] + alpha * ( 2 * eij * Q[k][j] - (beta * P[i][k]))
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - (beta * Q[k][j]))

        #compute total error
        error = 0
        #for each user
        for i in xrange(X.shape[0]):
            #for each item
            for j in xrange(X.shape[1]):
                if X[i][j] > 0:
                    error += np.power(X[i][j] - np.dot(P[i,:],Q[:,j]),2)
        if error < 0.001:
            break
    return P, Q.T
```

#### main function


```python
def main(X,K):
    #no of users
    N= X.shape[0]
    #no of movies
    M = X.shape[1]
    #P: an initial matrix of dimension N x K, where is n is no of users and k is hidden latent features
    P = np.random.rand(N,K)
    #Q : an initial matrix of dimension M x K, where M is no of movies and K is hidden latent features
    Q = np.random.rand(M,K)
    #steps : the maximum number of steps to perform the optimisation, hardcoding the values
    #alpha : the learning rate, hardcoding the values
    #beta  : the regularization parameter, hardcoding the values
    steps = 5000
    alpha = 0.0002
    beta = float(0.02)
    estimated_P, estimated_Q = matrix_factorization(X,P,Q,K,steps,alpha,beta)
    #Predicted numpy array of users and movie ratings
    modeled_X = np.dot(estimated_P,estimated_Q.T)
    np.savetxt('mf_result.txt', modeled_X, delimiter=',')
```


```python
if __name__ == '__main__':
    #MatrixFactorization.py <rating file>  <no of hidden features>  <movie mapping file>
    if len(sys.argv) == 4:
        ratings_file =  sys.argv[1]
        no_of_features = int(sys.argv[2])
        movies_mapping_file = sys.argv[3]

        #build a dictionary of movie id mapping with counter of no of movies
        movies_dict = build_movies_dict(movies_mapping_file)
        #read data and return a numpy array
        numpy_arr = read_data(ratings_file,movies_dict)
        #main function
        main(numpy_arr,no_of_features)
```

#### recommend movies for users who have rated some of the movies
recommend 50 tops movies for each user based on his/her unrated movies. Implemented this seperately from building model as once the model is built, we can use it many times.


```python
def dict_with_user_unrated_movies(rating_file,movie_mapping_id):
    #no of users
    users = 718
    #users = 5
    #no of movie ids
    #movies = 4
    movies = 8927
    dict_with_unrated_movies_users ={}
    X = np.zeros(shape=(users,movies))
    i = 0
    with open(rating_file,'r') as f:
        for line in f:
            if i == 0:
                i = i +1
            else:
                user,movie,rating,timestamp = line.split(',')
                id = movie_mapping_id[int(movie)]
                #print "user movie rating",user, movie, rating, i
                X[int(user)-1,id] = float(rating)
                i = i+1
    #print X
    for row in xrange(X.shape[0]):
        unrated_movi_ids = np.nonzero(X[row] == 0)
        #print "user",row+1, "has unrated movies", list(unrated_movi_ids[0])
        unrated_movi_ids = list(unrated_movi_ids[0])
        unrated_movi_ids = map(lambda x: x+1,unrated_movi_ids)
        dict_with_unrated_movies_users[row+1] = unrated_movi_ids
    #print "dict with unrated movies",dict_with_unrated_movies_users
    return dict_with_unrated_movies_users


#recommend top 25 movies for user specified
def top_25_recommended_movies(pred_rating_file,users,unrated_movies_per_user,movies_mapping_names,movie_mapping_id):
    #dicitonary with numpy movie id as key and actual movie id as value
    reverse_movie_id_mapping = {}
    for key,val in movie_mapping_id.items():
        reverse_movie_id_mapping[val] = key
    #for each user, predict top 25 movies
    for user in users:
        dict_pred_unrated_movies = {}
        unrated_movies = unrated_movies_per_user[int(user)]
        for unrated_movie in unrated_movies:
            dict_pred_unrated_movies[int(unrated_movie)] = pred_rating_file[int(user)-1][int(unrated_movie)-1]
        #recommend top k movies
        SortedMovies = sorted(dict_pred_unrated_movies.iteritems(), key=operator.itemgetter(1), reverse=True)
        print ("Top 25 movies recommendation for the user", user)
        for i in range(25):
            movie_id, rating = SortedMovies[i]
            actual_movie_id = reverse_movie_id_mapping[movie_id]
            #recommend movies only if the predicted rating is greater than 3.5
            if rating >= 3.5 :
                print ("{} ".format(movie))
            #print ("{} with Movie rating value {}".format(movies_mapping_names[actual_movie_id],rating))
        print ("\n")

#main method
def recommend_movies_for_users(orig_rating_file,pred_rating_file,movies_file,users):
    #method to get the mapping between movie names, actual movie id and numpy movie id
    movies_mapping_names,movie_mapping_id = dict_with_movie_and_id(movies_file)
    #build predicted numpy movie id from the saved predicted matrix of user and movie ratings
    predicted_rating_numpy_array = build_predicted_numpy_array(pred_rating_file)
    #dictionary of unrated movies for each user
    dict_with_unrated_movies_users = dict_with_user_unrated_movies(orig_rating_file,movie_mapping_id)
    #method which actually recommends top 25 unrated movies based on their the predicted score
    top_25_recommended_movies(predicted_rating_numpy_array,users,dict_with_unrated_movies_users,movies_mapping_names,movie_mapping_id)

if __name__ == '__main__':
    if len(sys.argv) == 5:
        #read the rating file for the missing
        orig_rating_file = sys.argv[1]
        pred_rating_file = sys.argv[2]
        movies_file = sys.argv[3]
        list_of_users = sys.argv[4]
        with open (list_of_users,'r') as f:
          users = f.readline().split(',')
        recommend_movies_for_users(orig_rating_file,pred_rating_file,movies_file,users)
```

### Part 2. Using Apache Spark to faciliate computing


```python

```

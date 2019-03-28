

```python
movies = sc.textFile("/FileStore/tables/movies.csv")
ratings= sc.textFile("/FileStore/tables/ratings.csv")
```


<style scoped>
  .ansiout {
    display: block;
    unicode-bidi: embed;
    white-space: pre-wrap;
    word-wrap: break-word;
    word-break: break-all;
    font-family: "Source Code Pro", "Menlo", monospace;;
    font-size: 13px;
    color: #555;
    margin-left: 4px;
    line-height: 19px;
  }
</style>
<div class="ansiout"></div>



```python
ratings=ratings.map(lambda x:x.split(","))
```


<style scoped>
  .ansiout {
    display: block;
    unicode-bidi: embed;
    white-space: pre-wrap;
    word-wrap: break-word;
    word-break: break-all;
    font-family: "Source Code Pro", "Menlo", monospace;;
    font-size: 13px;
    color: #555;
    margin-left: 4px;
    line-height: 19px;
  }
</style>
<div class="ansiout"></div>



```python
ratings.take(2)

```


<style scoped>
  .ansiout {
    display: block;
    unicode-bidi: embed;
    white-space: pre-wrap;
    word-wrap: break-word;
    word-break: break-all;
    font-family: "Source Code Pro", "Menlo", monospace;;
    font-size: 13px;
    color: #555;
    margin-left: 4px;
    line-height: 19px;
  }
</style>
<div class="ansiout"><span class="ansired">Out[</span><span class="ansired">154</span><span class="ansired">]: </span>[[&apos;userId&apos;, &apos;movieId&apos;, &apos;rating&apos;, &apos;timestamp&apos;], [&apos;1&apos;, &apos;1&apos;, &apos;5.0&apos;, &apos;847117005&apos;]]
</div>



```python
ratings=ratings.filter(lambda x:"userId" not in x).map(lambda x:(x[0],x[1],x[2]))
```


<style scoped>
  .ansiout {
    display: block;
    unicode-bidi: embed;
    white-space: pre-wrap;
    word-wrap: break-word;
    word-break: break-all;
    font-family: "Source Code Pro", "Menlo", monospace;;
    font-size: 13px;
    color: #555;
    margin-left: 4px;
    line-height: 19px;
  }
</style>
<div class="ansiout"></div>



```python
ratings.take(2)

```


<style scoped>
  .ansiout {
    display: block;
    unicode-bidi: embed;
    white-space: pre-wrap;
    word-wrap: break-word;
    word-break: break-all;
    font-family: "Source Code Pro", "Menlo", monospace;;
    font-size: 13px;
    color: #555;
    margin-left: 4px;
    line-height: 19px;
  }
</style>
<div class="ansiout"><span class="ansired">Out[</span><span class="ansired">156</span><span class="ansired">]: </span>[(&apos;1&apos;, &apos;1&apos;, &apos;5.0&apos;), (&apos;1&apos;, &apos;2&apos;, &apos;3.0&apos;)]
</div>



```python
training_RDD, validation_RDD, test_RDD = ratings.randomSplit([6, 2, 2])

validation_for_predict_RDD = validation_RDD.map(lambda x: (x[0], x[1]))
test_for_predict_RDD = test_RDD.map(lambda x: (x[0], x[1]))

```


<style scoped>
  .ansiout {
    display: block;
    unicode-bidi: embed;
    white-space: pre-wrap;
    word-wrap: break-word;
    word-break: break-all;
    font-family: "Source Code Pro", "Menlo", monospace;;
    font-size: 13px;
    color: #555;
    margin-left: 4px;
    line-height: 19px;
  }
</style>
<div class="ansiout"></div>



```python
from pyspark.mllib.recommendation import ALS
import math
iterations = 10
regularization_parameter = 0.1
ranks = [4, 8, 12]
errors = [0, 0, 0]
err = 0
tolerance = 0.02

min_error = float('inf')
best_rank = -1
best_iteration = -1
for rank in ranks:
    model = ALS.train(training_RDD, rank,  iterations=iterations,
                      lambda_=regularization_parameter)
    predictions = model.predictAll(validation_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
    rates_and_preds = validation_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
    error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
    errors[err] = error
    err += 1
    print ('For rank %s the RMSE is %s' % (rank, error))
    if error < min_error:
        min_error = error
        best_rank = rank

print ('The best model was trained with rank %s' % best_rank)
```


<style scoped>
  .ansiout {
    display: block;
    unicode-bidi: embed;
    white-space: pre-wrap;
    word-wrap: break-word;
    word-break: break-all;
    font-family: "Source Code Pro", "Menlo", monospace;;
    font-size: 13px;
    color: #555;
    margin-left: 4px;
    line-height: 19px;
  }
</style>
<div class="ansiout">For rank 4 the RMSE is 0.9359056108007288
For rank 8 the RMSE is 0.9382743636404246
For rank 12 the RMSE is 0.9389290854027168
The best model was trained with rank 4
</div>



```python
predictions.take(3)
```


<style scoped>
  .ansiout {
    display: block;
    unicode-bidi: embed;
    white-space: pre-wrap;
    word-wrap: break-word;
    word-break: break-all;
    font-family: "Source Code Pro", "Menlo", monospace;;
    font-size: 13px;
    color: #555;
    margin-left: 4px;
    line-height: 19px;
  }
</style>
<div class="ansiout"><span class="ansired">Out[</span><span class="ansired">159</span><span class="ansired">]: </span>
[((44, 3272), 3.9103419004701716),
 ((618, 7184), 2.9601086695162566),
 ((264, 52328), 3.3141828739969803)]
</div>



```python
model = ALS.train(training_RDD, best_rank,  iterations=iterations,
                      lambda_=regularization_parameter)
predictions = model.predictAll(test_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
rates_and_preds = test_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
    
print ('For testing data the RMSE is %s' % (error))
```


<style scoped>
  .ansiout {
    display: block;
    unicode-bidi: embed;
    white-space: pre-wrap;
    word-wrap: break-word;
    word-break: break-all;
    font-family: "Source Code Pro", "Menlo", monospace;;
    font-size: 13px;
    color: #555;
    margin-left: 4px;
    line-height: 19px;
  }
</style>
<div class="ansiout">For testing data the RMSE is 0.9457633345043662
</div>



```python
# Recommend to user movies which are unrated by himself, recommend movie for ID:2 as an example.
# First find all unrated movies for ID:2
personInfo=ratings.filter(lambda x:x[0]=='2')
movieRated=personInfo.map(lambda x:x[1]).collect()
movieUnrated=ratings.filter(lambda x:x[1] not in movieRated).map(lambda x:x[1]).distinct()
```


<style scoped>
  .ansiout {
    display: block;
    unicode-bidi: embed;
    white-space: pre-wrap;
    word-wrap: break-word;
    word-break: break-all;
    font-family: "Source Code Pro", "Menlo", monospace;;
    font-size: 13px;
    color: #555;
    margin-left: 4px;
    line-height: 19px;
  }
</style>
<div class="ansiout"></div>



```python
print(personInfo.count())
print(len(movieRated))
print(movies.count())
print(movieUnrated.count())
```


<style scoped>
  .ansiout {
    display: block;
    unicode-bidi: embed;
    white-space: pre-wrap;
    word-wrap: break-word;
    word-break: break-all;
    font-family: "Source Code Pro", "Menlo", monospace;;
    font-size: 13px;
    color: #555;
    margin-left: 4px;
    line-height: 19px;
  }
</style>
<div class="ansiout">50
50
8928
8865
</div>



```python
movieUnrated.count()
```


<style scoped>
  .ansiout {
    display: block;
    unicode-bidi: embed;
    white-space: pre-wrap;
    word-wrap: break-word;
    word-break: break-all;
    font-family: "Source Code Pro", "Menlo", monospace;;
    font-size: 13px;
    color: #555;
    margin-left: 4px;
    line-height: 19px;
  }
</style>
<div class="ansiout"><span class="ansired">Out[</span><span class="ansired">163</span><span class="ansired">]: </span>8865
</div>



```python
moviePredict=movieUnrated.map(lambda x:['2',x])
predictRes=model.predictAll(moviePredict)
```


<style scoped>
  .ansiout {
    display: block;
    unicode-bidi: embed;
    white-space: pre-wrap;
    word-wrap: break-word;
    word-break: break-all;
    font-family: "Source Code Pro", "Menlo", monospace;;
    font-size: 13px;
    color: #555;
    margin-left: 4px;
    line-height: 19px;
  }
</style>
<div class="ansiout"></div>



```python
predictRes=predictRes.map(lambda x:(x.product,x.rating))
```


<style scoped>
  .ansiout {
    display: block;
    unicode-bidi: embed;
    white-space: pre-wrap;
    word-wrap: break-word;
    word-break: break-all;
    font-family: "Source Code Pro", "Menlo", monospace;;
    font-size: 13px;
    color: #555;
    margin-left: 4px;
    line-height: 19px;
  }
</style>
<div class="ansiout"></div>



```python
movies=movies.filter(lambda x:'movieId' not in x).map(lambda x:x.split(',')).map(lambda x:(x[0],(x[1],x[2])))

```


<style scoped>
  .ansiout {
    display: block;
    unicode-bidi: embed;
    white-space: pre-wrap;
    word-wrap: break-word;
    word-break: break-all;
    font-family: "Source Code Pro", "Menlo", monospace;;
    font-size: 13px;
    color: #555;
    margin-left: 4px;
    line-height: 19px;
  }
</style>
<div class="ansiout"></div>



```python
movies.take(3)
```


<style scoped>
  .ansiout {
    display: block;
    unicode-bidi: embed;
    white-space: pre-wrap;
    word-wrap: break-word;
    word-break: break-all;
    font-family: "Source Code Pro", "Menlo", monospace;;
    font-size: 13px;
    color: #555;
    margin-left: 4px;
    line-height: 19px;
  }
</style>
<div class="ansiout"><span class="ansired">Out[</span><span class="ansired">167</span><span class="ansired">]: </span>
[(&apos;1&apos;, (&apos;Toy Story (1995)&apos;, &apos;Adventure|Animation|Children|Comedy|Fantasy&apos;)),
 (&apos;2&apos;, (&apos;Jumanji (1995)&apos;, &apos;Adventure|Children|Fantasy&apos;)),
 (&apos;3&apos;, (&apos;Grumpier Old Men (1995)&apos;, &apos;Comedy|Romance&apos;))]
</div>



```python
predictRes=predictRes.map(lambda x:(str(x[0]),(x[1]))).join(movies)
```


<style scoped>
  .ansiout {
    display: block;
    unicode-bidi: embed;
    white-space: pre-wrap;
    word-wrap: break-word;
    word-break: break-all;
    font-family: "Source Code Pro", "Menlo", monospace;;
    font-size: 13px;
    color: #555;
    margin-left: 4px;
    line-height: 19px;
  }
</style>
<div class="ansiout"></div>



```python
predictRes.take(3)

```


<style scoped>
  .ansiout {
    display: block;
    unicode-bidi: embed;
    white-space: pre-wrap;
    word-wrap: break-word;
    word-break: break-all;
    font-family: "Source Code Pro", "Menlo", monospace;;
    font-size: 13px;
    color: #555;
    margin-left: 4px;
    line-height: 19px;
  }
</style>
<div class="ansiout"><span class="ansired">Out[</span><span class="ansired">169</span><span class="ansired">]: </span>
[(&apos;27706&apos;,
  (3.761019669094141,
   (&quot;Lemony Snicket&apos;s A Series of Unfortunate Events (2004)&quot;,
    &apos;Adventure|Children|Comedy|Fantasy&apos;))),
 (&apos;37240&apos;, (2.4144329602095578, (&apos;Why We Fight (2005)&apos;, &apos;Documentary&apos;))),
 (&apos;45183&apos;,
  (4.492474207135306,
   (&apos;Protector  The (a.k.a. Warrior King) (Tom yum goong) (2005)&apos;,
    &apos;Action|Comedy|Crime|Thriller&apos;)))]
</div>



```python
predictRes=predictRes.map(lambda x:(x[1][1][0],x[1][0]))
```


<style scoped>
  .ansiout {
    display: block;
    unicode-bidi: embed;
    white-space: pre-wrap;
    word-wrap: break-word;
    word-break: break-all;
    font-family: "Source Code Pro", "Menlo", monospace;;
    font-size: 13px;
    color: #555;
    margin-left: 4px;
    line-height: 19px;
  }
</style>
<div class="ansiout"></div>



```python
predictRes.take(1)
```


<style scoped>
  .ansiout {
    display: block;
    unicode-bidi: embed;
    white-space: pre-wrap;
    word-wrap: break-word;
    word-break: break-all;
    font-family: "Source Code Pro", "Menlo", monospace;;
    font-size: 13px;
    color: #555;
    margin-left: 4px;
    line-height: 19px;
  }
</style>
<div class="ansiout"><span class="ansired">Out[</span><span class="ansired">171</span><span class="ansired">]: </span>[(&quot;Lemony Snicket&apos;s A Series of Unfortunate Events (2004)&quot;, 3.761019669094141)]
</div>



```python
top_movies = predictRes.takeOrdered(25, key=lambda x: -x[1])
top_movies
```


<style scoped>
  .ansiout {
    display: block;
    unicode-bidi: embed;
    white-space: pre-wrap;
    word-wrap: break-word;
    word-break: break-all;
    font-family: "Source Code Pro", "Menlo", monospace;;
    font-size: 13px;
    color: #555;
    margin-left: 4px;
    line-height: 19px;
  }
</style>
<div class="ansiout"><span class="ansired">Out[</span><span class="ansired">172</span><span class="ansired">]: </span>
[(&apos;Paulie (1998)&apos;, 6.010185255833662),
 (&apos;Sweet Land (2005)&apos;, 5.791408996053072),
 (&apos;Stir Crazy (1980)&apos;, 5.76573694136966),
 (&apos;Westerner  The (1940)&apos;, 5.549436023244587),
 (&apos;Battlestar Galactica (2003)&apos;, 5.533985629970289),
 (&apos;Truly  Madly  Deeply (1991)&apos;, 5.483704154666285),
 (&apos;Auntie Mame (1958)&apos;, 5.380268674819099),
 (&apos;School Daze (1988)&apos;, 5.363982694591471),
 (&apos;Bugs Bunny / Road Runner Movie  The (a.k.a. The Great American Chase) (1979)&apos;,
  5.358809646115851),
 (&apos;Cashback (2006)&apos;, 5.358809646115851),
 (&apos;Memoirs of a Geisha (2005)&apos;, 5.317983563349337),
 (&apos;About Time (2013)&apos;, 5.316004836486595),
 (&apos;Vicious Kind  The (2009)&apos;, 5.285894518338154),
 (&apos;Spread (2009)&apos;, 5.285894518338154),
 (&apos;New Rose Hotel (1998)&apos;, 5.27282052920912),
 (&apos;Cocaine Cowboys (2006)&apos;, 5.264844765867043),
 (&quot;Empire of the Wolves (L&apos;empire des loups) (2005)&quot;, 5.242122655555558),
 (&apos;Repo! The Genetic Opera (2008)&apos;, 5.242122655555558),
 (&apos;Visitors  The (Visiteurs  Les) (1993)&apos;, 5.24178791470284),
 (&apos;Aguirre: The Wrath of God (Aguirre  der Zorn Gottes) (1972)&apos;,
  5.23498039535383),
 (&apos;Bloody Sunday (2002)&apos;, 5.2286529970954305),
 (&apos;Mrs. Miniver (1942)&apos;, 5.216295450453579),
 (&apos;Chaos (2001)&apos;, 5.212268024919439),
 (&quot;Twelve O&apos;Clock High (1949)&quot;, 5.207334792061248),
 (&apos;American Pimp (1999)&apos;, 5.207111214164087)]
</div>


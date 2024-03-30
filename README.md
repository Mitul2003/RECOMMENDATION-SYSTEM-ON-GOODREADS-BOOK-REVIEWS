# Recommendation-System-on-Goodreads-Book-Reviews



## Problem Statement

In some industries, the use of recommender systems is crucial because, when implemented well, they can be extremely profitable and set themselves apart from their competitors. Online book selling websites nowadays are competing with each other by many means. One of the most effective strategies for increasing sales, enhancing customer experience and retaining customers is building an efficient Recommendation system. The book recommendation system must recommend books that are of interest to buyers. **Content based filtering** approach is used in this project to build book recommendation systems.

## Content-Based Filtering System

Content-based recommender systems, are based on the items, and not necessarily the users. This method builds an understanding of similarity between items. That is, it recommends items that are similar to each other in terms of properties.

![](https://pritamaich.github.io/Books-Recommendation-System/images/filtering.png)

## :book: Dataset information </h2>

Dataset used in this project is the **Goodreads Book Reviews** group datasets collected by [Julian McAuley](https://cseweb.ucsd.edu/~jmcauley/), UCSD and which is publicly available at https://cseweb.ucsd.edu/~jmcauley/datasets.html#goodreads. 

These datasets contain reviews from the Goodreads book review website, and a variety of attributes describing the items. Critically, these datasets have multiple levels of user interaction, ranging from adding to a "shelf", rating, and reading.

There are three groups of datasets: 

1. **Meta-Data of Books** : 
-  Detailed book graph (~2gb, about 2.3m books):  [goodreads_books.json.gz](https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/goodreads/goodreads_books.json.gz)
-   Detailed information of authors:  [goodreads_book_authors.json.gz](https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/goodreads/goodreads_book_authors.json.gz)
-   Detailed information of works (i.e., the abstract version of a book regardless any particular editions)

2. **User-Book Interactions** (users' public shelves) :
-   Complete user-book interactions in 'csv' format (~4.1gb):  [goodreads_interactions.csv](https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/goodreads/goodreads_interactions.csv)
-   Detailed information of the complete user-book interactions (~11gb, ~229m records):  [goodreads_interactions_dedup.json.gz](https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/goodreads/goodreads_interactions_dedup.json.gz)

3. **Users' detailed Book Reviews** :
-   Complete book reviews (~15m multilingual reviews about ~2m books and 465k users):  [goodreads_reviews_dedup.json.gz](https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/goodreads/goodreads_reviews_dedup.json.gz)


> For the purpose of this project, from the wide range of genre provided we took “**Children**” genre. The dataset consists of **24,082** books and **734,640** detailed reviews. 
> The reason for this is to make the data processing and system implementation feasible with available resources.
> Also we have read the data into chunks and then processed it individually for ease in computation and decreasing complexity.

## 🛠️ Tools and Technologies used


The programming language used in this project is **Python** . The following libraries were used for data analysis, data visualization and to build a recommendation system.

* **Pandas** :  For loading the dataset and performing data wrangling
* **Matplotlib**: For  data visualization.
* **Seaborn**: For data visualization.
* **NumPy**: For some math operations in predictions.
* **Sklearn**:  For the purpose of analysis, prediction and evaluation.
*  **NLTK**: For NLP tasks such as classification, stemming, tagging, parsing, semantic reasoning, and tokenization.

##  📑 Steps involved

* **Data Preprocessing** : For the Data Processing, we have created a pipeline for filtering out the dataset for appropriate use. It defines a Class with methods to process book and user review datasets. We filtered out only English language books, converts author information format, and applies stemming to text data , removing any unnecessary columns, null values etc before storing it in a pickle file. 

> We have used **pickle files** so that we could load only the data we require filtered on a predefined condition instead of loading the whole data first and then filtering,  thus reducing overhead and processsing time.

Overall, this code streamlines the preparation of datasets crucial for the recommendation system's functioning. 

* **Exploratory Data Analysis** : We did Exploratory Data Analysis on the processed data to get more information about the dataset we are working with. EDA basically talks about Distribution of Average_Rating, Top ten books, Top ten Authors, Word count distribution, User distribution about reviews, Correlation between Books and rating  and many more.

## 💻 Implementation

**1.  Item Representation** :

* For the item representation we have created separate classes for our two methods, first for TF-IDF (`BookData_ItemRepresentation_TFIDF`) and second Word2Vec (`BookData_ItemRepresentation_Word2vec`), which serve the purpose of generating item representations for book data using distinct methods.

* `Tf-Idf` class initializes paths for input and output data, along with a TF-IDF vectorizer configured with specified parameters. It offers methods to fit a TF-IDF vectorizer on book descriptions, retrieve feature names from the vectorizer, and process the book data to create TF-IDF representations combined with book identifiers.

* `Word2Vec` class focuses on generating item representations using Word2Vec embeddings. It initializes paths for input and output data and loads a pre-trained Word2Vec model using the “`genism`” library. We use an api “`word2vec-google-news-300`". The primary method processes book data chunks, computes average embeddings for each book description, and stores the resulting representations combined with book identifiers.

**2.  User Profiles** :

* For the user profile generation, we have created `Build_user_profiles` class facilitates the construction of user profiles based on book reviews and item representations. It initializes paths for input data and output profiles. The `build_user_profiles()` method iterates through review data chunks, retrieves unique user IDs, gathers user-book interactions, and merges them with item representations. User profiles are then computed by averaging book representations per user. The resulting profiles are stored as pickle files. The `read_user()` method reads and concatenates user profiles from the stored files. Overall, the class efficiently generates user profiles from review and item representation data, providing a basis for personalized recommendation systems.

**3.  Content Based Filtering** :

* For creating the final recommendation system, we  have created `ContentBasedFiltering` class which provides methods for user profile retrieval, finding similar books, and generating recommendations. First, we retrieve the user profile based on the provided user ID from the stored user profiles. Secondly, we compute the most similar books to the given user profile using cosine similarity between the user profile and item representations, return the top similar books. And at last we generate book recommendations for the user by finding the most similar books and filtering out those already reviewed by the user, return the user's review data along with recommended book details, make it compact.
After retrieving the top similar books we collected all the reviews present in the dataset for that particular `book_id` and for that reviews we basically went on doing sentiment analysis so that we only recommend only positively reviews books to our user.

**3.  Sentimental Analysis** :

* We have done Sentimental Analysis for the book reviews to divide them into positive and negative categories. This functionality offers valuable insights into user perceptions and preferences regarding recommended books, aiding in the refinement of recommendation algorithms for enhanced user satisfaction and engagement.

* For this task, we have defined a class called `ReviewsSentiment`, which facilitates sentiment analysis on reviews associated with recommended books from the Goodreads children dataset. This method aggregates reviews for each recommended book into a single string, ensuring efficient analysis. Utilizing the `TextBlob` library, the `get_sentiment_score` method computes the sentiment polarity score for each review, indicating its positivity or negativity. Subsequently, the `get_sentiment` method categorizes books based on their sentiment scores, segregating them into dataframes of positive and negative sentiment.

**4. Evaluation** :

* For evaluating the system, we first find similar books using `find_most_similar_books2`, compare recommended books with actual interactions in the test set.
-   Calculates precision and recall metrics using `precision_score` and `recall_score`.
-   Returns precision and recall values.


## References

-   Mengting Wan, Julian McAuley, "[Item Recommendation on Monotonic Behavior Chains](https://mengtingwan.github.io/paper/recsys18_mwan.pdf)", in RecSys'18. [[bibtex](https://dblp.uni-trier.de/rec/conf/recsys/WanM18.html?view=bibtex)]
-   Mengting Wan, Rishabh Misra, Ndapa Nakashole, Julian McAuley, "[Fine-Grained Spoiler Detection from Large-Scale Review Corpora](https://mengtingwan.github.io/paper/acl19_mwan.pdf)", in ACL'19. [[bibtex](https://dblp.uni-trier.de/rec/conf/acl/WanMNM19.html?view=bibtex)]
-  https://mengtingwan.github.io/data/goodreads
- https://pritamaich.github.io/Books-Recommendation-System/

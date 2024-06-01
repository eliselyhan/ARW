#arxiv_read
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from sklearn.decomposition import TruncatedSVD
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
import re

class readArxiv:
    
    def __init__(self):
        # TODO: change number of rows to an argument; check order; read_json intervals
        #self.arxiv_df = pd.read_json('arxiv-metadata-oai-snapshot.json', lines=True)# nrows=rows)

        #create a set of stopwords
        gist_file = open("../arxiv_data/stopwords.txt", "r")
        try:
            content = gist_file.read()
            stopwords = content.split(",")
        finally:
            gist_file.close()
        stopwords=[i.replace('"',"").strip() for i in stopwords]
        self.stopwords = set(stopwords)
        self.lemmatizer = WordNetLemmatizer()  
    
    #get DataFrame given category, and time interval(optional)

    def get_words_from_abstract(self, abstract):

        """ get a list of words from a given abstract(string)

        Args:
            abstract (string): abstract of a paper
        Returns:
            words: a list of words; lemmatized, lowercased, and without stopwords
        """
        words = []
        for word in abstract.split():
            lemma = self.lemmatizer.lemmatize(word.lower())
            lemma = re.sub((r'[!.,;?]+$'),'',lemma)
            if bool(re.match(r'^[a-zA-Z]+$', lemma)) and lemma not in self.stopwords:
                words.append(lemma)
        return words

    def get_df_by_category(self, category, **kwargs):

        """ get a DataFrame given category, and time interval(optional)
        Args:
            category (string): category name, e.g. 'cs_AI'
            start (string): start date of the interval, e.g. '20090501'
            end (string): end date of the interval, e.g. '20121208'
        Returns:
            df: a DataFrame of papers in the given category
        """

        start = kwargs.get('start', None)
        end = kwargs.get('end', None)
        if start == None:
            return pd.read_json('../arxiv_data/categories_json/{}.json'.format(category), lines=True)
        else:
            df = pd.read_json('../arxiv_data/categories_json/{}.json'.format(category), lines=True)
            #get every row where the first version was created within the time interval
            df = df[df['versions'].apply(lambda x: int(start) <= self.get_date_from_string(x[0]['created']) and int(end) >= self.get_date_from_string(x[0]['created']))]
            return df
        
    #get a list of abstracts given category, and time interval(optional)
    def get_abstracts_by_category(self, list_of_categories, **kwargs): 


        """ get a list of abstracts given category, and time interval(required)
        Args:
            list_of_categories (list): a list of category names, e.g. ['cs_AI', 'cs_LG']
            start (string): start date of the interval, e.g. '20090501'
            end (string): end date of the interval, e.g. '20121208'
            B (int): number of abstracts to be sampled from each category

        Returns:
            list_of_abstracts: a list of abstracts of papers in the given categories
        """

        start = kwargs.get('start')
        end = kwargs.get('end')
        B = kwargs.get('B', None)

        list_of_abstracts = []
        #'.' replace by '_'
        #interval: e.g. '20090501, 20121208'; inclusive of the end date

        for category in list_of_categories:
            df = self.get_df_by_category(category, start = start, end = end)
            
            for abstract in df['abstract'].tolist():
                list_of_abstracts.append(self.get_words_from_abstract(abstract))
        return list_of_abstracts    
    
    #get the number of abstracts given category, and time interval(optional)
    def get_abstracts_number_by_category(self, category, **kwargs):
        """ get the number of abstracts given category, and time interval(optional)
        Args:
            category (string): category name, e.g. 'cs_AI'
            start (string): start date of the interval, e.g. '20090501'
            end (string): end date of the interval, e.g. '20121208'
        Returns:
            len(df): number of abstracts in the given category
        """

        start = kwargs.get('start', None)
        end = kwargs.get('end', None)
        if start == None:
            return len(self.get_df_by_category(category))
        else:
            df = self.get_df_by_category(category, start = start, end=end)
            return len(df)
    
    #plot the number of abstracts over each half-year interval for a given category
    def plot_counts_by_category_and_time(self, category):
        time_intervals=['20070101, 20070630', '20070701, 20071231', '20080101, 20080630', '20080701, 20081231', '20090101, 20090630', '20090701, 20091231',
                '20100101, 20100630', '20100701, 20101231', '20110101, 20110630', '20110701, 20111231', '20120101, 20120630', '20120701, 20121231',
                '20130101, 20130630', '20130701, 20131231', '20140101, 20140630', '20140701, 20141231', '20150101, 20150630', '20150701, 20151231', 
                '20160101, 20160630', '20160701, 20161231', '20170101, 20170630', '20170701, 20171231', '20180101, 20180630', '20180701, 20181231', 
                '20190101, 20190630', '20190701, 20191231', '20200101, 20200630', '20200701, 20201231', '20210101, 20210630', '20210701, 20211231', 
                '20220101, 20220630', '20220701, 20221231', '20230101, 20230630']

        interval_numbers = []
        for interval in time_intervals:
            interval_numbers.append(self.get_abstracts_number_by_category(category, interval=interval))
        
        x_axis = np.arange(len(time_intervals))/2+2007
        plt.bar(x_axis, height = np.array(interval_numbers))
        plt.title('Number of abstracts published in {} by half-year intervals'.format(category))
        return plt.show()
    
    #get a set of vocabulary given a list of abstracts
    def get_vocab(self, abstracts): 
        """ get a set of vocabulary given a list of abstracts
        Args:
            abstracts (list): a list of abstracts
        Returns:
            vocab (set): a set of lemmatized vocabulary
        """
        vocab = set()
        for abstract in abstracts:
            for word in abstract.split():
                lemma = self.lemmatizer.lemmatize(word.lower())
                lemma = re.sub((r'[!.,;?]+$'),'',lemma)
                if bool(re.match(r'^[a-zA-Z]+$', lemma)) and lemma not in self.lemmatized_stopwords:
                    vocab.add(lemma)
        return vocab
    
    #return a set of vocabularies given a list of abstracts of given categories; e.g. [list_of_cat_1_abstracts, list_of_cat_2_abstracts]
    def get_all_vocab(self, category_abstracts): 
        """ get a set of vocabulary given a list of abstracts
        Args:
            category_abstracts (list): a list of categories, each of which is a list of abstracts

        Returns:
            all_vocab (set): a set of lemmatized vocabulary
        """
        
        all_vocab = set()
        for category_abstract in category_abstracts:
            vocab = self.get_vocab(category_abstract)
            all_vocab = all_vocab.union(vocab)

        return all_vocab

    #return a bag of words matrix given a set of vocabularies and a list of abstracts
    def get_bow_matrix(self, vocab, abstracts): 
        """ get a bag of words matrix given a set of vocabularies and a list of abstracts
        Args:
            vocab (set): a set of vocabulary
            abstracts (list): a list of abstracts
        Returns:
            bow_matrix (numpy matrix): (shape n, d); BOW matrix
        """

        bow = {word: 0 for word in vocab}
        bow_matrix = np.zeros((len(abstracts), len(bow)))

        for i in range(len(abstracts)):
            #create a local copy of vocabulary as keys of the hashtable
            bow_copy = bow.copy()

            #lemmatize; remove punctuation, non-alphabetic symbols, and all stopwords
            for word in abstracts[i].split():
                lemma = self.lemmatizer.lemmatize(word.lower())
                lemma = re.sub((r'[!.,;?]+$'),'',lemma)
                if bool(re.match(r'^[a-zA-Z]+$', lemma)) and lemma not in self.stopwords:
                    bow_copy[lemma] += 1

            #convert the hashtable to a row in the numpy matrix 
            bow_matrix[i, :] = np.array(list(bow_copy.values()))

        return bow_matrix
    

    #return the svd of a BOW matrix given the number of components k
    def get_svd(self, bow_matrix, k): 
        svd = TruncatedSVD(n_components=k)
        U = svd.fit_transform(bow_matrix)
        s = svd.singular_values_
        V = svd.components_ 
        return U, s, V
    
    #get a date of the format 'yyyymmdd' from a string of the format 'Sat 12 May 2001'
    def get_date_from_string(self, string_date):
        """ get a date of the format 'yyyymmdd' from a string of the format 'Sat 12 May 2001'
        Args:
            string_date (string): a string of the format 'Sat 12 May 2001'
        Returns:
            int_date (int): an integer of the format 'yyyymmdd'
        """
        date = string_date.split()
        switcher = {
            "Jan": "01", "Feb": "02", "Mar": "03", "Apr": "04", "May": "05", "Jun": "06",
            "Jul": "07", "Aug": "08", "Sep": "09", "Oct": "10", "Nov": "11", "Dec": "12"
        }
        if len(date[1]) < 2:
            return int(date[3] + switcher.get(date[2], "Invalid month") + '0' + date[1])
        return int(date[3] +  switcher.get(date[2], "Invalid month") + date[1])
    
    def count_ngrams_in_abstracts(self, abstracts, ngram_list):
        """ get a list of word counts given a list of abstracts and a word
        Args:
            abstracts (list): a list of abstracts
            ngram_list (list of tuple of strings): list of ngrams, e.g. [('llm',), ('neural', 'network')]
                #note for singleton words, we have to use ('word',) with a comma!
        Returns:
            counts (np.array): shape (len(abstracts),); a list of ngram counts for each abstract
        """
        counts = []
        from collections import Counter

        for abstract in abstracts:

            ngram_counts = Counter()

            # Iterate over each n-gram in the list
            for ngram in ngram_list:
                # Calculate specific n-grams using zip
                ngrams = zip(*(abstract[i:] for i in range(len(ngram))))

                # Update Counter with occurrences of the specific n-gram

                #modification for proportion
                if Counter(ngrams)[ngram] > 0:
                    ngram_counts[ngram] = 1

                #ngram_counts[ngram] += Counter(ngrams)[ngram]

            if sum(ngram_counts.values()) > 0:
                counts.append(1)
            else: 
                counts.append(0)
            
            #counts.append(sum(ngram_counts.values()))
        return np.array(counts)
       
        
    def compute_proportion(self, abstracts, ngram_list):
        """ compute the proportion of abstracts containing a given word
        Args:
            abstract (a list of list of words): list of abstract words
            ngram_list (list of tuples of strings): list of ngrams, e.g. [('llm',), ('neural', 'network')]
        Returns:
            proportion of abstracts containing at least one of the ngrams
        """
        from collections import Counter
        count = 0
        for abstract in abstracts:

            ngram_counts = Counter()

            # Iterate over each n-gram in the list
            for ngram in ngram_list:
                # Calculate specific n-grams using zip
                ngrams = zip(*(abstract[i:] for i in range(len(ngram))))

                # Update Counter with occurrences of the specific n-gram
                ngram_counts[ngram] += Counter(ngrams)[ngram]

            if sum(ngram_counts.values()) > 0:
                count += 1

        return count/len(abstracts)

    
    def get_window(self, **kwargs):
        """get the start and end of period given end date and window size
        Args:
            end_date (string): end date of the period, e.g. '20221231'
            d (int): window size in days, e.g. 30
        Returns:

        """
        end_date, d = kwargs.get('end_date', None), kwargs.get('d', None)
        start_date = datetime.strptime(end_date, "%Y%m%d") - timedelta(days=d)
        return start_date.strftime("%Y%m%d")
    
    def get_abstracts(self, **kwargs):
        """return a list of abstracts given num_periods, category, interval and sample size
        Args:
            num_periods (int): number of periods
            end (string): end date of the period, e.g. '20221231'
            window_width (int): window width in days, e.g. 30
            category_list (list): e.g. ['cs_AI']
        Returns:
            abs_list (list of periods, each of which is a list of abstracts) 
        """

        num_periods = kwargs.get('num_periods')
        end = kwargs.get('end'); window_width = kwargs.get('window_width')
        category_list = kwargs.get('category_list')

        abs_list = []
        B_arr = np.zeros(num_periods)

        for i in range(num_periods-1, -1, -1):

            start = self.get_window(end_date = end, d=window_width)
            abstracts = self.get_abstracts_by_category(category_list, start = start, end = end)

            abs_list.append(abstracts)
            B_arr[i] = len(abstracts)

            end = start

        #reverse the list to put the earliest period first
        abs_list.reverse() 

        return abs_list, B_arr


    def get_X_data(self, **kwargs):

        """ get the data matrix X

        Args:
            abs_list: a list of periods, each of which is a list of abstracts
            task (string): either 'count' or 'proportion'
            word (string): e.g. 'neural'        
        Returns:
            X: word count matrix(n,B) or proportion array(n,)
        """

        abs_list = kwargs.get('abs_list')
        task = kwargs.get('task') #either 'count' or 'proportion'
        ngrams = kwargs.get('ngrams') #list of ngrams to be counted; e.g. [('llm',), ('neural', 'network')]
        
        num_periods = len(abs_list); 

        #initialize 1D list to store data across all periods
        X = []

        for t in range(num_periods):

            #get abstracts from period t
            abs_t = abs_list[t]

            #carry out tasks
            if task == 'count':
                X.append(self.count_ngrams_in_abstracts(abs_t, ngrams))
            elif task == 'proportion':
                X.append(self.compute_proportion(abs_t, ngrams))

        return np.concatenate(X)

    #train-test split
    def X_train_test_split(self, **kwargs):
        X = kwargs.get('X'); train_size= kwargs.get('train_size')
        num_periods, B = X.shape
        
        #round trainsize to the nearest integer
        train_size = int(train_size * B)

        #for each period, extract a numpy array of size train_size
        X_train = []
        X_test = []
        for t in range(num_periods):
            X_t = X[t]
            #sample random indices
            train_indices = np.random.choice(B, train_size, replace=False)
            X_train.append(X_t[train_indices])
            X_test.append(np.delete(X_t, train_indices))
                    
        return np.array(X_train), np.array(X_test)
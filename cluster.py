import re
import tweetcollector
import math
import sys
import pickle
import random
import numpy as np
from nltk.corpus import stopwords

document_frequency_map = {}
query_tf_idf_map = {}
stop = stopwords.words('english')

def collect_tweets(tweet_collector_object):

    tweets_file = open('tweets.txt', 'wb')
    query_tweet_map = {}

    query_list = ['#katyperry', '#katycats', '#darkhorse', '#iHeartRadio', '#ladygaga', '#TaylorSwift', '#sxsw', 'Rolling Stone',
                  '@DwightHoward', '#rockets', 'jeremy lin', 'toyota center', 'kevin mchale', 'houston nba', 'James Harden', 'linsanity',
                  'Jan Koum', 'WhatsApp', '#SEO', 'facebook', '#socialmedia', 'Zuckerberg', 'user privacy', '#Instagram',
                  'Obama', '#tcot', 'Russia', 'Putin', 'White House', 'Ukraine', 'Rand Paul', 'foreign policy']

    for query in query_list:
        query_tweet_map[query] = tweet_collector_object.get_first50_tweets(query)
        tweet_with_query = query +": "+str(query_tweet_map[query])
        tweets_file.write(tweet_with_query)
        tweets_file.write('\n')

    tweets_file.close()
    return query_tweet_map


def compute_term_and_document_frequencies(query_tweet_map):
    global stop
    global query_tf_idf_map
    global document_frequency_map
    for query in query_tweet_map:
        token_set = set()
        token_frequency_map = {}
        tweet_list = query_tweet_map[query]
        for tweet in tweet_list:
            tweet  = tweet.lower()
            tokens = re.sub('[^0-9a-zA-Z@#]+', ' ', tweet).split()
            for token in tokens:
                if token in stop:
                    # Don't process stop words
                    continue
                token_frequency_map[token] = token_frequency_map.get(token, 0) + 1
                token_set.add(token)
        for token in token_set:
            document_frequency_map[token] = document_frequency_map.get(token, 0) + 1

        query_tf_idf_map[query] = token_frequency_map

def compute_tf_idf():
    global query_tf_idf_map
    global document_frequency_map
    for query in query_tf_idf_map:
        sum_of_squares = 0
        token_frequency_map = query_tf_idf_map[query]
        for token in token_frequency_map:
            token_frequency_val =  1 + math.log10(token_frequency_map[token])
            token_idf_val        = math.log10(len(query_tf_idf_map)*1.0/document_frequency_map[token])
            token_frequency_map[token] = token_frequency_val * token_idf_val
            sum_of_squares += math.pow(token_frequency_map[token], 2)

        for token in token_frequency_map:
            token_frequency_map[token] = token_frequency_map[token]/math.sqrt(sum_of_squares)

        query_tf_idf_map[query] = token_frequency_map

def compute_modulus(input_map):
    sum_of_squares = 0
    for k in input_map:
        sum_of_squares += math.pow(input_map[k], 2)
    return math.sqrt(sum_of_squares)

def add_vectors(vector_1, vector_2):
    sum_vector = {}
    token_list = vector_1.keys() + vector_2.keys()
    for token in token_list:
        sum_vector[token] = vector_1.get(token, 0) + vector_2.get(token, 0)

    return sum_vector

def build_unit_vector(input_map):
    sumOfSquares = 0
    for token in input_map:
            sumOfSquares += math.pow(input_map[token], 2)

    for token in input_map:
            input_map[token] = input_map[token]/math.sqrt(sumOfSquares)

    return input_map

def compute_cosine_similarity(vector_1, vector_2):
    vector_1 = build_unit_vector(vector_1)
    vector_2 = build_unit_vector(vector_2)
    similarity_val = 0
    token_list = vector_1.keys()
    for token in token_list:
        similarity_val += vector_1.get(token, 0) * vector_2.get(token, 0)
    return similarity_val

def compute_euclidean_distance(vector_1, vector_2):
    similarity_val = 0
    token_list = vector_1.keys() + vector_2.keys()
    for token in token_list:
        similarity_val += math.pow(vector_1.get(token, 0) - vector_2.get(token, 0), 2)
    return math.sqrt(similarity_val)

def run_K_means(query_map, k, use_cosine, purity_threshold = None):

    class_1 = set(['#katyperry', '#katycats', '#darkhorse', '#iHeartRadio', '#ladygaga', '#TaylorSwift', '#sxsw', 'Rolling Stone'])
    class_2 = set(['@DwightHoward', '#rockets', 'jeremy lin', 'toyota center', 'kevin mchale', 'houston nba', 'James Harden', 'linsanity'])
    class_3 = set(['Jan Koum', 'WhatsApp', '#SEO', 'facebook', '#socialmedia', 'Zuckerberg', 'user privacy', '#Instagram'])
    class_4 = set(['Obama', '#tcot', 'Russia', 'Putin', 'White House', 'Ukraine', 'Rand Paul', 'foreign policy'])

    max_cluster_purity = 0
    max_purity_clusters = []
    best_cluster_RSS = 0

    for i in range(0, 50):

        # Choosing the initial selection of seeds
        cluster_centres = []
        random_set = set()
        query_list = query_map.keys()
        while len(random_set) < k:
            random_set.add(random.choice(query_list))
        for element in random_set:
            cluster_centres.append(query_map[element])

        #Runs till convergence
        while True:
            cluster_elements = {}
            for i in range(0, k):
                cluster_elements[i] = []
            for query in query_map:
                similarity_values = []
                for i in range(0, k):
                    similarity_values.append(0)
                query_vector = query_map[query]
                for j in range (0, len(cluster_centres)):
                    if use_cosine:
                        similarity_values[j] = compute_cosine_similarity(query_vector, cluster_centres[j])
                    else:
                        similarity_values[j] = compute_euclidean_distance(query_vector, cluster_centres[j])

                if use_cosine:
                    cluster_number = np.argmax(similarity_values)
                else:
                    cluster_number = np.argmin(similarity_values)
                cluster_elements[cluster_number].append(query)

            # Compute the new cluster centres
            prev_cluster_centres = cluster_centres
            for i in range (0, k):
                new_cluster_centre = {}
                query_list = cluster_elements[i]
                for query in query_list:
                    query_vector = query_map[query]
                    new_cluster_centre = add_vectors(new_cluster_centre, query_vector)

                for token in new_cluster_centre:
                    new_cluster_centre[token] /= len(query_list)

                new_cluster_centre = build_unit_vector(new_cluster_centre)
                cluster_centres[i] = new_cluster_centre

            # RSS Calculation
            overall_RSS_value = 0
            for i in range(0, k):
                query_list = cluster_elements[i]
                for query in query_list:
                    query_vector = query_map[query]
                    if use_cosine:
                        overall_RSS_value += math.pow(1-compute_cosine_similarity(query_vector, cluster_centres[i]), 2)
                    else:
                        overall_RSS_value += math.pow(compute_euclidean_distance(query_vector, cluster_centres[i]), 2)

            # Purity Calculation. Purity makes sense only for k = 4.
            overall_cluster_purity = 0
            cluster_purity = []
            for i in range(0, k):
                cluster_purity.append(0)
            for i in range(0, k):
                cluster_element_set = set(cluster_elements[i])
                max_intersection = max(len(cluster_element_set.intersection(class_1)), len(cluster_element_set.intersection(class_2)), len(cluster_element_set.intersection(class_3)), len(cluster_element_set.intersection(class_4)))
                cluster_purity[i] = (max_intersection * 1.0)
            for i in range(0, k):
                overall_cluster_purity += (cluster_purity[i] * 1.0)/len(query_map)

            if overall_cluster_purity > max_cluster_purity:
                max_cluster_purity  = overall_cluster_purity
                max_purity_clusters = cluster_elements
                best_cluster_RSS    = overall_RSS_value

            if prev_cluster_centres == cluster_centres:
                # If the centroids don't change, it means the algorithm has converged
                break


        if purity_threshold:
            if max_cluster_purity > purity_threshold:
                break

    return (max_cluster_purity, max_purity_clusters, best_cluster_RSS)

def main():
    global query_tf_idf_map
    global document_frequency_map
    tweet_collector = tweetcollector.TwitterCrawler()
    print 'Collecting tweets....'
    query_tweet_map = collect_tweets(tweet_collector)
    compute_term_and_document_frequencies(query_tweet_map)
    compute_tf_idf()

    #Write to file
    output_file = open ('query_tf_idf_map.txt', 'wb')
    pickle.dump(query_tf_idf_map, output_file)
    output_file.close()

    #Read from file
    query_map = pickle.load(open('query_tf_idf_map.txt', 'rb'))

    for k in [2, 4, 6, 8]:
        (max_purity_cosine, best_clustering_cosine, best_cluster_RSS_cosine) = run_K_means(query_map, k, True)
        (max_purity_euclid, best_clustering_euclid, best_cluster_RSS_euclid) = run_K_means(query_map, k, False)

        print 'Clustering Using Cosine Similarity with k = '+str(k)
        print '===================================================='
        if k == 4:
            print 'Purity of best clustering using Cosine Similarity:'+str(max_purity_cosine)
            print ''
        print 'RSS Value of best clustering using Cosine Similarity: '+str(best_cluster_RSS_cosine)
        print ''
        for i in range(0, len(best_clustering_cosine)):
            print 'Cluster '+str(i+1)+':'+str(best_clustering_cosine[i])

        print ''
        print ''

        print 'Clustering Using Euclidean Distance with k = '+str(k)
        print '===================================================='
        if k == 4:
            print 'Purity of best clustering using Euclidean distance:'+str(max_purity_euclid)
            print ''
        print 'RSS Value of best clustering using Euclidean distance: '+str(best_cluster_RSS_euclid)
        print ''
        for i in range(0, len(best_clustering_euclid)):
            print 'Cluster '+str(i+1)+':'+str(best_clustering_euclid[i])

        print ''
        print ''

if __name__ == '__main__':
    main()

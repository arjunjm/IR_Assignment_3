import re
import tweetcollector
import math
import sys
import pickle
import random
import numpy as np

document_frequency_map = {}
query_tf_idf_map = {}

def collect_tweets(tweet_collector_object):

    query_tweet_map = {}

    query_list = ['#katyperry', '#katycats', '#darkhorse', '#iHeartRadio', '#ladygaga', '#TaylorSwift', '#sxsw', 'Rolling Stone',
                  '@DwightHoward', '#rockets', 'jeremy lin', 'toyota center', 'kevin mchale', 'houston nba', 'James Harden', 'linsanity',
                  'Jan Koum', 'WhatsApp', '#SEO', 'facebook', '#socialmedia', 'Zuckerberg', 'user privacy', '#Instagram',
                  'Obama', '#tcot', 'Russia', 'Putin', 'White House', 'Ukraine', 'Rand Paul', 'foreign policy']

    for query in query_list:
        query_tweet_map[query] = tweet_collector_object.get_first50_tweets(query)

    return query_tweet_map


def compute_term_and_document_frequencies(query_tweet_map):
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


def run_K_means(query_map, k):

    # Choosing the initial selection of seeds
    cluster_centres = []
    random_set = set()
    query_list = query_map.keys()
    while len(random_set) < k:
        random_set.add(random.choice(query_list))
    for element in random_set:
        cluster_centres.append(query_map[element])

    print random_set
    for count in range (0, 100):
        cluster_elements = {}
        for i in range(0, k):
            cluster_elements[i] = []
        for query in query_map:
            cosine_values = []
            for i in range(0, k):
                cosine_values.append(0)
            query_vector = query_map[query]
            for j in range (0, len(cluster_centres)):
                cosine_values[j] = compute_cosine_similarity(query_vector, cluster_centres[j])

            cluster_number = np.argmax(cosine_values)
            cluster_elements[cluster_number].append(query)

        # Compute the new cluster centres
        for i in range (0, k):
            new_cluster_centre = {}
            query_list = cluster_elements[i]
            for query in query_list:
                query_vector = query_map[query]
                new_cluster_centre = add_vectors(new_cluster_centre, query_vector)

            for token in new_cluster_centre:
                new_cluster_centre[token] /= len(query_list)

            cluster_centres[i] = new_cluster_centre
            print cluster_elements

    #print cluster_elements

def main():
    global query_tf_idf_map
    global document_frequency_map
    tweet_collector = tweetcollector.TwitterCrawler()
    print 'Created Tweet Collector Object!'
    print 'Creating query-tweet map'
    #query_tweet_map = collect_tweets(tweet_collector)
    #compute_term_and_document_frequencies(query_tweet_map)
    #compute_tf_idf()

    #Write to file
    #output_file = open ('query_tf_idf_map.txt', 'wb')
    #pickle.dump(query_tf_idf_map, output_file)
    #output_file.close()

    #Read from file
    query_map = pickle.load(open('query_tf_idf_map.txt', 'rb'))
    #print query_map

    k = 4
    run_K_means(query_map, k)

if __name__ == '__main__':
    main()

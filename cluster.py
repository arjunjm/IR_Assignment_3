import re
import tweetcollector
import math
import sys
import pickle

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

def main():
    global query_tf_idf_map
    global document_frequency_map
    tweet_collector = tweetcollector.TwitterCrawler()
    print 'Created Tweet Collector Object!'
    print 'Creating query-tweet map'
    query_tweet_map = collect_tweets(tweet_collector)
    compute_term_and_document_frequencies(query_tweet_map)
    compute_tf_idf()

    #Write to file
    output_file = open ('query_tf_idf_map.txt', 'wb')
    pickle.dump(query_tf_idf_map, output_file)
    output_file.close()

    #Read from file
    query_map = pickle.load(open('query_tf_idf_map.txt', 'rb'))
    print query_map

if __name__ == '__main__':
    main()

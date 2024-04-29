train_tweets = 50000
test_tweets = 10000
longest_sequence = 0

with open("origin_tweets.txt", "r", encoding="utf-8") as origin_tweets:
    with open("dataset/tweets.txt", "w", encoding="utf-8") as tweets_train:
        with open("dataset/testdata/tweets_test.txt", "w", encoding="utf-8") as tweets_test:
            for tweet in origin_tweets:
                seq_length = len(tweet.split(' '))

                if seq_length > 60:
                    break
                if seq_length > longest_sequence:
                    print(tweet)
                    longest_sequence = seq_length
                if train_tweets > 0:
                    tweets_train.write(tweet)
                    train_tweets -= 1
                if test_tweets > 0:
                    tweets_test.write(tweet)
                    test_tweets -= 1
                if train_tweets <= 0 and test_tweets <= 0:
                    break

print(f"Longest sequence: {longest_sequence}")

# from examples.train_criteo import train
# from examples.train_gc import train
# from data.criteo import process_raw_data, prepare_train_test_data
# from data.jaychou_lyrics import load_data_jay_lyrics, data_iter_consecutive
# from examples.train_jaychou_lyrics import train
# from examples.train_sms import train, test, train_bert, test_bert
# from data.india_sms import prepare_toy_data, get_india_sms_data, tokenize_sms_data, \
#         prepare_sms_data, bert_tokenize_sms_data, bert_prepare_sms_data
from examples.train_cartpole import train_model

# data_dir = '/data-0/yideng/criteo/'
# data_dir = '/Users/cashbus/Desktop/india_data/Ner_1024/data/annotation_results/round_2_results_bank'

if __name__ == '__main__':
        # process_raw_data(data_dir)
        # prepare_train_test_data(data_dir)
        # train(data_dir)

        # get_india_sms_data(data_dir)
        # tokenize_sms_data()
        # prepare_sms_data()
        # bert_tokenize_sms_data()
        # bert_prepare_sms_data()
        # train()
        # test()
        # train_bert()
        # test_bert()
        train_model()

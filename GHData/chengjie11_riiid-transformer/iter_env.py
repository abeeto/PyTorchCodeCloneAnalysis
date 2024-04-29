#%%
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import torch

from transformer.transformer import *
from utils import *
get_system()

HOME = os.path.abspath(os.path.join('.', os.pardir))
# HOME =  "/home/scao/Documents/kaggle-riiid-test/"
MODEL_DIR = os.path.join(HOME,  'model')
DATA_DIR = os.path.join(HOME,  'data')
PRIVATE = False
DEBUG = False
LAST_N = 100
VAL_BATCH_SIZE = 4096
SIMU_PUB_SIZE = 25_000

#%%
class Iter_Valid(object):
    def __init__(self, df, max_user=1000):
        df = df.reset_index(drop=True)
        self.df = df
        self.user_answer = df['user_answer'].astype(str).values
        self.answered_correctly = df['answered_correctly'].astype(str).values
        df['prior_group_responses'] = "[]"
        df['prior_group_answers_correct'] = "[]"
        self.sample_df = df[df['content_type_id'] == 0][['row_id']]
        self.sample_df['answered_correctly'] = 0
        self.len = len(df)
        self.user_id = df.user_id.values
        self.task_container_id = df.task_container_id.values
        self.content_type_id = df.content_type_id.values
        self.max_user = max_user
        self.current = 0
        self.pre_user_answer_list = []
        self.pre_answered_correctly_list = []

    def __iter__(self):
        return self
    
    def fix_df(self, user_answer_list, answered_correctly_list, pre_start):
        df= self.df[pre_start:self.current].copy()
        sample_df = self.sample_df[pre_start:self.current].copy()
        df.loc[pre_start,'prior_group_responses'] = '[' + ",".join(self.pre_user_answer_list) + ']'
        df.loc[pre_start,'prior_group_answers_correct'] = '[' + ",".join(self.pre_answered_correctly_list) + ']'
        self.pre_user_answer_list = user_answer_list
        self.pre_answered_correctly_list = answered_correctly_list
        return df, sample_df

    def __next__(self):
        added_user = set()
        pre_start = self.current
        pre_added_user = -1
        pre_task_container_id = -1

        user_answer_list = []
        answered_correctly_list = []
        while self.current < self.len:
            crr_user_id = self.user_id[self.current]
            crr_task_container_id = self.task_container_id[self.current]
            crr_content_type_id = self.content_type_id[self.current]
            if crr_content_type_id == 1:
                # no more than one task_container_id of "questions" from any single user
                # so we only care for content_type_id == 0 to break loop
                user_answer_list.append(self.user_answer[self.current])
                answered_correctly_list.append(self.answered_correctly[self.current])
                self.current += 1
                continue
            if crr_user_id in added_user and ((crr_user_id != pre_added_user) or (crr_task_container_id != pre_task_container_id)):
                # known user(not prev user or differnt task container)
                return self.fix_df(user_answer_list, answered_correctly_list, pre_start)
            if len(added_user) == self.max_user:
                if  crr_user_id == pre_added_user and crr_task_container_id == pre_task_container_id:
                    user_answer_list.append(self.user_answer[self.current])
                    answered_correctly_list.append(self.answered_correctly[self.current])
                    self.current += 1
                    continue
                else:
                    return self.fix_df(user_answer_list, answered_correctly_list, pre_start)
            added_user.add(crr_user_id)
            pre_added_user = crr_user_id
            pre_task_container_id = crr_task_container_id
            user_answer_list.append(self.user_answer[self.current])
            answered_correctly_list.append(self.answered_correctly[self.current])
            self.current += 1
        if pre_start < self.current:
            return self.fix_df(user_answer_list, answered_correctly_list, pre_start)
        else:
            raise StopIteration()

def load_model(model_file, device='cuda'):
    # creating the model and load the weights
    model = TransformerModel(ninp=LAST_N, nhead=10, nhid=128, nlayers=4, dropout=0.3)
    model = model.to(device)
    model.load_state_dict(torch.load(model_file,map_location=device))

    return model

def find_best_model(model_dir = MODEL_DIR, model_file=None):
    # find the best AUC model, or a given model
    if model_file is None:
        model_files = find_files('transformer', model_dir)
        tmp = [s.rsplit('.')[-2] for s in model_files]
        model_file = model_files[argmax(tmp)]
        print(model_file)
    return model_file

if DEBUG:
    test_df = pd.read_pickle(DATA_DIR+'cv2_valid.pickle')
    test_df[:SIMU_PUB_SIZE].to_pickle(DATA_DIR+'test_pub_simu.pickle')

#%%
if __name__ == "__main__":
    print("Loading test set....")
    if PRIVATE:
        test_df = pd.read_pickle(DATA_DIR+'cv2_valid.pickle')
    else:
        test_df = pd.read_pickle(DATA_DIR+'test_pub_simu.pickle')
        test_df = test_df[:SIMU_PUB_SIZE]
        train_df = pd.read_parquet(DATA_DIR+'cv2_valid.parquet')
    print("Loaded test.")
    df_questions = pd.read_csv(DATA_DIR+'questions.csv')
    iter_test = Iter_Valid(test_df, max_user=1000)
    predicted = []
    def set_predict(df):
        predicted.append(df)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\n\n Using device: {device} \n\n')
    model_file = find_best_model()
    model_name = model_file.split('/')[-1]
    model = load_model(model_file)
    print(f'\nLoaded {model_name}.')
    model.eval()
    print(model)

    len_test = len(test_df)
    with tqdm(total=len_test) as pbar:
        previous_test_df = None
        for (current_test, current_prediction_df) in iter_test:
            if prev_test_df is not None:
                '''Making use of answers to previous questions'''
                answers = eval(current_test["prior_group_answers_correct"].iloc[0])
                responses = eval(current_test["prior_group_responses"].iloc[0])
                prev_test_df['answered_correctly'] = answers
                prev_test_df['user_answer'] = responses
                prev_test_df = prev_test_df[prev_test_df[CONTENT_TYPE_ID] == False]
                prev_group = prev_test_df[[USER_ID, CONTENT_ID, 
                                        PRIOR_QUESTION_TIME, PRIOR_QUESTION_EXPLAIN, TARGET]]\
                                        .groupby(USER_ID)\
                                        .apply(lambda r: (r[CONTENT_ID].values, 
                                                        r[PRIOR_QUESTION_TIME].values,
                                                        r[PRIOR_QUESTION_EXPLAIN].values,
                                                        r[TARGET].values))
                for prev_user_id in prev_group.index:
                    prev_group_content = prev_group[prev_user_id][0]
                    prev_group_ac = prev_group[prev_user_id][1]
                    prev_group_time = prev_group[prev_user_id][2]
                    prev_group_exp = prev_group[prev_user_id][3]
                    
                    if prev_user_id in train_group.index:
                        train_group[prev_user_id] = (np.append(train_group[prev_user_id][0],prev_group_content), 
                                            np.append(train_group[prev_user_id][1],prev_group_ac),
                                            np.append(train_group[prev_user_id][2],prev_group_time),
                                            np.append(train_group[prev_user_id][3],prev_group_exp))
        
                    else:
                        train_group[prev_user_id] = (prev_group_content,
                                            prev_group_ac,
                                            prev_group_time,
                                            prev_group_exp)
                    
                    if len(train_group[prev_user_id][0])>MAX_SEQ:
                        new_group_content = train_group[prev_user_id][0][-MAX_SEQ:]
                        new_group_ac = train_group[prev_user_id][1][-MAX_SEQ:]
                        new_group_time = train_group[prev_user_id][2][-MAX_SEQ:]
                        new_group_exp = train_group[prev_user_id][3][-MAX_SEQ:]
                        train_group[prev_user_id] = (new_group_content,
                                            new_group_ac,
                                            new_group_time,
                                            new_group_exp)
            '''No labels'''
            # d_test, user_id_to_idx = get_feats_test(current_test_df)
            # dataset_test = RiiidTest(d=d_test)
            # test_dataloader = DataLoader(dataset=dataset_test, batch_size=VAL_BATCH_SIZE, 
            #                             collate_fn=collate_fn_test, shuffle=False, drop_last=False)

            '''Labels for verification'''
            d_test, user_id_to_idx = get_feats_train(current_test_df)
            dataset_test = Riiid(d=d_test)
            test_dataloader = DataLoader(dataset=dataset_test, batch_size=VAL_BATCH_SIZE, 
                                        collate_fn=collate_fn, shuffle=False, drop_last=False)

            # the problem with current feature gen is that 
            # using groupby user_id sorts the user_id and makes it different from the 
            # test_df's order

            output_all = []
            labels_all = []
            for _, batch in enumerate(test_dataloader):
                content_id, _, part_id, prior_question_elapsed_time, mask, labels = batch
                target_id = batch[1].to(device).long()

                content_id = Variable(content_id.cuda())
                part_id = Variable(part_id.cuda())
                prior_question_elapsed_time = Variable(prior_question_elapsed_time.cuda())
                mask = Variable(mask.cuda())

                with torch.no_grad():
                    output = model(content_id, part_id, prior_question_elapsed_time, mask)

                pred_probs = torch.softmax(output[~mask], dim=1)
                output_all.extend(pred_probs[:,1].reshape(-1).data.cpu().numpy())
                labels_all.extend(labels[~mask].reshape(-1).data.numpy())
            '''prediction code ends'''

            current_test['answered_correctly'] = output_all
            set_predict(current_test.loc[:,['row_id', 'answered_correctly']])
            pbar.update(len(current_test))

    y_true = test_df[test_df.content_type_id == 0].answered_correctly
    y_pred = pd.concat(predicted).answered_correctly
    print('\nValidation auc:', roc_auc_score(y_true, y_pred))
    print('# iterations:', len(predicted))

# %%

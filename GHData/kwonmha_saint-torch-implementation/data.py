
import os.path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class KT1Data(object):
    """
        Loads all columns of data. Necessary columns will be selected

        """

    def __init__(self, path, remove_duplicated):
        columns = ["user_id", "question_id", "user_answer", "correct_answer", "part", "tags",
                   "timestamp", "elapsed_time", "solving_id", "explanation_id", "bundle_id",
                   "deployed_at"]
        self.path = path

        df = pd.read_csv(path)
        if remove_duplicated:
            df = df[~df.duplicated(keep='first')]
        # data format: 'q1111' -> 1111
        df["question_id"] = df["question_id"].apply(lambda qid: int(qid[1:]))
        df = df.sort_values(["timestamp"], ascending=True).reset_index(drop=True)

        dir_path = os.path.dirname(path)
        question_df = pd.read_csv(dir_path + "/questions.csv")
        question_df["question_id"] = question_df["question_id"].apply(lambda qid: int(qid[1:]))
        # n_q = question_df.question_id.nunique()  # n_data: 13169, max:18143
        self.n_q = question_df.question_id.max()

        merge_df = pd.merge(left=df, right=question_df, how="inner", on="question_id")

        self.data = merge_df[columns] \
            .groupby("user_id") \
            .apply(lambda r: (r.question_id.values, r.user_answer.values == r.correct_answer.values,
                              r.part.values, r.tags.values, r.timestamp.values, r.elapsed_time.values,
                              r.solving_id.values, r.explanation_id.values, r.bundle_id.values,
                              r.deployed_at.values))

        ######### code for checking length distribution ############
        # lengths = [group[id][-1] for id in group.index]
        # from collections import Counter
        # length_count = Counter(lengths)
        # length_count = sorted(length_count.items())
        # length_count = {length:count for length, count in length_count}
        # length_df = pd.DataFrame(length_count.values(), index=length_count.keys())
        # length_df["cum"] = length_df[0].cumsum()
        ##########################################
        # return group, n_q


class SaintDataset(Dataset):
    def __init__(self, data, max_seq, min_items):
        super(SaintDataset, self).__init__()
        self.max_seq = max_seq
        self.inputs = []

        # 모든 데이터를 list 형태의 self.inputs에 넣는다.
        for id in data.index:
            # qids: 유저가 푼 모든 문제 리스트
            qids, correct, part_ids = data[id][0], data[id][1], data[id][2]
            correct = np.array([1 if c else 0 for c in correct])

            if len(qids) > max_seq:
                # 학습 데이터의 길이가 긴 경우 여러 번
                for start in range((len(qids) + max_seq - 1) // max_seq):
                    self.inputs.append((qids[start:start+max_seq], part_ids[start:start+max_seq],
                                       correct[start:start+max_seq]))
            elif min_items <= len(qids) < max_seq:
                self.inputs.append((qids, part_ids, correct))
            # len(qids) < min_items
            else:
                continue

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        qids, part_ids, correct = self.inputs[index]
        seq_len = len(qids)

        qid_tensor = np.zeros(self.max_seq, dtype=int)
        part_ids_tensor = np.zeros(self.max_seq, dtype=int)
        input_correct_tensor = np.zeros(self.max_seq, dtype=int)
        target_correct_tensor = np.zeros(self.max_seq, dtype=int)

        qid_tensor[-seq_len:] = qids + 1
        part_ids_tensor[-seq_len:] = part_ids
        # 0, 1-> 1, 2 for response embedding id
        input_correct_tensor[-seq_len:] = np.concatenate(([3], (correct + 1)[:-1]))  # 3 for start token
        # input_correct_tensor[-seq_len:] = np.concatenate(([0], (correct + 1)[:-1]))

        # use 0, 1 for sigmoid cross entropy loss
        target_correct_tensor[-seq_len:] = correct

        input_tensor = {"qids": qid_tensor,
                        "part_ids": part_ids_tensor,
                        "correct": input_correct_tensor}
        return input_tensor, target_correct_tensor

    # For VSaint
    def sample_data(self, sample_size, device):
        index_arr = np.arange(len(self.inputs))
        np.random.shuffle(index_arr)
        batch_ids = index_arr[:sample_size]
        batch_data = {"qids": [], "part_ids": [], "correct": []}

        for index in batch_ids:
            input_tensor, _ = self.__getitem__(index)
            batch_data["qids"].append(input_tensor["qids"])
            batch_data["part_ids"].append(input_tensor["part_ids"])
            batch_data["correct"].append(input_tensor["correct"])

        qids = torch.chunk(torch.tensor(batch_data["qids"]).to(device), 2)
        part_ids = torch.chunk(torch.tensor(batch_data["part_ids"]).to(device), 2)
        correct = torch.chunk(torch.tensor(batch_data["correct"]).to(device), 2)
        return qids, part_ids, correct

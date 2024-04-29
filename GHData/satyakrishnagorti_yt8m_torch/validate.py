"""
This file contains code to do the validation pipeline and report mAP, recall scores
"""

import time
import metric
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import utils


class Validation:

    def __init__(self, dataloader, model, device, label_cache):
        self.device = device
        self.dataloader = dataloader
        self.model = model
        vocab = pd.read_csv('/data/yt8m/vocabulary.csv')
        self.indices = vocab.Index.tolist()
        self.label_cache = label_cache

    def write_output_to_file(self, final_video_vals, final_predictions, output_file):
        print('writing to file...')
        f = open(output_file, 'w')
        f.write('Class,Segments\n')
        for idx, c in enumerate(tqdm(self.indices)):
            video_ids = final_video_vals[final_predictions[:, idx].tolist()].tolist()
            segments_list = []
            for v in video_ids:
                segments_list.append(v)
            segments_str = ' '.join(segments_list)
            f.write(str(c) + ',' + segments_str.strip() + '\n')
            f.flush()


    def validation_helper(self, generate_blend_file=False, top_n=100_000, output_file="data/submission.csv"):

        final_id_to_class = []
        final_video_vals = []

        for batch_sample in tqdm(self.dataloader):

            start_time = time.time()
            batch_vids = batch_sample["video_ids"]
            batch_video_matrix = batch_sample["video_matrix"].to(self.device)
            batch_num_frames = batch_sample["video_num_frames"].to(self.device)
            A_norm = utils.get_feature_similarity_matrix(batch_video_matrix, batch_num_frames, self.device,
                                                         normalize=True)

            output = self.model(batch_video_matrix, batch_num_frames, A_norm)
            end_time = time.time()

            # print("processed {} samples in seconds {0:.2f}".format(batch_vids.shape[0], end_time - start_time))

            predictions_val = output.cpu().numpy()

            predictions_val = np.reshape(predictions_val, (-1, 60, 3862))
            # (B, 60, 1000)
            predictions_val = predictions_val[:, :, self.indices]
            video_id_batch_val = np.repeat(batch_vids, 60, axis=1)
            video_id_batch_val = video_id_batch_val.astype(str)

            segment_ids = np.repeat(np.reshape(np.arange(0, 300, 5), (1, 60)), video_id_batch_val.shape[0],
                                    axis=0).astype(str)
            colons = np.repeat([':'], 60)
            segment_ids = np.core.defchararray.add(colons, segment_ids)
            # (B, 60)
            final_segment_video_ids = np.core.defchararray.add(video_id_batch_val, segment_ids)

            np_batch_num_frames = batch_num_frames.cpu().numpy()
            for b in range(np_batch_num_frames.shape[0]):
                # (60, 1000)
                pred = predictions_val[b]
                until = int(np.ceil(np_batch_num_frames[b] / 5))
                pred = pred[:until, :]
                final_id_to_class.append(pred)

                # (60, )
                seg_video_id = final_segment_video_ids[b]
                seg_video_id = seg_video_id[:until]
                final_video_vals.append(seg_video_id)

        final_id_to_class = np.concatenate(final_id_to_class, axis=0)
        final_video_vals = np.concatenate(final_video_vals, axis=0)

        if generate_blend_file:
            print("generating blend file")
            final_id_to_class, final_video_vals = self.add_missing_values(final_id_to_class, final_video_vals,
                                                                          test=False)
            assert (len(final_id_to_class) == len(final_video_vals))
            id_to_class = final_id_to_class.astype('>f4')
            id_to_class.tofile('/data2/yt8m/storage/gcn_90_10/gcn_90_10.seg.valid' + '.bin')
            np.savetxt('/data2/yt8m/storage/gcn_90_10/gcn_90_10.seg.valid' + '.txt', final_video_vals,
                       delimiter=',', fmt="%s")
            return

        final_predictions = np.argsort(-final_id_to_class, axis=0)[:top_n, :]

        self.write_output_to_file(final_video_vals, final_predictions, output_file)

    def validation_pipeline(self, generate_blend_file=False, top_n=100_000, output_file="data/submission.csv"):

        self.validation_helper(generate_blend_file, top_n, output_file)
        return metric.run_metric(output_file, self.label_cache, top_n)

    def add_missing_values(self, video_preds, seg_names, test=False):
        # video_preds: (x, 1000)
        # seg_names: (x,) like [vid:0, vid:5, ...]
        excess_vids = ['jubx', 'THiN', 'zhjn', '6Gjv', 'DjkK',
                       '3Vlt',
                       'FVl4',
                       'WIpl',
                       '56vl',
                       'lRCD',
                       'M0DR',
                       'aPGu',
                       'C2Kv',
                       'MRNj',
                       'zOQj',
                       '5WSP',
                       '0gW4',
                       '3597']

        if test:
            excess_vids = [
                "rIwe",
                "uoGZ",
                "iyP2",
                "A7jB",
                "kr4K",
                "cfi9",
                "QSb1",
                "HsBL",
                "XoZM",
                "KEsw",
                "pX0E",
                "URvp",
                "oZHr",
                "kitF",
                "JOy4",
                "cy9v",
            ]

        for vid in excess_vids:
            place = np.where(seg_names == vid + ":" + "295")
            if len(place) == 0:
                import pdb;
                pdb.set_trace()
                raise Exception
            index = place[0] + 1
            seg_names = np.insert(seg_names, index, vid + ":" + "300")
            zeros = np.zeros(1000)
            video_preds = np.insert(video_preds, index, zeros, axis=0)

        return video_preds, seg_names

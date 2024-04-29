import unittest
import pandas as pd
from functional import seq

import evaluator

class TestBlinkDetectionEvaluator(unittest.TestCase):

    def setUp(self):
        self.evaluator = evaluator.BlinkDetectionEvaluator()
        self.dataframe = pd.read_csv('test_dataframe.csv')

    def test_evaluate(self):
        results = self.evaluator.evaluate(self.dataframe)

        self.assertEqual(results['tp'], 14)
        self.assertEqual(results['fp'], 2)
        self.assertEqual(results['fn'], 2)
        self.assertEqual(results['db'], 16)
        self.assertEqual(results['f1'], 0.875)
        self.assertEqual(results['precision'], 0.875)
        self.assertEqual(results['recall'], 0.875)

class TestBlinkCompletenessDetectionEvaluator(unittest.TestCase):

    def setUp(self):
        self.evaluator = evaluator.BlinkCompletenessDetectionEvaluator()
        self.dataframe = pd.read_csv('test_dataframe.csv')

    def test_evaluate(self):
        results_partial, results_complete = self.evaluator.evaluate(self.dataframe)

        self.assertEqual(results_partial['tp'], 4)
        self.assertEqual(results_partial['fp'], 4)
        self.assertEqual(results_partial['fn'], 2)
        self.assertEqual(results_partial['db'], 8)
        self.assertEqual(results_partial['f1'], 2 * (2/3) * 0.5 /((2/3) + 0.5 ) )
        self.assertEqual(results_partial['precision'], 0.5)
        self.assertEqual(results_partial['recall'], 2/3)

        self.assertEqual(results_complete['tp'], 6)
        self.assertEqual(results_complete['fp'], 2)
        self.assertEqual(results_complete['fn'], 4)
        self.assertEqual(results_complete['db'], 8)
        self.assertEqual(results_complete['f1'], 2 * 0.6 * 0.75 / ( 0.6 + 0.75 ) )
        self.assertEqual(results_complete['precision'], 0.75)
        self.assertEqual(results_complete['recall'], 0.6)

class TestEyeStateDetectionEvaluator(unittest.TestCase):

    def setUp(self):
        self.evaluator = evaluator.EyeStateDetectionEvaluator()
        self.dataframe = pd.read_csv('test_dataframe.csv')
        self.dataframe['target'] = (self.dataframe['target'] > 0).astype(int)
        self.dataframe['pred'] = (self.dataframe['pred'] > 0).astype(int)
    
    def test_evaluate(self):
        results = self.evaluator.evaluate(self.dataframe)
    
        tp = 44
        fp = 12
        fn = 16
        precision = tp/(tp+fp)
        recall = tp/ (tp+fn)
        f1 = 2 * precision * recall /(precision + recall)
        self.assertEqual(results['tp'], tp)
        self.assertEqual(results['fp'], fp)
        self.assertEqual(results['fn'], fn)
        self.assertEqual(results['f1'], f1)
        self.assertEqual(results['precision'], precision)
        self.assertEqual(results['recall'], recall)


class TestBlinkEvaluator(unittest.TestCase):

    def setUp(self):
        self.evaluator = evaluator.BlinkEvaluator()
        self.dataframe = pd.read_csv('test_dataframe.csv')
        self.raw_blinks_video = [
        [
            evaluator.Blink(1,  4, False, False, 1), 
            evaluator.Blink(6, 7, False, True, 1), 
            evaluator.Blink(8, 10, False, True, 1), 
            evaluator.Blink(13, 14, True, False, 1), 
            evaluator.Blink(17, 19, False, True, 1)
        ],
        [
            evaluator.Blink(0, 1, False, True, 2), 
            evaluator.Blink(4, 7, False, False, 2), 
            evaluator.Blink(11, 14, False, False, 2), 
            evaluator.Blink(18, 21, False, True, 2), 
            evaluator.Blink(24, 27, False, True, 2), 
        ]]
        self.processed_blinks = [
        [
            evaluator.Blink(1,  4, False, False, 1), 
            evaluator.Blink(6, 10, False, True, 1), 
            evaluator.Blink(17, 19, False, True, 1)
        ],
        [
            evaluator.Blink(0, 1, False, True, 2), 
            evaluator.Blink(4, 7, False, False, 2), 
            evaluator.Blink(11, 14, False, False, 2), 
            evaluator.Blink(18, 21, False, True, 2), 
            evaluator.Blink(24, 27, False, True, 2), 
        ]
        ]
        self.predicted_blinks = [
        [
            evaluator.Blink(0,  2, False, False, 1), 
            evaluator.Blink(6,  6, False, False, 1), 
            evaluator.Blink(16,  18, False, True, 1), 
        ],
        [
            evaluator.Blink(0,  1, False, True, 2), 
            evaluator.Blink(3,  7, False, False, 2), 
            evaluator.Blink(10,  14, False, True, 2), 
            evaluator.Blink(17,  21, False, True, 2), 
            evaluator.Blink(23,  26, False, False, 2), 

        ]
        ]

    def test_separate_left_right_eyes(self):
        self.assertEqual(len(self.dataframe), 100)
        left_eyes, right_eyes = self.evaluator.separate_left_right_eyes(self.dataframe)
        self.assertEqual(len(left_eyes), 50)
        self.assertEqual(len(right_eyes), 50)
        self.assertEqual(len(left_eyes[left_eyes['eye'] == 'LEFT']), 50)
        self.assertEqual(len(right_eyes[right_eyes['eye'] == 'RIGHT']), 50)

    def test_extract_blinks_per_video(self):
        left_eyes, right_eyes = self.evaluator.separate_left_right_eyes(self.dataframe)
        left_gt_blinks, left_pred_blinks = self.evaluator.extract_blinks_per_video(left_eyes)
        right_gt_blinks, right_pred_blinks = self.evaluator.extract_blinks_per_video(right_eyes)
        self.assertEqual(len(left_gt_blinks), 2)
        self.assertEqual(len(left_pred_blinks), 2)
        self.assertEqual(len(right_gt_blinks), 2)
        self.assertEqual(len(right_pred_blinks), 2)

        num_gt_per_video = [3, 5]
        num_pred_per_video = [3, 5]
        for video in range(2):
            self.assertEqual(len(left_gt_blinks[video]), num_gt_per_video[video])
            self.assertEqual(len(left_pred_blinks[video]), num_pred_per_video[video])
            self.assertEqual(len(right_gt_blinks[video]), num_gt_per_video[video])
            self.assertEqual(len(right_pred_blinks[video]), num_pred_per_video[video])

    def test_calculate_blinks_for_video(self):
        left_eyes, right_eyes = self.evaluator.separate_left_right_eyes(self.dataframe)
        num_gt_per_video = [3, 5]
        num_pred_per_video = [3, 5]
        for video in range(2):
            left_eyes_of_video = left_eyes[left_eyes['video'] == video + 1].reset_index()
            right_eyes_of_video = right_eyes[right_eyes['video'] == video + 1].reset_index()
            gt_left, pred_left = self.evaluator.calculate_blinks_for_video(left_eyes_of_video)
            gt_right, pred_right = self.evaluator.calculate_blinks_for_video(right_eyes_of_video)
            self.assertEqual(len(gt_left), num_gt_per_video[video])
            self.assertEqual(len(pred_left), num_pred_per_video[video])
            self.assertEqual(len(gt_right), num_gt_per_video[video])
            self.assertEqual(len(pred_right), num_pred_per_video[video])

    def test_convert_annotation_to_blinks(self):
        left_eyes, _ = self.evaluator.separate_left_right_eyes(self.dataframe)
        num_total_blinks_per_video = [5, 5]
        for video in range(2):
            left_eyes_of_video = left_eyes[left_eyes['video'] == video + 1].reset_index()
            left_blinks = self.evaluator.convert_annotation_to_blinks(left_eyes_of_video)
            self.assertEqual(len(left_blinks), num_total_blinks_per_video[video])
    
    def test_convert_predictions_to_blinks(self):
        left_eyes, _ = self.evaluator.separate_left_right_eyes(self.dataframe)
        num_total_blinks_per_video = [3, 5]
        for video in range(2):
            left_eyes_of_video = left_eyes[left_eyes['video'] == video + 1].reset_index()
            left_blinks = self.evaluator.convert_predictions_to_blinks(left_eyes_of_video)
            self.assertEqual(len(left_blinks), num_total_blinks_per_video[video])
    
    def test_delete_non_visible_blinks(self):
        self.assertEqual(len(self.raw_blinks_video[0]), 5)
        gt_visible_blinks = list(filter(lambda blink: not blink.not_visible, self.raw_blinks_video[0]))
        pred_visible_blinks = self.evaluator.delete_non_visible_blinks(self.raw_blinks_video[0])
        
        self.assertEqual(len(gt_visible_blinks), len(pred_visible_blinks))

    def test_merge_double_blinks(self):
        merged_blinks = self.evaluator.merge_double_blinks(self.raw_blinks_video[0])
        self.assertEqual(len(merged_blinks), 4)
        merged_blink = merged_blinks[1]
        self.assertEqual(merged_blink.start, 6)
        self.assertEqual(merged_blink.end, 10)

    def test_calculate_global_confussion_matrix(self):
        tp_all, fp_all, fn_all, db_all = self.evaluator.calculate_global_confussion_matrix(self.processed_blinks, self.predicted_blinks)
        self.assertEqual(tp_all, 7)
        self.assertEqual(fp_all, 1)
        self.assertEqual(fn_all, 1)
        self.assertEqual(db_all, 8)

        complete_processed_blinks = seq(self.processed_blinks).map(lambda video_blinks: seq(video_blinks).filter(lambda blink: blink.complete_blink).to_list()).to_list()
        partial_processed_blinks = seq(self.processed_blinks).map(lambda video_blinks: seq(video_blinks).filter(lambda blink: not blink.complete_blink).to_list()).to_list()

        complete_pred_blinks = seq(self.predicted_blinks).map(lambda video_blinks: seq(video_blinks).filter(lambda blink: blink.complete_blink).to_list()).to_list()
        partial_pred_blinks = seq(self.predicted_blinks).map(lambda video_blinks: seq(video_blinks).filter(lambda blink: not blink.complete_blink).to_list()).to_list()

        tp_complete_all, fp_complete_all, fn_complete_all, db_complete_all = self.evaluator.calculate_global_confussion_matrix(complete_processed_blinks, complete_pred_blinks)
        tp_partial_all, fp_partial_all, fn_partial_all, db_partial_all = self.evaluator.calculate_global_confussion_matrix(partial_processed_blinks, partial_pred_blinks)

        self.assertEqual(tp_complete_all, 3)
        self.assertEqual(fp_complete_all, 1)
        self.assertEqual(fn_complete_all, 2)
        self.assertEqual(db_complete_all, 4)

        self.assertEqual(tp_partial_all, 2)
        self.assertEqual(fp_partial_all, 2)
        self.assertEqual(fn_partial_all, 1)
        self.assertEqual(db_partial_all, 4)

    def test_calculate_video_confussion_matrix(self):
        tp_video_1, fp_video_1, fn_video_1, db_video_1 = self.evaluator.calculate_video_confussion_matrix(self.processed_blinks[0], self.predicted_blinks[0])
        tp_video_2, fp_video_2, fn_video_2, db_video_2 = self.evaluator.calculate_video_confussion_matrix(self.processed_blinks[1], self.predicted_blinks[1])


        self.assertEqual(tp_video_1, 2)
        self.assertEqual(fp_video_1, 1)
        self.assertEqual(fn_video_1, 1)
        self.assertEqual(db_video_1, 3)
        
        self.assertEqual(tp_video_2, 5)
        self.assertEqual(fp_video_2, 0)
        self.assertEqual(fn_video_2, 0)
        self.assertEqual(db_video_2, 5)
        
        complete_processed_blinks = seq(self.processed_blinks).map(lambda video_blinks: seq(video_blinks).filter(lambda blink: blink.complete_blink).to_list()).to_list()
        partial_processed_blinks = seq(self.processed_blinks).map(lambda video_blinks: seq(video_blinks).filter(lambda blink: not blink.complete_blink).to_list()).to_list()

        complete_pred_blinks = seq(self.predicted_blinks).map(lambda video_blinks: seq(video_blinks).filter(lambda blink: blink.complete_blink).to_list()).to_list()
        partial_pred_blinks = seq(self.predicted_blinks).map(lambda video_blinks: seq(video_blinks).filter(lambda blink: not blink.complete_blink).to_list()).to_list()
        
        tp_complete_video_1, fp_complete_video_1, fn_complete_video_1, db_complete_video_1 = self.evaluator.calculate_video_confussion_matrix(complete_processed_blinks[0],complete_pred_blinks[0])
        self.assertEqual(tp_complete_video_1, 1)
        self.assertEqual(fp_complete_video_1, 0)
        self.assertEqual(fn_complete_video_1, 1)
        self.assertEqual(db_complete_video_1, 1)

        tp_complete_video_2, fp_complete_video_2, fn_complete_video_2, db_complete_video_2 = self.evaluator.calculate_video_confussion_matrix(complete_processed_blinks[1],complete_pred_blinks[1])
        self.assertEqual(tp_complete_video_2, 2)
        self.assertEqual(fp_complete_video_2, 1)
        self.assertEqual(fn_complete_video_2, 1)
        self.assertEqual(db_complete_video_2, 3)

        tp_partial_video_1, fp_partial_video_1, fn_partial_video_1, db_partial_video_1 = self.evaluator.calculate_video_confussion_matrix(partial_processed_blinks[0],partial_pred_blinks[0])
        self.assertEqual(tp_partial_video_1, 1)
        self.assertEqual(fp_partial_video_1, 1)
        self.assertEqual(fn_partial_video_1, 0)
        self.assertEqual(db_partial_video_1, 2)

        tp_partial_video_2, fp_partial_video_2, fn_partial_video_2, db_partial_video_2 = self.evaluator.calculate_video_confussion_matrix(partial_processed_blinks[1],partial_pred_blinks[1])
        self.assertEqual(tp_partial_video_2, 1)
        self.assertEqual(fp_partial_video_2, 1)
        self.assertEqual(fn_partial_video_2, 1)
        self.assertEqual(db_partial_video_2, 2)

    def test_calculate_f1_precision_and_recall(self):
        tp = 7
        fp = 1
        fn = 1
        db = 8

        f1, precision, recall = self.evaluator.calculate_f1_precision_and_recall(tp, fp, fn, db)
        self.assertEqual(precision, 0.875)
        self.assertEqual(recall, 0.875)
        self.assertEqual(f1, 0.875)
        
        tp = 2
        fp = 2
        fn = 1
        db = 4

        f1, precision, recall = self.evaluator.calculate_f1_precision_and_recall(tp, fp, fn, db)
        self.assertEqual(precision, 0.5)
        self.assertEqual(recall, 2/3)
        self.assertEqual(f1, 2 * (2/3) * 0.5 /((2/3) + 0.5 ))
        
        tp = 3
        fp = 1
        fn = 2
        db = 4

        f1, precision, recall = self.evaluator.calculate_f1_precision_and_recall(tp, fp, fn, db)
        self.assertEqual(precision, 0.75)
        self.assertEqual(recall, 0.6)
        self.assertEqual(f1, 2 * 0.6 * 0.75 / ( 0.6 + 0.75 ))



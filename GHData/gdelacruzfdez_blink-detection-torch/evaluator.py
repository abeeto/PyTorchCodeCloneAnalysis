from sklearn import metrics

EYE_OPEN = 0
EYE_PARTIALLY_CLOSED = 1
EYE_CLOSED = 2

PARTIAL_BLINK = 0
COMPLETE_BLINK = 1


class Blink(object):
    start = 0
    end = 0
    not_visible = False
    complete_blink = False
    video = -1

    def __init__(self, start, end, not_visible, complete_blink, video):
        self.start = start
        self.end = end
        self.not_visible = not_visible
        self.complete_blink = complete_blink
        self.video = video
        
    def __str__(self):
        return '[ start:{}, end:{}, not_visible:{}, complete_blink:{}, video:{} ]'.format(self.start, self.end, self.not_visible, self.complete_blink, self.video)

    def __repr__(self):
        return self.__str__()


class EvaluatorInterface:

    def evaluate(self, dataframe):
        pass


class BlinkEvaluator(EvaluatorInterface):

    def calculate_f1_precision_and_recall(self, tp, fp, fn, db):
        precision = 0
        if db > 0:
            precision = tp/db

        recall = 0
        if((tp + fn) > 0):
            recall = tp/(tp + fn)

        f1 = 0

        if(precision + recall) > 0:
            f1 = 2*precision*recall/(precision + recall)

        return f1, precision, recall


    def separate_left_right_eyes(self, dataframe):
        left_eyes =  dataframe[dataframe['eye'] == 'LEFT'].reset_index()
        right_eyes = dataframe[dataframe['eye'] == 'RIGHT'].reset_index()
        return left_eyes, right_eyes

    def extract_blinks_per_video(self, dataframe):
        pass


    def calculate_global_confussion_matrix(self, gt_blinks_per_video, pred_blinks_per_video):
        num_videos = len(gt_blinks_per_video)

        db_all = fn_all = tp_all = fp_all = 0

        for i in range(0, num_videos):
            gt_blinks = gt_blinks_per_video[i]
            pred_blinks = pred_blinks_per_video[i]

            tp, fp, fn, db = self.calculate_video_confussion_matrix(gt_blinks, pred_blinks)

            db_all += db
            fn_all += fn
            tp_all += tp
            fp_all += fp
        
        return tp_all, fp_all, fn_all, db_all

    def calculate_video_confussion_matrix(self, gt_blinks, pred_blinks):
        fp = self.calc_fp(gt_blinks, pred_blinks)
        fn = self.calc_fp(pred_blinks, gt_blinks)
        db = len(pred_blinks)
        tp = db - fp
        return tp, fp, fn, db

    
    def calc_fp(self, gt_blinks, pred_blinks):
        pred_blinks_copy = pred_blinks.copy()
        gt_blinks_copy = gt_blinks.copy()
        i = 0
        j = 0
        fp_blinks_counter = 0
        iou_detection = 0.2
        while i < len(pred_blinks_copy) or j < len(gt_blinks_copy):
            if i == len(pred_blinks_copy) and j < len(gt_blinks_copy):
                break
            if i < len(pred_blinks_copy) and j == len(gt_blinks_copy):
                fp_blinks_counter += 1
                i += 1
                continue
            if self.iou(pred_blinks_copy[i], gt_blinks_copy[j]) > iou_detection:
                i3 = i
                iouArray3 = []
                k3 = 0
                while j < len(gt_blinks_copy) and i3 < len(pred_blinks_copy):
                    temp = self.iou(pred_blinks_copy[i3], gt_blinks_copy[j])
                    if temp > iou_detection:
                        iouArray3.append(temp)
                        k3 += 1
                        i3 += 1
                    else:
                        break
                if k3 > 1:
                    max = iouArray3[0]
                    index = 0
                    for f in range(1, k3):
                        if max < iouArray3[f]:
                            max = iouArray3[f]
                            index = f
                    del pred_blinks_copy[i+index]
                    del gt_blinks_copy[j]
                    continue
                else:
                    i += 1
                    j += 1
                    continue
            if pred_blinks_copy[i].end < gt_blinks_copy[j].end:
                fp_blinks_counter += 1
                i += 1
            else:
                j += 1
        return fp_blinks_counter

    def iou(self, blink1, blink2):
        # intervals are 3 digits, middle one tells about the partial-0 / full-1 blink property
        min = blink1.start
        diffMin = blink2.start - blink1.start
        if min > blink2.start:
            diffMin = blink1.start - blink2.start
            min = blink2.start
        max = blink1.end
        diffMax = blink1.end - blink2.end
        if max < blink2.end:
            diffMax = blink2.end - blink1.end
            max = blink2.end
        unionCount = max-min-diffMin-diffMax+1
        if unionCount <= 0:
            return 0
        return(unionCount/float(max-min+1))


    def calculate_blinks_for_video(self, video_dataframe):
        pass
        


    def convert_annotation_to_blinks(self, dataframe):
        i = 0
        blinks = []
        while i < len(dataframe):
            if(dataframe.loc[i]['blink_id'] != -1):
                id = dataframe.loc[i]['blink_id']
                fully_closed = False
                not_visible = False
                start = dataframe.loc[i]['frameId']
                while i < len(dataframe) and dataframe.loc[i]['blink_id'] == id:
                    if dataframe.loc[i]['blink'] == 1:
                        fully_closed = True
                    if dataframe.loc[i]['NV'] == 1:
                        not_visible = True
                    i += 1
                i -= 1
                end = dataframe.loc[i]['frameId']
                blink = Blink(start, end, not_visible, fully_closed, dataframe.loc[i]['video'])
                blinks.append(blink)
            i += 1
        return blinks

    def convert_predictions_to_blinks(self, dataframe):
        i = 0
        blinks = []
        while i < len(dataframe):
            if(dataframe.loc[i]['pred'] > 0):
                fully_closed = False
                not_visible = False
                start = dataframe.loc[i]['frameId']
                while i < len(dataframe) and dataframe.loc[i]['pred'] > 0:
                    if dataframe.loc[i]['pred'] == EYE_CLOSED:
                        fully_closed = True
                    if dataframe.loc[i]['NV'] == 1:
                        not_visible = True
                    i += 1
                i -= 1
                end = dataframe.loc[i]['frameId']
                blink = Blink(start, end, not_visible, fully_closed, dataframe.loc[i]['video'])
                blinks.append(blink)
            i += 1
        return blinks

    def delete_non_visible_blinks(self, blinks):
        visible_blinks = []
        for blink in blinks:
            if blink.not_visible == False:
                visible_blinks.append(blink)
        return visible_blinks

    def merge_double_blinks(self, blinks):
        i = 1
        while i < len(blinks):
            if blinks[i-1].end == blinks[i].start - 1:
                blinks[i].start = blinks[i-1].start
                i -= 1
                blinks.pop(i)
            i += 1
        if len(blinks) > 1 and blinks[-2].end == blinks[-1].start - 1:
            blinks[-1].start = blinks[-1].start
            blinks.pop(-2)
        return blinks


class BlinkDetectionEvaluator(BlinkEvaluator):

    def evaluate(self, dataframe):
        left_eyes, right_eyes = self.separate_left_right_eyes(dataframe)
        gt_blinks_left, pred_blinks_left = self.extract_blinks_per_video(left_eyes)
        gt_blinks_right, pred_blinks_right = self.extract_blinks_per_video(right_eyes)

        tp_left, fp_left, fn_left, db_left = self.calculate_global_confussion_matrix(gt_blinks_left, pred_blinks_left)
        tp_right, fp_right, fn_right, db_right = self.calculate_global_confussion_matrix(gt_blinks_right, pred_blinks_right)

        tp = tp_left + tp_right
        fp = fp_left + fp_right
        fn = fn_left + fn_right
        db = db_left + db_right

        f1, precision, recall = self.calculate_f1_precision_and_recall(tp, fp, fn, db)


        ret_dict = {'f1': f1, 'precision': precision,
                    'recall': recall, 'tp': tp, 'fp': fp, 'fn': fn, 'db': db}
        return ret_dict

    def extract_blinks_per_video(self, dataframe):
        num_videos = dataframe['video'].max()
        
        all_gt_blinks = []
        all_pred_blinks = []

        for i in range(1, num_videos + 1):
            video_dataframe = dataframe[dataframe['video'] == i].reset_index()
            gt_blinks, pred_blinks = self.calculate_blinks_for_video(video_dataframe)

            all_gt_blinks.append(gt_blinks)
            all_pred_blinks.append(pred_blinks)
        
        return all_gt_blinks, all_pred_blinks
    
    def calculate_blinks_for_video(self, video_dataframe):

        gt_blinks = self.convert_annotation_to_blinks(video_dataframe)
        gt_blinks = self.delete_non_visible_blinks(gt_blinks)
        gt_blinks = self.merge_double_blinks(gt_blinks)

        pred_blinks = self.convert_predictions_to_blinks(video_dataframe)
        pred_blinks = self.delete_non_visible_blinks(pred_blinks)
        pred_blinks = self.merge_double_blinks(pred_blinks)
        
        return gt_blinks, pred_blinks
        


class BlinkCompletenessDetectionEvaluator(BlinkEvaluator):

    def evaluate(self, dataframe):
        left_eyes, right_eyes = self.separate_left_right_eyes(dataframe)
        partial_gt_left_blinks, complete_gt_left_blinks, partial_pred_left_blinks, complete_pred_left_blinks = self.extract_blinks_per_video(left_eyes)
        partial_gt_right_blinks, complete_gt_right_blinks, partial_pred_right_blinks, complete_pred_right_blinks = self.extract_blinks_per_video(right_eyes)


        tp_partial_left, fp_partial_left, fn_partial_left, db_partial_left = self.calculate_global_confussion_matrix(partial_gt_left_blinks, partial_pred_left_blinks)
        tp_complete_left, fp_complete_left, fn_complete_left, db_complete_left = self.calculate_global_confussion_matrix(complete_gt_left_blinks, complete_pred_left_blinks)
        tp_partial_right, fp_partial_right, fn_partial_right, db_partial_right = self.calculate_global_confussion_matrix(partial_gt_right_blinks, partial_pred_right_blinks)
        tp_complete_right, fp_complete_right, fn_complete_right, db_complete_right = self.calculate_global_confussion_matrix(complete_gt_right_blinks, complete_pred_right_blinks)

        tp_partial = tp_partial_left + tp_partial_right
        fp_partial = fp_partial_left + fp_partial_right
        fn_partial = fn_partial_left + fn_partial_right
        db_partial = db_partial_left + db_partial_right


        tp_complete = tp_complete_left + tp_complete_right
        fp_complete = fp_complete_left + fp_complete_right
        fn_complete = fn_complete_left + fn_complete_right
        db_complete = db_complete_left + db_complete_right

        f1_partial, precision_partial, recall_partial = self.calculate_f1_precision_and_recall(tp_partial, fp_partial, fn_partial, db_partial)
        f1_complete, precision_complete, recall_complete = self.calculate_f1_precision_and_recall(tp_complete, fp_complete, fn_complete, db_complete)


        ret_dict_partial = {'f1': f1_partial, 'precision': precision_partial,
                    'recall': recall_partial, 'tp': tp_partial, 'fp': fp_partial, 'fn': fn_partial, 'db': db_partial}

        ret_dict_complete = {'f1': f1_complete, 'precision': precision_complete,
                    'recall': recall_complete, 'tp': tp_complete, 'fp': fp_complete, 'fn': fn_complete, 'db': db_complete}
        return ret_dict_partial, ret_dict_complete
    
    def extract_blinks_per_video(self, dataframe):
        num_videos = dataframe['video'].max()
        
        all_partial_gt_blinks = []
        all_complete_gt_blinks = []
        all_partial_pred_blinks = []
        all_complete_pred_blinks = []

        for i in range(1, num_videos + 1):
            video_dataframe = dataframe[dataframe['video'] == i].reset_index()
            partial_gt_blinks, complete_gt_blinks, partial_pred_blinks, complete_pred_blinks = self.calculate_blinks_for_video(video_dataframe)

            all_partial_gt_blinks.append(partial_gt_blinks)
            all_complete_gt_blinks.append(complete_gt_blinks)

            all_partial_pred_blinks.append(partial_pred_blinks)
            all_complete_pred_blinks.append(complete_pred_blinks)
        
        return all_partial_gt_blinks, all_complete_gt_blinks, all_partial_pred_blinks, all_complete_pred_blinks
    
    def calculate_blinks_for_video(self, video_dataframe):

        gt_blinks = self.convert_annotation_to_blinks(video_dataframe)
        partial_gt_blinks, complete_gt_blinks = self.divide_partial_and_full_blinks(gt_blinks)
        partial_gt_blinks = self.merge_double_blinks(self.delete_non_visible_blinks(partial_gt_blinks))
        complete_gt_blinks = self.merge_double_blinks(self.delete_non_visible_blinks(complete_gt_blinks))

        pred_blinks = self.convert_predictions_to_blinks(video_dataframe)
        partial_pred_blinks, complete_pred_blinks = self.divide_partial_and_full_blinks(pred_blinks)
        partial_pred_blinks = self.merge_double_blinks(self.delete_non_visible_blinks(partial_pred_blinks))
        complete_pred_blinks = self.merge_double_blinks(self.delete_non_visible_blinks(complete_pred_blinks))
        
        return partial_gt_blinks, complete_gt_blinks, partial_pred_blinks, complete_pred_blinks


    def divide_partial_and_full_blinks(self, blinks):
        partial = []
        full = []
        for b in blinks:
            if PARTIAL_BLINK == b.complete_blink:
                partial.append(b)
            else:
                full.append(b)
        return partial, full


class EyeStateDetectionEvaluator(EvaluatorInterface):

    def evaluate(self, dataframe):
        preds = dataframe['pred']
        targets = dataframe['target']
        print(metrics.classification_report(targets, preds, target_names=['Open', 'Closed']))
        confussion_matrix = metrics.confusion_matrix(targets, preds).ravel()
        print(confussion_matrix)
        precisionRecallF1 = metrics.precision_recall_fscore_support(targets, preds, average='binary')
        print(precisionRecallF1)
        results = {'f1': precisionRecallF1[2], 'precision':precisionRecallF1[0], 'recall': precisionRecallF1[1], 'fp':confussion_matrix[1], 'fn': confussion_matrix[2], 'tp':confussion_matrix[3], 'db': 0}
        fpr, tpr, thresholds = metrics.roc_curve(targets, preds)
        auc = metrics.auc(fpr, tpr)
        print('AUC: ' + str(auc))
        return results


import csv
EYE_OPEN = 0
EYE_PARTIALLY_CLOSED = 1
EYE_CLOSED = 2

PARTIAL_BLINK = 0
COMPLETE_BLINK = 1


def evaluate(dataframe):
    results = extractBlinks(dataframe)
    return results


def evaluatePartialBlinks(dataframe):
    return extractPartialCompleteBlinks(dataframe)


def extractPartialCompleteBlinks(dataframe):
    leftFrames = dataframe[dataframe['eye'] == 'LEFT'].reset_index()
    rightFrames = dataframe[dataframe['eye'] == 'RIGHT'].reset_index()

    numVideos = dataframe['video'].max()

    allPartialLeftBlinks = []
    allPartialRightBlinks = []
    predictedPartialLeftBlinks = []
    predictedPartialRightBlinks = []
    allCompleteLeftBlinks = []
    allCompleteRightBlinks = []

    db_partial = 0
    fn_partial = 0
    fp_partial = 0
    tp_partial = 0

    db_complete = 0
    fn_complete = 0
    fp_complete = 0
    tp_complete = 0

    for i in range(1, numVideos+1):
        left = leftFrames[leftFrames['video'] == i].reset_index()
        right = rightFrames[rightFrames['video'] == i].reset_index()

        partial_left_blinks, complete_left_blinks = extractPartialAndFullBlinks(
            convertAnnotationToBlinks(left))
        partial_right_blinks, complete_right_blinks = extractPartialAndFullBlinks(
            convertAnnotationToBlinks(right))
        pred_partial_left_blinks, pred_complete_left_blinks = extractPartialAndFullBlinks(
            convertPredictionsToBlinks(left))
        pred_partial_right_blinks, pred_complete_right_blinks = extractPartialAndFullBlinks(
            convertPredictionsToBlinks(right))

        partial_left_blinks = mergeDoubleBlinks(
            deleteNonVisibleBlinks(partial_left_blinks))
        partial_right_blinks = mergeDoubleBlinks(
            deleteNonVisibleBlinks(partial_right_blinks))
        complete_left_blinks = mergeDoubleBlinks(
            deleteNonVisibleBlinks(complete_left_blinks))
        complete_right_blinks = mergeDoubleBlinks(
            deleteNonVisibleBlinks(complete_right_blinks))

        pred_partial_left_blinks = mergeDoubleBlinks(
            deleteNonVisibleBlinks(pred_partial_left_blinks))
        pred_partial_right_blinks = mergeDoubleBlinks(
            deleteNonVisibleBlinks(pred_partial_right_blinks))
        pred_complete_left_blinks = mergeDoubleBlinks(
            deleteNonVisibleBlinks(pred_complete_left_blinks))
        pred_complete_right_blinks = mergeDoubleBlinks(
            deleteNonVisibleBlinks(pred_complete_right_blinks))

        allPartialLeftBlinks.extend(partial_left_blinks)
        allPartialRightBlinks.extend(partial_right_blinks)
        predictedPartialLeftBlinks.extend(pred_partial_left_blinks)
        predictedPartialRightBlinks.extend(pred_partial_right_blinks)
        allCompleteLeftBlinks.extend(complete_left_blinks)
        allCompleteRightBlinks.extend(complete_right_blinks)

        fp_partial_left, fn_partial_left, db_partial_left, tp_partial_left = calculateConfussionMatrix(
            pred_partial_left_blinks, partial_left_blinks)
        fp_partial_right, fn_partial_right, db_partial_right, tp_partial_right = calculateConfussionMatrix(
            pred_partial_right_blinks, partial_right_blinks)
        fp_complete_left, fn_complete_left, db_complete_left, tp_complete_left = calculateConfussionMatrix(
            pred_complete_left_blinks, complete_left_blinks)
        fp_complete_right, fn_complete_right, db_complete_right, tp_complete_right = calculateConfussionMatrix(
            pred_complete_right_blinks, complete_right_blinks)

        db_partial += db_partial_left + db_partial_right
        fn_partial += fn_partial_left + fn_partial_right
        fp_partial += fp_partial_left + fp_partial_right
        tp_partial += tp_partial_left + tp_partial_right

        db_complete += db_complete_left + db_complete_right
        fn_complete += fn_complete_left + fn_complete_right
        fp_complete += fp_complete_left + fp_complete_right
        tp_complete += tp_complete_left + tp_complete_right

    print(len(allPartialLeftBlinks), len(allPartialRightBlinks),
          len(allCompleteLeftBlinks), len(allCompleteRightBlinks))

    results_partial = calculateStatistics(fp_partial, fn_partial, db_partial, tp_partial)
    results_complete = calculateStatistics(fp_complete, fn_complete, db_complete, tp_complete)
    return (results_partial, results_complete)


def extractBlinks(dataframe):
    leftFrames = dataframe[dataframe['eye'] == 'LEFT'].reset_index()
    rightFrames = dataframe[dataframe['eye'] == 'RIGHT'].reset_index()

    numVideos = dataframe['video'].max()

    all_left_blinks = []
    all_right_blinks = []
    all_pred_left_blinks = []
    all_pred_right_blinks = []

    db = 0
    fn = 0
    fp = 0
    tp = 0

    for i in range(1, numVideos+1):
        left = leftFrames[leftFrames['video'] == i].reset_index()
        right = rightFrames[rightFrames['video'] == i].reset_index()

        left_blinks = mergeDoubleBlinks(
            deleteNonVisibleBlinks(convertAnnotationToBlinks(left)))
        right_blinks = mergeDoubleBlinks(
            deleteNonVisibleBlinks(convertAnnotationToBlinks(right)))
        pred_left_blinks = mergeDoubleBlinks(
            deleteNonVisibleBlinks(convertPredictionsToBlinks(left)))
        pred_right_blinks = mergeDoubleBlinks(
            deleteNonVisibleBlinks(convertPredictionsToBlinks(right)))

        fp_left, fn_left, db_left, tp_left = calculateConfussionMatrix(
            pred_left_blinks, left_blinks)
        fp_right, fn_right, db_right, tp_right = calculateConfussionMatrix(
            pred_right_blinks, right_blinks)

        db += db_left + db_right
        fn += fn_left + fn_right
        fp += fp_left + fp_right
        tp += tp_left + tp_right

        all_left_blinks.extend(left_blinks)
        all_right_blinks.extend(right_blinks)
        all_pred_left_blinks.extend(pred_left_blinks)
        all_pred_right_blinks.extend(pred_right_blinks)

    print('TOTAL BLINKS:', len(all_left_blinks), len(all_right_blinks),
          len(all_pred_left_blinks), len(all_pred_right_blinks))
    return calculateStatistics(fp, fn, db, tp)


def extractBlinksFromPredictions(dataframe):
    leftFrames = dataframe[dataframe['eye'] == 'LEFT'].reset_index()
    rightFrames = dataframe[dataframe['eye'] == 'RIGHT'].reset_index()

    numVideos = dataframe['video'].max()

    all_left_blinks = []
    all_right_blinks = []

    for i in range(1, numVideos+1):
        left = leftFrames[leftFrames['video'] == i].reset_index()
        right = rightFrames[rightFrames['video'] == i].reset_index()

        left_blinks = mergeDoubleBlinks(
            deleteNonVisibleBlinks(convertPredictionsToBlinks(left)))
        right_blinks = mergeDoubleBlinks(
            deleteNonVisibleBlinks(convertPredictionsToBlinks(right)))

        all_left_blinks.extend(left_blinks)
        all_right_blinks.extend(right_blinks)

    return all_left_blinks, all_right_blinks


def realBlinks(annotations):
    blinks = []
    index = 0
    blink_id = -1
    while index < len(annotations.index):
        row = annotations.loc[index]
        if row['blink_id'] > 0:
            blink_id = row['blink_id']
            start = index
            notVisible = False
            if row['NV'] == True:
                notVisible = True
            while index < len(annotations.index) and row['blink_id'] == blink_id:
                index += 1
                row = annotations.loc[index]
            index -= 1
            end = index
            blinks.append({'start': start, 'end': end,
                           'notVisible': notVisible, 'video': row['video']})
            blink_id = -1
        index += 1
    return blinks


def mergeDoubleBlinks(blinks):
    i = 1
    while i < len(blinks):
        if blinks[i-1]['end'] == blinks[i]['start'] - 1:
            blinks[i]['start'] = blinks[i-1]['start']
            i -= 1
            blinks.pop(i)
        i += 1
    if len(blinks) > 1 and blinks[-2]['end'] == blinks[-1]['start']-1:
        blinks[-1]['start'] = blinks[-1]['start']
        blinks.pop(-2)
    return blinks


def convertAnnotationToBlinks(annotations):
    i = 0
    blinks = []
    while i < len(annotations):
        if(annotations.loc[i]['blink_id'] != -1):
            id = annotations.loc[i]['blink_id']
            fullyClosed = False
            notVisible = False
            start = annotations.loc[i]['frameId']
            while i < len(annotations) and annotations.loc[i]['blink_id'] == id:
                if annotations.loc[i]['blink'] == 1:
                    fullyClosed = True
                if annotations.loc[i]['NV'] == 1:
                    notVisible = True
                i += 1
            i -= 1
            end = annotations.loc[i]['frameId']
            blinks.append({'start': start, 'end': end, 'notVisible': notVisible,
                           'completeBlink': fullyClosed, 'video': annotations.loc[i]['video']})
        i += 1
    return blinks


def convertPredictionsToBlinks(annotations):
    i = 0
    blinks = []
    while i < len(annotations):
        if(annotations.loc[i]['pred'] > 0):
            fullyClosed = False
            notVisible = False
            start = annotations.loc[i]['frameId']
            while i < len(annotations) and annotations.loc[i]['pred'] > 0:
                if annotations.loc[i]['pred'] == EYE_CLOSED:
                    fullyClosed = True
                if annotations.loc[i]['NV'] == 1:
                    notVisible = True
                i += 1
            i -= 1
            end = annotations.loc[i]['frameId']
            blinks.append({'start': start, 'end': end, 'notVisible': notVisible,
                           'completeBlink': fullyClosed, 'video': annotations.loc[i]['video']})
        i += 1
    return blinks


def convertAnnotationToBlinksOld(annotations, blink_col):
    blinks = []
    index = 0
    while index < len(annotations.index):
        row = annotations.loc[index]
        if row[blink_col] > 0:
            id = row[blink_col]
            start = index
            notVisible = False
            if row['NV'] == True:
                notVisible = True
            while index < len(annotations.index) and row[blink_col] > 0:
                if row['NV'] == True:
                    notVisible = True
                index += 1
                row = annotations.loc[index]
            index -= 1
            end = index
            blinks.append({'start': start, 'end': end,
                           'notVisible': notVisible})
        index += 1
    return blinks


def deleteNonVisibleBlinks(blinks):
    newBlinks = []
    for blink in blinks:
        if blink['notVisible'] == False:
            newBlinks.append(blink)
    return newBlinks


def calcFP(detectedBlinksOriginal, groundTruthOriginal):
    detectedBlinks = detectedBlinksOriginal.copy()
    groundTruth = groundTruthOriginal.copy()
    i = 0
    j = 0
    blinkFPCounter = 0
    iou_detection = 0.2
    while i < len(detectedBlinks) or j < len(groundTruth):
        if i == len(detectedBlinks) and j < len(groundTruth):
            break
        if i < len(detectedBlinks) and j == len(groundTruth):
            blinkFPCounter += 1
            i += 1
            continue
        if iou(detectedBlinks[i], groundTruth[j]) > iou_detection:
            i3 = i
            iouArray3 = []
            k3 = 0
            while j < len(groundTruth) and i3 < len(detectedBlinks):
                temp = iou(detectedBlinks[i3], groundTruth[j])
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
                del detectedBlinks[i+index]
                del groundTruth[j]
                continue
            else:
                i += 1
                j += 1
                continue
        if detectedBlinks[i]['end'] < groundTruth[j]['end']:
            blinkFPCounter += 1
            i += 1
        else:
            j += 1
    return blinkFPCounter


def iou(blink1, blink2):
    # intervals are 3 digits, middle one tells about the partial-0 / full-1 blink property
    min = blink1['start']
    diffMin = blink2['start']-blink1['start']
    if min > blink2['start']:
        diffMin = blink1['start']-blink2['start']
        min = blink2['start']
    max = blink1['end']
    diffMax = blink1['end']-blink2['end']
    if max < blink2['end']:
        diffMax = blink2['end']-blink1['end']
        max = blink2['end']
    unionCount = max-min-diffMin-diffMax+1
    if unionCount <= 0:
        return 0
    return(unionCount/float(max-min+1))


def calculateConfussionMatrix(pred_blinks, true_blinks):
    fp = calcFP(pred_blinks, true_blinks)
    fn = calcFP(true_blinks, pred_blinks)
    db = len(pred_blinks)
    tp = db - fp
    return fp, fn, db, tp


def calculateStatistics(fp, fn, db, tp):
    precision = 0
    if db > 0:
        precision = tp/db

    recall = 0
    if((tp + fn) > 0):
        recall = tp/(tp + fn)

    f1 = 0

    if(precision + recall) > 0:
        f1 = 2*precision*recall/(precision + recall)

    ret_dict = {'f1': f1, 'precision': precision,
                'recall': recall, 'tp': tp, 'fp': fp, 'fn': fn, 'db': db}
    return ret_dict


def mergeNeighbourBlinks(blinks):
    results = []
    i = 1
    while i < len(blinks):
        if blinks[i-1]['end'] + 1 != blinks[i]['start']:
            results.append(blinks[i-1])
            i += 1
        else:
            blink = {'start': blinks[i-1]['start'], 'notVisible': blinks[i-1]
                     ['notVisible'] or blinks[i]['notVisible'], 'end': blinks[i]['end']}
            i += 1
            while i < len(blinks) and blink['end']+1 == blinks[i]['start']:
                blink = {'start': blinks[i-1]['start'], 'notVisible': blinks[i-1]
                         ['notVisible'] or blinks[i]['notVisible'], 'end': blinks[i]['end']}
                i += 1
            blinks[i-1] = blink

        if i == len(blinks):
            results.append(blinks[i-1])
    return results


def extractPartialAndFullBlinks(blinks):
    partial = []
    full = []
    for b in blinks:
        if PARTIAL_BLINK == b['completeBlink']:
            partial.append(b)
        else:
            full.append(b)
    return partial, full

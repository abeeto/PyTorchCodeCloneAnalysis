

rootPath = "/local-scratch/aarab/CTStroke/segmentation/GroundTruthNii"
import nibabel as nib
import numpy as np

#accuracy  = []
#precisions = []
#recalls = []
#for imgNum in [2,3,4,6,8,9,10]:
#       print imgNum
#       img1Path = "{}/ACT{:02}/pitchperfectResult/final.nii.gz.nii.gz".format(rootPath, imgNum)
#       img2Path = "{}/ACT{:02}/groundTruth/img.nii.gz".format(rootPath, imgNum)
#
#       img1= nib.load(img1Path)
#       img2= nib.load(img2Path)
#
#       V1 = img1.get_data()
#       V2 = img2.get_data()
#
#       V1Indc =  np.nonzero(V1)
#       V1Set = set()
#       print V1Indc
#       for i in range(0, len(V1Indc[0])):
#               V1Set.add((V1Indc[0][i] , V1Indc[1][i], V1Indc[2][i]))
#       print len(V1Set)
#
#
#       V2Indc =  np.nonzero(V2)
#       V2Set = set()
#       for i in range(0, len(V2Indc[0])):
#               V2Set.add((V2Indc[0][i],V2Indc[1][i], V2Indc[2][i]))    
#       print len(V2Set)
#
#
#       intersect = V1Set & V2Set
#       union = V1Set | V2Set
#
#
#       jaccard = (2.0*len(intersect))/(len(V1Set)+len(V2Set))
#       accuracy.append(jaccard)
#
#
#
#
#       dim = V1.shape
#       I = dim[0]
#       J = dim[1]
#       K = dim[2]
#
#       TP = set()
#       TN = set()
#       FP = set()
#       FN = set()
#
#       for i in range(0, I):
#               for j in range(0,J):
#                       for k in range(0,K):
#                               if (V1[i,j,k]!=0) and (V2[i,j,k]!=0):
#                                       TP.add((i,j,k))
#                               if (V1[i,j,k]!=0) and (V2[i,j,k]==0):
#                                       FP.add((i,j,k))
#                               if (V1[i,j,k]==0) and (V2[i,j,k]!=0):
#                                       FN.add((i,j,k))
#                               # if V1[i,j,k]==0 and V2[i,j,k]==0:
#                               #       TN.add((i,j,k))
#
#
#
#       precision = (len(TP)*1.0)/(len(TP)+len(FP))
#       recall = (len(TP)*1.0)/(len(TP)+len(FN))
#
#
#       precisions.append(precision)
#       print "p:", precision
#       recalls.append(recall)
#       print "r:", recall
#
#
#
#
#print np.mean(accuracy)
#print np.mean(precisions)
#print np.mean(recalls)



#rootPath = "/home/ali/Documents/SFU/Research/CTStroke/segmentation/GroundTruthNii/"


imagesList = [1,2,3,4,5,6,7,8,9,10]
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

def computeAccuracyMeasures():
        #print "compute accuracy measures:"
        jaccards  = []
        precisions = []
        recalls = []
        dices = []

        for subject in  ["ACT01","ACT02","ACT03","ACT04","ACT05","ACT06","ACT07","ACT08","ACT09","ACT10"]:
                print(subject)
                # print imgNum
                img1Path = "data/prediction/{}/prediction.nii.gz".format(subject)
                img2Path = "data/prediction/{}/truth.nii.gz".format(subject)

                img1= nib.load(img1Path)
                img2= nib.load(img2Path)

                V1 = img1.get_data()
                V2 = img2.get_data()


                prediction=V1.flatten()
                truth=V2.flatten()

                p=precision_score(truth, prediction)
                precisions.append(p)
                r = recall_score(truth, prediction)
                recalls.append(r)

                """
                V1Indc =  np.nonzero(V1)
                V1Set = set()
                # print V1Indc
                for i in range(0, len(V1Indc[0])):
                        V1Set.add((V1Indc[0][i] , V1Indc[1][i], V1Indc[2][i]))
                # print len(V1Set)


                V2Indc =  np.nonzero(V2)
                V2Set = set()
                for i in range(0, len(V2Indc[0])):
                        V2Set.add((V2Indc[0][i],V2Indc[1][i], V2Indc[2][i]))    
                # print len(V2Set)


                intersect = V1Set & V2Set
                union = V1Set | V2Set


                dice = (2.*len(intersect))/(len(V1Set)+len(V2Set))
                # dice= 2*jaccard/(1+jaccard)
                # jaccards.append(jaccard)
                dices.append(dice)




                dim = V1.shape
                I = dim[0]
                J = dim[1]
                K = dim[2]

                TP = set()
                TN = set()
                FP = set()
                FN = set()

                for i in range(0, I):
                        for j in range(0,J):
                                for k in range(0,K):
                                        if (V1[i,j,k]!=0) and (V2[i,j,k]!=0):
                                                TP.add((i,j,k))
                                        if (V1[i,j,k]!=0) and (V2[i,j,k]==0):
                                                FP.add((i,j,k))
                                        if (V1[i,j,k]==0) and (V2[i,j,k]!=0):
                                                FN.add((i,j,k))
                                        if V1[i,j,k]==0 and V2[i,j,k]==0:
                                                TN.add((i,j,k))



                precision = (len(TP)*1.0)/(len(TP)+len(FN))
                recall = (len(TN)*1.0)/(len(TN)+len(FP))


                precisions.append(precision)
                #print ("p:".format(precision))
                recalls.append(recall)
                #print ("r:".format(recall))

                print ("img: {}:, precision: {} , recall: {}".format(imgNum, precision, recall))

"""

        # print np.mean(accuracy)
        #print (np.mean(dices))
        print (np.mean(precisions))
        print (np.std(precisions))
        print (np.mean(recalls))
        print (np.std(recalls))

computeAccuracyMeasures()

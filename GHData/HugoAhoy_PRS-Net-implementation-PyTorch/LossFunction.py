import torch
import torch.nn as nn
import torchsnooper

class SymmetryDistanceLoss(nn.Module):
    def __init__(self):
        super(SymmetryDistanceLoss, self).__init__()
    
    def forward(self, output, target):
        batchSize = output.shape[0]
        self.loss = torch.zeros(1).cuda()
        # self.totalSamplePoints = 0
        for batch in range(batchSize):
            self.Q = target['points'][batch]
            # self.totalSamplePoints = self.totalSamplePoints + len(self.Q)
            self.ClosestGrid = target['closest'][batch]
            self.batch = batch
            self.SymPoints = []
            for i in range(3):
                self.ReflectiveDistance(output[batch][i])

            for i in range(3,6):
                self.RotationDistance(output[batch][i])
            
            self.loss = self.loss + self.totalDis()/len(self.Q)
        
        return self.loss/batchSize

    def ReflectiveDistance(self, ReflectivePlane):
        # get normal vector of the plane
        nv = ReflectivePlane[0:3]
        d = ReflectivePlane[3]

        for k in range(len(self.Q)):
            q = self.Q[k]
            dis = (torch.dot(nv, q)+d)/(torch.dot(nv, nv))
            q_sym = q - 2*(dis)*nv
            self.SymPoints.append(q_sym)

    def RotationDistance(self, RotationQuater):
        for k in range(len(self.Q)):
            q = self.Q[k]
            q_hat = torch.zeros(4).cuda()
            q_hat[1:] = q
            q_sym = self.QuaternionProduct(self.QuaternionProduct(RotationQuater,q_hat), self.QuaternionInverse(RotationQuater))[1:]
            self.SymPoints.append(q_sym)

    # @torchsnooper.snoop()
    def totalDis(self):
        totalDis = torch.zeros(1).cuda()
        for i in range(len(self.SymPoints)):
            x, y, z = self.SymPoints[i]
            # print(x, y, z)
            if x < 0:
                x = 0
            elif x >= 32:
                x = 31
            
            if y < 0:
                y = 0
            elif y >= 32:
                y = 31

            if z < 0:
                z = 0
            elif z >= 32:
                z = 31

            CP = self.ClosestGrid[int(x)*32*32+int(y)*32+int(z)]
            totalDis = totalDis + torch.norm(self.SymPoints[i] - CP)

        return totalDis

    # 四元数乘法
    # q1q2 = (s1s2 - v1·v2) +s1v2 + s2v1 + v1Xv2
    def QuaternionProduct(self, Qa, Qb):
        Qres = torch.zeros(4).cuda()
        Qres[0] = Qa[0]*Qb[0] - torch.dot(Qa[1:], Qb[1:])
        Qres[1:] = Qa[0]*Qb[1:] + Qb[0]*Qa[1:] + torch.cross(Qa[1:], Qb[1:])

        return Qres


    def QuaternionInverse(self, Quaternion):
        Qi = torch.zeros(4).cuda()
        norm = torch.norm(Quaternion)
        Qi[1:] = -Quaternion[1:]/norm
        Qi[0] = Quaternion[0]/norm
        return Qi

class RegularizationLoss(nn.Module):
    def __init__(self):
        super(RegularizationLoss, self).__init__()
    
    def forward(self, output):
        batchSize = output.shape[0]
        self.loss = torch.zeros(1).cuda()
        for batch in range(batchSize):
            M1 = torch.zeros(3,3).cuda()
            M2 = torch.zeros(3,3).cuda()
            for i in range(3):
                M1[i-3] = output[batch][i][1:]/torch.norm(output[batch][i][1:])

            for i in range(3, 6):
                M2[i-3] = output[batch][i][1:]/torch.norm(output[batch][i][1:])

            I = torch.eye(3).cuda()
            A = torch.mm(M1, torch.t(M1)) - I
            B = torch.mm(M2, torch.t(M2)) - I

            self.loss = self.loss + torch.norm(A, p='fro')**2 + torch.norm(B, p = 'fro')**2

        return self.loss/batchSize

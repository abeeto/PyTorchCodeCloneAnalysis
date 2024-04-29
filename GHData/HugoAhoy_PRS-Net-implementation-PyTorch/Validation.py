import math

def Validate(ouput, target):
    Q = target['points']
    ClosestGrid = target[batch]['closest']
    epsilon = 4e-4
    cosThreshold = math.cos（math.pi/6)
    Plane = output[:3, :3]
    saveP = [True, True, True]
    PD = [0, 0, 0]
    Axes = output[3:]
    saveA = [True, True, True]
    AD = [0,0,0]
    for i in range(3):
        PD[i] = ReflectiveDistance(Plane[i], Q, ClosestGrid)
        if PD[i] > epsilon:
            saveP[i] = False

    for i in range(3):
        AD[i] = RotationDistance(Axes[i], Q, ClosestGrid)
        if AD[i] > epsilon:
            saveA[i] = False

    for i in range(3):
        for j in range(i+1,3):
            if saveP[i] and saveP[j]:
                cosVal = torch.dot(Plane[i][:3], Plane[j][:3])/(torch.norm(Plane[i][:3])*torch.norm(Plane[j][:3]))
                if cosVal < cosThreshold:
                    if PD[i] > PD[j]:
                        saveP[i] = False
                    else:
                        saveP[j] = False
    
    for i in range(3):
        for j in range(i+1,3):
            if saveA[i] and saveA[j]:
                cosVal = torch.dot(Axes[i][1:], Axes[j][1:])/(torch.norm(Axes[i][1:])*torch.norm(Axes[j][1:]))
                if cosVal < cosThreshold:
                    if AD[i] > AD[j]:
                        saveA[i] = False
                    else:
                        saveA[j] = False
    
    # todo
    # particular for rotation axes


    def ReflectiveDistance(ReflectivePlane, Q, ClosestGrid):
        # get normal vector of the plane
        nv = ReflectivePlane[0:3]
        d = ReflectivePlane[3]
        SymPoints = []
        for k in range(len(Q)):
            q = Q[k]
            dis = (torch.dot(nv, torch.tensor(q))+d)/(torch.dot(nv, nv))
            q_sym = q - 2*(dis)*nv
            SymPoints.append(q_sym)
        return totalDis(SymPoints, ClosestGrid)
        

    def RotationDistance(RotationQuater, Q, ClosestGrid):
        SymPoints = []
        for k in range(len(Q)):
            q = Q[k]
            q_hat = torch.zeros(4)
            q_hat[1:] = q
            q_sym = QuaternionProduct(QuaternionProduct(RotationQuater,q_hat), QuaternionInverse(RotationQuater))[1:]
            SymPoints.append(q_sym)
        return totalDis(SymPoints, ClosestGrid)
        
    def QuaternionProduct(Qa, Qb):
        Qres = torch.zeros(4)
        Qres[0] = Qa[0]*Qb[0] - torch.dot(Qa[1:], Qb[1:])
        Qres[1:] = Qa[0]*Qb[1:] + Qb[0]*Qa[1:] + torch.cross(Qa[1:], Qb[1:])

        return Qres


    def QuaternionInverse(Quaternion):
        Qi = torch.zeros(4)
        Qi[1:] = -Quaternion[1:]
        Qi[0] = Quaternion[0]
        Qi = Qi/ torch.norm(Quaternion)
        return Qi


    def totalDis(SymPoints, ClosestGrid):
        totalDis = 0
        for i in range(SymPoints):
            x, y, z = SymPoints[i]
            if x < 0:
                x = 0
            elif x > 32:
                x = 32
            
            if y < 0:
                y = 0
            elif y > 32:
                y = 32
            if z < 0:
                z = 0
            elif z > 32:
                z = 32

            CP = torch.tensor(ClosestGrid[int(x)*32*32+int(y)*32+int(z)])
            totalDis = totalDis + torch.norm(SymPoints[i] - CP)

        return totalDis/len(SymPoints)

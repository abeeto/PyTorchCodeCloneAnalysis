from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
from torch import nn
import numpy as pd
import pandas as pd
from pandas import DataFrame
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

teams = [
"Virtus Bologna", 
"Banco di Sardegna Sassari", 
"Gemani Basket Brescia", 
"EA7 Emporio Armani Milano", 
"Enel Brindisi", 
"Vanoli Cremona", 
"Umana Venezia", 
"Dolomiti Energia Trento", 
"Fortituto Kontatto Bologna", 
"OpenJobMetis Varese", 
"Red October Cantu",
"Grissin Bon Reggio Emilia",
"Universo De'Longhi Treviso",
"Virtus Roma",
"Flexx Pistoia",
"Pallacanestro Trieste",
"Consultinvest VL Pesaro"
]


#Show Game Results
frame = pd.read_csv("serieAUpdated.csv") 
columns = [ 'Team', 'PFH', 'PAH', 'PFA', 'PAA', 'eFGFH', 'eFGAH', 'eFGFA', 'eFGAA', 'TOFH', 'TOAH', 'TOFA', 'TOAA', 'ORFH', 'ORAH', 'ORFA', 'ORAA', 'FTRFH', 'FTRAH', 'FTRFA', 'FTRAA']
data = []
print (frame)

#Loop Through Game Results to Create Team Averages
for i in teams:
    row = []
    homeFrame = frame[frame.homeTeam.str.contains(i)]
    awayFrame = frame[frame.awayTeam.str.contains(i)]
    PFH = homeFrame["homeScore"].mean()
    PAH = homeFrame["awayScore"].mean()
    PFA = awayFrame["awayScore"].mean()
    PAA = awayFrame["homeScore"].mean()
    eFGFH = homeFrame["heFG%"].mean()
    eFGAH = homeFrame["aeFG%"].mean()
    eFGFA = awayFrame["aeFG%"].mean()
    eFGAA = awayFrame["heFG%"].mean()
    TOFH = homeFrame["hTO%"].mean()
    TOAH = homeFrame["aTO%"].mean()
    TOFA = awayFrame["aTO%"].mean()
    TOAA = awayFrame["hTO%"].mean()
    ORFH = homeFrame["hOR%"].mean()
    ORAH = homeFrame["aOR%"].mean()
    ORFA = awayFrame["aOR%"].mean()
    ORAA = awayFrame["hOR%"].mean()
    FTRFH = homeFrame["hFTR"].mean()
    FTRAH = homeFrame["aFTR"].mean()
    FTRFA = awayFrame["aFTR"].mean()
    FTRAA = awayFrame["hFTR"].mean()
    row.append(i)
    row.append(PFH)
    row.append(PAH)
    row.append(PFA)
    row.append(PAA)
    row.append(eFGFH)
    row.append(eFGAH)
    row.append(eFGFA)
    row.append(eFGAA)
    row.append(TOFH)
    row.append(TOAH)
    row.append(TOFA)
    row.append(TOAA)
    row.append(ORFH)
    row.append(ORAH)
    row.append(ORFA)
    row.append(ORAA)
    row.append(FTRFH)
    row.append(FTRAH)
    row.append(FTRFA)
    row.append(FTRAA)
    data.append(row)


#Create DataFrame of Team Stats
teamStats = pd.DataFrame(data, columns = columns)

refHomeTeams = teamStats[['Team', 'PFH', 'PAH', 'eFGFH', 'eFGAH', 'TOFH', 'TOAH', 'ORFH', 'ORAH', 'FTRFH', 'FTRAH']].copy()
refAwayTeams = teamStats[['Team', 'PFA', 'PAA', 'eFGFA', 'eFGAA', 'TOFA', 'TOAA', 'ORFA', 'ORAA', 'FTRFA', 'FTRAA']].copy()


#Create Dataframe that adds stats to each team's games
addHome = pd.merge(frame, refHomeTeams, left_on = 'homeTeam', right_on = 'Team')
addAway = pd.merge(addHome, refAwayTeams, left_on = 'awayTeam', right_on = 'Team')
finalFrame = addAway.drop(['Team_x', 'Team_y'], axis = 1)

#Make a Frame to Make A tensor From
preTensorFrame = finalFrame[['PFH', 'PAH', 'eFGFH', 'eFGAH', 'TOFH', 'TOAH', 'ORFH', 'ORAH', 'FTRFH', 'FTRAH', 'PFA', 'PAA', 'eFGFA', 'eFGAA', 'TOFA', 'TOAA', 'ORFA', 'ORAA', 'FTRFA', 'FTRAA']]

#Create Tensors for Inputs
inputsArray = preTensorFrame.to_numpy(dtype='float32')
inputs = torch.from_numpy(inputsArray)

#Create Tensors for Targets
targetFrame = frame[['awayScore', 'homeScore']]
targetsArray = targetFrame.to_numpy(dtype='float32')
targets = torch.from_numpy(targetsArray)


print (inputs, targets)
import pandapower as pp
import pandapower.networks as pn
from numba import jit
import pandas as pd
import numpy as np
import copy
import random as rnd

#action = [rnd.uniform(-1,1) for i in range(9)]

net = pn.case5()



class Grid(object):
    def __init__(self, net,max_loading=1, Vbusmax=1.03, Vbusmin=0.98, max_shedding =0.2, T_max=21):
        """
        Initialize the environment
        :param net: Power grid (for example IEEE 24-bus)
        :param max_loading: maximum line thermal limit
        :param Vbusmax: Maximum upper bound for Bus magnitude voltage
        :param Vbusmin : Minimum lower bound for Bus magnitude voltage
        """
        self.net_origin= net
        self.net = copy.deepcopy(net)
        self.max_loading = max_loading
        self.Vbusmax = Vbusmax
        self.Vbusmin = Vbusmin
        self.max_shedding = max_shedding
        self.T_max= T_max
        self.t= 0
        self.reward_neg = 0

    def LftoState(self,grid):
        # pp.runpp(grid, lumba=False)
        pij = np.zeros((grid.bus.shape[0], grid.bus.shape[0]))
        qij = np.zeros((grid.bus.shape[0], grid.bus.shape[0]))
        for l, res in zip(grid.line.iterrows(), grid.res_line.iterrows()):
            pij[l[1].from_bus, l[1].to_bus] = res[1].p_from_mw
            pij[l[1].to_bus, l[1].from_bus] = -res[1].p_to_mw
            qij[l[1].from_bus, l[1].to_bus] = res[1].q_from_mvar
            qij[l[1].to_bus, l[1].from_bus] = -res[1].q_to_mvar
        return np.concatenate((grid.res_bus.values, pij, qij), axis=1)

    def Attack(self,ind=-1):
        if ind not in np.arange(self.net.line.shape[0]):
            print('You choose wrong number ({}) for line under attack'.format(ind))
            ind = np.random.choice(np.arange(self.net.line.shape[0]))
        print('Line with index -->{}<-- between bus {} and {} is attacked '.
              format(ind,self.net.line.from_bus[ind],self.net.line.to_bus[ind]))
        self.net.line.in_service[ind] = False

    def assessment(self, grid,line_limit, v_upper, v_lower):
        free_cap = list(map(lambda x: line_limit-x if x < line_limit else 0 , grid.res_line.loading_percent.values/100))
        overload = list(map(lambda x : x-line_limit if x > line_limit else 0, grid.res_line.loading_percent.values/100))
        overvoltage = list(map(lambda x : x-v_upper if x > v_upper else 0, grid.res_bus.vm_pu.values))
        undervoltage = list(map(lambda x : v_lower-x if x < v_lower else 0, grid.res_bus.vm_pu.values))
        return free_cap, overload , overvoltage , undervoltage

    def InitState(self):
        pp.runpp(self.net,algorithm='nr', numba= False)
        tmp, _, _, _ = self.assessment(self.net, self.max_loading, self.Vbusmax, self.Vbusmin)
        self.base_cap= copy.deepcopy(sum(tmp))
        tmp = sum(self.net.res_gen.p_mw.values)
        self.base_power = copy.deepcopy(tmp)
        self.base_voltage = copy.deepcopy(sum(self.net.res_gen.vm_pu.values))
        self.base_load = copy.deepcopy(sum(self.net.load.scaling.values))
        return self.LftoState(self.net)

    def Nline(self):
        return self.net.line.shape[0]

    def StateFeatures(self):
        return [self.net.bus.shape[0],4+2*self.net.bus.shape[0]]

    def ActionFeature(self):
        return 2*self.net.gen.shape[0] + self.net.load.shape[0]


    def take_action(self,action):
        Ng= self.net.res_gen.vm_pu.values.shape[0]
        Nload= self.net.load.scaling.shape[0]

        # self.net.gen.vm_pu= self.net.res_gen.vm_pu.values + np.array([action[i]*(self.Vbusmax-self.net.res_gen.vm_pu.values[i])
        #                                                      if action[i]>0 else action[i]*(self.net.res_gen.vm_pu.values[i]-self.Vbusmin)
        #                                                      for i in range(Ng)])
        delta_vg = np.array(action[0:Ng]*(self.Vbusmax-self.Vbusmin))
        v_before = self.net.res_gen.vm_pu.values
        self.net.gen.vm_pu = self.net.res_gen.vm_pu.values + delta_vg
        v_now = self.net.gen.vm_pu.values
        delta_vg_perc = (v_now - v_before) / v_before

        voltage_punish= 0
        for i in range(Ng):
            if self.net.gen.vm_pu[i] > self.Vbusmax:
                voltage_punish = voltage_punish + self.Vbusmax-self.net.gen.vm_pu[i]
                self.net.gen.vm_pu[i] = self.Vbusmax
            elif self.net.gen.vm_pu[i] < self.Vbusmin:
                voltage_punish = voltage_punish + self.Vbusmin-self.net.gen.vm_pu[i]
                self.net.gen.vm_pu[i] = self.Vbusmin

        print('Network new Voltages are :----> \n')
        print(self.net.gen.vm_pu)


        # self.net.gen.p_mw = self.net.res_gen.p_mw.values +np.array([action[i+Ng]*(self.net.gen.max_p_mw.values[i]-self.net.res_gen.p_mw[i])
        #                                                     if action[i+Ng] > 0 else
        #                                                     action[i+Ng] * (self.net.res_gen.p_mw[i]-self.net.gen.min_p_mw.values[i])
        #                                                     for i in range(Ng)])

        delta_pg = np.array(action[Ng:2*Ng]*(self.net.gen.max_p_mw-(self.net.gen.max_p_mw/3)))
        p_before =  self.net.res_gen.p_mw.values
        self.net.gen.p_mw = self.net.res_gen.p_mw.values + delta_pg
        p_now = self.net.gen.p_mw.values
        delta_pg_perc = (p_now - p_before) / p_before

        power_punish= 0
        for i in range(Ng):
            if self.net.gen.p_mw[i] > self.net.gen.max_p_mw[i]:
                power_punish = power_punish+ (self.net.gen.p_mw[i]-self.net.gen.max_p_mw[i])/(self.net.gen.max_p_mw[i])
                self.net.gen.p_mw[i] = self.net.gen.max_p_mw[i]
            elif self.net.gen.p_mw[i] < self.net.gen.max_p_mw[i]/3:
                power_punish = power_punish + (self.net.gen.max_p_mw[i]/3)-self.net.gen.p_mw[i]/(self.net.gen.max_p_mw[i]/3)
                self.net.gen.p_mw[i] = self.net.gen.max_p_mw[i]/3

        print('Network new Active Power are :----> \n')
        print(self.net.gen.p_mw)

        # self.net.load.scaling = self.net.load.scaling.values - \
        #                         np.array([action[2*Ng+i]*self.max_shedding if self.net.load.scaling.values[i]-action[2*Ng+i]*self.max_shedding>0 else 0
        #                         for i in range(Ng)])
        delta_d = np.array(action[2*Ng:2*Ng+Nload]*self.max_shedding)
        d_before = self.net.load.scaling.values
        self.net.load.scaling = self.net.load.scaling.values + delta_d

        d_now = self.net.load.scaling.values

        delta_d_perc = (d_now - d_before) / d_before

        load_punish = 0
        for i in range(Nload):
            if self.net.load.scaling[i] > 1+self.max_shedding:
                load_punish=load_punish + (self.net.load.scaling[i] - (1+self.max_shedding))
                self.net.load.scaling[i] = 1+self.max_shedding
            elif self.net.load.scaling[i] < 1-self.max_shedding:
                load_punish = load_punish + (1-self.max_shedding)-self.net.load.scaling[i]
                self.net.load.scaling[i] = 1 - self.max_shedding


        # self.net.load.scaling = self.net.load.scaling.values - np.array(action[2*Ng:self.ActionFeature()])*self.max_shedding
        # self.net.load.q_mvar = action[2*self.net.gen.shape[0]:self.ActionFeature()]
        pp.runpp(self.net,'nr',lumba=False)
        self.t = self.t +1
        State = self.LftoState(self.net)
        free_cap, overload, overvoltage, undervoltage = self.assessment(self.net, self.max_loading, self.Vbusmax, self.Vbusmin)
        conditions = sum(overload) + sum(overvoltage) + sum(undervoltage)
        punish_total = voltage_punish + power_punish+ load_punish
        print('<<------Base values are are : -------->>')
        print ([self.base_cap, self.base_voltage,self.base_power,self.base_load])
        print('<<------The punishments are : -------->>')
        print([voltage_punish, power_punish, load_punish])

        done = 0
        if conditions == 0:
            done = 1

            # tmp = abs(sum(free_cap))+ abs(sum(delta_vg_perc*action[0:Ng]))+abs(sum(delta_pg_perc*action[Ng:2*Ng]))+\
            #       abs(sum(delta_d_perc*action[2*Ng:2*Ng+Nload]))+abs(punish_total)


            reward = 1*(sum(free_cap)/self.base_cap)*(self.T_max-self.t)-1*(sum(delta_vg_perc*action[0:Ng])/self.base_voltage)-1*(sum(delta_pg_perc*action[Ng:2*Ng])/self.base_power)\
                     +1*(sum(delta_d_perc*action[2*Ng:2*Ng+Nload])/self.base_load) - abs(voltage_punish/self.base_voltage)-\
                     abs(power_punish/self.base_power)-abs(load_punish/self.base_load)

            # -1 * (punish_total / self.base_cap)


            reward_com=[sum(free_cap), sum(delta_vg_perc*action[0:Ng]), sum(delta_pg_perc*action[Ng:2*Ng]), sum(delta_d_perc*action[2*Ng:2*Ng+Nload])]
            # reward= 0.8*(sum(free_cap)/tmp) - 0.2*(sum(action[2*Ng:self.ActionFeature()])/tmp)
        else:
            reward = -((conditions)/(self.base_cap))* (self.t) - abs(voltage_punish/self.base_voltage)-\
                     abs(power_punish/self.base_power)-abs(load_punish/self.base_load)
                     # -sum(action[0:Ng])-sum(action[Ng:2*Ng])+sum(action[2*Ng:self.ActionFeature()])-\
                     # (voltage_punish +power_punish+ load_punish)
            reward_com= [0,0,0,0]
        return State, reward, done, reward_com

    def reset(self):
        self.net = copy.deepcopy(self.net_origin)
        self.t = 0
        self.reward_neg = 0






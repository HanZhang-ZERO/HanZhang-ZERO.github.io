# -*- coding: UTF-8 -*-
import numpy as np
import pickle


class Instances:
    def __init__(self, fp_listParameters):
        '''
        @parameters: 0:iSitesNum, 1:iScenNum, 2:iDemandLB, 3:iDemandUB, 4:iFixedCostLB, 5:iFixedCostUP, 6:iCoordinateLB, 7:iCoordinateUB, 8:fFaciFailProb, 9:fScenProb
        '''
        self.iSitesNum = fp_listParameters[0]
        self.iScenNum = fp_listParameters[1]
        self.iDemandLB = fp_listParameters[2]
        self.iDemandUB = fp_listParameters[3]
        self.iFixedCostLB = fp_listParameters[4]
        self.iFixedCostUB = fp_listParameters[5]
        self.iCoordinateLB = fp_listParameters[6]
        self.iCoordinateUB = fp_listParameters[7]
        self.fFaciFailProb = fp_listParameters[8]
        self.fScenProb = fp_listParameters[9]

        self.listaiDemands = []  # Store customers' demands in different scenes

        self.a_2d_SitesCoordi = np.zeros((self.iSitesNum, 2))
        self.aiDemands = np.zeros(self.iSitesNum, dtype=np.int)
        self.aiFixedCost = np.zeros(self.iSitesNum, dtype=np.int)
        self.af_2d_TransCost = np.zeros((self.iSitesNum, self.iSitesNum))

    def funGenerateInstances(self):
        # generate the x and y coordinates of candidate sites
        self.a_2d_SitesCoordi = self.iCoordinateLB + (
            self.iCoordinateUB - self.iCoordinateLB) * np.random.rand(
                self.iSitesNum, 2)

        fSpan = (self.iDemandUB - self.iDemandLB)/self.iScenNum
        for i in range(self.iScenNum):
            self.aiDemands = np.random.randint(self.iDemandLB + fSpan * i,
                                               self.iDemandLB + fSpan * (i + 1),
                                               size=self.iSitesNum)
            self.listaiDemands.append(self.aiDemands)

        self.aiFixedCost = np.random.randint(self.iFixedCostLB,
                                             self.iFixedCostUB,
                                             size=self.iSitesNum)
        for i in range(self.iSitesNum):
            for j in range(i + 1, self.iSitesNum):
                temCost = np.linalg.norm(self.a_2d_SitesCoordi[i] -
                                         self.a_2d_SitesCoordi[j])
                self.af_2d_TransCost[i][j] = self.af_2d_TransCost[j][
                    i] = temCost


if __name__ == '__main__':
    '''
    Test the code.
    listPara:
    0:iSitesNum, 1:iScenNum, 2:iDemandLB, 3:iDemandUB, 4:iFixedCostLB, 5:iFixedCostUP, 6:iCoordinateLB, 7:iCoordinateUB, 8:fFaciFailProb, 9:fScenProb
    '''
    iInsNum = 8
    iSitesNum = 100
    iScenNum = 3
    iDemandLB = 0
    iDemandUB = 1000
    iFixedCostLB = 500
    iFixedCostUB = 1500
    iCoordinateLB = 0
    iCoordinateUB = 1
    fFaciFailProb = 0.05
    fScenProb = 1/iScenNum
    listPara = [iSitesNum, iScenNum, iDemandLB, iDemandUB, iFixedCostLB, iFixedCostUB, iCoordinateLB, iCoordinateUB, fFaciFailProb, fScenProb]
    f = open('100-nodeInstances_3scenes', 'wb')
    for i in range(iInsNum):
        generateInstances = Instances(listPara)
        generateInstances.funGenerateInstances()
        pickle.dump(generateInstances, f)
    f.close()
    # f = open('30-nodeInstances', 'rb')
    # ins1 = pickle.load(f)
    # ins2 = pickle.load(f)
    # print(ins1.aiFixedCost)
    # print(ins2.aiFixedCost)
    # print("trans cost: \n", generateInstances.af_2d_TransCost)
    # print("fixed cost: \n", generateInstances.aiFixedCost)
    # print("coordinate: \n", generateInstances.a_2d_SitesCoordi)
    # print("demands: \n", generateInstances.aiDemands)

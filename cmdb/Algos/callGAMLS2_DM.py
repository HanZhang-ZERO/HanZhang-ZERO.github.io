import numpy as np
import pickle
import instanceGenerationRFLP
import instanceGenerationRRFLP
from instanceGenerationRFLP import Instances
from instanceGenerationRRFLP import Instances
import matplotlib.pyplot as plt
import time
import GAMLS2_DM_RFLP
import GAMLS2_DM_RRFLP

listlist_bestSolsOfAllIns_locationIndex = []  # 如果有重复多次运行，就取最好的结果作为最终输出结果,每个dict是一个问题实例的解，多个问题实例的解组成一个list. 放在这里是个全局变量，每次运行得到的最好解会不断地加入到这个变量中，如果不想这样，可以把改变量放到方法内部

# iActualInsNum = 1
# iInsNum = 8
# iRunsNum = 8
# iCandidateFaciNum = 10
# insName = '10-nodeInstances'
# fileName = 'ex10-node'


def funGAMLS2_RFLP_generateIns(listInsPara, listGAParameters, iRunsNum):
    '''
    调用GAMLS2解决Reliable FLP问题

    listGAParameters = [0:iGenNum, 1:iPopSize, 2:iIndLen, 3:fCrosRate, 4:fMutRate, 5:fAlpha, 6:boolAllo2Faci]

    listInstPara=[0:iSitesNum, 1:iScenNum, 2:iDemandLB, 3:iDemandUB, 4:iFixedCostLB, 5:iFixedCostUP, 6:iCoordinateLB, 7:iCoordinateUB, 8:fFaciFailProb]

    The value of  2:iIndLen and 0:iSitesNum should be equal.
    '''
    # 计划每次只计算一个问题实例，所以把下面两个参数固定为1
    iInsNum = 1
    iActualInsNum = 1
    iGenNum = listGAParameters[0]
    iCandidateFaciNum = listInsPara[0]
    # generate instance
    obInstance = instanceGenerationRFLP.Instances(listInsPara)
    obInstance.funGenerateInstances()

    # genetic algorithm
    listfAveFitnessEveryIns = np.zeros((iInsNum,)).tolist()
    listfAveObjValueEveryIns = np.zeros((iInsNum,)).tolist()
    listfAveCPUTimeEveryIns = np.zeros((iInsNum,)).tolist()
    a_2d_fEveInsEveRunObjValue = np.zeros((iInsNum, iRunsNum))

    for i in range(iActualInsNum):
        # genetic algorithm
        listfAllDiffGenBestIndFitness = np.zeros((iGenNum + 1,)).tolist()  # 算上第0代
        listiAllDiffGenDiversityMetric1 = np.zeros((iGenNum + 1,)).tolist()  # 算上第0代
        listiAllDiffGenDiversityMetric2 = np.zeros((iGenNum + 1,)).tolist()  # 算上第0代
        listiFitEvaNumByThisGen_AllRunsAve = np.zeros((iGenNum + 1,)).tolist()
        listfEveGenProportion_belongToLocalSearchedIndNeighbor_exceptCurrLocalSearchedInd_AllRunsAve = np.zeros((iGenNum + 1,)).tolist()
        listfEveGenProportion_belongToNeighborOfAllLocalSearchedInd_AllRunsAve = np.zeros((iGenNum + 1,)).tolist()
        dict_bestIndOfThisIns = {'chromosome': 0,
                                 'fitness': 0}  # 存储该instance下的最好解
        for j in range(iRunsNum):  # Every instance has 10 runs experiments.
            print("Begin: ins " + str(i) + ", Runs " + str(j))
            print("Running......")
            cpuStart = time.process_time()
            # 调用GADM求解
            local_state = np.random.RandomState()
            GeneticAlgo = GAMLS2_DM_RFLP.GA(listGAParameters, obInstance, local_state)
            listdictFinalPop, listGenNum, listfBestIndFitnessEveGen, listiDiversityMetric1, listiDiversityMetric2, listiFitEvaNumByThisGen, listfEveGenProportion_belongToLocalSearchedIndNeighbor_exceptCurrLocalSearchedInd, listfEveGenProportion_belongToOnlyCurrGenLocalSearchedIndNeighbor, listfEveGenProportion_belongToNeighborOfAllLocalSearchedInd, listiEveGenLocalSearchedIndNum = GeneticAlgo.funGA_main()
            cpuEnd = time.process_time()
            # 记录CPU time，累加
            listfAveCPUTimeEveryIns[i] += (cpuEnd - cpuStart)

            print("Objective value:", listdictFinalPop[0]['objectValue'])
            # 更新dict_bestInd
            if listdictFinalPop[0]['fitness'] > dict_bestIndOfThisIns['fitness']:
                dict_bestIndOfThisIns['chromosome'] = listdictFinalPop[0]['chromosome']
                dict_bestIndOfThisIns['fitness'] = listdictFinalPop[0]['fitness']
            if listfBestIndFitnessEveGen[-1] != 1/listdictFinalPop[0]['objectValue']:
                print("Wrong. Please check funLS_DM().")
            # 记录最终种群中最好个体的fitness和目标函数值，累加
            a_2d_fEveInsEveRunObjValue[i][j] = listdictFinalPop[0]['objectValue']
            listfAveFitnessEveryIns[i] += listfBestIndFitnessEveGen[-1]
            listfAveObjValueEveryIns[i] += listdictFinalPop[0]['objectValue']
            # 为绘图准备
            new_listfBestIndFitness = [fitness * 1000 for fitness in listfBestIndFitnessEveGen]
            for g in range(len(listGenNum)):
                listfAllDiffGenBestIndFitness[g] += new_listfBestIndFitness[g]
                listiAllDiffGenDiversityMetric1[g] += listiDiversityMetric1[g]
                listiAllDiffGenDiversityMetric2[g] += listiDiversityMetric2[g]
                listiFitEvaNumByThisGen_AllRunsAve[g] += listiFitEvaNumByThisGen[g]
                listfEveGenProportion_belongToLocalSearchedIndNeighbor_exceptCurrLocalSearchedInd_AllRunsAve[g] += listfEveGenProportion_belongToLocalSearchedIndNeighbor_exceptCurrLocalSearchedInd[g]
                listfEveGenProportion_belongToNeighborOfAllLocalSearchedInd_AllRunsAve[g] += listfEveGenProportion_belongToNeighborOfAllLocalSearchedInd[g]

        print("End: ins " + str(i) + ", Runs " + str(j) + "\n")

        # 将个体的chromosome从0和1转化为被选中地址的索引,并取被选中地址的横纵坐标
        list_BestSolOfThisIns_LocationIndex = []
        list_facility_x = []  # x-coordinate
        list_facility_y = []  # y-coordinate
        for c in range(iCandidateFaciNum):
            if dict_bestIndOfThisIns['chromosome'][c] == 1:
                list_BestSolOfThisIns_LocationIndex.append(c)
                list_facility_x.append(obInstance.a_2d_SitesCoordi[c, 0])
                list_facility_y.append(obInstance.a_2d_SitesCoordi[c, 1])

        # 画散点图
        plt.figure()
        x = obInstance.a_2d_SitesCoordi[:, 0]  # 横坐标数组
        y = obInstance.a_2d_SitesCoordi[:, 1]  # 纵坐标数组
        plt.scatter(x, y, alpha=0.6, s=20, label='Candidate Locations')  # 绘制散点图，透明度为0.6（这样颜色浅一点，比较好看），s控制点的大小
        plt.scatter(list_facility_x, list_facility_y, alpha=1, s=30, marker='*', c='r', label='Selected Locations')
        plt.legend()
        plt.savefig("static/imgs/1-scatter-RFLP.svg")

        # 存储每个实例多次运行后最好的运行结果,因为决定每次只运行一个实例，所以这个变量实际上用不着
        listlist_bestSolsOfAllIns_locationIndex.append(list_BestSolOfThisIns_LocationIndex)
        print("Best sol:", listlist_bestSolsOfAllIns_locationIndex)  # 输出选中的地址索引

        # 平均每次运行的时间
        listfAveCPUTimeEveryIns[i] /= iRunsNum
        # 平均fitness和目标函数值
        listfAveFitnessEveryIns[i] /= iRunsNum
        listfAveObjValueEveryIns[i] /= iRunsNum
        # 绘图
        listfAveBestIndFitnessEveryGen = [fitness / iRunsNum for fitness in listfAllDiffGenBestIndFitness]
        listiAveDiversityMetric1EveGen = [diversity / iRunsNum for diversity in listiAllDiffGenDiversityMetric1]
        # listiAveDiversityMetric2EveGen = [diversity / iRunsNum for diversity in listiAllDiffGenDiversityMetric2]
        listiFitEvaNumByThisGen_AllRunsAve = [diversity / iRunsNum for diversity in listiFitEvaNumByThisGen_AllRunsAve]
        listfEveGenProportion_belongToLocalSearchedIndNeighbor_exceptCurrLocalSearchedInd_AllRunsAve = [diversity / iRunsNum for diversity in listfEveGenProportion_belongToLocalSearchedIndNeighbor_exceptCurrLocalSearchedInd_AllRunsAve]
        listfEveGenProportion_belongToNeighborOfAllLocalSearchedInd_AllRunsAve = [diversity / iRunsNum for diversity in listfEveGenProportion_belongToNeighborOfAllLocalSearchedInd_AllRunsAve]

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        l1, = ax1.plot(listGenNum, listfAveBestIndFitnessEveryGen, marker='*')
        ax1.set_xlabel("# of Generation")
        ax1.set_ylabel("Fitness Of Best Individual")
        # 右方Y轴
        ax2 = ax1.twinx()  # 与ax1共用1个x轴，在右方生成自己的y轴
        l2, = ax2.plot(listGenNum, listiAveDiversityMetric1EveGen, 'r',)
        # l5, = ax2.plot(listGenNum, listiAveDiversityMetric2EveGen, 'go:')
        l3, = ax2.plot(listGenNum, listfEveGenProportion_belongToLocalSearchedIndNeighbor_exceptCurrLocalSearchedInd_AllRunsAve, 'purple', marker='p', linestyle='--')
        # l4, = ax2.plot(listGenNum, listfEveGenProportion_belongToNeighborOfAllLocalSearchedInd_AllRunsAve, 'darkslategray', linestyle='-.')
        ax2.set_ylabel("Diversity Metric")
        # 上方X轴
        ax3 = ax1.twiny()  # 与ax1共用1个y轴，在上方生成自己的x轴
        ax3.set_xlabel("# of Fitness Evaluation")
        listfFeIndex = list(np.linspace(0, iGenNum, num=10+1))
        listFeXCoordinate = []
        for i in range(len(listfFeIndex)):
            listFeXCoordinate.append(listiFitEvaNumByThisGen[int(listfFeIndex[i])])
        ax3.plot(listGenNum, listfAveBestIndFitnessEveryGen, '--')
        ax3.set_xticks(listfFeIndex)
        ax3.set_xticklabels(listFeXCoordinate)
        plt.legend(handles=[l1, l2, l3], labels=['l1:Fitness curve', 'l2:0-HDR', 'l3-value'], loc='best')
        plt.savefig("static/imgs/2-convergenceCurves-RFLP.svg")
        # plt.show()
        # print("Final Solution:", listdictFinalPop[0]['chromosome'])
        # print("Average Objective value:", 1/listfAveBestIndFitnessEveryGen[-1])
        # print("Average Objective value:", listfAveObjValueEveryIns)

        # 被选中的地址的坐标
        listlist_locationCoordinate = []
        for index in list_BestSolOfThisIns_LocationIndex:
            listlist_locationCoordinate.append(list(np.around(obInstance.a_2d_SitesCoordi[index], decimals=3)))  # 保留小数点后3位

    return list_BestSolOfThisIns_LocationIndex, listlist_locationCoordinate


def funGAMLS2_RRFLP_generateIns(listInsPara, listGAParameters, iRunsNum):
    '''
    调用GAMLS2解决Reliable&Robust FLP问题

    listGAParameters = [0:iGenNum, 1:iPopSize, 2:iIndLen, 3:fCrosRate, 4:fMutRate, 5:fAlpha, 6:boolAllo2Faci]

    listInstPara=[0:iSitesNum, 1:iScenNum, 2:iDemandLB, 3:iDemandUB, 4:iFixedCostLB, 5:iFixedCostUP, 6:iCoordinateLB, 7:iCoordinateUB, 8:fFaciFailProb]

    The value of  2:iIndLen and 0:iSitesNum should be equal.
    '''
    # 计划每次只计算一个问题实例，所以把下面两个参数固定为1
    iInsNum = 1
    iActualInsNum = 1
    iGenNum = listGAParameters[0]
    iCandidateFaciNum = listInsPara[0]
    fScenProb = 1.0/listInsPara[1]
    listInsPara.append(fScenProb)  # fScenProb这个参数暂时不由用户指定
    # generate instance
    obInstance = instanceGenerationRRFLP.Instances(listInsPara)
    obInstance.funGenerateInstances()

    # genetic algorithm
    listfAveFitnessEveryIns = np.zeros((iInsNum,)).tolist()
    listfAveObjValueEveryIns = np.zeros((iInsNum,)).tolist()
    listfAveCPUTimeEveryIns = np.zeros((iInsNum,)).tolist()
    a_2d_fEveInsEveRunObjValue = np.zeros((iInsNum, iRunsNum))

    for i in range(iActualInsNum):
        # genetic algorithm
        listfAllDiffGenBestIndFitness = np.zeros((iGenNum + 1,)).tolist()  # 算上第0代
        listiAllDiffGenDiversityMetric1 = np.zeros((iGenNum + 1,)).tolist()  # 算上第0代
        listiAllDiffGenDiversityMetric2 = np.zeros((iGenNum + 1,)).tolist()  # 算上第0代
        listiFitEvaNumByThisGen_AllRunsAve = np.zeros((iGenNum + 1,)).tolist()
        listfEveGenProportion_belongToLocalSearchedIndNeighbor_exceptCurrLocalSearchedInd_AllRunsAve = np.zeros((iGenNum + 1,)).tolist()
        listfEveGenProportion_belongToNeighborOfAllLocalSearchedInd_AllRunsAve = np.zeros((iGenNum + 1,)).tolist()
        dict_bestIndOfThisIns = {'chromosome': 0,
                                 'fitness': 0}  # 存储该instance下的最好解
        for j in range(iRunsNum):  # Every instance has 10 runs experiments.
            print("Begin: ins " + str(i) + ", Runs " + str(j))
            print("Running......")
            cpuStart = time.process_time()
            # 调用GADM求解
            local_state = np.random.RandomState()
            GeneticAlgo = GAMLS2_DM_RRFLP.GA(listGAParameters, obInstance, local_state)
            listdictFinalPop, listGenNum, listfBestIndFitnessEveGen, listiDiversityMetric1, listiDiversityMetric2, listiFitEvaNumByThisGen, listfEveGenProportion_belongToLocalSearchedIndNeighbor_exceptCurrLocalSearchedInd, listfEveGenProportion_belongToOnlyCurrGenLocalSearchedIndNeighbor, listfEveGenProportion_belongToNeighborOfAllLocalSearchedInd, listiEveGenLocalSearchedIndNum = GeneticAlgo.funGA_main()
            cpuEnd = time.process_time()
            # 记录CPU time，累加
            listfAveCPUTimeEveryIns[i] += (cpuEnd - cpuStart)

            print("Objective value:", listdictFinalPop[0]['objectValue'])
            # 更新dict_bestInd
            if listdictFinalPop[0]['fitness'] > dict_bestIndOfThisIns['fitness']:
                dict_bestIndOfThisIns['chromosome'] = listdictFinalPop[0]['chromosome']
                dict_bestIndOfThisIns['fitness'] = listdictFinalPop[0]['fitness']
            if listfBestIndFitnessEveGen[-1] != 1/listdictFinalPop[0]['objectValue']:
                print("Wrong. Please check funLS_DM().")
            # 记录最终种群中最好个体的fitness和目标函数值，累加
            a_2d_fEveInsEveRunObjValue[i][j] = listdictFinalPop[0]['objectValue']
            listfAveFitnessEveryIns[i] += listfBestIndFitnessEveGen[-1]
            listfAveObjValueEveryIns[i] += listdictFinalPop[0]['objectValue']
            # 为绘图准备
            new_listfBestIndFitness = [fitness * 1000 for fitness in listfBestIndFitnessEveGen]
            for g in range(len(listGenNum)):
                listfAllDiffGenBestIndFitness[g] += new_listfBestIndFitness[g]
                listiAllDiffGenDiversityMetric1[g] += listiDiversityMetric1[g]
                listiAllDiffGenDiversityMetric2[g] += listiDiversityMetric2[g]
                listiFitEvaNumByThisGen_AllRunsAve[g] += listiFitEvaNumByThisGen[g]
                listfEveGenProportion_belongToLocalSearchedIndNeighbor_exceptCurrLocalSearchedInd_AllRunsAve[g] += listfEveGenProportion_belongToLocalSearchedIndNeighbor_exceptCurrLocalSearchedInd[g]
                listfEveGenProportion_belongToNeighborOfAllLocalSearchedInd_AllRunsAve[g] += listfEveGenProportion_belongToNeighborOfAllLocalSearchedInd[g]

        print("End: ins " + str(i) + ", Runs " + str(j) + "\n")

        # 将个体的chromosome从0和1转化为被选中地址的索引,并取被选中地址的横纵坐标
        list_BestSolOfThisIns_LocationIndex = []
        list_facility_x = []  # x-coordinate
        list_facility_y = []  # y-coordinate
        for c in range(iCandidateFaciNum):
            if dict_bestIndOfThisIns['chromosome'][c] == 1:
                list_BestSolOfThisIns_LocationIndex.append(c)
                list_facility_x.append(obInstance.a_2d_SitesCoordi[c, 0])
                list_facility_y.append(obInstance.a_2d_SitesCoordi[c, 1])

        # 画散点图
        plt.figure()
        x = obInstance.a_2d_SitesCoordi[:, 0]  # 横坐标数组
        y = obInstance.a_2d_SitesCoordi[:, 1]  # 纵坐标数组
        plt.scatter(x, y, alpha=0.6, s=20, label='Candidate Locations')  # 绘制散点图，透明度为0.6（这样颜色浅一点，比较好看），s控制点的大小
        plt.scatter(list_facility_x, list_facility_y, alpha=1, s=30, marker='*', c='r', label='Selected Locations')
        plt.legend()
        plt.savefig("static/imgs/1-scatter-RRFLP.svg")

        # 存储每个实例多次运行后最好的运行结果,因为决定每次只运行一个实例，所以这个变量实际上用不着
        listlist_bestSolsOfAllIns_locationIndex.append(list_BestSolOfThisIns_LocationIndex)
        print("Best sol:", listlist_bestSolsOfAllIns_locationIndex)  # 输出选中的地址索引

        # 平均每次运行的时间
        listfAveCPUTimeEveryIns[i] /= iRunsNum
        # 平均fitness和目标函数值
        listfAveFitnessEveryIns[i] /= iRunsNum
        listfAveObjValueEveryIns[i] /= iRunsNum
        # 绘图
        listfAveBestIndFitnessEveryGen = [fitness / iRunsNum for fitness in listfAllDiffGenBestIndFitness]
        listiAveDiversityMetric1EveGen = [diversity / iRunsNum for diversity in listiAllDiffGenDiversityMetric1]
        # listiAveDiversityMetric2EveGen = [diversity / iRunsNum for diversity in listiAllDiffGenDiversityMetric2]
        listiFitEvaNumByThisGen_AllRunsAve = [diversity / iRunsNum for diversity in listiFitEvaNumByThisGen_AllRunsAve]
        listfEveGenProportion_belongToLocalSearchedIndNeighbor_exceptCurrLocalSearchedInd_AllRunsAve = [diversity / iRunsNum for diversity in listfEveGenProportion_belongToLocalSearchedIndNeighbor_exceptCurrLocalSearchedInd_AllRunsAve]
        listfEveGenProportion_belongToNeighborOfAllLocalSearchedInd_AllRunsAve = [diversity / iRunsNum for diversity in listfEveGenProportion_belongToNeighborOfAllLocalSearchedInd_AllRunsAve]

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        l1, = ax1.plot(listGenNum, listfAveBestIndFitnessEveryGen, marker='*')
        ax1.set_xlabel("# of Generation")
        ax1.set_ylabel("Fitness Of Best Individual")
        # 右方Y轴
        ax2 = ax1.twinx()  # 与ax1共用1个x轴，在右方生成自己的y轴
        l2, = ax2.plot(listGenNum, listiAveDiversityMetric1EveGen, 'r',)
        # l5, = ax2.plot(listGenNum, listiAveDiversityMetric2EveGen, 'go:')
        l3, = ax2.plot(listGenNum, listfEveGenProportion_belongToLocalSearchedIndNeighbor_exceptCurrLocalSearchedInd_AllRunsAve, 'purple', marker='p', linestyle='--')
        # l4, = ax2.plot(listGenNum, listfEveGenProportion_belongToNeighborOfAllLocalSearchedInd_AllRunsAve, 'darkslategray', linestyle='-.')
        ax2.set_ylabel("Diversity Metric")
        # 上方X轴
        ax3 = ax1.twiny()  # 与ax1共用1个y轴，在上方生成自己的x轴
        ax3.set_xlabel("# of Fitness Evaluation")
        listfFeIndex = list(np.linspace(0, iGenNum, num=10+1))
        listFeXCoordinate = []
        for i in range(len(listfFeIndex)):
            listFeXCoordinate.append(listiFitEvaNumByThisGen[int(listfFeIndex[i])])
        ax3.plot(listGenNum, listfAveBestIndFitnessEveryGen, '--')
        ax3.set_xticks(listfFeIndex)
        ax3.set_xticklabels(listFeXCoordinate)
        plt.legend(handles=[l1, l2, l3], labels=['l1:Fitness curve', 'l2:0-HDR', 'l3-value'], loc='best')
        plt.savefig("static/imgs/2-convergenceCurves-RRFLP.svg")
        # plt.show()
        # print("Final Solution:", listdictFinalPop[0]['chromosome'])
        # print("Average Objective value:", 1/listfAveBestIndFitnessEveryGen[-1])
        # print("Average Objective value:", listfAveObjValueEveryIns)

        # 被选中的地址的坐标
        listlist_locationCoordinate = []
        for index in list_BestSolOfThisIns_LocationIndex:
            listlist_locationCoordinate.append(list(np.around(obInstance.a_2d_SitesCoordi[index], decimals=3)))  # 保留小数点后3位

    return list_BestSolOfThisIns_LocationIndex, listlist_locationCoordinate


def funGAMLS2_RFLP_selectExitIns():  # 未完成
    listfAveFitnessEveryIns = np.zeros((iInsNum,)).tolist()
    listfAveObjValueEveryIns = np.zeros((iInsNum,)).tolist()
    listfAveCPUTimeEveryIns = np.zeros((iInsNum,)).tolist()
    # listAllRunsSumFitEvaNumByThisGen = np.zeros((iGenNum + 1,)).tolist()
    a_2d_fEveInsEveRunObjValue = np.zeros((iInsNum, iRunsNum))
    f = open(insName, 'rb')
    listIns = []
    fig = plt.figure()
    for i in range(iInsNum):  # 8 instances
        ins = pickle.load(f)
        listIns.append(ins)
    for i in range(iActualInsNum):
        # genetic algorithm
        listfAllDiffGenBestIndFitness = np.zeros((iGenNum + 1,)).tolist()  # 算上第0代
        listiAllDiffGenDiversityMetric1 = np.zeros((iGenNum + 1,)).tolist()  # 算上第0代
        listiAllDiffGenDiversityMetric2 = np.zeros((iGenNum + 1,)).tolist()  # 算上第0代
        listiFitEvaNumByThisGen_AllRunsAve = np.zeros((iGenNum + 1,)).tolist()
        listfEveGenProportion_belongToLocalSearchedIndNeighbor_exceptCurrLocalSearchedInd_AllRunsAve = np.zeros((iGenNum + 1,)).tolist()
        listfEveGenProportion_belongToNeighborOfAllLocalSearchedInd_AllRunsAve = np.zeros((iGenNum + 1,)).tolist()
        dict_bestIndOfThisIns = {'chromosome': 0,
                                 'fitness': 0}  # 存储该instance下的最好解
        for j in range(iRunsNum):  # Every instance has 10 runs experiments.
            print("Begin: ins " + str(i) + ", Runs " + str(j))
            print("Running......")
            cpuStart = time.process_time()
            # 调用GADM求解
            local_state = np.random.RandomState()
            GeneticAlgo = GAMLS2_DM.GA(listGAParameters, listIns[3], local_state)
            listdictFinalPop, listGenNum, listfBestIndFitnessEveGen, listiDiversityMetric1, listiDiversityMetric2, listiFitEvaNumByThisGen, listfEveGenProportion_belongToLocalSearchedIndNeighbor_exceptCurrLocalSearchedInd, listfEveGenProportion_belongToOnlyCurrGenLocalSearchedIndNeighbor, listfEveGenProportion_belongToNeighborOfAllLocalSearchedInd, listiEveGenLocalSearchedIndNum = GeneticAlgo.funGA_main()
            cpuEnd = time.process_time()
            # 记录CPU time，累加
            listfAveCPUTimeEveryIns[i] += (cpuEnd - cpuStart)

            print("Objective value:", listdictFinalPop[0]['objectValue'])
            # 更新dict_bestInd
            if listdictFinalPop[0]['fitness'] > dict_bestIndOfThisIns['fitness']:
                dict_bestIndOfThisIns['chromosome'] = listdictFinalPop[0]['chromosome']
                dict_bestIndOfThisIns['fitness'] = listdictFinalPop[0]['fitness']
            if listfBestIndFitnessEveGen[-1] != 1/listdictFinalPop[0]['objectValue']:
                print("Wrong. Please check funLS_DM().")
            # 记录最终种群中最好个体的fitness和目标函数值，累加
            a_2d_fEveInsEveRunObjValue[i][j] = listdictFinalPop[0]['objectValue']
            listfAveFitnessEveryIns[i] += listfBestIndFitnessEveGen[-1]
            listfAveObjValueEveryIns[i] += listdictFinalPop[0]['objectValue']
            # 为绘图准备
            new_listfBestIndFitness = [fitness * 1000 for fitness in listfBestIndFitnessEveGen]
            for g in range(len(listGenNum)):
                listfAllDiffGenBestIndFitness[g] += new_listfBestIndFitness[g]
                listiAllDiffGenDiversityMetric1[g] += listiDiversityMetric1[g]
                listiAllDiffGenDiversityMetric2[g] += listiDiversityMetric2[g]
                listiFitEvaNumByThisGen_AllRunsAve[g] += listiFitEvaNumByThisGen[g]
                listfEveGenProportion_belongToLocalSearchedIndNeighbor_exceptCurrLocalSearchedInd_AllRunsAve[g] += listfEveGenProportion_belongToLocalSearchedIndNeighbor_exceptCurrLocalSearchedInd[g]
                listfEveGenProportion_belongToNeighborOfAllLocalSearchedInd_AllRunsAve[g] += listfEveGenProportion_belongToNeighborOfAllLocalSearchedInd[g]

        print("End: ins " + str(i) + ", Runs " + str(j) + "\n")

        # 将个体的chromosome从01转化为被选中地址的索引
        list_bestSolOfThisIns = []
        for c in range(iCandidateFaciNum):
            if dict_bestIndOfThisIns['chromosome'][c] == 1:
                list_bestSolOfThisIns.append(c)

        # 存储每个实例多次运行后最好的运行结果
        listlist_bestSolsOfAllIns_locationIndex.append(list_bestSolOfThisIns)
        print("Best sol:", listlist_bestSolsOfAllIns_locationIndex)
        # 平均每次运行的时间
        listfAveCPUTimeEveryIns[i] /= iRunsNum
        # 平均fitness和目标函数值
        listfAveFitnessEveryIns[i] /= iRunsNum
        listfAveObjValueEveryIns[i] /= iRunsNum
        # 绘图
        listfAveBestIndFitnessEveryGen = [fitness / iRunsNum for fitness in listfAllDiffGenBestIndFitness]
        listiAveDiversityMetric1EveGen = [diversity / iRunsNum for diversity in listiAllDiffGenDiversityMetric1]
        # listiAveDiversityMetric2EveGen = [diversity / iRunsNum for diversity in listiAllDiffGenDiversityMetric2]
        listiFitEvaNumByThisGen_AllRunsAve = [diversity / iRunsNum for diversity in listiFitEvaNumByThisGen_AllRunsAve]
        listfEveGenProportion_belongToLocalSearchedIndNeighbor_exceptCurrLocalSearchedInd_AllRunsAve = [diversity / iRunsNum for diversity in listfEveGenProportion_belongToLocalSearchedIndNeighbor_exceptCurrLocalSearchedInd_AllRunsAve]
        listfEveGenProportion_belongToNeighborOfAllLocalSearchedInd_AllRunsAve = [diversity / iRunsNum for diversity in listfEveGenProportion_belongToNeighborOfAllLocalSearchedInd_AllRunsAve]

        ax1 = fig.add_subplot(111)
        l1, = ax1.plot(listGenNum, listfAveBestIndFitnessEveryGen, marker='*')
        ax1.set_xlabel("# of Generation")
        ax1.set_ylabel("Fitness Of Best Individual")
        # 右方Y轴
        ax2 = ax1.twinx()  # 与ax1共用1个x轴，在右方生成自己的y轴
        l2, = ax2.plot(listGenNum, listiAveDiversityMetric1EveGen, 'r',)
        # l5, = ax2.plot(listGenNum, listiAveDiversityMetric2EveGen, 'go:')
        l3, = ax2.plot(listGenNum, listfEveGenProportion_belongToLocalSearchedIndNeighbor_exceptCurrLocalSearchedInd_AllRunsAve, 'purple', marker='p', linestyle='--')
        # l4, = ax2.plot(listGenNum, listfEveGenProportion_belongToNeighborOfAllLocalSearchedInd_AllRunsAve, 'darkslategray', linestyle='-.')
        ax2.set_ylabel("Diversity Metric")
        # 上方X轴
        ax3 = ax1.twiny()  # 与ax1共用1个y轴，在上方生成自己的x轴
        ax3.set_xlabel("# of Fitness Evaluation")
        listfFeIndex = list(np.linspace(0, iGenNum, num=10+1))
        listFeXCoordinate = []
        for i in range(len(listfFeIndex)):
            listFeXCoordinate.append(listiFitEvaNumByThisGen[int(listfFeIndex[i])])
        ax3.plot(listGenNum, listfAveBestIndFitnessEveryGen, '--')
        ax3.set_xticks(listfFeIndex)
        ax3.set_xticklabels(listFeXCoordinate)
        plt.legend(handles=[l1, l2, l3], labels=['l1:Fitness curve', 'l2:0-HDR', 'l3-value'], loc='best')
        plt.show()
        # print("Final Solution:", listdictFinalPop[0]['chromosome'])
        print("Average Objective value:", 1/listfAveBestIndFitnessEveryGen[-1])
        print("Average Objective value:", listfAveObjValueEveryIns)


if __name__ == '__main__':
    iRunsNum = 8
    '''listInstPara=[0:iSitesNum, 1:iScenNum, 2:iDemandLB, 3:iDemandUB, 4:iFixedCostLB, 5:iFixedCostUP, 6:iCoordinateLB, 7:iCoordinateUB, 8:fFaciFailProb]'''
    iCandidateFaciNum = 10
    iScenNum = 1
    iDemandLB = 0
    iDemandUB = 1000
    iFixedCostLB = 500
    iFixedCostUB = 1500
    iCoordinateLB = 0
    iCoordinateUB = 1
    fFaciFailProb = 0.05
    listInsPara = [iCandidateFaciNum, iScenNum, iDemandLB, iDemandUB, iFixedCostLB, iFixedCostUB, iCoordinateLB, iCoordinateUB, fFaciFailProb]

    '''
    @listGAParameters = [0:iGenNum, 1:iPopSize, 2:iIndLen, 3:fCrosRate, 4:fMutRate, 5:fAlpha, 6:boolAllo2Faci]
    '''
    iGenNum = 10
    iPopSize = 10
    fCrosRate = 0.9
    fMutRate = 0.1
    fAlpha = 1.0
    boolAllo2Faci = True
    listGAParameters = [iGenNum, iPopSize, iCandidateFaciNum, fCrosRate, fMutRate, fAlpha, boolAllo2Faci]

    funGAMLS2_RFLP_generateIns(listInsPara, listGAParameters, iRunsNum)

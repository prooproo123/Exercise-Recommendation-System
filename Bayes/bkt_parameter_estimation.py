import random
import math
import sys


class computeKTparams_SA:
    students = []
    skill = []
    right = []

    skillends = []
    skillnum = -1

    lnminus1_estimation = False
    bounded = True
    L0Tbunded = False

    top = dict()
    stepSize = 0.05
    minVal = 0.000001
    totalSteps = 1000000

    def read_data(self, infile):
        actnum = 0
        self.skillnum = -1
        prevskill = "FLURG"

        for line in open(infile).readlines():

            words = line.split("\t")
            i = 0

            tt = words[i]  # num
            i += 1

            if tt == "":
                prevskill = self.skill[actnum - 1]
                if self.skillnum > -1:
                    self.skillends[self.skillnum] = actnum - 1
                break

            if tt == "num" or tt == "order_id":
                continue

            tt = words[i]  # student
            self.students.append(tt)
            i += 1

            tt = words[i]  # skill
            self.skill.append(tt)
            i += 1

            tt = words[i]  # right
            self.right.append(tt)

            actnum += 1

            if self.skill[actnum - 1] != prevskill:
                prevskill = self.skill[actnum - 1]
                if self.skillnum > -1:
                    self.skillends.append(actnum - 2)
                self.skillnum += 1

        if self.skillnum > -1:
            self.skillends.append(actnum - 1)

    def findGOOF(self, start, end, params):
        SSR = 0
        prevStudent = "FWORPLEJOHN"
        prevL = 0
        prevLgivenresult = 0
        newL = 0
        likelihoodcorrect = 0

        count = 0

        for i in range(start, end + 1):

            if self.students[i] != prevStudent:
                prevL = params.L0
                prevStudent = self.students[i]

            if self.lnminus1_estimation:
                likelihoodcorrect = prevL
            else:
                likelihoodcorrect = (prevL * (1 - params.S)) + (1 - prevL) * params.G

            self.right[i] = int(self.right[i])
            if self.right[i] != -1:
                SSR += (self.right[i] - likelihoodcorrect) * (self.right[i] - likelihoodcorrect)
                count += 1

            if self.right[i] == -1:
                prevLgivenresult = prevL
            else:
                prevLgivenresult = self.right[i] * ((prevL * (1 - params.S)) /
                                                    ((prevL * (1 - params.S)) + ((1 - prevL) * params.G)))
                prevLgivenresult += (1 - self.right[i]) * ((prevL * params.S) /
                                                           ((prevL * params.S) + ((1 - prevL) * (1 - params.G))))

            newL = prevLgivenresult + (1 - prevLgivenresult) * params.T
            prevL = newL

        if count == 0:
            return 0

        return math.sqrt(SSR / count)

    def fit_skill_model(self, currentskill, f):
        if self.L0Tbunded:
            self.top['L0'] = 0.85
            self.top['T'] = 0.3
        else:
            self.top['L0'] = 0.999999
            self.top['T'] = 0.999999

        if self.bounded:
            self.top['G'] = 0.3
            self.top['S'] = 0.1
        else:
            self.top['G'] = 0.999999
            self.top['S'] = 0.999999

        oldParams = BKTParams(init=-1, top=self.top)
        bestParams = BKTParams(init=0.01)

        oldRMSE = 1
        newRMSE = 1

        bestRMSE = 9999999
        prevBestRMSE = 9999999

        temp = 0.005
        startact = 0

        if currentskill > 0:
            startact = self.skillends[currentskill - 1] + 1

        endact = self.skillends[currentskill]

        oldRMSE = self.findGOOF(startact, endact, oldParams)

        for i in range(self.totalSteps):
            newParams = BKTParams(copy=oldParams, randStep=True, top=self.top, stepSize=self.stepSize,
                                  minVal=self.minVal)
            newRMSE = self.findGOOF(startact, endact, newParams)

            if random.random() <= math.exp((oldRMSE - newRMSE) / temp):
                oldParams = BKTParams.BKTParamsCopy(oldParams, copy=newParams)
                oldRMSE = newRMSE

            if newRMSE < bestRMSE:
                bestParams = BKTParams.BKTParamsCopy(bestParams, copy=newParams)
                bestRMSE = newRMSE

            if i % 10000 == 0 and i > 0:
                if bestRMSE == prevBestRMSE:
                    break
                prevBestRMSE = bestRMSE
                temp = temp / 2

        print(self.skill[startact] + "\t" + str(bestParams.L0) + "\t"
              + str(bestParams.G) + "\t" + str(bestParams.S) + "\t"
              + str(bestParams.T) + "\t" + str(bestRMSE) + "\t")

        f.write(self.skill[startact] + "\t" + str(bestParams.L0) + "\t"
                + str(bestParams.G) + "\t" + str(bestParams.S) + "\t"
                + str(bestParams.T) + "\t" + str(bestRMSE) + "\n")

        return (self.skill[startact], bestParams.L0, bestParams.G, bestParams.S, bestParams.T)

    def computelzerot(self, data, f):
        data.to_csv("Dataset.csv", sep="\t", index=None)
        infile = "Dataset.csv"
        self.read_data(infile)

        print("skill\tL0\tG\tS\tT\tRMSE\t")

        params = {}

        for currentskill in range(0, self.skillnum + 1):
            parameters = self.fit_skill_model(currentskill, f)
            params[str(parameters[0])] = list(parameters[1:5])

        return params

class BKTParams:

    def __init__(self, copy = None, randStep = None, top = None, stepSize = None, minVal = None, init = None):

        if init is not None:
            if init < 0:
                self.L0 = random.random() * top['L0']
                self.G = random.random() * top['G']
                self.S = random.random() * top['S']
                self.T = random.random() * top['T']
            else:
                self.L0 = init
                self.G = init
                self.S = init
                self.T = init
        else:
            self.L0 = copy.L0
            self.G = copy.G
            self.S = copy.S
            self.T = copy.T

            if randStep:
                randomChange = random.random()
                thisStep = 2. * (random.random()-0.5) * stepSize

                if randomChange <= 0.25:
                    self.L0 = max(min(self.L0 + thisStep, top['L0']), minVal)
                elif randomChange <= 0.5:
                    self.T = max(min(self.T + thisStep, top['T']), minVal)
                elif randomChange <= 0.75:
                    self.G = max(min(self.G + thisStep, top['G']), minVal)
                else:
                    self.S = max(min(self.S + thisStep, top['S']), minVal)

    def BKTParamsCopy(self, copy):
        return BKTParams(copy, False)


def estimate_parameters(data):

  f = open("BKT_parameters.txt", "w")
  model = computeKTparams_SA()
  params = model.computelzerot(data, f)

#  print(params)
  return params
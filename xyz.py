
row_counter = 0;

accuracy_incrementer = 0;


fileob = open('Face_C.csv')
Face_C = fileob.readlines()
fileob = open('Face_G.csv')
Face_G = fileob.readlines();
fileob = open('Finger_li.csv')
Finger_li = fileob.readlines();
fileob = open('Finger_ri.csv')
Finger_r i= fileob.readlines()
fileob = open('t.csv')
check = fileob.readlines()
for i in range(len(check)):
    check[i] = int(check[i])

f17 =[i for i in range(1, 518)]

dk1 = {}
dk2 = {}
while row_counter < 517:

    r1 = Face_G[row_counter].split(',')
    r2 = Face_C[row_counter].split(',')
    r3 = Finger_li[row_counter].split(',')
    r4 = Finger_ri[row_counter].split(',')

    for i in range(517):
        r1[i] = float(r1[i])
        r2[i] = float(r2[i])
        r3[i] = float(r3[i])
        r4[i] = float(r4[i])

    pt = {}

    pt[0] = r1
    pt[1] = r2
    pt[2] = r3
    pt[3] = r4

    import random

    for i in range(4, 100):
        random.shuffle(f17)
        tmp = []
        for j in range(517):
            tmp.append(f17[j])
        pt[i] = tmp

    fitness = {}

    for i in range(100):
        f1 = 0
        f2 = 0
        f3 = 0
        f4 = 0
        tmp = pt[i]
        mp = {};
        mp1 = {};
        mp2 = {};
        mp3 = {};
        mp4 = {}
        for j in range(517):
            mp1[r1[j]] = j + 1
            mp2[r2[j]] = j + 1
            mp3[r3[j]] = j + 1
            mp4[r4[j]] = j + 1
            mp[tmp[j]] = j + 1

        for j in range(1, 518):
            f1 = f1 + (1 - (min(mp[j], mp1[j]) - 1) / 517) * abs(mp[j] - mp1[j]);
            f2 = f2 + (1 - (min(mp[j], mp2[j]) - 1) / 517) * abs(mp[j] - mp2[j]);
            f3 = f3 + (1 - (min(mp[j], mp3[j]) - 1) / 517) * abs(mp[j] - mp3[j]);
            f4 = f4 + (1 - (min(mp[j], mp4[j]) - 1) / 517) * abs(mp[j] - mp4[j]);

        fitness[i] = [f1, f2, f3, f4]

    rank = {}

    for i in range(100):
        count = 0
        for j in range(100):
            if i != j:
                if fitness[j][0] < fitness[i][0] and fitness[j][1] < fitness[i][1] and fitness[j][2] < fitness[i][2] and \
                        fitness[j][3] < fitness[i][3]:
                    count += 1
        rank[i] = count

    qt = {}
    sizeofqt = 0

    while sizeofqt < 100:
        ind1 = random.randint(0, 99)
        ind2 = random.randint(0, 99)
        winner1 = 0
        winner1 = 0
        if fitness[ind1][0] < fitness[ind2][0] and fitness[ind1][1] < fitness[ind2][1] and fitness[ind1][2] < \
                fitness[ind2][2] and fitness[ind1][3] < fitness[ind2][3]:
            winner1 = 1
        else:
            winner2 = 1

        parent1 = None
        if winner1 == 1:
            parent1 = pt[ind1]
        else:
            parent1 = pt[ind2]

        ind1 = random.randint(0, 99)
        ind2 = random.randint(0, 99)
        winner1 = 0
        winner1 = 0
        if fitness[ind1][0] < fitness[ind2][0] and fitness[ind1][1] < fitness[ind2][1] and fitness[ind1][2] < \
                fitness[ind2][2] and fitness[ind1][3] < fitness[ind2][3]:
            winner1 = 1
        else:
            winner2 = 1

        parent2 = None
        if winner1 == 1:
            parent2 = pt[ind1]
        else:
            parent2 = pt[ind2]

        crossoverpoint = random.randint(0, 516)

        o1 = []
        o2 = []

        for i in range(crossoverpoint):
            o1.append(parent1[i])
            o2.append(parent2[i])

        for i in range(crossoverpoint, 517):
            o1.append(parent2[i])
            o2.append(parent1[i])

        mp1 = {}
        for i in range(1, 518):
            mp1[i] = 0

        for i in range(517):
            mp1[o1[i]] += 1

        notfound = []
        foundtwice = []

        for i in range(1, 518):
            if mp1[i] == 0:
                notfound.append(i)
            elif mp1[i] == 2:
                foundtwice.append(i)

        for i in range(crossoverpoint, 517):
            if mp1[o1[i]] == 2:
                o1[i] = notfound[0]
                notfound.pop(0)
                mp1[o1[i]] = 1

        mp1 = {}
        for i in range(1, 518):
            mp1[i] = 0

        for i in range(517):
            mp1[o2[i]] += 1

        notfound = []
        foundtwice = []

        for i in range(1, 518):
            if mp1[i] == 0:
                notfound.append(i)
            elif mp1[i] == 2:
                foundtwice.append(i)

        for i in range(crossoverpoint, 517):
            if mp1[o2[i]] == 2:
                o2[i] = notfound[0]
                notfound.pop(0)
                mp1[o2[i]] = 1

        mutationpoints = []
        for i in range(100):
            x = random.uniform(0, 1)
            if x <= 0.01:
                mutationpoints.append(i)

        if len(mutationpoints) % 2 == 1:
            mutationpoints.pop()

        for i in range(0, len(mutationpoints), 2):
            ind1 = mutationpoints[i]
            ind2 = mutationpoints[i + 1]

            tmp = o1[ind1]
            o1[ind1] = o1[ind2]
            o1[ind2] = tmp

        mutationpoints = []
        for i in range(100):
            x = random.uniform(0, 1)
            if x <= 0.01:
                mutationpoints.append(i)

        if len(mutationpoints) % 2 == 1:
            mutationpoints.pop()

        for i in range(0, len(mutationpoints), 2):
            ind1 = mutationpoints[i]
            ind2 = mutationpoints[i + 1]

            tmp = o2[ind1]
            o2[ind1] = o2[ind2]
            o2[ind2] = tmp

        qt[sizeofqt] = o1
        sizeofqt += 1
        qt[sizeofqt] = o2
        sizeofqt += 1

    gen = 0
    v1prev = [0, 0, 0, 0]
    v2prev = [0, 0, 0, 0]
    saturationcounter = 0
    graph1 = []
    graph2 = []
    graph3 = []
    graph4 = []

    while gen <= 1000:

        concatpopulation = {}

        for i in range(100):
            concatpopulation[i] = pt[i]

        for i in range(100):
            concatpopulation[100 + i] = qt[i]

        fitnesscat = {}

        for i in range(200):
            f1 = 0
            f2 = 0
            f3 = 0
            f4 = 0
            tmp = concatpopulation[i]
            mp = {};
            mp1 = {};
            mp2 = {};
            mp3 = {};
            mp4 = {}
            for j in range(517):
                mp1[r1[j]] = j + 1
                mp2[r2[j]] = j + 1
                mp3[r3[j]] = j + 1
                mp4[r4[j]] = j + 1
                mp[tmp[j]] = j + 1

            for j in range(1, 518):
                f1 = f1 + (1 - (min(mp[j], mp1[j]) - 1) / 517) * abs(mp[j] - mp1[j]);
                f2 = f2 + (1 - (min(mp[j], mp2[j]) - 1) / 517) * abs(mp[j] - mp2[j]);
                f3 = f3 + (1 - (min(mp[j], mp3[j]) - 1) / 517) * abs(mp[j] - mp3[j]);
                f4 = f4 + (1 - (min(mp[j], mp4[j]) - 1) / 517) * abs(mp[j] - mp4[j]);

            fitnesscat[i] = [f1, f2, f3, f4]

        rankcat = {}
        for i in range(200):
            count = 0
            for j in range(200):
                if i != j:
                    if fitnesscat[j][0] < fitnesscat[i][0] and fitnesscat[j][1] < fitnesscat[i][1] and fitnesscat[j][
                        2] < fitnesscat[i][2] and \
                            fitnesscat[j][3] < fitnesscat[i][3]:
                        count += 1
            rankcat[i] = count

        srank = []

        for i in range(200):
            srank.append(rankcat[i])

        usrank = list(set(srank))
        usrank.sort()

        cdist = {}
        for i in range(200):
            cdist[i] = 0

        pt1 = {}
        newselectedpopulation = []
        frontbreak = []
        found = 0

        for i in range(len(usrank)):
            tmpfront = []

            for j in range(200):
                if rankcat[j] == usrank[i]:
                    tmpfront.append(j)

            if len(newselectedpopulation) + len(tmpfront) <= 100:
                for j in range(len(tmpfront)):
                    newselectedpopulation.append(tmpfront[j])

            if len(newselectedpopulation) + len(tmpfront) > 100 and found == 0:
                for j in range(len(tmpfront)):
                    frontbreak.append(tmpfront[j])

                found = 1

            if len(tmpfront) == 1:
                cdist[tmpfront[0]] = 10000000
                pass
            if len(tmpfront) == 2:
                cdist[tmpfront[0]] = 10000000
                cdist[tmpfront[1]] = 10000000
                pass

            for m in range(4):
                fitnesstmpfront = []
                tmpfront_map_fitnesstmpfront = {}

                for ii in range(len(tmpfront)):
                    fitnesstmpfront.append(fitnesscat[tmpfront[ii]][m])
                    tmpfront_map_fitnesstmpfront[tmpfront[ii]] = fitnesscat[tmpfront[ii]][m]

                usfitnesstmpfront = list(set(fitnesstmpfront))
                usfitnesstmpfront.sort(reverse=True)

                sortedtmpfront = []

                for ii in range(len(usfitnesstmpfront)):
                    for j in range(len(tmpfront)):
                        if tmpfront_map_fitnesstmpfront[tmpfront[j]] == usfitnesstmpfront[ii]:
                            sortedtmpfront.append(tmpfront[j])

                cdist[sortedtmpfront[0]] = 10000000
                cdist[sortedtmpfront[len(sortedtmpfront) - 1]] = 10000000

                mxfitness = tmpfront_map_fitnesstmpfront[sortedtmpfront[0]]
                mnfitness = tmpfront_map_fitnesstmpfront[sortedtmpfront[len(sortedtmpfront) - 1]]

                for ii in range(1, len(sortedtmpfront) - 1):
                    try:
                        cdist[sortedtmpfront[ii]] += abs((tmpfront_map_fitnesstmpfront[sortedtmpfront[ii + 1]] -
                                                          tmpfront_map_fitnesstmpfront[sortedtmpfront[ii - 1]]) \
                                                             (mxfitness - mnfitness))
                    except:
                        cdist[sortedtmpfront[ii]] = 10000000

        if len(newselectedpopulation) < 100:
            distance = []
            for i in range(len(frontbreak)):
                distance.append(cdist[frontbreak[i]])

            udistance = list(set(distance))
            usdistance = udistance.sort(reverse=True)

            for i in range(len(usdistance)):
                for j in range(len(frontbreak)):
                    if cdist[frontbreak[j]] == usdistance[i]:
                        newselectedpopulation.append(frontbreak[j])
                    if len(newselectedpopulation) == 100:
                        break
                if len(newselectedpopulation) == 100:
                    break

        for i in range(100):
            pt1[i] = concatpopulation[newselectedpopulation[i]]

        qt1 = {}
        sizeofqt1 = 0

        while sizeofqt1 < 100:
            ind1 = random.randint(0, 99)
            ind2 = random.randint(0, 99)
            winner1 = 0
            winner1 = 0
            if rankcat[newselectedpopulation[ind1]] < rankcat[newselectedpopulation[ind2]]:
                winner1 = 1

            if rankcat[newselectedpopulation[ind1]] > rankcat[newselectedpopulation[ind2]]:
                winner2 = 1

            if rankcat[newselectedpopulation[ind1]] == rankcat[newselectedpopulation[ind2]]:
                if cdist[newselectedpopulation[ind1]] > cdist[newselectedpopulation[ind2]]:
                    winner1 = 1
                elif cdist[newselectedpopulation[ind2]] > cdist[newselectedpopulation[ind1]]:
                    winner2 = 1
                else:
                    winner1 = 1

            parent1 = None
            if winner1 == 1:
                parent1 = concatpopulation[newselectedpopulation[ind1]]
            else:
                parent1 = concatpopulation[newselectedpopulation[ind2]]

            ind1 = random.randint(0, 99)
            ind2 = random.randint(0, 99)
            winner1 = 0
            winner1 = 0
            if rankcat[newselectedpopulation[ind1]] < rankcat[newselectedpopulation[ind2]]:
                winner1 = 1

            if rankcat[newselectedpopulation[ind1]] > rankcat[newselectedpopulation[ind2]]:
                winner2 = 1

            if rankcat[newselectedpopulation[ind1]] == rankcat[newselectedpopulation[ind2]]:
                if cdist[newselectedpopulation[ind1]] > cdist[newselectedpopulation[ind2]]:
                    winner1 = 1
                elif cdist[newselectedpopulation[ind2]] > cdist[newselectedpopulation[ind1]]:
                    winner2 = 1
                else:
                    winner1 = 1

            parent2 = None
            if winner1 == 1:
                parent2 = concatpopulation[newselectedpopulation[ind1]]
            else:
                parent2 = concatpopulation[newselectedpopulation[ind2]]

            crossoverpoint = random.randint(0, 516)

            o1 = []
            o2 = []

            for i in range(crossoverpoint):
                o1.append(parent1[i])
                o2.append(parent2[i])

            for i in range(crossoverpoint, 517):
                o1.append(parent2[i])
                o2.append(parent1[i])
            mp1 = {}
            for i in range(1, 518):
                mp1[i] = 0

            for i in range(517):
                mp1[o1[i]] += 1

            notfound = []
            foundtwice = []

            for i in range(1, 518):
                if mp1[i] == 0:
                    notfound.append(i)
                elif mp1[i] == 2:
                    foundtwice.append(i)

            for i in range(crossoverpoint, 517):
                if mp1[o1[i]] == 2:
                    o1[i] = notfound[0]
                    notfound.pop(0)
                    mp1[o1[i]] = 1

            mp1 = {}
            for i in range(1, 518):
                mp1[i] = 0

            for i in range(517):
                mp1[o2[i]] += 1

            notfound = []
            foundtwice = []

            for i in range(1, 518):
                if mp1[i] == 0:
                    notfound.append(i)
                elif mp1[i] == 2:
                    foundtwice.append(i)

            for i in range(crossoverpoint, 517):
                if mp1[o2[i]] == 2:
                    o2[i] = notfound[0]
                    notfound.pop(0)
                    mp1[o2[i]] = 1

            mutationpoints = []
            for i in range(100):
                x = random.uniform(0, 1)
                if x <= 0.01:
                    mutationpoints.append(i)

            if len(mutationpoints) % 2 == 1:
                mutationpoints.pop()

            for i in range(0, len(mutationpoints), 2):
                ind1 = mutationpoints[i]
                ind2 = mutationpoints[i + 1]

                tmp = o1[ind1]
                o1[ind1] = o1[ind2]
                o1[ind2] = tmp

            mutationpoints = []
            for i in range(100):
                x = random.uniform(0, 1)
                if x <= 0.01:
                    mutationpoints.append(i)

            if len(mutationpoints) % 2 == 1:
                mutationpoints.pop()

            for i in range(0, len(mutationpoints), 2):
                ind1 = mutationpoints[i]
                ind2 = mutationpoints[i + 1]

                tmp = o2[ind1]
                o2[ind1] = o2[ind2]
                o2[ind2] = tmp

            qt[sizeofqt] = o1
            sizeofqt += 1
            qt[sizeofqt] = o2
            sizeofqt += 1

        prevparentfitness = {}

        for i in range(100):
            f1 = 0
            f2 = 0
            f3 = 0
            f4 = 0
            tmp = pt[i]
            mp = {};
            mp1 = {};
            mp2 = {};
            mp3 = {};
            mp4 = {}
            for j in range(517):
                mp1[r1[j]] = j + 1
                mp2[r2[j]] = j + 1
                mp3[r3[j]] = j + 1
                mp4[r4[j]] = j + 1
                mp[tmp[j]] = j + 1

            for j in range(1, 518):
                f1 = f1 + (1 - (min(mp[j], mp1[j]) - 1) / 517) * abs(mp[j] - mp1[j]);
                f2 = f2 + (1 - (min(mp[j], mp2[j]) - 1) / 517) * abs(mp[j] - mp2[j]);
                f3 = f3 + (1 - (min(mp[j], mp3[j]) - 1) / 517) * abs(mp[j] - mp3[j]);
                f4 = f4 + (1 - (min(mp[j], mp4[j]) - 1) / 517) * abs(mp[j] - mp4[j]);

            prevparentfitness[i] = [f1, f2, f3, f4]

        currentparentfitness = {}
        for i in range(100):
            f1 = 0
            f2 = 0
            f3 = 0
            f4 = 0
            tmp = pt1[i]
            mp = {};
            mp1 = {};
            mp2 = {};
            mp3 = {};
            mp4 = {}
            for j in range(517):
                mp1[r1[j]] = j + 1
                mp2[r2[j]] = j + 1
                mp3[r3[j]] = j + 1
                mp4[r4[j]] = j + 1
                mp[tmp[j]] = j + 1

            for j in range(1, 518):
                f1 = f1 + (1 - (min(mp[j], mp1[j]) - 1) / 517) * abs(mp[j] - mp1[j]);
                f2 = f2 + (1 - (min(mp[j], mp2[j]) - 1) / 517) * abs(mp[j] - mp2[j]);
                f3 = f3 + (1 - (min(mp[j], mp3[j]) - 1) / 517) * abs(mp[j] - mp3[j]);
                f4 = f4 + (1 - (min(mp[j], mp4[j]) - 1) / 517) * abs(mp[j] - mp4[j]);

            currentparentfitness[i] = [f1, f2, f3, f4]
        v1cur = [0, 0, 0, 0]
        for i in range(100):
            tf = 1
            if currentparentfitness[i][0] < prevparentfitness[i][0] and currentparentfitness[i][1] < \
                    prevparentfitness[i][1] and \
                    currentparentfitness[i][2] < prevparentfitness[i][2] and currentparentfitness[i][3] < \
                    prevparentfitness[i][3]:
                pt[i] = pt1[i]
                tf = 0
                for k in range(4):
                    v1cur[k] += currentparentfitness[i][k]

            if tf == 1:
                for k in range(4):
                    v1cur[k] += prevparentfitness[i][k]

        prevchildfitness = {}
        for i in range(100):
            f1 = 0
            f2 = 0
            f3 = 0
            f4 = 0
            tmp = qt[i]
            mp = {};
            mp1 = {};
            mp2 = {};
            mp3 = {};
            mp4 = {}
            for j in range(517):
                mp1[r1[j]] = j + 1
                mp2[r2[j]] = j + 1
                mp3[r3[j]] = j + 1
                mp4[r4[j]] = j + 1
                mp[tmp[j]] = j + 1

            for j in range(517):
                f1 = f1 + (1 - (min(mp[j], mp1[j]) - 1) / 517) * abs(mp[j] - mp1[j]);
                f2 = f2 + (1 - (min(mp[j], mp2[j]) - 1) / 517) * abs(mp[j] - mp2[j]);
                f3 = f3 + (1 - (min(mp[j], mp3[j]) - 1) / 517) * abs(mp[j] - mp3[j]);
                f4 = f4 + (1 - (min(mp[j], mp4[j]) - 1) / 517) * abs(mp[j] - mp4[j]);

            prevchildfitness[i] = [f1, f2, f3, f4]

        currentchildfitness = {}
        for i in range(100):
            f1 = 0
            f2 = 0
            f3 = 0
            f4 = 0
            tmp = qt1[i]
            mp = {};
            mp1 = {};
            mp2 = {};
            mp3 = {};
            mp4 = {}
            for j in range(517):
                mp1[r1[j]] = j + 1
                mp2[r2[j]] = j + 1
                mp3[r3[j]] = j + 1
                mp4[r4[j]] = j + 1
                mp[tmp[j]] = j + 1

            for j in range(517):
                f1 = f1 + (1 - (min(mp[j], mp1[j]) - 1) / 517) * abs(mp[j] - mp1[j]);
                f2 = f2 + (1 - (min(mp[j], mp2[j]) - 1) / 517) * abs(mp[j] - mp2[j]);
                f3 = f3 + (1 - (min(mp[j], mp3[j]) - 1) / 517) * abs(mp[j] - mp3[j]);
                f4 = f4 + (1 - (min(mp[j], mp4[j]) - 1) / 517) * abs(mp[j] - mp4[j]);

            currentchildfitness[i] = [f1, f2, f3, f4]

        v2cur = [0, 0, 0, 0]

        for i in range(100):
            tf = 1
            if currentchildfitness[i][0] < prevchildfitness[i][0] and currentchildfitness[i][1] < \
                    prevchildfitness[i][1] and \
                    currentchildfitness[i][2] < prevchildfitness[i][2] and currentchildfitness[i][3] < \
                    prevchildfitness[i][3]:
                qt[i] = qt1[i]
                tf = 0
                for k in range(4):
                    v2cur[k] += currentchildfitness[i][k]

            if tf == 1:
                for k in range(4):
                    v2cur[k] += prevchildfitness[i][k]

        cnd1 = 0
        diffv1 = 0
        for k in range(4):
            diffv1 += abs(v1cur[k] - v1prev[k])

        ck1 = 0.01 * sum(v1cur)

        if diffv1 <= ck1:
            cnd1 = 1

        cnd2 = 0
        diffv2 = 0
        for k in range(4):
            diffv2 += abs(v2cur[k] - v2prev[k])

        ck2 = 0.01 * sum(v2cur)

        if diffv2 <= ck2:
            cnd2 = 1

        if cnd1 == 1 and cnd2 == 1 and saturationcounter == 100:
            dk1[row_counter] = gen
            break

        if cnd1 == 1 and cnd2 == 1:
            saturationcounter += 1
        if cnd1 != 1 or cnd2 != 1:
            saturationcounter = 0

        v1prev = v1cur;
        v2prev = v2cur;
        graph1.append(v1cur[0])
        graph2.append(v1cur[1])
        graph3.append(v1cur[3])
        graph4.append(v1cur[4])

        gen += 1

    """
    import matplotlib.pyplot as pl
    xis = [i for i in range(100)]
    pl.plot(xis,graph1)
    pl.plot(xis,graph2)
    pl.plot(xis,graph3)
    pl.plot(xis,graph4)
    pl.show()
    """

    finalpopulation = {}

    for i in range(100):
        finalpopulation[i] = pt[i]
    for i in range(100):
        finalpopulation[i + 100] = qt[i]

    finalfitness = {}

    for i in range(200):
        f1 = 0
        f2 = 0
        f3 = 0
        f4 = 0
        tmp = finalpopulation[i]
        for j in range(517):
            f1 += abs(tmp[j] - r1[j])
            f2 += abs(tmp[j] - r2[j])
            f3 += abs(tmp[j] - r3[j])
            f4 += abs(tmp[j] - r4[j])

        finalfitness[i] = [f1, f2, f3, f4]
    for i in range(200):
        f1 = 0
        f2 = 0
        f3 = 0
        f4 = 0
        tmp = finalpopulation[i]
        mp = {};
        mp1 = {};
        mp2 = {};
        mp3 = {};
        mp4 = {}
        for j in range(517):
            mp1[r1[j]] = j + 1
            mp2[r2[j]] = j + 1
            mp3[r3[j]] = j + 1
            mp4[r4[j]] = j + 1
            mp[tmp[j]] = j + 1

        for j in range(1, 518):
            f1 = f1 + (1 - (min(mp[j], mp1[j]) - 1) / 517) * abs(mp[j] - mp1[j]);
            f2 = f2 + (1 - (min(mp[j], mp2[j]) - 1) / 517) * abs(mp[j] - mp2[j]);
            f3 = f3 + (1 - (min(mp[j], mp3[j]) - 1) / 517) * abs(mp[j] - mp3[j]);
            f4 = f4 + (1 - (min(mp[j], mp4[j]) - 1) / 517) * abs(mp[j] - mp4[j]);

        finalfitness[i] = [f1, f2, f3, f4]

    finalrank = {}

    for i in range(200):
        count = 0
        for j in range(200):
            if i != j:
                if finalfitness[j][0] < finalfitness[i][0] and finalfitness[j][1] < finalfitness[i][1] and \
                        finalfitness[j][2] < finalfitness[i][2] and \
                        finalfitness[j][3] < finalfitness[i][3]:
                    count += 1
        finalrank[i] = count

    res = []

    for i in range(200):
        if finalrank[i] == 1:
            res.append(i)

    linmap = {}
    for i in range(200):
        linmap[i] = 10000000

    for i in range(len(res)):
        linmap[res[i]] = finalfitness[res[i]][0] + finalfitness[res[i]][1] + finalfitness[res[i]][2] + \
                         finalfitness[res[i]][3]

    mn = 10000000
    found = []

    for i in range(len(res)):
        if mn > linmap[res[i]]:
            mn = linmap[res[i]]

    for i in range(len(res)):
        if mn == linmap[res[i]]:
            found = res[i]
            break

    # print('found->',found)

    found_individual = finalpopulation[found]
    dk2[row_counter] = finalpopulation[found]
    original_individual = check[row_counter]

    if original_individual == found_individual:
        accuracy_incrementer = accuracy_incrementer + 1

    print(
        '--------------------------------------------------------------------------------------------------------------------------------------------------------')
    print('found_individual        ->', found_individual);
    print('original_invividual     ->', original_individual);
    print('row ->%d accuracy        ->', row_counter, accuracy_incrementer)
    print('iteration end            ->', dk1[row_counter])
    print('Fused List               ->', dk2[row_counter])

    print(
        '---------------------------------------------------------------------------------------------------------------------------------------------------------')

    row_counter = row_counter + 1

accuracy = (accuracy_incrementer / 517) * 100

print(accuracy)







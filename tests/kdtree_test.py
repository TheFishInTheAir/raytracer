from random import uniform
import copy
import time
print("This is just practice for implementing the k-d tree in c")

class Sphere:
    def __init__(self, x, y, z, size):
        self.pos = (x,y,z)
        self.r = size
    def getPosByAxis(self, dim):
        return self.pos[dim]
    def getInfoStr(self):
        return "Sphere: (%8.4f, %8.4f, %8.4f), radius: %7.4f" % (self.pos[0], self.pos[1], self.pos[2], self.r)
    def collidesPlane(self, axis, wB):
        if (self.pos[axis] + self.r > wB and
            self.pos[axis] - self.r < wB):
            return True
        return False
    def isLeftOfPlane(self, axis, wB):
        if (self.pos[axis] < wB):
            return True
        return False

class Event:
    def __init__(self, sphere, b, k, eType):
        self.sphere = sphere
        self.b = b
        self.k = k
        self.eType = eType
    def __lt__(self, other):
        return ((self.b < other.b) or
                (self.b == other.b and self.eType < other.eType))

class KDTreeNode:
    def __init__(self, k):
        self.splitPlane = 0 #0 = x, 1 = y, 2 = z
        self.k = k
        self.depth = 0
        self.childFirst = None
        self.childSecond = None
        self.spheres = []
        self.b = 0.5 #Spatial Median default
        self.voxel = None

    def isLeaf(self):
        return (self.childFirst == None) and (self.childSecond == None)

class Voxel:
    def __init__(self, minV, maxV):
        self.minV = minV
        self.maxV = maxV

    #def __init__(self, sphere):
    def initFromSphere(self, sphere):
        self.minV = list(simdAddScalar(sphere.pos, -sphere.r, 3))
        self.maxV = list(simdAddScalar(sphere.pos,  sphere.r, 3))
        #print(self.getInfoStr())

    def collides(self, sphere):
        if (simdMoreThan(simdAddScalar(sphere.pos, sphere.r, 3),  self.minV,  3) and
            simdMoreThan(self.maxV, simdAddScalar(sphere.pos, -sphere.r, 3),  3)):
            return True
        return False
    def getInfoStr(self):
        return "Voxel: (%8.4f, %8.4f, %8.4f) to (%8.4f, %8.4f, %8.4f)" % (
            self.minV[0], self.minV[1], self.minV[2],
            self.maxV[0], self.maxV[1], self.maxV[2])

    def isPlanar(self):
        for i in range(3):
            if(self.minV[i]==self.maxV[i]):
                return True
        return False

    #def bToWorld(self, axis, b):
    #    return self.maxV[0][axis] + (self.minV[axis] - self.maxV[axis])*b

    def divide(self, axis, b):
        divMin = list(self.minV)
        divMax = list(self.maxV)


        divMin[axis] = self.minV[axis] + (self.maxV[axis] - self.minV[axis])*b
        divMax[axis] = self.minV[axis] + (self.maxV[axis] - self.minV[axis])*(b)

        return (Voxel(self.minV, divMax),
                Voxel(divMin, self.maxV))

    def iLerp(self, worldB, axis):
        return (worldB - self.minV[axis]) / (self.maxV[axis] - self.minV[axis])

    def divideWorld(self, axis, worldB):

        #b = self.iLerp(worldB, axis)
        #print(worldB)
        #print(b)

        divMin = list(self.minV)
        divMax = list(self.maxV)
        #divMin[axis] = self.minV[axis] + (self.maxV[axis] - self.minV[axis])*b
        #divMax[axis] = self.minV[axis] + (self.maxV[axis] - self.minV[axis])*(b)
        divMin[axis] = worldB
        divMax[axis] = worldB

        #print("%d: %f to %f" % (axis, divMin[axis], divMax[axis]))

        return (Voxel(self.minV, divMax),
                Voxel(divMin, self.maxV))
class KDTree:
    def __init__(self):
        self.root = None #KDTreeNode
        self.k = 3
        self.maxRecurse = 30
        self.KT = 1.0 #TRAVERSAL    COST GENERIC DEFAULT
        self.KI = 1.5 #INTERSECTION COST GENERIC DEFAULT

    def C(self, PL, PR, NL, NR):
        return perfLambda(NL,NR, PL, PR)*(self.KT + self.KI*(PL*NL + PR*NR))

    def SAH(self, axis, b, V, NL, NR, NP):
        #print("SAH START")
        newVox = V.divide(axis, b)
        VL = newVox[0]
        VR = newVox[1]
        PL = SA_VOXEL(VL)/SA_VOXEL(V)
        PR = SA_VOXEL(VR)/SA_VOXEL(V)
        #print(VL.getInfoStr())
        #print("PL/PR: %f %f" % (PL, PR))
        #print("VLSA/VRSA: %f %f" % (SA_VOXEL(VL), SA_VOXEL(VR)))
        CPL = self.C(PL, PR, NL+NP, NR)
        CPR = self.C(PL, PR, NL, NR+NP)
        #print("SAH END")
        if CPL < CPR:
            return (CPL, 1) #1 is LEFT
        else:
            return (CPR, 2) #2 is RIGHT

    def FindPlaneAlg5(N, V, E):
        bestC = 10000000
        bestPSide = None
        pInfo = None

        NL = [None]*self.k
        NP = [None]*self.k
        NR = [None]*self.k
        for k in range(self.k):
            NL[k] = 0
            NP[k] = 0
            NR[k] = N
        i = 0
        LE = len(E)
        while i < LE:
            p = E[i] #(E[i].p, E[i].k) I THINK?
            Ps = 0
            Pe = 0
            Pp = 0
            while i < len(E) and E[i].k == p.k and E[i].b == p.b and E[i].eType == 0: #End
                Pe += 1
                i  += 1
            while i < len(E) and E[i].k == p.k and E[i].b == p.b and E[i].eType == 1: #Planar
                Pp += 1
                i  += 1
            while i < len(E) and E[i].k == p.k and E[i].b == p.b and E[i].eType == 2: #Start
                Ps += 1
                i  += 1
            NP[p.k] =  Pp #Planar
            NR[p.k] -= Pp #Planar
            NR[p.k] -= Pe #End
            dataSAH = SAH(p.k, p.b, V, NL[p.k], NR[p.k], NP[p.k]) # I THINK, THIS PAPER DOESN'T MAKE SENSE?
            C = dataSAH[0]
            side = dataSAH[1]
            if C < bestC:
                bestC = C
                bestPSide = side
                pInfo = p
            NL[p.k] += Ps #Start
            NL[p.k] += Pp #Planar
            NP[p.k] = 0

            return (pInfo, bestPSide)



    def N2SAHPartition(self, spheres, voxel):
        bestC = 10000000
        bestPSide = None
        pInfo = None
        for s in spheres:
            for p in perfectSplits(s, voxel):

                newV = voxel.divideWorld(p[0], p[1])
                VL = newV[0]
                VR = newV[1]
                Ts = classify(spheres, VL, VR, p)
                dataSAH = self.SAH(p[0], voxel.iLerp(p[1], p[0]),
                              voxel, len(Ts[0]), len(Ts[1]), len(Ts[2]))
                #print(dataSAH[0])
                if(dataSAH[0] <= bestC):

                    bestC = dataSAH[0]
                    bestPSide = dataSAH[1]
                    pInfo = (VL, VR, Ts, p)



        #print("GOAABA %f: %f" % (pInfo[3][1], bestC))
        if bestPSide == 1: #LEFT
            pInfo[2][0].extend(pInfo[2][2])
            return (pInfo[3], pInfo[2][0], pInfo[2][1]) # Plane, SpheresL, SpheresR
        elif bestPSide ==2: #RIGHT
            pInfo[2][1].extend(pInfo[2][2])
            return (pInfo[3], pInfo[2][0], pInfo[2][1])
        else:
            print("BAD THING: Attempt to do SAH Partition on empty node?nn")


    def terminate(self, spheres, voxel, depth):

        if voxel.isPlanar():
            return True
        if depth==self.maxRecurse:
            return True
        if len(spheres)<=1:
            return True
        #include other cases
        return False


    def findPlane(self, spheres, voxel):
        return 0.5



    def findPlaneAlg4(self, spheres, voxel):
        bestC = 10000000
        bestPSide = None
        pInfo = None

        for k in range(3):
            E = []
            for s in spheres:
                sVox = Voxel([0,0,0], [0,0,0])
                sVox.initFromSphere(s)
                B = clip(sVox, voxel)

                if(B.isPlanar()):
                    E.append(Event(s, B.minV[k], k, 1)) #1 is Planar
                else:
                    E.append(Event(s, B.minV[k], k, 2)) #2 is Start
                    E.append(Event(s, B.minV[k], k, 0)) #0 is End
            E.sort()

            #Sweep over split candidates
            NL = 0
            NP = 0
            NR = len(spheres)

            i = 0
            while (i < len(E)):
                p = E[i]
                Ps = 0
                Pe = 0
                Pp = 0
                while i < len(E) and E[i].b == p.b and E[i].eType == 0: #End
                    Pe += 1
                    i  += 1
                while i < len(E) and E[i].b == p.b and E[i].eType == 1: #Planar
                    Pp += 1
                    i  += 1
                while i < len(E) and E[i].b == p.b and E[i].eType == 2: #Start
                    Ps += 1
                    i  += 1
                NP =  Pp
                NR -= Pp
                NR -= Pe

                dataSAH = self.SAH(k, p.b, voxel, NL, NR, NP)
                C = dataSAH[0]
                if C < bestC:
                    bestC = C
                    bestPSide = dataSAH[1]
                    pInfo = (k, p.b)
                NL += Ps
                NL += Pp
                NP = 0
                i += 1
        return (pInfo, bestPSide)

    #Algorithm 4 from https://webserver2.tecgraf.puc-rio.br/~psantos/inf2602_2008_2/pesquisa/hierarchy/kdtree/On%20building%20fast%20kd-Trees%20for%20Ray%20Tracing%20and%20on%20doing%20that%20in%20O(N%20log%20N).pdf
    def _simpleInsertF4(self, spheres, voxel, depth):
        #axis = depth % self.k

        node = KDTreeNode(self.k)
        #node.splitPlane = axis
        node.depth = depth
        node.voxel = voxel
        if self.terminate(spheres, voxel, depth):
            node.spheres = spheres
            return node

        #node.b = self.findPlane(spheres, voxel)
        #newVox = voxel.divide(axis, node.b)
        data = self.findPlaneAlg4(spheres, voxel)
        axis = data[0][0]
        wB   = data[0][1]

        #node.b = voxel.iLerp(wB, axis)
        #node.splitPlane = axis


        newVox = voxel.divideWorld(axis, wB)

        VL = newVox[0]
        VR = newVox[1]

        sData = classify(spheres, VL, VR, (axis, wB))

        SL = sData[0]
        SR = sData[1]
        SP = sData[2]

        if(data[1] == 1):
            SL.extend(SP)
        else:
            SR.extend(SP)
        #SL = [s for s in spheres if VL.collides(s)]
        #SR = [s for s in spheres if VR.collides(s)]
        #print("%d, %d" % (len(SL), len(SR)))
        node.childFirst  = self._simpleInsertF3(SL, VL, depth+1)
        node.childSecond = self._simpleInsertF3(SR, VR, depth+1)

        return node


    #Algorithm 3 from https://webserver2.tecgraf.puc-rio.br/~psantos/inf2602_2008_2/pesquisa/hierarchy/kdtree/On%20building%20fast%20kd-Trees%20for%20Ray%20Tracing%20and%20on%20doing%20that%20in%20O(N%20log%20N).pdf
    def _simpleInsertF3(self, spheres, voxel, depth):
        #axis = depth % self.k

        node = KDTreeNode(self.k)
        #node.splitPlane = axis
        node.depth = depth
        node.voxel = voxel
        if self.terminate(spheres, voxel, depth):
            node.spheres = spheres
            return node

        #node.b = self.findPlane(spheres, voxel)
        #newVox = voxel.divide(axis, node.b)
        data = self.N2SAHPartition(spheres, voxel)
        axis = data[0][0]
        wB   = data[0][1]

        node.b = voxel.iLerp(wB, axis)
        node.splitPlane = axis


        newVox = voxel.divideWorld(axis, wB)

        VL = newVox[0]
        VR = newVox[1]

        SL = data[1]
        SR = data[2]

        #print("%d, %d" % (len(SL), len(SR)))
        node.childFirst  = self._simpleInsertF3(SL, VL, depth+1)
        node.childSecond = self._simpleInsertF3(SR, VR, depth+1)

        return node

    #Algorithm 1 from https://webserver2.tecgraf.puc-rio.br/~psantos/inf2602_2008_2/pesquisa/hierarchy/kdtree/On%20building%20fast%20kd-Trees%20for%20Ray%20Tracing%20and%20on%20doing%20that%20in%20O(N%20log%20N).pdf
    def _simpleInsertF2(self, spheres, voxel, depth):
        axis = depth % self.k

        node = KDTreeNode(self.k)
        node.splitPlane = axis
        node.depth = depth
        node.voxel = voxel
        if self.terminate(spheres, voxel, depth):
            node.spheres = spheres
            return node

        node.b = self.findPlane(spheres, voxel)

        newVox = voxel.divide(axis, node.b)
        VL = newVox[0]
        VR = newVox[1]

        SL = [s for s in spheres if VL.collides(s)]#VL.collides(s)]
        SR = [s for s in spheres if VR.collides(s)]

        node.childFirst  = self._simpleInsertF2(SL, VL, depth+1)
        node.childSecond = self._simpleInsertF2(SR, VR, depth+1)

        return node

    def _simpleInsertF(self, spheres, depth):
        axis = depth % self.k

        node = KDTreeNode(self.k)
        node.splitPlane = axis
        node.depth = depth

        if (depth == self.maxRecurse) or len(spheres)==1:
            node.spheres = spheres;
            return node

        bounds = getSphereBounds(spheres) # This should be calculated once at the top, but tired and lazy

        #  min.? + (max.? - min.?)*b, in this implementation we are dividing accross the spatial median because we are trash
        worldSpaceB = bounds[0][axis] + (bounds[1][axis] - bounds[0][axis])*node.b

        spheresFirst  = []
        spheresSecond = []
        for s in spheres:
            if (s.getPosByAxis(axis)-s.r)>worldSpaceB:
                spheresFirst.append(s)
            elif (s.getPosByAxis(axis)+s.r)<worldSpaceB:
                spheresSecond.append(s)
            else:
                spheresFirst.append(s)
                spheresSecond.append(s)
                print("PLANE INTERSECTS SPHERE")


        node.childFirst  = self._simpleInsertF(spheresFirst, depth+1)
        node.childSecond = self._simpleInsertF(spheresSecond, depth+1)

        return node


    def _spewNodeF(self, node):
        depthSpace(node.depth)
        print("Entering Node (depth: %d, axis: %s) ((%4.2f, %4.2f, %4.2f)(%4.2f, %4.2f, %4.2f))" %
              (node.depth, ('x', 'y', 'z')[node.splitPlane],
               node.voxel.minV[0],  node.voxel.minV[1],  node.voxel.minV[2],
               node.voxel.maxV[0],  node.voxel.maxV[1],  node.voxel.maxV[2]))
        if(node.isLeaf()):
            for s in node.spheres:
                depthSpace(node.depth)
                print("- "+s.getInfoStr())

        else:
            self._spewNodeF(node.childFirst)
            self._spewNodeF(node.childSecond)

    def spewTree(self):
        print("Spewing k-d tree:")
        self._spewNodeF(self.root)

    def simpleInsert(self, spheres):
        #Using new Alg
        print("Starting k-d tree construction")
        start = time.time()
        self.root = self._simpleInsertF4(spheres, getSphereVoxel(spheres), 0);
        end = time.time()
        print("k-d Construction took: %f ms" % ((end - start)*1000))

def getSphereVoxel(spheres): #This doesn't have to be recalculated each time
    minmax = getSphereBounds(spheres)
    return Voxel(minmax[0], minmax[1])

def getSphereBounds(spheres): #This doesn't have to be recalculated each time
    maxV = [-10000.0,-10000.0,-10000.0]
    minV = [ 10000.0, 10000.0, 10000.0]

    for s in spheres:
        for i in range(3):
            if(s.pos[i]>maxV[i]):
                maxV[i] = s.pos[i]
            if(s.pos[i]<minV[i]):
                minV[i] = s.pos[i]
    return (minV,maxV)


def clip(v1, v2):
    B = copy.deepcopy(v1)
    for i in range(3):
        if(B.minV[i] < v2.minV[i]):
            B.minV[i] = v2.minV[i]
        if(B.maxV[i] > v2.maxV[i]):
            B.maxV[i] = v2.maxV[i]
    return B


def perfLambda(NL, NR, PL, PR):
    if (NL == 0 or NR == 0) and not (PL == 1 or PR == 1):
        #return 1.0
        return 0.8
    #if (PL == 1 or PR == 1):
        #return 10000.4 #SNEAK!
    return 1.0

def SA_VOXEL(V):
    diff = simdDiff(V.minV, V.maxV, 3)
    return (diff[0]*diff[1]*2 +
            diff[1]*diff[2]*2 +
            diff[0]*diff[2]*2)



def perfectSplits(sphere, voxel):
    sVox = Voxel([0,0,0], [0,0,0])
    sVox.initFromSphere(sphere)
    B = clip(sVox, voxel)

    P = []
    for k in range(3):
        P.append((k,   B.minV[k]))
        P.append((k, B.maxV[k]))
    return P

def classify(S, VL, VR, P):
    SL = []
    SR = []
    SP = []
    for s in S:
        if(s.collidesPlane(P[0], P[1])):
            SP.append(s)
        else:
            if(s.isLeftOfPlane(P[0], P[1])):
                SL.append(s)
            else:
                SR.append(s)
    return (SL, SR, SP)


def getSurfaceArea(spheres, axis, side):
    print ("implement surface area")
    return 0

class World:
    def __init__(self):
        self.spheres = []
        self.tree = None
    def fillWithRandSpheres(self, numSpheres):
        for i in range(numSpheres):
            self.spheres.append(Sphere(
                uniform(-10, 10),
                uniform(-10, 10),
                uniform(-10, 10),
                uniform(0, 1)))
    def spewSpheres(self):
        for s in self.spheres:
            print("Sphere: (%8.4f, %8.4f, %8.4f), radius: %7.4f" % (s.pos[0], s.pos[1], s.pos[2], s.r))




def depthSpace(depth):
    for i in range(depth):
        print('    ', end='')

def simdAdd(a, b, d): #It would be ideal if it was actually simd... but I don't really have time to look up how to do simd python instructions so this will do.
    retVar = [None]*d
    for i in range(d):
        retVar[i] = a[i]+b[i]
    return tuple(retVar)

def simdDiff(a, b, d): #It would be ideal if it was actually simd... but I don't really have time to look up how to do simd python instructions so this will do.
    retVar = [None]*d
    for i in range(d):
        retVar[i] = b[i]-a[i]
    return tuple(retVar)

def simdMoreThan(a, b, d): #It would be ideal if it was actually simd... but I don't really have time to look up how to do simd python instructions so this will do.
    for i in range(d):
        if a[i] < b[i]:
            return False
    return True

def simdAddScalar(a, b, d): #It would be ideal if it was actually simd... but I don't really have time to look up how to do simd python instructions so this will do.
    retVar = [None]*d
    for i in range(d):
        retVar[i] = a[i]+b
    return tuple(retVar)

def Main():
    world = World()
    world.fillWithRandSpheres(500)
    world.spewSpheres()
    world.tree = KDTree()
    world.tree.simpleInsert(world.spheres)
    world.tree.spewTree()

Main()

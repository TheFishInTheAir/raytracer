from random import uniform
print("This is just practice for implementing the k-d tree in c")

class Sphere:
    def __init__(self, x, y, z, size):
        self.pos = (x,y,z)
        self.r = size
    def getPosByAxis(self, dim):
        return self.pos[dim]
    def getInfoStr(self):
        return "Sphere: (%8.4f, %8.4f, %8.4f), radius: %7.4f" % (self.pos[0], self.pos[1], self.pos[2], self.r)
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

    def collides(self, sphere):
        if (simdMoreThan(simdAddScalar(sphere.pos, sphere.r, 3),  self.minV,  3) and
            simdMoreThan(self.maxV, simdAddScalar(sphere.pos, -sphere.r, 3),  3)):
            return True
        return False

    #def bToWorld(self, axis, b):
    #    return self.maxV[0][axis] + (self.minV[axis] - self.maxV[axis])*b

    def divide(self, axis, b):
        divMin = list(self.minV)
        divMax = list(self.maxV)
        divMin[axis] = self.maxV[axis] + (self.minV[axis] - self.maxV[axis])*b
        divMax[axis] = self.maxV[axis] + (self.minV[axis] - self.maxV[axis])*(1-b)
        return (Voxel(self.minV, divMax),
                Voxel(divMin, self.maxV))
class KDTree:
    def __init__(self):
        self.root = None #KDTreeNode
        self.k = 3
        self.maxRecurse = 10
        self.KT = 1.0 #TRAVERSAL    COST GENERIC DEFAULT
        self.KI = 1.5 #INTERSECTION COST GENERIC DEFAULT

    def perfLambda(self, NL, NR):
        if NL == 0 or NR == 0 :
            return 0.8
        return 1.0

    def SA_VOXEL(V):
        diff = simdDiff(V.minV, V.maxV, 3)
        return (diff[0]*diff[1]*2 +
                diff[1]*diff[2]*2 +
                diff[0]*diff[2]*2)

    def C(PL, PR, NL, NR):
        return perfLambda(NL,NR)*(self.KT + self.KI*(PL*NL + PR*NR))

    def SAH(axis, b, V, NL, NR, NP):
        newVox = V.divide(axis, b)
        VL = newVox[0]
        VR = newVox[1]
        PL = SA_VOXEL(VL)/SA_VOXEL(V)
        PR = SA_VOXEL(VR)/SA_VOXEL(V)
        CPL = C(PL, PR, NL+NP, NR)
        CPR = C(PL, PR, NL, NR+NP)
        if CPL < CPR:
            return (CPL, 1) #1 is LEFT
        else:
            return (CPR, 2) #2 is RIGHT

    def terminate(self, spheres, voxel, depth):
        if (depth==self.maxRecurse) or (len(spheres)<=1):
            return True
        #include other cases
        return False


    def findPlane(self, spheres, voxel):
        return 0.5

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
        self.root = self._simpleInsertF2(spheres, getSphereVoxel(spheres), 0);

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
                uniform(0, 0)))
    def spewSpheres(self):
        for s in self.spheres:
            print("Sphere: (%8.4f, %8.4f, %8.4f), radius: %7.4f" % (s.pos[0], s.pos[1], s.pos[2], s.r))

def depthSpace(depth):
    for i in range(depth):
        print('    ', end='')

def simdAdd(a, b, d):
    retVar = [None]*d
    for i in range(d):
        retVar[i] = a[i]+b[i]
    return tuple(retVar)

def simdDiff(a, b, d):
    retVar = [None]*d
    for i in range(d):
        retVar[i] = b[i]+(a[i]-b[i])
    return tuple(retVar)

def simdMoreThan(a, b, d):
    for i in range(d):
        if a[i] < b[i]:
            return False
    return True

def simdAddScalar(a, b, d):
    retVar = [None]*d
    for i in range(d):
        retVar[i] = a[i]+b
    return tuple(retVar)

def Main():
    world = World()
    world.fillWithRandSpheres(10)
    world.spewSpheres()
    world.tree = KDTree()
    world.tree.simpleInsert(world.spheres)
    world.tree.spewTree()

Main()

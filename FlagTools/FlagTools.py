#import scipy as sp
import numpy as np
import copy
from itertools import permutations
#import dask.array as da

""" We will identify a (complete) flag F of R^n as a matrix by choosing linearly
independent vectors b_1,b_2,...,b_n in R^n (with coordinates relative to
the standard basis) such that the ith subspace of F is spanned by
{b_1,...,b_i}. All the functions we implement here do not depend on this
choice. """

def tripleRatio(flags, triple):
    #Input sanity checks
    dim = np.shape(flags[0])[0]
    for flag in flags:
        assert np.shape(flag) == (dim,dim), "Input flags are not of the correct dimension."
    assert len(flags) == 3 and len(triple) == 3, "Inputs must be arrays of size 3."
    assert np.sum(triple) == dim, "Input triple does not sum to {}.".format(dim)
    for number in triple:
        assert 0 < number <= dim-2, "Numbers in the triple must be between 1 and {}.".format(dim-2)

    flagE=flags[0]
    flagF=flags[1]
    flagG=flags[2]
    a=triple[0]
    b=triple[1]
    c=triple[2]

    if c==1:
        numeratorOne = np.linalg.det(np.hstack([flagE[0:dim,0:a+1], flagF[0:dim,0:b]]))
        denominatorTwo= np.linalg.det(np.hstack([flagE[0:dim,0:a], flagF[0:dim,0:b+1]]))
    else:
        numeratorOne = np.linalg.det(np.hstack([np.hstack([flagE[0:dim,0:a+1], flagF[0:dim,0:b]]), flagG[0:dim,0:c-1]]))
        denominatorTwo= np.linalg.det(np.hstack([np.hstack([flagE[0:dim,0:a], flagF[0:dim,0:b+1]]), flagG[0:dim,0:c-1]]))
    if b==1:
        numeratorTwo = np.linalg.det(np.hstack([flagE[0:dim,0:a], flagG[0:dim,0:c+1]]))
        denominatorThree= np.linalg.det(np.hstack([flagE[0:dim,0:a+1], flagG[0:dim,0:c]]))
    else:
        numeratorTwo = np.linalg.det(np.hstack([np.hstack([flagE[0:dim,0:a], flagF[0:dim,0:b-1]]), flagG[0:dim,0:c+1]]))
        denominatorThree= np.linalg.det(np.hstack([np.hstack([flagE[0:dim,0:a+1], flagF[0:dim,0:b-1]]), flagG[0:dim,0:c]]))
    if a==1:
        numeratorThree = np.linalg.det(np.hstack([flagF[0:dim,0:b+1], flagG[0:dim,0:c]]))
        denominatorOne= np.linalg.det(np.hstack([flagF[0:dim,0:b], flagG[0:dim,0:c+1]]))
    else:
        numeratorThree = np.linalg.det(np.hstack([np.hstack([flagE[0:dim,0:a-1], flagF[0:dim,0:b+1]]), flagG[0:dim,0:c]]))
        denominatorOne= np.linalg.det(np.hstack([np.hstack([flagE[0:dim,0:a-1], flagF[0:dim,0:b]]), flagG[0:dim,0:c+1]]))
    return (numeratorOne*numeratorTwo*numeratorThree)/(denominatorOne*denominatorTwo*denominatorThree)

""" Returns a list of all triple ratios of a given triple of flags. """

def tripleRatios(flags):
    dim = np.shape(flags[0])[0]
    tripleRatios = []
    for i in range(1,dim-1):
        for j in range(1,dim-i):
            k = dim-i-j
            tripleRatios.append(tripleRatio(flags,(i,j,k)))
    return tripleRatios

""" Returns a list of all triple ratios of a given triple of flags
paired with the corresponding triple (i,j,k) as tuples. """

def tripleRatiosPairedWithTriples(flags):
    dim = np.shape(flags[0])[0]
    tripleRatios = []
    for i in range(1,dim-1):
        for j in range(1,dim-i):
            k = dim-i-j
            tripleRatios.append((tripleRatio(flags,(i,j,k)), (i,j,k)))
    return tripleRatios

""" Returns a list of matrices E_s_1, ..., E_s_{dim-1} that we will need for
the following function, getFlagRepresentative. The notation follows the
paper of Subhadip Dey."""
def generateEMatrices(dim):
    EMatrices = []
    for i in range(1,dim):
        E_s_i = np.zeros((dim,dim))
        E_s_i[i-1][i] = 1
        EMatrices.append(E_s_i)
    return(EMatrices)

def getLusztigCoefficient(i,j,componentMatrix):
    minusOneForArrayNotation = -1
    epsilon_ij=componentMatrix[j+minusOneForArrayNotation][j+(i-1)+minusOneForArrayNotation]
    if epsilon_ij == 0:
        return(1)
    else:
        return(-1)

def getUnipotent(componentMatrix, dim,EMatrices):
    lusztigMatrix = np.identity(dim)
    minusOneForArrayNotation = -1
    for line in range(1,dim):  #line refers to the equation line in (6) of p.10
        for i in range(1, dim-line+1):
            t = getLusztigCoefficient(i,dim-line-i+1,componentMatrix)
            lusztigMatrix = np.matmul(lusztigMatrix, np.add(np.identity(dim),t*EMatrices[i+minusOneForArrayNotation]))
    return lusztigMatrix


"""" Returns a representative flag contained in the component of the standard
double Schubert cell corresponding to the given component matrix (as in
Shapiro-Shapiro-Vainshtein). The standard double Schubert cell is
parametrized as a subset of the group of unipotent upper triangular matrices
since it acts simply transitively on the space of flags antipodal to FMinus,
where FMinus is the standard ascending flag. Thus any flag in the standard
double Schubert cell can be written as u.FPlus, where FPlus is the standard
descending flag and u is a unique unipotent upper triangular matrix. Since
experimental data sets are typically large and dimensions vary according to
the dimension of the ambient Euclidean space, we ask the user to
provide 'FPlus' and 'dim' in the argument to reduce computations.

The correspondence between the given component matrix and u that we
implement below is described in p.10-11 of Subhadip Dey's paper, "On Borel
Anosov Subgroups of SL(d,R)."

"""
def getFlagRepresentative(componentMatrix, FPlus, dim, EMatrices):
    lusztigMatrix = np.identity(dim)
    minusOneForArrayNotation = -1

    for line in range(1,dim):  #line refers to the equation line in (6) of p.10
        for i in range(1, dim-line+1):
            t = getLusztigCoefficient(i,dim-line-i+1,componentMatrix)
            lusztigMatrix = np.matmul(lusztigMatrix, np.add(np.identity(dim),t*EMatrices[i+minusOneForArrayNotation]))
    return(np.matmul(getUnipotent(componentMatrix,dim,EMatrices), FPlus))

""" Helper function for the functioxwn 'generateComponentMatrices.' """    
def generatorFunction(componentMatrix, componentMatrices, i, j, dimMinusOne):
        if i == dimMinusOne:
            componentMatrices.append(copy.deepcopy(componentMatrix))
            return()
        if j == dimMinusOne:
            componentMatrixCopy = copy.deepcopy(componentMatrix)
            generatorFunction(componentMatrixCopy, componentMatrices,i+1,i+1,dimMinusOne)
            return()
        else:
            componentMatrixCopy = copy.deepcopy(componentMatrix)
            generatorFunction(componentMatrixCopy, componentMatrices, i, j+1, dimMinusOne)
            componentMatrixCopy = copy.deepcopy(componentMatrix)
            componentMatrixCopy[i][j]=1
            generatorFunction(componentMatrixCopy, componentMatrices, i, j+1, dimMinusOne)
            return()
                                      

""" Returns a list of all unipotent upper triangular (dim-1)x(dim-1)
matrices with entries in Z/2Z as in Shapiro-Shapiro-Vainshtein. Each matrix
determines a component in the double Schubert cell C(F_-,F_+). The list is
not in one-to-one correspondence with the components since many matrices will
determine the same component.

WARNING: This process will usually be killed when dim>7. Instead, use
the modified function following this one which instead writes the component
matrices to a SQLite database.
"""
def generateComponentMatrices(dim):
    dimMinusOne=dim-1
    componentMatrices = []
    generatorFunction(np.zeros((dimMinusOne,dimMinusOne)),componentMatrices,0,0, dimMinusOne)
    return(componentMatrices)



""" Given a flag F antipodal to both FMinus and FPlus, returns the transversality
vector of the middle triple (FMinus, F, FPlus).
"""
def getTransversalityVector(triple):
    dim = np.shape(triple[0])[0]
    transversalityVector = []
    for k in range(0,dim):
        transversalityVector.append(np.linalg.det(np.hstack([triple[1][0:dim,0:k], triple[2][0:dim, 0:dim-k]])))
    return transversalityVector

def getNormalizedTransversalityVector(triple):
    transversalityVector = getTransversalityVector(triple)
    normalizedTransversalityVector = []
    for entry in transversalityVector:
        normalizedTransversalityVector.append(np.sign(entry))
    return normalizedTransversalityVector

def generatorFunctionForNormalizedTransversalityVectors(componentMatrix,normalizedTransversalityVectors,FMinus,FPlus,i,j,dim,EMatrices):
    dimMinusOne = dim-1

    if i == dimMinusOne:
        F = getFlagRepresentative(componentMatrix,FPlus,dim,EMatrices)
        triple = (FMinus,F,FPlus)
        normalizedTransversalityVector = getNormalizedTransversalityVector(triple)
        if normalizedTransversalityVector not in normalizedTransversalityVectors:
                normalizedTransversalityVectors.append(normalizedTransversalityVector)
        return()
    if j == dimMinusOne:
        F = getFlagRepresentative(componentMatrix,FPlus,dim,EMatrices)
        triple = (FMinus,F,FPlus)
        normalizedTransversalityVector = getNormalizedTransversalityVector(triple)
        if normalizedTransversalityVector not in normalizedTransversalityVectors:
            normalizedTransversalityVectors.append(normalizedTransversalityVector)
        componentMatrixCopy = copy.deepcopy(componentMatrix)
        generatorFunctionForNormalizedTransversalityVectors(componentMatrixCopy,normalizedTransversalityVectors,FMinus,FPlus,i+1,i+1,dim,EMatrices)
        return()
    else:
        componentMatrixCopy = copy.deepcopy(componentMatrix)
        generatorFunctionForNormalizedTransversalityVectors(componentMatrixCopy,normalizedTransversalityVectors,FMinus,FPlus,i,j+1,dim,EMatrices)
        componentMatrixCopy = copy.deepcopy(componentMatrix)
        componentMatrixCopy[i][j]=1
        generatorFunctionForNormalizedTransversalityVectors(componentMatrixCopy,normalizedTransversalityVectors,FMinus,FPlus,i,j+1,dim,EMatrices)
        return()
    
def generateAllNormalizedTransversalityVectors(dim):
    normalizedTransversalityVectors = []
    EMatrices = generateEMatrices(dim)
    generatorFunctionForNormalizedTransversalityVectors(np.zeros((dim-1,dim-1)),normalizedTransversalityVectors,np.identity(dim), np.flip(np.identity(dim),0), 0,0, dim, EMatrices)
    return normalizedTransversalityVectors

def twoThreeRatio(triple,k):
    flagF = triple[0]
    flagG = triple[1]
    flagH = triple[2]
    dim = np.shape(flagF)[0]
    numeratorOne = np.linalg.det(np.hstack([flagF[0:dim,0:k], flagG[0:dim,0:dim-k]]))
    denominatorOne = np.linalg.det(np.hstack([flagF[0:dim,0:k], flagH[0:dim,0:dim-k]]))
    numeratorTwo = np.linalg.det(np.hstack([flagG[0:dim,0:k], flagH[0:dim,0:dim-k]]))
    denominatorTwo = np.linalg.det(np.hstack([flagG[0:dim,0:k], flagF[0:dim,0:dim-k]]))
    numeratorThree = np.linalg.det(np.hstack([flagH[0:dim,0:k], flagF[0:dim,0:dim-k]]))
    denominatorThree = np.linalg.det(np.hstack([flagH[0:dim,0:k], flagG[0:dim,0:dim-k]]))
    return (numeratorOne*numeratorTwo*numeratorThree)/(denominatorOne*denominatorTwo*denominatorThree)

def getTwoThreeVector(triple):
    dim = np.shape(triple[0])[0]
    twoThreeVector = []
    for k in range(0,dim-1):
        twoThreeVector.append(twoThreeRatio(triple,k))
    return twoThreeRatio

def getNormalizedTwoThreeVector(triple):
    dim = np.shape(triple[0])[0]
    normalizedTwoThreeVector = []
    for k in range(0,dim-1):
        normalizedTwoThreeVector.append(np.sign(twoThreeRatio(triple,k)))
    return normalizedTwoThreeVector

#==============================================================================#
# Daniel Nichol - 08/06/2015.
#
# Implements the Markov Chain model for SSWM dyanmics evolution on fitness 
# landscapes presented in: 
#
# Exploiting evolutionary non-commutativity to prevent the emergence of bacterial antibiotic resistance
# Daniel Nichol , Peter Jeavons , Alexander G Fletcher , Robert A Bonomo , Philip K Maini , 
# Jerome L Paul , Robert A Gatneby , Alexander RA Anderson , Jacob G Scott
# bioRxiv doi: http://dx.doi.org/10.1101/007542
#==============================================================================#
import numpy as np
from math import log
from copy import deepcopy
from random import random
from random import sample

#==============================================================================#
# Helper functions
#==============================================================================#

# Computes the hamming distance between two genotypes.
def hammingDistance(s1, s2):
    assert len(s1) == len(s2)
    return sum(ch1 != ch2 for ch1, ch2 in zip(s1, s2))

# Takes a genotype and converts it to an integer for use indexing the fitness landscape list 
def convertGenotypeToInt(genotype):
	out = 0
	for bit in genotype:
		out = (out << 1) | bit
	return out

# Converts an integer to a genotype by taking the binary value and padding to the left by 0s		
def convertIntToGenotype(anInt, pad):
	offset = 2**pad
	return [int(x) for x in bin(offset+anInt)[3:]]	

# Function which returns all genotypes at Hamming distance 1 from a specified genotype
def getOneStepNeighbours(genotype):
	neighbours = []
	for x in range(0, len(genotype)):
		temp = deepcopy(genotype)
		temp[x] = (genotype[x]+1) %2 #There is some inefficiency here.
		neighbours.append(temp)
	return neighbours

#==============================================================================#

#==============================================================================#
# Defining a fitness landscape class
#==============================================================================#
class FitnessLandscape:
	def __init__(self, landscapeValues, name=None):
		self.landscape = landscapeValues
		self.name = name

	def getFitness(self, genotype):
		fitness = self.landscape[convertGenotypeToInt(genotype)]
		return fitness

	def genotypeLength(self):
		return int(log(len(self.landscape), 2))

	def numGenotypes(self):
		return len(self.landscape)

	def isPeak(self, g):
		peak = True
		for h in getOneStepNeighbours(g):
			if self.getFitness(g) < self.getFitness(h):
				peak = False
				break
		return peak

	def getPeaks(self):
		peaks = []

		allGenotypes = []
		N =self.genotypeLength()
		for x in range(0, 2**N):
			allGenotypes.append(convertIntToGenotype(x, self.genotypeLength()))

		for g in allGenotypes:
			if self.isPeak(g):
				peaks.append(g)
		
		return peaks

	def getGlobalPeak(self):
		return convertIntToGenotype(np.argmax(self.landscape), self.genotypeLength())

	def getLowestFitnessPeak(self):
		# Finds the peaks of the landscape
		peak_genotypes = self.getPeaks()
		lowest_peak_genotype = peak_genotypes[np.argmin([self.getFitness(g) for g in peak_genotypes])]
		return lowest_peak_genotype

#==============================================================================#

#==============================================================================#
# Building the Markov Chain 
#==============================================================================#

################################################################################
# Given two genotypes and a landscape, computes the transition probability		
# Pr(g1->g2) in the markov chain transition matrix (eqns 2 and 3 of the paper)
################################################################################
def transProbSSWM(g1, g2, landscape, r=0):
	#If the genotypes are more than one mutation apart, then 0
	if hammingDistance(g1,g2) > 1:
		return 0

	#Else compute Pr(g1->g2) from eqn 2 of the paper
	elif hammingDistance(g1,g2) == 1:
		if landscape.getFitness(g1) >= landscape.getFitness(g2):
			return 0
		else:
			num = (landscape.getFitness(g2) - landscape.getFitness(g1))**r
			den = 0.
			for genotype in getOneStepNeighbours(g1):
				fitDiff = (landscape.getFitness(genotype) - landscape.getFitness(g1))
				if fitDiff > 0:
					den += fitDiff**r
			return num / den

	#Finally add in those Pr(g1->g1)=1 for g1 a local optima (eqn 3 of the paper)
	else:
		isPeak = landscape.isPeak(g1)
		return int(isPeak)

################################################################################
# Builds the transition matrix for a given landscape
################################################################################
def buildTransitionMatrix(landscape, r=0):
	genomeLen = landscape.genotypeLength()
	matList = [[transProbSSWM(convertIntToGenotype(i,genomeLen), convertIntToGenotype(j, genomeLen), landscape, r) for j in range(0, 2**genomeLen)] for i in range(0, 2**genomeLen)]
	return np.matrix(matList)

################################################################################
# Given a stochastic matrix P, finds the limit matrix 
################################################################################
def limitMatrix(P):
	Q = np.identity(len(P))
	while not np.array_equal(Q,P):
		Q = deepcopy(P)
		P = P*P #Square P until it no longer changes.
	return P

#===========================================================================================#

#===========================================================================================#
# An implementation of Kauffman's NK models for generating tunably rugged landscapes
#===========================================================================================#

# For each possible value of (x_i;x_{i_1}, ... , x_{i_K}) we have a random number sampled from [0,1)
def geneWeights(K,N):
	return [[random() for x in range(2**(K+1))] for y in range(N)]

# Given a genotype length (N) and the number of alleles (K) this function randomly choses K alleles 
# from positions {1,...N}\{i} which interact epistatically with the ith position	
def buildInfringersTable(N,K):
	return [sample(range(i)+range(i+1,N),K) for i in range(N)]

# Builds a tuple for look up in the fitness table from the infringers list
def buildTuple(allele, i, infringers):
	tp = [allele[i]]
	for j in infringers:
		tp += [allele[j]]
	return tp

# Given an allele computes the fitness by building a tuple of revelant infringers
# and looks them up in the gene weights table
def alleleFitness(allele, gw, infrs, N, K):
	s = 0.
	for i in range(N):
		index = buildTuple(allele,i,infrs[i])
		s += gw[i][convertGenotypeToInt(index)]
	s = s / N
	return s

#Generates an N-K landscape from Kauffman's method.
def generateNKLandscape(N,K):
	gw = geneWeights(K,N)
	infrs = buildInfringersTable(N,K)
	landscape = [alleleFitness(convertIntToGenotype(a,N), gw, infrs, N, K) for a in range(2**N)]
	return FitnessLandscape(landscape)
	
#==========================================================================================#
#==========================================================================================#

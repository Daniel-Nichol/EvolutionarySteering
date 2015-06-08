#==============================================================================#
# Daniel Nichol - 08/06/2015.
#
# Uses the fitness landscapes for E. Coli reported by Mira et. al. in:
#
# Rational Design of Antibiotic Treatment Plans: A Treatment Strategy for Managing Evolution and Reversing Resistance
# doi: 10.1371/journal.pone.0122283
#
# To generate the results presented in our manuscript:
# "Exploiting evolutionary non-commutativity to prevent the emergence of bacterial antibiotic resistance"
#==============================================================================#
import markovModel
from markovModel import FitnessLandscape
from markovModel import convertGenotypeToInt
from markovModel import convertIntToGenotype
import numpy as np

#==============================================================================#
# The landscapes reported by Mira et. al.
#==============================================================================#
AMP = FitnessLandscape([1.851, 2.082, 1.948, 2.434, 2.024, 2.198, 2.033, 0.034, 1.57, 2.165, 0.051, 0.083, 2.186, 2.322, 0.088, 2.821], "Ampicillin")
AM  = FitnessLandscape([1.778, 1.782, 2.042, 1.752, 1.448, 1.544, 1.184, 0.063, 1.72, 2.008, 1.799, 2.005, 1.557, 2.247, 1.768, 2.047], "Amoxicillin")
CEC	= FitnessLandscape([2.258, 1.996, 2.151, 2.648, 2.396, 1.846, 2.23, 0.214, 0.234, 0.172, 2.242, 0.093, 2.15, 0.095, 2.64, 0.516], "Cefaclor")
CTX = FitnessLandscape([0.16, 0.085, 1.936, 2.348, 1.653, 0.138, 2.295, 2.269, 0.185, 0.14, 1.969, 0.203, 0.225, 0.092, 0.119, 2.412], "Cefotaxime")
ZOX = FitnessLandscape([0.993, 0.805, 2.069, 2.683, 1.698, 2.01, 2.138, 2.688, 1.106, 1.171, 1.894, 0.681, 1.116, 1.105, 1.103, 2.591], "Ceftizoxime")
CXM = FitnessLandscape([1.748, 1.7, 2.07, 1.938, 2.94, 2.173, 2.918, 3.272, 0.423, 1.578, 1.911, 2.754, 2.024, 1.678, 1.591, 2.923], "Cefuroxime")
CRO = FitnessLandscape([1.092, 0.287, 2.554, 3.042, 2.88, 0.656, 2.732, 0.436, 0.83, 0.54, 3.173, 1.153, 1.407, 0.751, 2.74, 3.227], "Ceftriaxone")
AMC = FitnessLandscape([1.435, 1.573, 1.061, 1.457, 1.672, 1.625, 0.073, 0.068, 1.417, 1.351, 1.538, 1.59, 1.377, 1.914, 1.307, 1.728], "Amoxicillin + Clav")
CAZ = FitnessLandscape([2.134, 2.656, 2.618, 2.688, 2.042, 2.756, 2.924, 0.251, 0.288, 0.576, 1.604, 1.378, 2.63, 2.677, 2.893, 2.563], "Ceftazidime")
CTT = FitnessLandscape([2.125, 1.922, 2.804, 0.588, 3.291, 2.888, 3.082, 3.508, 3.238, 2.966, 2.883, 0.89, 0.546, 3.181, 3.193, 2.543], "Cefotetan")
SAM = FitnessLandscape([1.879, 2.533, 0.133, 0.094, 2.456, 2.437, 0.083, 0.094, 2.198, 2.57, 2.308, 2.886, 2.504, 3.002, 2.528, 3.453], "Ampicillin +Sulbactam")
CPR = FitnessLandscape([1.743, 1.662, 1.763, 1.785, 2.018, 2.05, 2.042, 0.218, 1.553, 0.256, 0.165, 0.221, 0.223, 0.239, 1.811, 0.288], "Cefprozil")
CPD = FitnessLandscape([0.595, 0.245, 2.604, 3.043, 1.761, 1.471, 2.91, 3.096, 0.432, 0.388, 2.651, 1.103, 0.638, 0.986, 0.963, 3.268], "Cefpodoxime")
TZP = FitnessLandscape([2.679, 2.906, 2.427, 0.141, 3.038, 3.309, 2.528, 0.143, 2.709, 2.5, 0.172, 0.093, 2.453, 2.739, 0.609, 0.171], "Piperacillin + Tazobactam")
FEP = FitnessLandscape([2.59, 2.572, 2.393, 2.832, 2.44, 2.808, 2.652, 0.611, 2.067, 2.446, 2.957, 2.633, 2.735, 2.863, 2.796, 3.203], "Cefepime")

landscapes = [AMP, AM, CEC, CTX, ZOX, CXM, CRO, AMC, CAZ, CTT, SAM, CPR, CPD, TZP, FEP]
#==============================================================================#
# The limits of the Markov chain matrices corresponding to these landscapes (r=0)
#==============================================================================#
L_AMP = markovModel.limitMatrix(markovModel.buildTransitionMatrix(AMP))
L_AM  = markovModel.limitMatrix(markovModel.buildTransitionMatrix(AM))
L_CEC = markovModel.limitMatrix(markovModel.buildTransitionMatrix(CEC))
L_CTX = markovModel.limitMatrix(markovModel.buildTransitionMatrix(CTX))
L_ZOX = markovModel.limitMatrix(markovModel.buildTransitionMatrix(ZOX))
L_CXM = markovModel.limitMatrix(markovModel.buildTransitionMatrix(CXM))
L_CRO = markovModel.limitMatrix(markovModel.buildTransitionMatrix(CRO))
L_AMC = markovModel.limitMatrix(markovModel.buildTransitionMatrix(AMC))
L_CAZ = markovModel.limitMatrix(markovModel.buildTransitionMatrix(CAZ))
L_CTT = markovModel.limitMatrix(markovModel.buildTransitionMatrix(CTT))
L_SAM = markovModel.limitMatrix(markovModel.buildTransitionMatrix(SAM))
L_CPR = markovModel.limitMatrix(markovModel.buildTransitionMatrix(CPR))
L_CPD = markovModel.limitMatrix(markovModel.buildTransitionMatrix(CPD))
L_TZP = markovModel.limitMatrix(markovModel.buildTransitionMatrix(TZP))
L_FEP = markovModel.limitMatrix(markovModel.buildTransitionMatrix(FEP))

limit_matrices = [L_AMP, L_AM, L_CEC, L_CTX, L_ZOX, L_CXM, L_CRO, L_AMC, L_CAZ, L_CTT, L_SAM, L_CPR, L_CPD, L_TZP, L_FEP]

#==============================================================================#
# The initial population vector. Each genotype is considered equally likely to 
# constitute the initial genotype.
#==============================================================================#
init_pop = np.array([1./2**4 for i in range(16)])

#==============================================================================#
# Functions which generate the results of table 1 in the manuscript
#==============================================================================#

# Determines for each drug the probability of ending at the highest fitness peak
# genotype when starting from the initial distribution init_pop.
def probHighestPeak(init_pop):
	highest_pgs_probs = []
	for t in range(len(limit_matrices)):
		dist = np.array(init_pop * limit_matrices[t])[0]
		global_peak_index = convertGenotypeToInt(landscapes[t].getGlobalPeak())
		highest_pgs_probs.append(dist[global_peak_index])
		print landscapes[t].name+":", highest_pgs_probs[t]

	return highest_pgs_probs

# Determines for each drug the single steering drug which minimizes the probability
# that evolution proceeds to the highest fitness peak.
def highPeakBestSingle(init_pop):
	best_steerers = []
	for t in range(len(limit_matrices)):
		global_peak_index = convertGenotypeToInt(landscapes[t].getGlobalPeak())
		best_prob = 1.0
		best_steerer = -1
		for s in range(len(limit_matrices)):
			dist = np.array(init_pop * limit_matrices[s] * limit_matrices[t])[0]
			if dist[global_peak_index] < best_prob:
				best_prob = dist[global_peak_index]
				best_steerer = s
		best_steerers.append((landscapes[s], best_prob))

		print landscapes[t].name+":", landscapes[best_steerer].name, best_prob
	return best_steerers

# Determines for each drug the ordered pair of steering drugs which minimizes the probability
# that evolution proceeds to the highest fitness peak.
def highPeakBestDouble(init_pop):
	best_steerers = []
	for t in range(len(limit_matrices)):
		global_peak_index = convertGenotypeToInt(landscapes[t].getGlobalPeak())
		best_prob = 1.0
		best_s1 = -1
		best_s2 = -1
		for s1 in range(len(limit_matrices)):
			for s2 in range(len(limit_matrices)):
				dist = np.array(init_pop * limit_matrices[s1] * limit_matrices[s2] * limit_matrices[t])[0]
				if dist[global_peak_index] < best_prob:
					best_prob = dist[global_peak_index]
					best_s1 = s1
					best_s2 = s2

		print landscapes[t].name+":", landscapes[best_s1].name+" ---> "+landscapes[best_s2].name+", ", best_prob
		best_steerers.append((landscapes[best_s2], landscapes[best_s2], best_prob))
	return best_steerers

# Determines for each drug the ordered triple of steering drugs which minimizes the probability
# that evolution proceeds to the highest fitness peak.
def highPeakBestTriple(init_pop):
	best_steerers = []
	for t in range(len(limit_matrices)):
		global_peak_index = convertGenotypeToInt(landscapes[t].getGlobalPeak())
		best_prob = 1.0
		best_s1 = -1
		best_s2 = -1
		best_s3 = -1
		for s1 in range(len(limit_matrices)):
			for s2 in range(len(limit_matrices)):
				for s3 in range(len(limit_matrices)):
					dist = np.array(init_pop * limit_matrices[s1] * limit_matrices[s2] * limit_matrices[s3] * limit_matrices[t])[0]
					if dist[global_peak_index] < best_prob:
						best_prob = dist[global_peak_index]
						best_s1 = s1
						best_s2 = s2
						best_s3 = s3


		print landscapes[t].name+":", landscapes[best_s1].name+" ---> "+landscapes[best_s2].name+" ---> "+landscapes[best_s3].name+", ", best_prob
		best_steerers.append((landscapes[best_s1], landscapes[best_s2], landscapes[best_s3], best_prob))

	return best_steerers

#==============================================================================#
# Functions which generate the results of table 2 in the manuscript
#==============================================================================#

# Determines for each drug the probability of ending at the lowest fitness peak
# genotype when starting from the initial distribution init_pop.
def probLowestPeak(init_pop):
	lowest_pgs_probs = []
	for t in range(len(limit_matrices)):
		dist = np.array(init_pop * limit_matrices[t])[0]
		lpg = landscapes[t].getLowestFitnessPeak()
		lowest_pgs_probs.append(dist[convertGenotypeToInt(lpg)])		
		print landscapes[t].name+":", lowest_pgs_probs[t]

	return lowest_pgs_probs

# Determines for each drug the single steering drug which maximizes the probability
# that evolution proceeds to the lowest fitness peak.
def lowestPeakBestSingle(init_pop):
	best_steerers = [] 
	for t in range(len(limit_matrices)):
		lowest_peak_index = convertGenotypeToInt(landscapes[t].getLowestFitnessPeak())

		best_steerer = -1
		best_prob = 0.0

		#For each possible steerer
		for s in range(len(limit_matrices)):
			dist = np.array(init_pop * limit_matrices[s] * limit_matrices[t])[0]
			if dist[lowest_peak_index] > best_prob:
				best_prob = dist[lowest_peak_index]
				best_steerer = s
		best_steerers.append((landscapes[best_steerer], best_prob))

		print landscapes[t].name+":", best_steerers[t][0].name, best_steerers[t][1]

	return best_steerers

# Determines for each drug the ordered pair of steering drugs which maximizes the probability
# that evolution proceeds to the lowest fitness peak.
def lowestPeakBestDouble(init_pop):
	best_steerers = [] 
	for t in range(len(limit_matrices)):
		lowest_peak_index = convertGenotypeToInt(landscapes[t].getLowestFitnessPeak())

		best_s1 = -1
		best_s2 = -1
		best_prob = 0.0
		for s1 in range(len(limit_matrices)):
			for s2 in range(len(limit_matrices)):
				dist = np.array(init_pop * limit_matrices[s1] * limit_matrices[s2] * limit_matrices[t])[0]
				if dist[lowest_peak_index] > best_prob:
					best_prob = dist[lowest_peak_index]
					best_s1 = s1
					best_s2 = s2
		best_steerers.append((landscapes[best_s1], landscapes[best_s2], best_prob))
		print landscapes[t].name+":  ", landscapes[best_s1].name+" ---> "+landscapes[best_s2].name+", ", best_prob

	return best_steerers

# Determines for each drug the ordered triple of steering drugs which maximizes the probability
# that evolution proceeds to the lowest fitness peak.
def lowestPeakBestTriple(init_pop):
	best_steerers = []
	for t in range(len(limit_matrices)):
		lowest_peak_index = convertGenotypeToInt(landscapes[t].getLowestFitnessPeak())

		best_s1 = -1
		best_s2 = -1
		best_s3 = -1
		best_prob = 0.0
		for s1 in range(len(limit_matrices)):
			for s2 in range(len(limit_matrices)):
				for s3 in range(len(limit_matrices)):
					dist = np.array(init_pop * limit_matrices[s1] * limit_matrices[s2] * limit_matrices[s3] * limit_matrices[t])[0]
					if dist[lowest_peak_index] > best_prob:
						best_prob = dist[lowest_peak_index]
						best_s1 = s1
						best_s2 = s2
						best_s3 = s3
		best_steerers.append((landscapes[best_s1], landscapes[best_s2], landscapes[best_s3], best_prob))
		print landscapes[t].name+":  ", landscapes[best_s1].name+" ---> "+landscapes[best_s2].name+" ---> "+landscapes[best_s3].name+", ", best_prob

	return best_steerers

#==============================================================================#
# Functions which generate the results of table 3 in the manuscript
#==============================================================================#

# Determines the percentage of single steering drugs that decrease and increase the
# probability of evolution to the highest peak of the landscape.
def singleSteererPercentages(init_pop, allowSAM = True):

	steering_stats = []

	overall_better = 0
	overall_worse = 0
	overall_same = 0

	for t in range(len(limit_matrices)):
		if t!=10 or allowSAM:

			better = 0
			worse = 0
			same = 0

			peak_genotype = landscapes[t].getGlobalPeak()
			peak_genotype_index = convertGenotypeToInt(peak_genotype)
			peak_fitness = landscapes[t].getFitness(peak_genotype)

			for s in range(len(limit_matrices)): 
				steered_pop = (init_pop * limit_matrices[s] * limit_matrices[t]).tolist()[0]
				straight_pop = (init_pop * limit_matrices[t]).tolist()[0]

				#Floating point arithematic is nasty, check for what is essential equality
				if np.abs(steered_pop[peak_genotype_index] - straight_pop[peak_genotype_index]) < 10**-15:
					same+=1
				elif steered_pop[peak_genotype_index] > straight_pop[peak_genotype_index]:
					worse+=1
				elif steered_pop[peak_genotype_index] < straight_pop[peak_genotype_index]:
					better+=1

			steering_stats.append((better,worse,same))


			pc_better = 100*float(better) / (better+worse+same)
			pc_worse  = 100*float(worse) / (better+worse+same)
			print landscapes[t].name+", better:", str(better)+" ("+str(pc_better)+"%)", ", worse:", str(worse)+" ("+str(pc_worse)+"%)"

			overall_better+=better
			overall_worse+=worse
			overall_same+=same
			pc_ov_better = 100*float(overall_better) / (overall_better+overall_worse+overall_same)
			pc_ov_worse = 100*float(overall_worse) / (overall_better+overall_worse+overall_same)



	print "Overall, better:", str(overall_better)+" ("+str(pc_ov_better)+"%)", ", worse:", str(overall_worse)+" ("+str(pc_ov_worse)+"%)"
	return steering_stats


# Determines the percentage of ordered pairs of steering drugs that decrease and increase the
# probability of evolution to the highest peak of the landscape.
def doubleSteererPercentages(init_pop, allowSAM = True):

	steering_stats = []

	overall_better = 0
	overall_worse = 0
	overall_same = 0

	for t in range(len(limit_matrices)):
		if t!=10 or allowSAM:

			better = 0
			worse = 0
			same = 0

			peak_genotype = landscapes[t].getGlobalPeak()
			peak_genotype_index = convertGenotypeToInt(peak_genotype)
			peak_fitness = landscapes[t].getFitness(peak_genotype)

			for k1 in range(len(limit_matrices)):
				for k2 in range(len(limit_matrices)): 
					steered_pop = (init_pop * limit_matrices[k1] * limit_matrices[k2] * limit_matrices[t]).tolist()[0]
					straight_pop = (init_pop * limit_matrices[t]).tolist()[0]

				
					#Floating point arithematic is nasty, check for what is essential equality
					if np.abs(steered_pop[peak_genotype_index] - straight_pop[peak_genotype_index]) < 10**-15:
						same+=1
					elif steered_pop[peak_genotype_index] > straight_pop[peak_genotype_index]:
						worse+=1
					elif steered_pop[peak_genotype_index] < straight_pop[peak_genotype_index]:
						better+=1

			steering_stats.append((better,worse,same))


			pc_better = 100*float(better) / (better+worse+same)
			pc_worse  = 100*float(worse) / (better+worse+same)
			print landscapes[t].name+", better:", str(better)+" ("+str(pc_better)+"%)", ", worse:", str(worse)+" ("+str(pc_worse)+"%)"

			overall_better+=better
			overall_worse+=worse
			overall_same+=same
			pc_ov_better = 100*float(overall_better) / (overall_better+overall_worse+overall_same)
			pc_ov_worse = 100*float(overall_worse) / (overall_better+overall_worse+overall_same)



	print "Overall, better:", str(overall_better)+" ("+str(pc_ov_better)+"%)", ", worse:", str(overall_worse)+" ("+str(pc_ov_worse)+"%)"
	return steering_stats


# Determines the percentage of ordered triples of steering drugs that decrease and increase the
# probability of evolution to the highest peak of the landscape.
def tripleSteererPercentages(init_pop, allowSAM = True):

	steering_stats = []

	overall_better = 0
	overall_worse = 0
	overall_same = 0

	for t in range(len(limit_matrices)):
		if t!=10 or allowSAM:

			better = 0
			worse = 0
			same = 0

			peak_genotype = landscapes[t].getGlobalPeak()
			peak_genotype_index = convertGenotypeToInt(peak_genotype)
			peak_fitness = landscapes[t].getFitness(peak_genotype)

			for k1 in range(len(limit_matrices)):
				for k2 in range(len(limit_matrices)): 
					for k3 in range(len(limit_matrices)):

						steered_pop = (init_pop * limit_matrices[k1] * limit_matrices[k2] * limit_matrices[k3] * limit_matrices[t]).tolist()[0]
						straight_pop = (init_pop * limit_matrices[t]).tolist()[0]

						#Floating point arithematic is nasty, check for what is essential equality
						if np.abs(steered_pop[peak_genotype_index] - straight_pop[peak_genotype_index]) < 10**-15:
							same+=1
						elif steered_pop[peak_genotype_index] > straight_pop[peak_genotype_index]:
							worse+=1
						elif steered_pop[peak_genotype_index] < straight_pop[peak_genotype_index]:
							better+=1

			steering_stats.append((better,worse,same))


			pc_better = 100*float(better) / (better+worse+same)
			pc_worse  = 100*float(worse) / (better+worse+same)
			print landscapes[t].name+", better:", str(better)+" ("+str(pc_better)+"%)", ", worse:", str(worse)+" ("+str(pc_worse)+"%)"

			overall_better+=better
			overall_worse+=worse
			overall_same+=same
			pc_ov_better = 100*float(overall_better) / (overall_better+overall_worse+overall_same)
			pc_ov_worse = 100*float(overall_worse) / (overall_better+overall_worse+overall_same)



	print "Overall, better:", str(overall_better)+" ("+str(pc_ov_better)+"%)", ", worse:", str(overall_worse)+" ("+str(pc_ov_worse)+"%)"
	return steering_stats

#==============================================================================#
# Generates Results
#==============================================================================#

print "Table 1 results: \n"
probHighestPeak(init_pop)
print 
highPeakBestSingle(init_pop)
print 
highPeakBestDouble(init_pop)
print 
highPeakBestTriple(init_pop)
print
print "Table 2 results: \n"
probLowestPeak(init_pop)
print 
lowestPeakBestSingle(init_pop)
print 
lowestPeakBestDouble(init_pop)
print
lowestPeakBestTriple(init_pop)
print 
print "Table 3 results: \n"
singleSteererPercentages(init_pop)
print 
doubleSteererPercentages(init_pop)
print
tripleSteererPercentages(init_pop)
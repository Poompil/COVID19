import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import random
from scipy import optimize
import time

import mplcyberpunk
plt.style.use("cyberpunk")


'''
References

1) https://scipython.com/book/chapter-8-scipy/additional-examples/the-sir-epidemic-model/
2) https://arxiv.org/pdf/2002.06563.pdf
3) http://www.public.asu.edu/~hnesse/classes/seir.html?Beta=0.3&Gamma=0.1&Sigma=0.2&Mu=0&Nu=0&initialS=999&initialE=1&initialI=0&initialR=0&iters=160
4) https://triplebyte.com/blog/modeling-infectious-diseases

'''


class SEIR_Modified():
	def __init__(self, N=1000, P0=0, D0=0, R0=0, Q0=0, I0=1, E0=0, t=365):
		self.N = N
		
		self.P0 = P0
		self.D0 = D0
		self.R0 = R0
		self.Q0 = Q0
		self.I0 = I0
		self.E0 = E0
		self.S0 = self.N - self.E0 - self.I0 - self.Q0 - self.R0 - self.D0 - self.P0
		
		self.t = np.linspace(0, int(t), int(t))
		
	def Params(self, t):
		#also change params based on country's age distribution?
		#i.e., Params(self, t, age):
		
		alpha 	= 0.085
		beta 	= 1.0
		gamma 	= 1.0 / 2.0
		delta 	= 1.0 / 14.0
		lam 	= 0.08 #+ 0.0001*t
		kappa 	= 0.004
		
		return alpha, beta, gamma, delta, lam, kappa
		
	
	def ODE(self, y, t):
		S, E, I, Q, R, D, P = y
		try:
			alpha, beta, gamma, delta, lam, kappa = self.parameters
		except:
			alpha, beta, gamma, delta, lam, kappa = self.Params(t)

		dSdt = (- beta * S * I / self.N) - (alpha * S)
		dEdt = (beta * S * I / self.N) - (gamma * E)
		dIdt = (gamma * E) - (delta * I) - (lam * I) - (kappa * I)
		dQdt = (delta * I) - (lam * Q) - (kappa * Q)
		dRdt = (lam * Q) + (lam * I)
		dDdt = (kappa * Q) + (kappa * I)
		dPdt = alpha * S

		return dSdt, dEdt, dIdt, dQdt, dRdt, dDdt,dPdt
		
	def SolveODE(self, *argv):
		
		y0 = self.S0, self.E0, self.I0, self.Q0, self.R0, self.D0, self.P0

		try:
			self.parameters = argv[0]
		except:
			pass
			
		# Integrate the SIR equations over the time grid, t.
		res = odeint(self.ODE, y0, self.t)
		self.S, self.E, self.I, self.Q, self.R, self.D, self.P = res.T
		self.Total = self.S + self.E + self.I + self.Q + self.R + self.D + self.P  
		self.C = self.D + self.R + self.I + self.Q #Everyone except exposed is confirmed

		return
		
	def PlotResults(self, Normalized=False):
		# Plot the data on three separate curves for S(t), I(t) and R(t)
		if Normalized == True:
			N = self.N
		else:
			N = 1.0
			
		fig = plt.figure(figsize=(7,9))
		
		try:
			fig.suptitle('Model vs. Data - ' + self.Data.country + ' - ' + self.Data.state)
		except:
			fig.suptitle('Model vs. Data - ' + self.Data.country)


		ax = fig.add_subplot(211, axisbelow=True)
		#ax.plot(self.t, self.S/N, 'blue', alpha=0.5, lw=2, label='Susceptible')
		#ax.plot(self.t, self.E/N, 'yellow', alpha=0.5, lw=2, label='Exposed')
		#ax.plot(self.t, self.I/N, 'red', alpha=0.5, lw=2, label='Infected')
		#ax.plot(self.t, self.Q/N, 'orange', alpha=0.5, lw=2, label='Quarantined')
		#ax.plot(self.t, self.P/N, 'teal', alpha=0.5, lw=2, label='Insusceptible')
		#ax.plot(self.t, self.Total/N, 'brown', alpha=0.5, lw=2, label='Total')
		
		ax.plot(self.t, self.C/N, label='Confirmed (Model)')
		ax.plot(self.t, self.D/N, label='Dead (Model)')

		try:
			ax.scatter(self.Data.Days, self.Data.C/N, s=5, label='Confirmed (Data)')
			ax.scatter(self.Data.Days, self.Data.D/N, s=5, label='Death (Data)')
			
			if self.Data.country != 'Canada':
				ax.plot(self.t, self.R/N, label='Recovered (Model)')
				ax.scatter(self.Data.Days, self.Data.R/N, s=5, label='Recovered (Data)')			
		except:
			pass

		#ax.set_xlabel('Time (days)')
		ax.set_ylabel('Number (log)')
		#ax.set_ylim(0,1.2)
		ax.yaxis.set_tick_params(length=0)
		ax.xaxis.set_tick_params(length=0)
		#ax.grid(b=True, which='major', c='w', lw=2, ls='-')
#		legend = ax.legend()
#		legend.get_frame().set_alpha(0.5)
		for spine in ('top', 'right', 'bottom', 'left'):
			ax.spines[spine].set_visible(False)

		ax.set_yscale('log')
		############################
		ax2 = fig.add_subplot(212, axisbelow=True)
		ax2.plot(self.t, self.C/N, label='Confirmed (Model)')
		ax2.plot(self.t, self.D/N, label='Dead (Model)')
		
		try:
			ax2.scatter(self.Data.Days, self.Data.C/N, s=5, label='Confirmed (Data)')
			ax2.scatter(self.Data.Days, self.Data.D/N, s=5, label='Death (Data)')
			
			if self.Data.country != 'Canada':
				ax2.plot(self.t, self.R/N, label='Recovered (Model)')
				ax2.scatter(self.Data.Days, self.Data.R/N, s=5, label='Recovered (Data)')
		
		except:
			pass
		
		ax2.set_xlabel('Time (days)')
		ax2.set_ylabel('Number')
		ax2.yaxis.set_tick_params(length=0)
		ax2.xaxis.set_tick_params(length=0)
		
		legend = ax2.legend()
		legend.get_frame().set_alpha(0.5)
		for spine in ('top', 'right', 'bottom', 'left'):
			ax2.spines[spine].set_visible(False)
		
		##############
		mplcyberpunk.make_lines_glow(ax)
		mplcyberpunk.make_lines_glow(ax2)
		plt.show()
		
	def AddData(self, Data):
		self.Data = Data
		#self.Data_Day = Data[:,0]
		#self.Data_C = Data[:,1]
		#self.Data_D = Data[:,2]
		#self.Data_R = Data[:,3]
			
	def EvalFit(self):
		self.NRMSE_C = np.sqrt(np.mean((self.Data.C - self.C)**2)) / np.mean(self.Data.C)
		self.NRMSE_D = np.sqrt(np.mean((self.Data.D - self.D)**2)) / np.mean(self.Data.D)
		
		if self.Data.country != 'Canada':
			self.NRMSE_R = np.sqrt(np.mean((self.Data.R - self.R)**2)) / np.mean(self.Data.R)
		else:
			self.NRMSE_R = 0.0
		
		
		#self.CORR_C = np.corrcoef(self.Data_C, self.C)
		#self.CORR_D = np.corrcoef(self.Data_D, self.D)
		#self.CORR_R = np.corrcoef(self.Data_R, self.R)
		
	def FitData(self, params):
		self.SolveODE(params)
		self.EvalFit()
	
		return  self.NRMSE_C + self.NRMSE_D + self.NRMSE_R
		#return  self.CORR_C + self.CORR_D + self.CORR_R

class Data():
	def __init__(self, country=None, state=None):
		#https://github.com/datasets/covid-19
		
		self.country = country
		self.state = state

		headers = ['date','country','state','c','r','d']
		parse_dates = ['date']
		df = pd.read_csv('data/time-series-19-covid-combined.csv', usecols=[0,1,2,5,6,7], header=0, names=headers, parse_dates=parse_dates)
		
		countryList = df.country.unique()
		dataDuration = (df.date.max() - df.date.min()).days + 1

		if state==None:
			print ('sup')
			df_reduced = df[(df.country == country) & df.state.isnull()]
		else:
			df_reduced = df[(df.country == country) & (df.state == state)]
			
		print (df_reduced)
			
		if df_reduced.shape[0] != dataDuration:
			raise ValueError('Please narrow down selection by entering a state/province in ' + country)
		else:
			df = df_reduced

			#TRIM DATA TO WHEN FIRST CASE OCCURS
			#df = df.loc[(df[['c','r','d']] != 0).any(axis=1)]
			df = df.loc[(df[['d']] != 0).any(axis=1)]
			dataDuration = (df.date.max() - df.date.min()).days + 1

			Initial = df[df.date == df.date.min()]
			
			self.C0 = int(Initial.c)
			self.D0 = int(Initial.d)
			
			try:
				self.R0 = int(Initial.r)
			except:
				self.R0 = 0
			
			self.Duration = dataDuration

			self.Days = np.arange(dataDuration)
			
			df = df.fillna(0)
			
			self.C = df['c'].values
			self.D = df['d'].values
			self.R = df['r'].values
			
			#print (self.R)
						

#make this work fo rthe way canada data is formatted
data = Data('US',)
#US: 328200000
#Italy: 60360000
#Wuhan: 58500000
#Canada: 37590000
#Ontario: 14570000
#British Columbia: 5071000
model = SEIR_Modified(328200000, 0, data.D0, data.R0, 0, data.C0, 0, data.Duration)
model.AddData(data)
#model.PlotResults()

bounds = [(0.0,1.0), (0.0,1.0), (0.0,1.0), (0.0,1.0), (0.0,1.0), (0.0,1.0)]
x0 = np.random.rand(6,1)
results = {}

t0 = time.time()
print ('\nRUNNING: L-BFGS-B')
results['L-BFGS-B'] = optimize.minimize(model.FitData, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter':3000})
t1 = time.time()
print ('Time taken: \t' + str(t1 - t0))
print (results['L-BFGS-B'].x, results['L-BFGS-B'].fun)

model.t = np.linspace(0, int(365), int(365))
model.SolveODE(results['L-BFGS-B'].x)
#model.EvalFit()
model.PlotResults()


'''

#Hubei
b = SEIR_Modified(58500000, 0, 17, 28, 0, 444, 0, 60)	
HubeiData = np.loadtxt('Hubei.csv', delimiter=',', skiprows=1)
b.AddData(HubeiData)

##USA
#b = SEIR_Modified(58500000, 0, 17, 28, 0, 444, 0, 60)	
#USAData = np.loadtxt('USA.csv', delimiter=',', skiprows=1)
#b.AddData(USAData)

#b.SolveODE([0.085,1.0,0.5,1.0/14.0,0.08,0.004])
#b.EvalFit()
#b.PlotResults()




bounds = [(0.0,1.0), (0.0,1.0), (0.0,1.0), (0.0,1.0), (0.0,1.0), (0.0,1.0)]
x0 = np.random.rand(6,1)

results = {}

t0 = time.time()
print '\nRUNNING: L-BFGS-B'
results['L-BFGS-B'] = optimize.minimize(b.FitData, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter':3000})
t1 = time.time()
print 'Time taken: \t' + str(t1 - t0)
print results['L-BFGS-B'].x, results['L-BFGS-B'].fun
b.SolveODE(results['L-BFGS-B'].x)
#b.EvalFit()
#print b.CORR_C, b.CORR_D, b.CORR_R
b.PlotResults()
'''
#	FOLLOWING ARE ALL TIME-SUCKS...GRADIENT OPTIMIZATION IS THE CLEAR BEST
#	print '\nRUNNING: SHGO'
#	results['SHGO'] = optimize.shgo(b.FitData, bounds=bounds, n=200, iters=5, sampling_method='sobol')
#	t2 = time.time()
#	print 'Time taken: \t' + str(t2 - t1)
#	print results['SHGO'].x, results['SHGO'].fun
#	b.SolveODE(results['SHGO'].x)
#	#b.PlotResults()
#	
#	print '\nRUNNING: DIFFERENTIAL EVOLUTION (DE)'
#	results['DE'] = optimize.differential_evolution(b.FitData, bounds=bounds, maxiter=3000, popsize=50)
#	t3 = time.time()
#	print 'Time taken: \t' + str(t3 - t2)
#	print results['DE'].x, results['DE'].fun
#	b.SolveODE(results['DE'].x)
#	#b.PlotResults()
#	
#	print '\nRUNNING: DUAL ANNEALING (DA)'
#	results['DA'] = optimize.dual_annealing(b.FitData, bounds=bounds, maxiter=3000)
#	t4 = time.time()
#	print 'Time taken: \t' + str(t4 - t3)
#	print results['DA'].x, results['DA'].fun
#	b.SolveODE(results['DA'].x)
#	#b.PlotResults()

'''
RUNNING: L-BFGS-B
Time taken:     3.31200003624
[0.05447533 1.         0.86607059 0.29648546 0.04015081 0.0028479 ] 0.520614540533312

RUNNING: SHGO
Time taken:     10.9570000172
[0.05046408 1.         0.93714384 0.32637476 0.04012678 0.00284661] 0.520640401486222

RUNNING: DIFFERENTIAL EVOLUTION (DE)
Time taken:     126.101999998
[0.05532742 0.99955477 0.85261549 0.29011112 0.04016078 0.00284855] 0.5206257533256043

RUNNING: DUAL ANNEALING (DA)
Time taken:     437.100999832
[0.05349788 0.99998268 0.88321992 0.3037168  0.04013765 0.00284711] 0.520612097967827
'''
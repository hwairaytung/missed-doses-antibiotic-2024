# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 08:32:25 2023

@author: R
"""

#Increase EC50 or decrease E_max for resistant? 

import numpy as np
import scipy.integrate as spint
import matplotlib.pyplot as plt
import scipy.stats as sstats
import scipy.special as smisc

def whoops(dose_history, num_doses, mode, args= []):
    if mode == 'adherence':
        return 1
    #args[0] is p(remember)
    if mode == 'random':
        return int(np.random.uniform()<args[0])
    #forgets every args[0] th dose
    if mode == 'periodic':
        return int((len(dose_history)+1)%args[0] != 0) 
    if mode == 'forget_then_remember':
        if len(dose_history) == 0:
            return int(np.random.uniform()<args[0])#1
        return int(len(dose_history) == 0 or dose_history[-1] == 0 or np.random.uniform()<args[0])
    if mode == 'forget_then_double':
        if len(dose_history) == 0:
            return int(np.random.uniform()<args[0])#1
        if dose_history[-1] == 0:
            return 2
        return int(np.random.uniform()<args[0])

def generate_double_or_not_dosing(num_doses, args= []):
    dose_history = []
    while np.sum(dose_history) < num_doses:
        dose_history.append(whoops(dose_history, num_doses, 'forget_then_remember', args=args))
    double_dose_history = [dose_history[0]]
    index = 1
    while np.sum(double_dose_history) < num_doses:
        if double_dose_history[-1] == 0:
            double_dose_history.append(2)
        else:
            double_dose_history.append(dose_history[index])
        index = index + 1
    if np.sum(double_dose_history) == num_doses + 1:
        double_dose_history[-1] = 1
    return dose_history, double_dose_history

def antibiotic_concentration(dose_history, lam, T):
    points_per_dosing = 100
    time1 = [0]
    res1 = [dose_history[0]]
    for i in range(1, len(dose_history) +1):
        for j in range(points_per_dosing):
            time1.append(T*(j/points_per_dosing + i-1))
            #print(res1[-1])
            res1.append(res1[-1]*0.5**(T/points_per_dosing/lam))
        time1.append((i)*T)
        if i < len(dose_history):
            res1.append(res1[-1] + dose_history[i])
        else:
            res1.append(res1[-1])
    return time1[:-1], res1[:-1]

def plot_antibiotic_concentration(dose_history, double_dose_history, lam, T):
    t1, r1 = antibiotic_concentration(dose_history, lam, T)
    t2, r2 = antibiotic_concentration(double_dose_history, lam, T)
    plt.plot(t1, r1, label = 'single dosing')
    plt.plot(t2, r2, label = 'double dosing')
    plt.legend(fontsize=12)
    plt.xlim(0,120)
    plt.xlabel("time (hours)", fontsize=12)
    plt.ylabel("antibiotic concentration", fontsize=12)
    plt.savefig("antibact_conc.png", dpi = 600, bbox_inches = 'tight')
    plt.show()
    plt.close()

########################################################################################

def lol(y, t, lam, T, pMIC, r, d, kappa, K, c0):
    zmic = 0.5**(pMIC*T/lam)/(1-0.5**(T/lam))
    if K == np.inf:
        return r-(r+d)*(c0*0.5**(t/lam)/zmic)**kappa/(d/r + (c0*0.5**(t/lam)/zmic)**kappa)
    return r*(1-np.exp(y)/K)-(r+d)*(c0*0.5**(t/lam)/zmic)**kappa/(d/r + (c0*0.5**(t/lam)/zmic)**kappa)

def bacteria_concentration(dose_hist, lam, T, pMIC, r, d, kappa, K):
    res_bact = [10*np.log(10)]
    res_time = [0]
    time = np.array(range(101))/100*T
    c0 = 0
    for dose_num in range(len(dose_hist)):
        yay = spint.odeint(lol, res_bact[-1], time, args = (lam, T, pMIC, r, d, kappa, K, c0 + dose_hist[dose_num]))
        c0 = (c0 + dose_hist[dose_num])*0.5**(T/lam)
        res_bact = res_bact + list(list(zip(*yay))[0])
        res_time = res_time + list(time+dose_num*T)
    return res_time, res_bact


def plot_bacteria_concentration(dose_history, double_dose_history, lam, T, pMIC, r, d, kappa, K):
    t1, r1 = bacteria_concentration(dose_history, lam, T, pMIC, r, d, kappa, K)
    t2, r2 = bacteria_concentration(double_dose_history, lam, T, pMIC, r, d, kappa, K)
    t3, r3 = bacteria_concentration([1]*np.sum(dose_history), lam, T, pMIC, r, d, kappa, K)
    r1, r2, r3 = np.array(r1)*np.log10(np.e), np.array(r2)*np.log10(np.e), np.array(r3)*np.log10(np.e)
    plt.plot(t1, r1-r1[0], label = 'single dosing')
    plt.plot(t2, r2-r2[0], label = 'double dosing')
    plt.plot(t3, r3-r3[0], label = 'adherent')
    plt.legend(fontsize=12)
    #plt.yscale('log',base=10) 
    plt.xlim(0,120)
    plt.xlabel("time (hours)", fontsize=12)
    plt.ylabel("log10 bacteria concentration", fontsize=12)
    plt.savefig("bact_conc.png", dpi = 600, bbox_inches = 'tight')
    plt.show()
    plt.close()

def s(i, lam, T, pMIC, r, d, kappa):
    return r*T + (r+d)*lam/kappa*np.log2((d+r*(2**((pMIC-2)*T/lam)*(2**(T/lam)-1)*i)**kappa)/(d+r*(2**((pMIC-1)*T/lam)*(2**(T/lam)-1)*i)**kappa))

p = 0.8
num_doses = 15
lam, T = 1.5, 6

K = 10**10
pMIC = 0.5
r, d = 0.7, 4
kappa = 2

s0, s1, s2 = s(0, lam, T, pMIC, r, d, kappa), s(1, lam, T, pMIC, r, d, kappa), s(2, lam, T, pMIC, r, d, kappa)
print("S", (s2-2*s1)/s0)

#example for fig
dose_history =        [1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1]
double_dose_history = [1, 1, 1, 1, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 1]

dose_history =        [1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]
double_dose_history = [1, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 1, 1, 1, 1]

plot_antibiotic_concentration(dose_history, double_dose_history, lam, T)
plot_bacteria_concentration(dose_history, double_dose_history, lam, T, pMIC, r, d, kappa, K)


def double_cdf(s0, s1, s2, n, p):
    probs = []
    vals = []
    for n2 in range(n//2 + 1):
        probs.append(smisc.comb(n-n2, n2)*p**(n-2*n2)*(1-p)**n2)
        vals.append(n2*(s2-2*s1+s0))
        if n2<n/2:
            probs.append(smisc.comb(n-n2-1, n2)*p**(n-2*n2-1)*(1-p)**(n2+1))
            vals.append(s0+n2*(s2-2*s1+s0))
    vals, probs = (list(t) for t in zip(*sorted(zip(vals, probs))))
    probs = 1-np.cumsum(probs)
    
    xs = [vals[0]]
    for i in range(1, len(vals)):
        xs = xs + 2*[vals[i]]
    ys = []
    for i in range(len(probs)):
        ys = ys + 2*[probs[i]]
    ys = ys[:-1]
    return xs, ys
        


first_n_remember = 4

single_bact = []
double_bact = []
perf_bact = bacteria_concentration([1]*np.sum(dose_history), lam, T, pMIC, r, d, kappa, K)[-1][-1]
for i in range(1000):
    dose_history, double_dose_history = generate_double_or_not_dosing(num_doses-first_n_remember, args= [p])
    dose_history, double_dose_history = first_n_remember*[1]+dose_history, first_n_remember*[1]+double_dose_history
    single_bact.append((bacteria_concentration(dose_history, lam, T, pMIC, r, d, kappa, K)[-1][-1]) - perf_bact)
    double_bact.append((bacteria_concentration(double_dose_history, lam, T, pMIC, r, d, kappa, K)[-1][-1]) - perf_bact)

single_bact, double_bact = np.array(single_bact)*np.log10(np.e), np.array(double_bact)*np.log10(np.e)
perf_bact = perf_bact*np.log10(np.e)

single_bact_sort = np.sort(single_bact)/(10-perf_bact)
single_bact_cdf = [np.sum(i < single_bact_sort)/len(single_bact_sort) for i in np.linspace(0,single_bact_sort[-1],1000)]
double_bact_sort = np.sort(double_bact)/(10-perf_bact)
double_bact_cdf = [np.sum(i < double_bact_sort)/len(double_bact_sort) for i in np.linspace(double_bact_sort[0],double_bact_sort[-1],1000)]


plt.plot(np.linspace(0,single_bact_sort[-1],1000), single_bact_cdf, color='blue', label = 'single dosing')
plt.plot(np.linspace(0,single_bact_sort[-1],1000), 1-sstats.binom.cdf(np.linspace(0,single_bact_sort[-1],1000)/(s0*np.log10(np.e))*(10-perf_bact), num_doses-first_n_remember, 1-p), color = 'green', linestyle='dashed', label = 'rw single dosing')

#plt.plot(np.linspace(0,single_bact_sort[-1],1000), 1-sstats.binom.cdf(np.linspace(0,single_bact_sort[-1],1000)/s0*(-num_doses*s1), num_doses-first_n_remember, 1-p), color = 'black', linestyle='dashed', label = 'rw actual single')



plt.plot(np.linspace(double_bact_sort[0],double_bact_sort[-1],1000), double_bact_cdf, color = 'orange', label = 'double dosing')
xs, ys = double_cdf(s0, s1, s2, num_doses-first_n_remember, p)
plt.plot(np.array(xs)*np.log10(np.e)/(10-perf_bact), ys, color = 'red', linestyle='dashed', label = 'rw double dosing')

#plt.plot(np.array(xs)/(-num_doses*s1), ys, color = 'black', linestyle='dashed', label = 'rw double actual')

plt.xlim([0, 1])
plt.ylim([0, 1])

plt.xlabel("$Y_i/X^{\mathrm{ode}}_{\mathrm{perf}}$", fontsize=12)
plt.ylabel("Survival probability", fontsize=12)

plt.legend(fontsize=12)

plt.savefig("survival"+str(first_n_remember)+".png", dpi = 600, bbox_inches = 'tight')

#s1_mid = s(1/(1-0.5**(T/lam)), lam, T, pMIC, r, d, kappa)
#xs_mid, ys_mid = double_cdf(s0, s1_mid, s2, num_doses-first_n_remember, p)
#plt.plot(np.array(xs_mid)*np.log10(np.e)/(10-perf_bact), ys_mid, color = 'grey')

plt.show()
plt.close()



first_n_remember = 0

single_bact = []
double_bact = []
perf_bact = bacteria_concentration([1]*np.sum(dose_history), lam, T, pMIC, r, d, kappa, K)[-1][-1]
for i in range(1000):
    dose_history, double_dose_history = generate_double_or_not_dosing(num_doses-first_n_remember, args= [p])
    dose_history, double_dose_history = first_n_remember*[1]+dose_history, first_n_remember*[1]+double_dose_history
    single_bact.append((bacteria_concentration(dose_history, lam, T, pMIC, r, d, kappa, K)[-1][-1]) - perf_bact)
    double_bact.append((bacteria_concentration(double_dose_history, lam, T, pMIC, r, d, kappa, K)[-1][-1]) - perf_bact)

single_bact, double_bact = np.array(single_bact)*np.log10(np.e), np.array(double_bact)*np.log10(np.e)
perf_bact = perf_bact*np.log10(np.e)

single_bact_sort = np.sort(single_bact)/(10-perf_bact)
single_bact_cdf = [np.sum(i < single_bact_sort)/len(single_bact_sort) for i in np.linspace(single_bact_sort[0],single_bact_sort[-1],1000)]
double_bact_sort = np.sort(double_bact)/(10-perf_bact)
double_bact_cdf = [np.sum(i < double_bact_sort)/len(double_bact_sort) for i in np.linspace(double_bact_sort[0],double_bact_sort[-1],1000)]


plt.plot(np.linspace(0,single_bact_sort[-1],1000), single_bact_cdf, color='blue', label = 'single dosing')
plt.plot([0, 0], [1, single_bact_cdf[0]], color='blue')
plt.plot(np.linspace(double_bact_sort[0],double_bact_sort[-1],1000), double_bact_cdf, color = 'orange', label = 'double dosing')
plt.plot(2*[double_bact_sort[0]], [1, double_bact_cdf[0]], color='orange')

#plt.xlim([0, 1])
plt.ylim([0, 1])

plt.xlabel("$Y_i^{ode}/X^{\mathrm{ode}}_{\mathrm{perf}}$", fontsize=12)
plt.ylabel("Survival probability", fontsize=12)

plt.legend(fontsize=12)

plt.savefig("survival"+str(first_n_remember)+".png", dpi = 600, bbox_inches = 'tight')

plt.show()
plt.close()

###############################################################################

first_n_remember = 4

single_bact = []
double_bact = []
perf_bact = bacteria_concentration([1]*np.sum(dose_history), lam, T, pMIC, r, d, kappa, K)[-1][-1]
for i in range(1000):
    dose_history, double_dose_history = generate_double_or_not_dosing(num_doses-first_n_remember, args= [p])
    dose_history, double_dose_history = first_n_remember*[1]+dose_history, first_n_remember*[1]+double_dose_history
    single_bact.append((bacteria_concentration(dose_history, lam, T, pMIC, r, d, kappa, K)[-1][-1]) - perf_bact)
    double_bact.append((bacteria_concentration(double_dose_history, lam, T, pMIC, r, d, kappa, K)[-1][-1]) - perf_bact)

single_bact, double_bact = np.array(single_bact)*np.log10(np.e), np.array(double_bact)*np.log10(np.e)
perf_bact = perf_bact*np.log10(np.e)

single_bact_sort = np.sort(single_bact)/(10-perf_bact)
single_bact_cdf = [np.sum(i < single_bact_sort)/len(single_bact_sort) for i in np.linspace(0,single_bact_sort[-1],1000)]
double_bact_sort = np.sort(double_bact)/(10-perf_bact)
double_bact_cdf = [np.sum(i < double_bact_sort)/len(double_bact_sort) for i in np.linspace(double_bact_sort[0],double_bact_sort[-1],1000)]


#plt.plot(np.linspace(0,single_bact_sort[-1],1000), single_bact_cdf, color='blue', label = 'single dosing')
#plt.plot(np.linspace(0,single_bact_sort[-1],1000), 1-sstats.binom.cdf(np.linspace(0,single_bact_sort[-1],1000)/(s0*np.log10(np.e))*(10-perf_bact), num_doses-first_n_remember, 1-p), color = 'green', linestyle='dashed', label = 'rw single dosing')

plt.plot(np.linspace(double_bact_sort[0],double_bact_sort[-1],1000), double_bact_cdf, color = 'orange', label = 'double dosing')
xs, ys = double_cdf(s0, s1, s2, num_doses-first_n_remember, p)
plt.plot(np.array(xs)*np.log10(np.e)/(10-perf_bact), ys, color = 'red', linestyle='dashed', label = 'rw double dosing $s_1$')

#plt.plot(np.array(xs)/(-num_doses*s1), ys, color = 'black', linestyle='dashed', label = 'rw double actual')

plt.xlim([0, 1])
plt.ylim([0, 1])

plt.xlabel("$Y_i/X^{\mathrm{ode}}_{\mathrm{perf}}$", fontsize=12)
plt.ylabel("Survival probability", fontsize=12)





s1_mid = s(1/(1-0.5**(T/lam)), lam, T, pMIC, r, d, kappa)
xs_mid, ys_mid = double_cdf(s0, s1_mid, s2, num_doses-first_n_remember, p)
plt.plot(np.array(xs_mid)*np.log10(np.e)/(10-perf_bact), ys_mid, color = 'grey', linestyle='dashed', label = 'rw double dosing $s_{1*}$')

#plt.plot(np.array(xs_mid)/(-num_doses*s1_mid), ys_mid, color = 'black', linestyle='dashed', label = 'rw double actual')

# =============================================================================
# s1_mid1 = s(1 + 0.5**(T/lam), lam, T, pMIC, r, d, kappa)
# xs_mid1, ys_mid1 = double_cdf(s0, s1_mid1, s2, num_doses-first_n_remember, p)
# plt.plot(np.array(xs_mid1)*np.log10(np.e)/(10-perf_bact), ys_mid1, color = 'grey', label = 'rw double dosing $s_{1*1}$')
# =============================================================================

plt.legend(fontsize=12)
plt.savefig("survival"+str(first_n_remember)+"sstar.png", dpi = 600, bbox_inches = 'tight')

plt.show()
plt.close()

print("zMIC", 0.5**(pMIC*T/lam)/(1-0.5**(T/lam)))

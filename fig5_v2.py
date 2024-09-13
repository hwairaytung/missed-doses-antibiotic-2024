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

import seaborn

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
    plt.legend()
    plt.xlim(0,120)
    plt.xlabel("time")
    plt.ylabel("antibiotic concentration")
    plt.savefig("antibact_conc.png", dpi = 600, bbox_inches = 'tight')
    plt.show()
    plt.close()

########################################################################################

def lol(y, t, lam, T, zmic, r, d, kappa, K, c0):
    if K == np.inf:
        return r-(r+d)*(c0*0.5**(t/lam)/zmic)**kappa/(d/r + (c0*0.5**(t/lam)/zmic)**kappa)
    return r*(1-np.exp(y)/K)-(r+d)*(c0*0.5**(t/lam)/zmic)**kappa/(d/r + (c0*0.5**(t/lam)/zmic)**kappa)

def bacteria_concentration(dose_hist, lam, T, zmic, r, d, kappa, K):
    res_bact = [10*np.log(10)]
    res_time = [0]
    time = np.array(range(101))/100*T
    c0 = 0
    for dose_num in range(len(dose_hist)):
        yay = spint.odeint(lol, res_bact[-1], time, args = (lam, T, zmic, r, d, kappa, K, c0 + dose_hist[dose_num]))
        c0 = (c0 + dose_hist[dose_num])*0.5**(T/lam)
        res_bact = res_bact + list(list(zip(*yay))[0])
        res_time = res_time + list(time+dose_num*T)
    return res_time, res_bact

#def s(i, lam, T, pMIC, r, d, kappa):
#    return r*T + (r+d)*lam/kappa*np.log2((d+r*(2**((pMIC-2)*T/lam)*(2**(T/lam)-1)*i)**kappa)/(d+r*(2**((pMIC-1)*T/lam)*(2**(T/lam)-1)*i)**kappa))

def s(i, lam, T, zmic, r, d, kappa):
    return r*T - (r+d)*lam/kappa*np.log2((d+r*(i/zmic)**kappa)/(d+r*(i/zmic*0.5**(T/lam))**kappa))

def n2_dist(n, p):
    probs = []
    vals = []
    for n2 in range(n//2 + 1):
        p1= smisc.comb(n-n2, n2)*p**(n-2*n2)*(1-p)**n2
        vals.append(n2)
        if n2<n/2:
            p1= p1+smisc.comb(n-n2-1, n2)*p**(n-2*n2-1)*(1-p)**(n2+1)
        probs.append(p1)
    return vals, probs

p = 0.8
num_doses = 15
lam, T = 1.5, 6

K = 10**10
zmic = 0.267
r, d = 0.7, 4
kappa = 2

s0, s1, s2 = s(0, lam, T, zmic, r, d, kappa), s(1, lam, T, zmic, r, d, kappa), s(2, lam, T, zmic, r, d, kappa)
print("S", (s2-2*s1)/s0)

first_n_remember = 4

single_bact = []
double_bact = []
results = []

rw_results = []

lam_vals = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2]
for lam_test in lam_vals:
    single_wins = 0
    for i in range(1000):
        dose_history, double_dose_history = generate_double_or_not_dosing(num_doses-first_n_remember, args= [p])
        dose_history, double_dose_history = first_n_remember*[1]+dose_history, first_n_remember*[1]+double_dose_history
        if bacteria_concentration(dose_history, lam_test, T, zmic, r, d, kappa, K)[-1][-1] <= bacteria_concentration(double_dose_history, lam_test, T, zmic, r, d, kappa, K)[-1][-1]:
            single_wins = single_wins + 1
    results.append(1-single_wins/1000)
    s0, s1, s2 = s(0, lam_test, T, zmic, r, d, kappa), s(1, lam_test, T, zmic, r, d, kappa), s(2, lam_test, T, zmic, r, d, kappa)
    S = (s2-2*s1)/s0
    print(S)
    n2_vals, n2_probs = n2_dist(num_doses-first_n_remember, p)
    probs = 0#n2_probs[0]
    for i in n2_vals:
        if i != 0:
            probs = probs + (1-sstats.binom.cdf(i*S, i, 1-p))*n2_probs[i]
    rw_results.append(probs)
    print(results)
    print(rw_results)
plt.plot(lam_vals, results, label = 'ODE sim')
plt.plot(lam_vals, rw_results, label = 'rw')
plt.legend()
plt.xlabel("$t_{1/2}$")
plt.ylabel('$P(X_2 < X_1)$')
plt.show()
plt.close()

lam_vals = list(np.linspace(0.8, 1.4, num=13))[::-1]#[0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
zmic_vals = list(np.linspace(.1, .2, num=11))#[0.4, 0.45, 0.5, 0.55, 0.6]
results, rw_results = [], []
for lam_test in lam_vals:
    resul = []
    rw_resul = []
    for zmic_test in zmic_vals:
        single_wins = 0
        for i in range(1000):
            dose_history, double_dose_history = generate_double_or_not_dosing(num_doses-first_n_remember, args= [p])
            dose_history, double_dose_history = first_n_remember*[1]+dose_history, first_n_remember*[1]+double_dose_history
            if bacteria_concentration(dose_history, lam_test, T, zmic_test, r, d, kappa, K)[-1][-1] <= bacteria_concentration(double_dose_history, lam_test, T, zmic_test, r, d, kappa, K)[-1][-1]:
                single_wins = single_wins + 1
        resul.append(1-single_wins/1000)
        s0, s1, s2 = s(0, lam_test, T, zmic_test, r, d, kappa), s(1, lam_test, T, zmic_test, r, d, kappa), s(2, lam_test, T, zmic_test, r, d, kappa)
        S = (s2-2*s1)/s0
        print(S)
        n2_vals, n2_probs = n2_dist(num_doses-first_n_remember, p)
        probs = 0#n2_probs[0]
        for i in n2_vals:
            if i != 0:
                probs = probs + (1-sstats.binom.cdf(i*S, i, 1-p))*n2_probs[i]
        rw_resul.append(probs)
    results.append(resul)
    rw_results.append(rw_resul)

def format_label(list1, dec_points):
    return [("{:."+str(dec_points)+"f}").format(i) for i in list1]

#zMIC_vals = 0.5**(pMIC_vals*T/lam_vals)/(1-0.5**(T/lam_vals))

seaborn.set(font_scale=1.3)

ax = seaborn.heatmap(results, cmap = 'RdBu', vmin = 0, vmax = 1, xticklabels = format_label(zmic_vals, 2), yticklabels=format_label(lam_vals, 2))
ax.set(xlabel='$zMIC$', ylabel = '$t_{\mathrm{half}}$ (hours)')
plt.savefig("heatmapODE.png", dpi = 600, bbox_inches = 'tight')
plt.show()
plt.close()

ax = seaborn.heatmap(rw_results, cmap = 'RdBu', vmin = 0, vmax = 1, xticklabels = format_label(zmic_vals, 2), yticklabels=format_label(lam_vals, 2))
ax.set(xlabel='$zMIC$', ylabel = '$t_{\mathrm{half}}$ (hours)')
plt.savefig("heatmapRW.png", dpi = 600, bbox_inches = 'tight')
plt.show()
plt.close()

# =============================================================================
# plt.plot(lam_vals, results, label = 'ODE sim')
# plt.plot(lam_vals, rw_results, label = 'rw')
# plt.legend()
# plt.xlabel("$t_{1/2}$")
# plt.ylabel('$P(X_2 < X_1)$')
# plt.show()
# plt.close()
# =============================================================================

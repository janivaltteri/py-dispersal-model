import sys
import json
import numpy
import pandas
import argparse
import scipy.integrate as integrate
from scipy.spatial import distance_matrix
from scipy.special import k0, k1, i0, i1
from numpy.linalg import inv
from math import sqrt, pow, pi

parser = argparse.ArgumentParser(description='generate migration matrices from patch data')
parser.add_argument('--infile', '-i', required=True, type=str,
                    help='input parameter file')
parser.add_argument('--outfile', '-o', required=True, type=str,
                    help='output file name')
args = parser.parse_args()

## the quantities P_ij, F_i|j, F_i|0, G_i|j, G_i|0 are needed for simulations
## they are called pm, fm, f0, gm, g0 in this code

## read infile

try:
    with open(args.infile) as parfile:
        jstr = parfile.read()
        par = json.loads(jstr)
except IOError as e:
    print(str(e) + " reading " + args.infile)
    sys.exit(1)
except:
    print("error")
    sys.exit(1)

## todo:
## check that par contains
## am cm km, abase, cbase, kbase, xlocs, ylocs, areas
    
## parameters for The Matrix

##par = {'am': 1.0/10.0, 'cm': 2.0/10.0, 'km': 1.0}
par['alpha_m'] = sqrt(par['cm']/par['am'])

## locations

n = 8

##coef = 1.0
##abase = 1.0/10.0
##cbase = 1.0/10.0
##kbase = 100.0

##xlocations = [ 0.0, -1.0, -0.5,  0.0,  0.5,  1.0,  0.0,  0.0]
##ylocations = [ 1.0,  0.0,  0.0,  0.0,  0.0,  0.0, -1.0, -1.5]
##areas = [ 0.01, 0.01 ,0.01, 0.005, 0.005, 0.01, 0.01, 0.01]

##rexp = lambda t: -log(t)/coef
##rexpfun = numpy.vectorize(rexp)

locs = pandas.DataFrame({'x': par['xlocs'],
                         'y': par['ylocs'],
                         'rad': [sqrt(i/pi) for i in par['areas']],
                         'a': numpy.repeat(par['abase'],n),
                         'c': numpy.repeat(par['cbase'],n),
                         'k': numpy.repeat(par['kbase'],n)})

locs['alpha'] = locs['c']/locs['a']
locs['delta'] = locs['k']*locs['alpha']*locs['a']

print('the locations are:')
print(locs)

## distance matrix

dlist = [[locs['x'][i],locs['y'][i]] for i in range(n)]

dij = distance_matrix(dlist,dlist)

## Z matrix
## N.B.: Zheng 2009 p. 1501 has r_i in the denominator
## whereas the confirm.nb has r_j instead
## -> Ovaskainen Cornell 2003 also has r_j so that must be right!

zm = numpy.zeros([n,n])
for i in range(n):
    for j in range(n):
        if(i==j):
            zm[i,j] = 1.0
        else:
            zm[i,j] = k0(par['alpha_m']*dij[i,j]) / k0(par['alpha_m']*locs['rad'][j])

print(zm)
            
## p1, Dp1, p2, Dp2, p2star, Dp2star

p1_iterable = ((par['alpha_m']*k1(par['alpha_m']*locs['rad'][i])/
                k0(par['alpha_m']*locs['rad'][i])) for i in range(n))
p1 = numpy.fromiter(p1_iterable,float)

Dp1 = p1*numpy.identity(n)

## in p2: should check if delta means delta_h or delta_m
## in confirm.nb it is delta_h = k_h * a_h * alpha_h (patch dependent)

p2_iterable = ((locs['delta'])[i]*i1(locs['alpha'][i] * locs['rad'][i]) /
               (par['am']*par['km']*i0(locs['alpha'][i] * locs['rad'][i])) for i in range(n))
p2 = numpy.fromiter(p2_iterable,float)

Dp2 = p2*numpy.identity(n)

p2star_iterable = (par['km']*par['alpha_m']*par['am']*i1(par['alpha_m']*locs['rad'][i])/
                   (par['am']*par['km']*i0(par['alpha_m']*locs['rad'][i])) for i in range(n))
p2star = numpy.fromiter(p2star_iterable,float)

Dp2star = p2star*numpy.identity(n)

## S matrix

sm = Dp1.dot(inv(zm))

## P matrix, Pstar

pm = numpy.zeros([n,n])
for i in range(n):
    for j in range(n):
        if(i==j):
            pm[i,j] = 0.0
        else:
            pm[i,j] = -sm[i,j]/(sm[i,i] + p2[i])

pmstar = numpy.zeros([n,n])
for i in range(n):
    for j in range(n):
        if(i==j):
            pmstar[i,j] = 0.0
        else:
            pmstar[i,j] = -sm[i,j]/(sm[i,i] + p2star[i])

## prob dying before hitting another patch

fii = numpy.ones([n]) - numpy.sum(pm,axis=1)

## X matrix

xm = sm + Dp2

## t1, t2, t2star, t4

## fixed error locs['rad'][i] * ...
## fixed error denominator should have am not alpha_m

t1_iterable = ((locs['k'])[i] * i1(locs['alpha'][i]*locs['rad'][i]) /
               (par['km']*par['am']*locs['alpha'][i]*i0((locs['alpha'][i]*locs['rad'][i])))
               for i in range(n))
t1 = numpy.fromiter(t1_iterable,float)

t2_iterable = ((((locs['k'])[i]*locs['rad'][i])/(2.0*par['am']*par['km']))*
               (1.0 - (pow(i1(locs['alpha'][i]*locs['rad'][i]),2)/
                       pow(i0(locs['alpha'][i]*locs['rad'][i]),2)))
               for i in range(n))
t2 = numpy.fromiter(t2_iterable,float)

t2star_iterable = (((par['km']*locs['rad'][i])/(2.0*par['am']*par['km']))*
                   (1.0 - (pow(i1(par['alpha_m']*locs['rad'][i]),2)/
                           pow(i0(par['alpha_m']*locs['rad'][i]),2)))
                   for i in range(n))
t2star = numpy.fromiter(t2star_iterable,float)

t4_iterable = ((pow(par['alpha_m'],2)*locs['rad'][i]*
                  (pow(k1(par['alpha_m']*locs['rad'][i]),2) -
                   pow(k0(par['alpha_m']*locs['rad'][i]),2)))/
                 (2.0*par['cm']*pow(k0(par['alpha_m']*locs['rad'][i]),2))
               for i in range(n))
t4 = numpy.fromiter(t4_iterable,float)

## F_i, tm

xinv = inv(xm)

Dt1 = t1*numpy.identity(n)

tm = xinv.dot(Dt1)

rm = numpy.zeros([n,n])
for i in range(n):
    for j in range(n):
        rm[i,j] = tm[i,j]/tm[j,j]


fi = numpy.zeros([n])
for i in range(n):
    fi[i] = t1[i]/xm[i,i]

fm = numpy.zeros([n,n])
for i in range(n):
    for j in range(n):
        if(i==j):
            fm[i,j] = 0.0
        else:
            fm[i,j] = t2[i]/xm[i,i]

f0_iterable = ((t1[i] - t2[i] + fii[i]*t2[i])/(xm[i,i]*fii[i]) for i in range(n))
f0 = numpy.fromiter(f0_iterable,float)

iimp = inv(numpy.eye(n)-pm)

vm = (1.0-numpy.eye(n))*((iimp.dot((pm*fm).dot(rm)))/rm)

## Capital functions

def P1fun(rad,s):
    res = (k0(par['alpha_m']*(rad + s))/
           k0(par['alpha_m']*rad))
    return res

def p1fun(rad):
    res = (par['alpha_m']*k1(par['alpha_m']*rad)/
           k0(par['alpha_m']*rad))
    return res

def T4fun(rad,s):
    res = integrate.quad(lambda x: x*(k1(x)*k1(x)-k0(x)*k0(x))/(2.0*par['cm']*k0(x)*k0(x)),
                         par['alpha_m']*rad,par['alpha_m']*(rad+s))
    return res[0]

def t4fun(rad):
    res = ((pow(par['alpha_m'],2)*rad*
            (pow(k1(par['alpha_m']*rad),2)-pow(k0(par['alpha_m']*rad),2)))/
           (2.0*par['cm']*pow(k0(par['alpha_m']*rad),2)))
    return(res)
    
## matrices A, Ainv, B, C, Y, Xstar

xmstar = sm + Dp2star

xmstarinv = inv(xmstar)

## confirm.nb contains two definitions of A matrix
a_version = False
am = numpy.zeros([n,n])
if(a_version):
    for i in range(n):
        for j in range(n):
            if(i==j):
                am[i,j] = 1.0
            else:
                am[i,j] = P1fun(locs['rad'][j],
                                dij[i,j] - locs['rad'][j])
else:
    for i in range(n):
        for j in range(n):
            am[i,j] = xmstarinv[i,j]/xmstarinv[j,j]

ainv = inv(am)

bm = numpy.zeros([n,n])
for i in range(n):
    for j in range(n):
        if(i==j):
            bm[i,j] = 0.0
        else:
            bm[i,j] = T4fun(locs['rad'][j],
                            dij[i,j] - locs['rad'][j])

## old version
#bm = numpy.zeros([n,n])
#for i in range(n):
#    for j in range(n):
#        if(i==j):
#            bm[i,j] = 0.0
#        else:
#            bm[i,j] = t4[j]*(dij[i,j] - locs['rad'][j])

## error fixed: not a dot product but elementwise multiplication!
cm = am*bm

ym = numpy.eye(n) * (((numpy.eye(n)-pmstar).dot(cm.dot(ainv)))/ainv)

## matrices Fstar, Gstar

## error fixed: diagonal entries should be zero
fmstar = numpy.zeros([n,n])
for i in range(n):
    for j in range(n):
        if(i==j):
            fmstar[i,j] = 0.0
        else:
            fmstar[i,j] = t2star[i]/xmstar[i,i]

gmstar = (((cm - ym - pmstar.dot(cm)).dot(ainv))/(numpy.eye(n) + pmstar)) - fmstar

## vector g_i

## just slightly different values in confirm.nb
## calculated differently, perhaps without nint or P1 other way
g = numpy.zeros([n])
for i in range(n):
    tsum = 0.0
    bsum = 0.0
    P1value = 0.0
    T4value = 0.0
    for j in range(n):
        if(j != i):
            P1value = P1fun(locs['rad'][i],
                            dij[i,j]-locs['rad'][i])
            T4value = T4fun(locs['rad'][i],
                            dij[i,j]-locs['rad'][i])
            tsum += sm[i,j]*P1value*(gmstar[i,j] + T4value)
            bsum += sm[i,j]*(P1value/xmstar[i,i])
    g[i] = (t4[i] + tsum)/(1.0 + bsum)

## old version
#g = numpy.zeros([n])
#for i in range(n):
#    tsum = 0.0
#    bsum = 0.0
#    for j in range(n):
#        if(j != i):
#            tsum += sm[i,j]*((1.0-(dij[i,j]-locs['rad'][i])*p1[i])*
#                               (gmstar[i,j]+t4[i]*(dij[i,j]-locs['rad'][i])))
#            bsum += sm[i,j]*((1.0-(dij[i,j]-locs['rad'][i])*p1[i])/
#                               xmstar[i,i])
#    g[i] = (t4[i] + tsum)/(1.0 + bsum)

# matrices small g_i,j, G

## this is called gil in confirm.nb
## fixed error: should divide by xmstar[i,i] not xmstar[i,j]
## values are not exactly the same - because of g[i] differences
smallgm = numpy.zeros([n,n])
for i in range(n):
    for j in range(n):
        smallgm[i,j] = gmstar[i,j] - (g[i]/xmstar[i,i])

## values are not exactly the same: g[i] and smallgm[i,j]
gm = numpy.zeros([n,n])
for i in range(n):
    for j in range(n):
        if(i==j):
            gm[i,j] = 0.0
        else:
            gm[i,j] = (g[i]/xm[i,i]) + smallgm[i,j]

um = (1-numpy.eye(n))*((iimp.dot((pm*gm).dot(rm)))/rm)
            
## G0

def T6fun(i,x):
    betah = (locs['a'])[i] * (locs['k'])[i] * locs['alpha'][i]
    betam = par['am'] * par['km'] * par['alpha_m']
    a = numpy.zeros([2,2])
    a[0,0] = i0(locs['alpha'][i] * locs['rad'][i])
    a[1,0] = betah*i1(locs['alpha'][i] * locs['rad'][i])
    a[0,1] = -k0(par['alpha_m']*locs['rad'][i])
    a[1,1] = betam*k1(par['alpha_m']*locs['rad'][i])
    ## b is markedly different in zheng2009 and confirm.nb
    ## this is now the confirm.nb version
    b = numpy.zeros([2,1])
    b[0] = (1.0/par['cm']) - (locs['rad'][i] *
                              k0(locs['alpha'][i]*locs['rad'][i]) *
                              i1(locs['alpha'][i] * locs['rad'][i]) /
                              ((locs['a'])[i] * locs['alpha'][i]))
    b[1] = ((locs['k'])[i] * locs['rad'][i] *
            k1(locs['alpha'][i] * locs['rad'][i]) *
            i1(locs['alpha'][i] * locs['rad'][i]))
    const = (inv(a)).dot(b)
    qh = ((1.0/(locs['c'])[i])*(1.0-locs['alpha'][i]*locs['rad'][i] *
                              i0(locs['alpha'][i] * x) *
                              k1(locs['alpha'][i]*locs['rad'][i])))
    if(x < locs['rad'][i]):
        return const[0]*i0(locs['alpha'][i]*x) + qh
    else:
        return const[1]*k0(par['alpha_m']*x) + (1.0/par['cm'])

t6m = numpy.zeros([n,n])
for i in range(n):
    for j in range(n):
        if(i==j):
            t6m[i,j] = T6fun(i,locs['rad'][i])
        else:
            t6m[i,j] = T6fun(i,dij[i,j])

t6_iterable = (t6m[i,i] for i in range(n))
t6 = numpy.fromiter(t6_iterable,float)

## again the same: values close but not exactly the same
g0 = (1.0/fii)*(t6 - numpy.sum((pm*(fm + gm + t6m)),axis=1)) - f0

gi = numpy.sum(pm*gm,axis=1) + fii*g0

li = (inv(numpy.eye(n) - pm)).dot(fi+gi)

out = {'N': n,
       'P': pm.tolist(),
       'Phi': fii.tolist(),
       'F': fm.tolist(),
       'F0': f0.tolist(),
       'G': gm.tolist(),
       'G0': g0.tolist()}

with open(args.outfile,'w') as outfile:
    json.dump(out,outfile)

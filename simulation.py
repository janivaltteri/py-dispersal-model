import json
import math
import heapq
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='')
parser.add_argument('--par', '-p', required=True, help='input file name')
parser.add_argument('--disp', '-d', required=True, help='landscape file name')
parser.add_argument('--seed', '-s', required=True, help='random seed', type=int)
args = parser.parse_args()

np.random.seed(args.seed)

parfile = args.par
with open(parfile) as jfile:
    jstr = jfile.read()
    par = json.loads(jstr)

dispfile = args.disp
with open(dispfile) as jfile:
    jstr = jfile.read()
    d = json.loads(jstr)

pm = np.array(d['P'])
phi = np.array(d['Phi'])
fm = np.array(d['F'])
f0 = np.array(d['F0'])
gm = np.array(d['G'])
g0 = np.array(d['G0'])

## pm : probability of hitting patch j next given that an individual
## is at patch i now
## (do not add up to 1.0 since one could die)

## phi : probability of dying before hitting another patch 

## fm : time that an individual is expected to spend in patch i
## (is independent of its destination patch j)

## f0 : time that an individual is going to spend in patch i
## given that it dies before hitting another patch

## gm : time that an individual is going to spend in the matrix
## given that its destination patch is j

## g0 : time that an individual is going to spend in the matrix
## given that it dies before hitting another patch

def normalise(v):
    """ used to normalise the event_props vector """
    summa = v.sum()
    if summa > 0.0:
        return v / summa
    else:
        print('error: normalise received a zero sum vector')
        return v

class individual:
    """ individual butterfly """

    def __init__(self,ii,ll,dd,dbnp,nat,sex):
        self.index = ii
        self.in_a_patch = True ## all individuals start in a patch
        self.departure_patch = ll
        self.destination_patch = dd
        self.dies_before_next_patch = dbnp ## boolean value
        self.next_action_time = nat
        self.is_female = sex ## boolean value
        self.has_mated = False ## all individuals start non mated

    def can_mate(self):
        if(self.is_female & self.in_a_patch & (not self.has_mated)):
            return True
        else:
            return False

    ## FIXME: should test whether the female has already laid eggs on the same day
    def can_lay(self):
        if(self.is_female & self.has_mated & self.in_a_patch):
            return True
        else:
            return False

    def __lt__(self,other):
        return self.next_action_time < other.next_action_time

class clutch:
    """ a clutch of eggs """

    def __init__(self,pp):
        self.patch = pp

    def __str__(self):
        return "a clutch at patch "+str(self.patch)
    
class writer:
    """ handles all the output """

    def __init__(self,filename,sp):
        self.eventlog = open(filename,'w')
        self.screenprint = sp

    def close(self):
        self.eventlog.close()

    def initial(self,ind,patch,destination):
        self.eventlog.write('0.0 '+str(ind)+' '+str(0)+' '+
                            str(patch)+' '+str(destination)+'\n')

    def in_patch_dies_before_next(self,a):
        if self.screenprint:
            print(' t '+str(round(a.next_action_time,3))+
                  ' i '+str(a.index)+
                  ' f '+str(a.departure_patch)+' -> d')
        self.eventlog.write(str(a.next_action_time)+' '+str(a.index)+
                            ' '+str(1)+' '+str(a.departure_patch)+' -2\n')

    def in_patch_enters_matrix(self,a):
        if self.screenprint:
            print(' t '+str(round(a.next_action_time,3))+
                  ' i '+str(a.index)+
                  ' f '+str(a.departure_patch)+
                  ' -> m ( -> ' + str(a.destination_patch) + ' )')
        self.eventlog.write(str(a.next_action_time)+' '+str(a.index)+
                            ' '+str(2)+' '+str(a.departure_patch)+' '+
                            str(a.destination_patch)+'\n')

    def enters_patch(self,a):
        if self.screenprint:
            print(' t '+str(round(a.next_action_time,3)) +
                  ' i '+str(a.index) +
                  ' f m -> '+str(a.departure_patch))
        self.eventlog.write(str(a.next_action_time)+' '+str(a.index)+
                            ' '+str(3)+' '+str(a.departure_patch)+' '+
                            str(a.destination_patch)+'\n')

    def enters_patch_dies_next(self,a):
        if self.screenprint:
            print(' t '+str(round(a.next_action_time,3)) +
                  ' i '+str(a.index) +
                  ' f m -> '+str(a.departure_patch))
        self.eventlog.write(str(a.next_action_time)+' '+str(a.index)+
                            ' '+str(4)+' '+str(a.departure_patch)+' -1\n')
    
## initialisations

wr = writer("out-events.txt",True)

n = 30
dt = 0.2
generations = 5
num_patches = len(par["areas"])

indivs = []
popcounter = 0
for i in range(n):
    initpatch = (np.random.choice(num_patches,1))[0]
    diesnext = np.random.uniform() < phi[initpatch]
    if diesnext:
        destination = None
        move_time= np.random.exponential(f0[initpatch])
    else:
        dest_probs = normalise(np.squeeze(pm[initpatch,:]))
        destination = (np.random.choice(num_patches,1,p=dest_probs))[0]
        move_time = np.random.exponential(fm[initpatch,destination])
    sex_female = np.random.uniform() < 0.5
    heapq.heappush(indivs,
                   individual(popcounter,
                              initpatch,
                              destination,
                              diesnext,
                              move_time,
                              sex_female))
    if(destination == None):
        wr.initial(popcounter,initpatch,-1)
    else:
        wr.initial(popcounter,initpatch,destination)
    popcounter += 1

## should work like this:
## progress in discretised steps dt
##  when we are at step t
##   check if the next event in movement queue should happen before t + dt
##    process events from the movement queue until the next event is not within t + dt
##   check for placing eggs within dt
##   check for matings within dt

## initialise some variables and set time increment
time = 0.0
step = 0
clutches = []
live = len(indivs)

## single generation simulation loop
for i in range(generations):
    while len(indivs) > 0:
        step += 1
        time += dt
        ## handle movements
        if indivs[0].next_action_time < time: # test if any move within dt
            while indivs[0].next_action_time < time: # if yes -> loop until no
                ## resolve acting individual
                acting = heapq.heappop(indivs)
                nt = acting.next_action_time
                depp = acting.departure_patch
                desp = acting.destination_patch
                dies = False
                if acting.in_a_patch:
                    if acting.dies_before_next_patch:
                        ##do nothing but print
                        wr.in_patch_dies_before_next(acting)
                        dies = True
                        live -= 1
                    else:
                        wr.in_patch_enters_matrix(acting)
                        acting.in_a_patch = False # go to matrix
                        acting.next_action_time = (nt +
                                                   np.random.exponential(gm[acting.departure_patch,
                                                                            acting.destination_patch]))
                else:
                    acting.in_a_patch = True # go to a patch
                    acting.departure_patch = acting.destination_patch
                    diesnext = np.random.uniform() < phi[acting.departure_patch]
                    if diesnext:
                        acting.destination_patch = None
                        move_time = np.random.exponential(f0[acting.departure_patch])
                        wr.enters_patch_dies_next(acting)
                    else:
                        dest_probs = normalise(np.squeeze(pm[acting.departure_patch,:]))
                        destination = (np.random.choice(num_patches,1,p=dest_probs))[0]
                        acting.destination_patch = destination
                        move_time = np.random.exponential(fm[acting.departure_patch,destination])
                        wr.enters_patch(acting)
                    acting.next_action_time = nt + move_time
                    acting.dies_before_next_patch = diesnext
                if not dies:
                    heapq.heappush(indivs,acting)
                if len(indivs) == 0:
                    break
        ## handle egg placement
        for j in range(live):
            if indivs[j].can_lay():
                laying_prob = 1.0 - math.exp(-par['lambda_l']*dt)
                if np.random.uniform() < laying_prob:
                    clutches.append(clutch(indivs[j].departure_patch))
                    print(str(j) + ' egg laying occurred')
        ## handle mating
        for j in range(live): # loop across all
            if indivs[j].can_mate():
                ## count males in the same patch
                malecounter = 0
                for k in range(live):
                    if((not indivs[k].is_female) &
                       indivs[k].in_a_patch &
                       (indivs[k].departure_patch == indivs[j].departure_patch)):
                        malecounter += 1
                mating_prob = 1.0 - math.exp(-par['lambda_m']*malecounter*dt)
                if np.random.uniform() < mating_prob: ## resolve mating
                    ## should choose a male for getting the alleles
                    print(str(malecounter) + ' mating occurred')
                    indivs[j].has_mated = True
        ##print('t ' + str(round(time,2)))
    print('the clutches are at ',end='')
    for i in range(len(clutches)):
        print(str(clutches[i].patch)+' ',end='')
    print('')
    if len(clutches) < 1:
        print('extinction!')
        break
    indivs = []
    popcounter = 0
    for i in range(len(clutches)):
        initpatch = clutches[i].patch
        diesnext = np.random.uniform() < phi[initpatch]
        if diesnext:
            destination = None
            move_time= np.random.exponential(f0[initpatch])
        else:
            dest_probs = normalise(np.squeeze(pm[initpatch,:]))
            destination = (np.random.choice(num_patches,1,p=dest_probs))[0]
            move_time = np.random.exponential(fm[initpatch,destination])
        sex_female = np.random.uniform() < 0.5 ## this should be determined in the clutch
        heapq.heappush(indivs,
                       individual(popcounter,
                                  initpatch,
                                  destination,
                                  diesnext,
                                  move_time,
                                  sex_female))
        if(destination == None):
            wr.initial(popcounter,initpatch,-1)
        else:
            wr.initial(popcounter,initpatch,destination)
        popcounter += 1
    time = 0.0
    step = 0
    clutches = []
    live = len(indivs)

print("done!")

wr.close()

import sys
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import *
import factoradic as fc
from fractions import Fraction
import math
from sklearn.cluster import k_means
from collections import Counter
import csv

from torch import true_divide

count = 0
if(len(sys.argv)<2):
    print("\nEnter your desired pancake stack size:\n")
    count = input()
else:
    count = sys.argv[1]

notInt=1
while(notInt):
    try:
        count = int(count)
        notInt = 0
    except ValueError:
        print("\n[Requires an integer] Enter your desired pancake stack size:\n")
        count = input()
# print([x for x in range(count)])
maxValue = fc.from_factoradic([x for x in range(count)])
# print(maxValue)
maxRank = (maxValue+1)*pow(2,count)-1
print("\nMatrix indices range from 0 to",maxRank)

def lehmer_to_perm(factoradic):
    permutation = factoradic.copy()
    for i in range(len(permutation)-1,-1,-1):
        for j in range(i+1,len(permutation)):
            if(permutation[j]>=permutation[i]):
                permutation[j]+=1
        # print(permutation)
    # print()
    return permutation
    
def lehmer_from_perm(permutation):
    factoradic = permutation.copy()
    for i in range(len(factoradic)):
        for j in range(i+1,len(factoradic)):
            if(factoradic[j]>factoradic[i]):
                factoradic[j]-=1
        # print(factoradic)
    # print()
    return factoradic

def pancake_flip(stack_size,stack_value,print_stack=False):
    sign,permutation = divmod(stack_value,(maxValue+1))
    sign = format(sign,'b')
    sign = "0"*(count-len(sign))+sign
    permutation = fc.to_factoradic(permutation)
    permutation = lehmer_to_perm(list(reversed(permutation + (count-len(permutation))*[0])))
    if(print_stack):
        print(stack_value," = ",[x+1 if sign[x]=='0' else -(x+1) for x in permutation])
    permutation = list(reversed(permutation[:stack_size]))+permutation[stack_size:]
    sign = ''.join(['1' if i == '0' else '0' for i in list(reversed(sign[:stack_size]))])+''.join(sign[stack_size:])
    sign = int(sign,2)
    return sign*(maxValue+1)+fc.from_factoradic(list(reversed(lehmer_from_perm(permutation))))

def print_stack(stack_value):
    sign,permutation = divmod(stack_value,(maxValue+1))
    sign = format(sign,'b')
    sign = "0"*(count-len(sign))+sign
    permutation = fc.to_factoradic(permutation)
    permutation = lehmer_to_perm(list(reversed(permutation + (count-len(permutation))*[0])))
    print(stack_value," = ",[(permutation[x]+1 if sign[x]=='0' else -(permutation[x]+1)) for x in range(count)])

def make_rational(dec,acc=100):
    positive = 1
    if dec == 0: return 0
    if dec < 0:
        positive = -1
    newdec = dec*positive
    series = []
    f = Fraction(0)
    while(np.product([x+1 for x in series])<pow(acc,2)):
        if(len(series)>=1):
            if(newdec>acc):
                for val in range(len(series)-1,0,-1):
                    f += series[val]
                    f = Fraction(1/f)
                f += series[0]
                return f*positive
        series.append(math.floor(newdec))
        if(newdec == math.floor(newdec)):
            for val in range(len(series)-1,0,-1):
                f += series[val]
                f = Fraction(1/f)
            f += series[0]
            return f*positive
        newdec = 1/(newdec-math.floor(newdec))
    return np.round(dec,5)

def cluster(evalues):
    best_centroid = []
    best_label = []
    prev_inertia = 0
    prev_gain = 0
    matrix = np.array(evalues).reshape(-1,1)
    top_i = 0
    for i in range(pow(2,count)*math.factorial(count-2),pow(2,count+1)*math.factorial(count-2), math.floor(math.sqrt(pow(2,count-1)*math.factorial(count-2)))):
        gain = 0
        _, _, inertia = k_means(matrix,i,algorithm="elkan")
        if i>pow(2,count-1)*math.factorial(count-2):
            gain = prev_inertia/inertia
            # print(gain)
            if 2*gain < prev_gain:
                # print("ding!")
                break
        prev_inertia = inertia
        prev_gain = gain
        top_i = i
    prev_gain = 0
    prev_inertia = 0
    for i in range(top_i-math.floor(math.sqrt(pow(2,count-1)*math.factorial(count-2)))-1,top_i+1):
        gain = 0
        centroid, label, inertia = k_means(matrix,i,algorithm="elkan")
        if i>pow(2,count-1)*math.factorial(count-2):
            gain = prev_inertia/inertia
            # print(gain)
            if 2*gain < prev_gain:
                # print("ding!")
                break
        best_centroid = centroid
        best_label = label
        prev_inertia = inertia
        prev_gain = gain
    return best_centroid,best_label

def summary(evalues):
    centroid,label = cluster(evalues)
    counts = Counter(label)
    sortable = []
    for i in range(len(centroid)):
        sortable.append([centroid[i],counts[i]])
    stay_summary = True
    while(stay_summary):
        print("\nSort eigenvalues by 's'/'d' (size/degree), 'a'/'d' (ascending/descending), 'r'/'d'  \n   (rational approx. [only converts to rational when accurate]/decimal approx.), or \n   anything else to exit:\nFor example, 'ddd' sorts by degree, descending, in decimal.\n")
        inp = input()
        if(len(inp)<3): break
        if(inp.lower()[:2] == 'sa'):
            sortable.sort(key=lambda x:x[0],reverse=False)
        elif(inp.lower()[:2] == 'sd'):
            sortable.sort(key=lambda x:x[0],reverse=True)
        elif(inp.lower()[:2] == 'da'):
            sortable.sort(key=lambda x:x[1],reverse=False)
        elif(inp.lower()[:2] == 'dd'):
            sortable.sort(key=lambda x:x[1],reverse=True)
        else: stay_summary = False
        if(inp.lower()[2] == 'r'):
            print("\nEigenvalues: ",end="")
            for i in range(len(sortable)-1):
                print("{} (x{}),  ".format(make_rational(sortable[i][0]),sortable[i][1]),end="")
            print("{} (x{})".format(make_rational(sortable[len(centroid)-1][0]),sortable[len(centroid)-1][1]))
        elif(inp.lower()[2] == 'd'):
            print("\nEigenvalues: ",end="")
            for i in range(len(sortable)-1):
                print("{} (x{}),  ".format(np.round(sortable[i][0],5),sortable[i][1]),end="")
            print("{} (x{})".format(np.round(sortable[len(centroid)-1][0],5),sortable[len(centroid)-1][1]))
        else: stay_summary = False

matrix = sparse.dok_matrix((maxRank+1,maxRank+1), dtype=np.int8)

for i in range(maxRank+1):
    for j in range(count+1):
        matrix[i,pancake_flip(j,i)] = 1

def to_csv():
    with open('data/burntpancake_'+str(count)+'.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['x', 'y', 'value'])
        # print(matrix.getnnz())
        for i in matrix.items():
            ((a,b),_) = i
            writer.writerow([a, b, 1])
    with open('data/burntpancake_eigen_'+str(count)+'.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['eigenvalue', 'multiplicity', 'estimated rational'])
        eigenvalues,_ = eigsh(matrix.tocoo().asfptype(),maxRank)
        centroid,label = cluster(eigenvalues)
        counts = Counter(label)
        sortable = []
        for i in range(len(centroid)):
            sortable.append([centroid[i],counts[i]])
        sortable.sort(key=lambda x:x[0],reverse=True)
        for i in range(len(sortable)):
            writer.writerow([sortable[i][0],sortable[i][1],make_rational(sortable[i][0])])

while(True):
    print("\nEnter an index to check its graph adjacencies, 'q' to quit, 'e X' to return X large \n   eigenvalues, 's' for summarized eigenvalue results, or 'csv' to export to csv:\n")
    ind = input()
    if ind.strip()=='' or ind.strip()=='e' or ind.strip()=="eigenvalues": continue
    try:
        if ind.lower() == 'csv':
            to_csv()
        elif ind.lower() == 's' or ind.lower() == 'summary':
            eigenvalues,_ = eigsh(matrix.tocoo().asfptype(),maxRank)
            summary(eigenvalues)
        elif ind.split()[0].lower() == 'e' or ind.split()[0].lower() == "eigenvalues":
            val = int(ind.split()[1])
            eigenvalues,eigenvectors = eigsh(matrix.tocoo().asfptype(),k=val)
            eigenvalues = [round(elem,5) for elem in eigenvalues]
            print("Eigenvalues = ", eigenvalues)
            print("\nWould you like to view rational approximations of these (only converts when accurate)?")
            ans = input()
            if ans.lower() == 'y' or ans.lower() == 'yes':
                # print(make_rational(math.pi))
                # print(Fraction(0.125))
                print("Eigenvalues = ",end='')
                for x in range(len(eigenvalues)-1):
                    print(make_rational(eigenvalues[x]),end=', ')
                print(make_rational(eigenvalues[-1]))
        else:
            ind = int(ind)
            if ind > maxRank:
                print("Value must be a valid integer.")
            else:
                for j in range(count+1):
                    print_stack(pancake_flip(j,ind))
    except ValueError:
        if(ind.lower() == 'q' or ind.lower() == 'quit'):
            sys.exit("Quit successfully.")
        print("Value must be a valid integer.")


# 2 8 - - 4
# 6 48 - - 8
# 12 384 - - 32
# 20 3840 - - 192

# 2*4*6*8*10
# 2*2*2*4*6
# 1 2 3 4 5
# 2^n *(n-2)!
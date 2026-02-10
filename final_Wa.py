import os, sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, Delaunay
from scipy.spatial.distance import pdist, squareform
from matplotlib.path import Path
import copy
import random
from difflib import ndiff
from itertools import groupby, permutations
import pickle
import gc

dir1 = r'C:\Users\praven.kumar\OneDrive - Cardinal Health\Desktop\HMM\SCSE_Run_2_V3\data'

dat = pd.read_csv(dir1 + r"\dat.csv")

# dat = pd.read_csv(dir1 + r"\full_data_v1.csv")

# columns_to_drop = ['Sold_To_Customer', 'C_SHIP_TO_KUNNR']

# dat.drop(columns=columns_to_drop, axis=1, inplace=True)

# dat.columns = ['shipto','SIG','sales','lvl3_name','lvl4_name','lvl5_name','city','ship_to_customer_name','state','address','lvl3','lvl4','lvl5','zip','lat','lon']

dat_zip = pd.read_excel(dir1 + r"\sales_rep_data_v3.xlsx")

SIG_names_2 = pd.read_csv(dir1 + r"\SCSE_withnames_V3.csv",index_col=False)

SIG_names_2 = SIG_names_2.sort_values(by='Entpr_ID_5')

SIG_names_2['lvl3_lvl_5_SIG'] = SIG_names_2['SIG'] + "_" + SIG_names_2['Entpr_ID_3'] +  "_" +SIG_names_2['Entpr_ID_5']

SIG_names_2["SIG_lvl5_int"] = (SIG_names_2.lvl3_lvl_5_SIG.astype('category')).cat.codes

SIG_names_2 = SIG_names_2.drop(['Entpr_ID_4'], axis =1)

dat_zip = pd.merge(dat_zip,SIG_names_2,left_on="Entpr_ID_5",right_on="Entpr_ID_5",how="left")

dat_zip = dat_zip.drop(['lvl3_lvl_5_SIG'], axis =1)

# dat_zip = dat_zip.drop(['LEVEL3_HRCHY_NODE_ID_y'], axis =1)

# dat_zip = dat_zip.rename(columns={'LEVEL3_HRCHY_NODE_ID_x': 'LEVEL3_HRCHY_NODE_ID'})

dat_zip = dat_zip.rename(columns={'lat_new': 'lat'})

dat_zip = dat_zip.rename(columns={'lon_new': 'lon'})

dat_zip['SIG_lvl5_int'] = dat_zip['SIG_lvl5_int'].astype('Int64')

n_lvl5_unq = list(pd.unique(dat_zip.SIG_lvl5_int))

n_lvl5 = len(n_lvl5_unq)

# dat['lvl3_lvl_5_SIG'] = dat['SIG'] + "_" + dat['lvl3'] +  "_" +dat['lvl5']

# dat = pd.merge(dat,SIG_names_2,left_on="lvl3_lvl_5_SIG",right_on="lvl3_lvl_5_SIG",how="left")

# dat = dat.reset_index(drop=True)

# dat = dat.rename(columns={'SIG_x': 'SIG'})

# dat = dat.drop_duplicates(subset=['shipto'], keep='first')

# dat = dat.drop(['Entpr_ID_3'], axis =1)
# dat = dat.drop(['Entpr_ID_5'], axis =1)
# dat = dat.drop(['lvl3_lvl_5_SIG'], axis =1)
# dat = dat.drop(['SIG_y'], axis =1)

# dat = dat.rename(columns={'SIG_lvl5_int': 'lvl5_int'})

# dat['lvl5_int'] = dat['lvl5_int'].astype('Int64')
# dat["shipto_int"] = (dat.shipto.astype('category')).cat.codes

# dat_2 = dat[~dat['lvl5_int'].isin(n_lvl5_unq)]

# dat_2 = dat_2.reset_index(drop=True)

# dat = dat[dat['lvl5_int'].isin(n_lvl5_unq)]

# dat = dat.dropna(subset=['lat'])

# dat = dat.reset_index(drop=True)

# no_sig = list(pd.unique(dat_2.lvl5))

# yes_sig = list(pd.unique(dat.lvl5))

# dat = dat.sample(frac=1).reset_index(drop=True)

# dat.to_csv(dir1 + r"\dat.csv",index = False)

# dat_2.to_csv(dir1 + r"\dat_2.csv",index = False)

# dat = pd.read_csv(dir1 + r"\dat_wa.csv")

# dat_2 = pd.read_csv(dir1 + r"\dat_2.csv")

column_array = dat['lvl5_int'].to_numpy()

print(column_array)

def round_list_values(float_list):
    return [int(x) for x in float_list]

region_dists_a = []
for i in n_lvl5_unq:
    sum_dist = 0.0
    sum_dist += np.sum(squareform(
        pdist(np.vstack([dat_zip.loc[np.where(dat_zip == i)[0], ["lat", "lon"]].to_numpy()[0], dat.loc[np.where(column_array == i)[0], ["lat", "lon"]].to_numpy()])))[0, 1:])
    region_dists_a.append(sum_dist)

rounded_list = round_list_values(region_dists_a)
rounded_list = np.array(rounded_list)


region_dists_avg = []
for i in n_lvl5_unq:
    sum_dist_avg = 0.0
    sum_dist_avg += np.mean(squareform(
        pdist(np.vstack([dat_zip.loc[np.where(dat_zip == i)[0], ["lat", "lon"]].to_numpy()[0], dat.loc[np.where(column_array == i)[0], ["lat", "lon"]].to_numpy()])))[0, 1:])
    region_dists_avg.append(sum_dist_avg)

rounded_list_avg = round_list_values(region_dists_avg)
rounded_list_avg = [x * 100 for x in rounded_list_avg]
rounded_list_avg = np.array(rounded_list_avg)
rounded_list_avg_10 = [(x * 10)/100 for x in rounded_list_avg]
rounded_list_avg_10 = round_list_values(rounded_list_avg_10)


actual_array = np.array(dat["lvl5_int"])
dat_lvl5 = dat[["lvl5","lvl5_int"]]
dat_lvl5 = dat_lvl5.drop_duplicates(subset='lvl5_int', keep="first")

SIG_LVL5_LVL3 = SIG_names_2[["Entpr_ID_5","Entpr_ID_3","SIG","SIG_lvl5_int"]]
SIG_LVL5_LVL3 = SIG_LVL5_LVL3.dropna(subset=['Entpr_ID_5'])
# SIG_LVL5_LVL3 = SIG_LVL5_LVL3.drop_duplicates(subset=['Entpr_ID_5'], keep='first')

with open('C:/Users/praven.kumar/OneDrive - Cardinal Health/Desktop/HMM/SCSE_Run_2_V3/data/lvl3tolvl5_dict_v4.pkl', 'rb') as f:
    SIG_names_3 = pickle.load(f)

final_rows_group = []
for g in range(0, len(SIG_names_3)):
    temp_SIG = SIG_names_3['rows_group'].iloc[g]
    temp_list = list(map(lambda x : x['SIG_lvl5_int'], temp_SIG))
    unique_list = list(set(temp_list))
    final_rows_group.append(unique_list)

final_rows_group = [item for sublist in final_rows_group for item in sublist]
list2 = list(dat_zip["SIG_lvl5_int"])
final_rows_group = [item for item in final_rows_group if item in list2]

POPULATION_SIZE = 100
CROSSOVER_PROBABILITY = 0.6
MUTATION_PROBABILITY = 0.4
MAX_GENERATIONS = 10000

from scipy import stats

low = 0.325
high = 0.625
mean = 0.25
stddev = 0.5

numbers = stats.truncnorm.rvs(low, high,
                             loc = mean, scale = stddev,
                             size = POPULATION_SIZE)


class Individual:
    counter = 0

    @classmethod
    def set_fitness_function(cls, fun):
        i = 0
        cls.fitness_function = fun

    @classmethod
    def generate_random(cls):
        # new_grid = np.random.choice(range(n_lvl5),n_shipto)
        # i=0
        new_grid = list(range(n_lvl5))
        new_grid = new_grid * int(np.ceil(n_shipto / n_lvl5))
        # new_grid = new_grid * int(np.ceil(n_soldto / n_lvl5))
        new_grid = np.array(new_grid[:n_shipto])
        # new_grid = np.array(new_grid[:n_soldto])
        np.random.shuffle(new_grid)
        x = pd.DataFrame(new_grid)
        x.columns = ["col1"]
        # x_2 = x.loc[x["col1"] != -1]
        i = Individual.counter
        j = __class__.counter
        if i>98:
            i=98
        # num = round(len(x) / 2)
        num = round(len(new_grid)*round(numbers[i],2))
        x_3 = x.sample(n=num)
        x_3.loc[x_3.index, 'col1'] = dat['lvl5_int']
        x.loc[x_3.index, 'col1'] = x_3['col1']
        new_grid = x['col1'].to_numpy()

        return Individual(new_grid)
    
    
    @classmethod
    def generate_random_2(cls):
        new_grid = dat['lvl5_int']
        return Individual(new_grid)
    

    def __init__(self, gene_list) -> None:
        self.gene_list = gene_list
        self.fitness = self.__class__.fitness_function(self.gene_list)
        self.__class__.counter += 1


def selection_rank_with_elite(individuals, elite_size=7):
    sorted_individuals = sorted(individuals, key=lambda ind: ind.fitness[3], reverse=True)
    rank_distance = 1 / len(individuals)
    ranks = [(1 - i * rank_distance) for i in range(len(individuals))]
    ranks_sum = sum(ranks)
    selected = sorted_individuals[0:elite_size]
    val = 0
    for i in range(len(sorted_individuals) - elite_size):
        shave = random.random() * ranks_sum
        rank_sum = 0
        for i in range(len(sorted_individuals)):
            rank_sum += ranks[i]
            if rank_sum > shave:
                if i <= 10:
                    val += 1
                selected.append(sorted_individuals[i])
                break

    return selected, val

def makeWheel(population):
    wheel = []
    total = sum(p.fitness[3] for p in population)
    top = 0
    for p in population:
        f = p.fitness[3]/total
        wheel.append((top, top+f, p))
        top += f
    return wheel

def binSearch(wheel, num):
    mid = len(wheel)//2
    low, high, answer = wheel[mid]
    if low<=num<=high:
        return answer, mid
    elif high < num:
        return binSearch(wheel[mid+1:], num)
    else:
        return binSearch(wheel[:mid], num)

def select_wheel(wheel, N):
    stepSize = 1.0/N
    answer = []
    r = random.random()
    ans1, mid1 = binSearch(wheel, r)
    answer.append(ans1)

    while len(answer) < N:
        r += stepSize
        if r>1:
            r %= 1

        ans2, mid2 = binSearch(wheel, r)
        answer.append(ans2)
    return answer, mid1, mid2


def swap(vals1, vals2, n):
    n_qtr = round(n/4)
    swap_idxs = np.random.randint(low=0, high=len(dat), size=(4,))
    
    while swap_idxs[1]-swap_idxs[0] < 25 or swap_idxs[2]-swap_idxs[1] < 25 or swap_idxs[3]-swap_idxs[2] < 25:
        swap_idxs = np.random.randint(low=0, high=len(dat), size=(4,))
        swap_idxs.sort()
    
    for i in swap_idxs:
        new = i+n_qtr
        swap_idxs = np.append(swap_idxs, new)
    
    swap_idxs.sort()
    
    swap_on = False
    for i in range(len(vals1)):
        if swap_on:
            if i in swap_idxs:
                swap_on = False
            else:
                vals1[i], vals2[i] = vals2[i], vals1[i]
        elif not swap_on and i in swap_idxs:
            swap_on = True

    return vals1, vals2


def crossover_n_point(p1, p2, n):
    c1, c2 = copy.deepcopy(p1), copy.deepcopy(p2)

    idx = np.random.choice(range(p1.shape[0]), n, replace=False)

    for i in range(0, n):
        c1[idx[i]] = p2[idx[i]]
        c2[idx[i]] = p1[idx[i]]

    return [c1, c2]


def crossover(parent1, parent2):
    child1_genes, child2_genes = crossover_n_point(parent1.gene_list, parent2.gene_list, 500)
    return Individual(child1_genes), Individual(child2_genes)


def crossover_n_point_2(p1, p2, n):
    c1, c2 = copy.deepcopy(p1), copy.deepcopy(p2)

    idx = np.random.choice(range(p1.shape[0]), n, replace=False)

    for i in range(0, n):
        c1[idx[i]] = p2[idx[i]]

    return c1


def crossover_2(parent1, parent2):
    child1_genes = crossover_n_point_2(parent1.gene_list, parent2.gene_list, 750)
    return Individual(child1_genes)


def crossover_operation(population, method, prob):
    crossed_offspring = []
    for ind1, ind2 in zip(population[::2], population[1::2]):
        if random.random() < prob:
            kid1, kid2 = method(ind1, ind2)
            crossed_offspring.append(kid1)
            crossed_offspring.append(kid2)
        else:
            crossed_offspring.append(ind1)
            crossed_offspring.append(ind2)
    return crossed_offspring

def crossover_operation_2(population, current, method, prob):
    crossed_offspring = []
    ind2 = current[0]
    for ind1 in population:
        if random.random() < prob:
            kid1 = method(ind1, ind2)
            crossed_offspring.append(kid1)
        else:
            crossed_offspring.append(ind1)
    return crossed_offspring



def find_outliers_std_dev(data_list, num_std_dev=2):
    """
    Finds outliers in a list that are more than a specified number of 
    standard deviations from the mean.

    Args:
        data_list (list): The input list of numerical data.
        num_std_dev (int or float): The number of standard deviations 
                                     to use as the outlier threshold.

    Returns:
        list: A list containing the identified outliers.
    """
    if not data_list:
        return []

    data_array = np.array(data_list)
    mean = np.mean(data_array)
    std_dev = np.std(data_array)

    lower_bound = mean - (num_std_dev * std_dev)
    upper_bound = mean + (num_std_dev * std_dev)

    outliers = [x for x in data_list if x < lower_bound or x > upper_bound]
    return outliers


def mutation_move(ind):
    mut = copy.deepcopy(ind)
    # idx = np.random.choice(range(mut.shape[0]),1)[0]

    idx_n = np.random.poisson(lam=1.0, size=1)[0] + 100
    reten_idx_n = round((idx_n)*(0.5))
    ch_idxn = idx_n - reten_idx_n
    for q in range(ch_idxn):
        idx = np.random.choice(range(mut.shape[0]), 1)[0]
        idx_lvl3 = dat['lvl3'][idx]
        # print(idx_lvl3)
        idx_lvl5 = dat['lvl5'][idx]
        # temp_SIG = SIG_names_3.loc[SIG_names_3['Entpr_ID_3'] == idx_lvl3]
        # temp_dict_SIG = temp_SIG['rows_group'].iloc[0]
        # temp_list = list(map(lambda x : x['SIG_lvl5_int'], temp_dict_SIG))
        delta = np.random.choice(final_rows_group, 1)[0]
        if mut[idx] == delta:
            delta = np.random.choice([i for i in final_rows_group if i != mut[idx]], 1)[0]
        mut[idx] = delta

    for r in range(reten_idx_n):
        idx = np.random.choice(range(mut.shape[0]), 1)[0]

        delta = dat['lvl5_int'][idx]
        
        mut[idx] = delta
        
    return mut


def mutate(ind):
    mutated_gene = mutation_move(ind.gene_list)
    return Individual(mutated_gene)


def mutation_operation(population, method, prob):
    mutated_offspring = []
    for mutant in population:
        if random.random() < prob:
            new_mutant = method(mutant)
            mutated_offspring.append(new_mutant)
        else:
            mutated_offspring.append(mutant)
    return mutated_offspring


def stats(population, best_ind, fit_avg, fit_best):
    best_of_generation = max(population, key=lambda ind: ind.fitness[3])
    if best_ind.fitness[3] < best_of_generation.fitness[3]:
        best_ind = best_of_generation
    fit_avg.append(sum([ind.fitness[3] for ind in population]) / len(population))
    fit_best.append(best_ind.fitness[3])
    return best_ind, fit_avg, fit_best


def plot_stats(fit_avg, fit_best, title):
    plt.plot(fit_avg, label="Average Fitness of Generation")
    plt.plot(fit_best, label="Best Fitness")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()
    plt.close()

def compare_lists(list1, list2):
    if len(list1) != len(list2):
        return []

    differences = []
    for i in range(len(list1)):
        differences.append(list1[i] - list2[i])
    return differences

def count_greater_than(data_list,rounded_list_avg_10,avg_diff_curr_prop,threshold):
    count = 0
    for i,j,k in zip(data_list,rounded_list_avg_10,avg_diff_curr_prop):
        if i > threshold:
            count += 1
        elif i < threshold and k<j:
            count += 1
    return count


def select(population):
    return selection_rank_with_elite(population, elite_size=7)


# currently only revenue...
def fitness_function(df):
    total_error = 0.0

    df_array = np.array(df)

    #################################
    #################################
    # create copy of data frame with gene list and do std dev of region spend based on that instead

    dat_fit = pd.DataFrame({'sales': dat.sales, 'lvl5_int': df_array})
    region_revs = dat_fit.groupby('lvl5_int')['sales'].sum()
    #################################
    #################################

    ##################################
    # scores_vp_rg = 0
    # scores_sig_rg = 0
    # acrual_array_df = pd.DataFrame({"array":actual_array})
    # pred_array_df = pd.DataFrame({"s_array":df_array})
    # final_array_df = pred_array_df.where(pred_array_df.s_array!=acrual_array_df.array)
    # final_array_df = final_array_df[final_array_df['s_array'].notna()]
    # final_array_df['s_array'] = final_array_df['s_array'].astype('Int64')
    # acrual_array_df = acrual_array_df[acrual_array_df.index.isin(final_array_df.index)]
    # acrual_array_df = pd.merge(acrual_array_df,dat_lvl5, left_on='array',right_on='lvl5_int',how='left')
    # # acrual_array_df = acrual_array_df.drop(['lvl5_int'], axis =1)
    # acrual_array_df_2 = pd.merge(acrual_array_df,SIG_LVL5_LVL3, left_on='lvl5_int',right_on='SIG_lvl5_int',how='left')
    # final_array_df = pd.merge(final_array_df,dat_lvl5, left_on='s_array',right_on='lvl5_int',how='left')
    # # final_array_df = final_array_df.drop(['lvl5_int'], axis =1)
    # final_array_df_2 = pd.merge(final_array_df,SIG_LVL5_LVL3, left_on='lvl5_int',right_on='SIG_lvl5_int',how='left')
    # VP_match = final_array_df_2['Entpr_ID_3'] != acrual_array_df_2['Entpr_ID_3']
    # SIG_match = final_array_df_2['SIG'] != acrual_array_df_2['SIG']
    # scores_vp_rg += sum(VP_match)*1000000
    # scores_sig_rg += sum(SIG_match)*10000000
    # print(scores_sig_rg)
    # print(scores_vp_rg)
    ###################################
    outliers_total = 0
    region_dists = []
    for i in n_lvl5_unq:
        sum_dist = 0.0
        dist_list=list(squareform(
            pdist(np.vstack([dat_zip.loc[np.where(dat_zip == i)[0], ["lat", "lon"]].to_numpy()[0], dat.loc[np.where(df_array == i)[0], ["lat", "lon"]].to_numpy()])))[0, 1:])
        outliers = find_outliers_std_dev(dist_list, num_std_dev=3)
        outliers_total += len(outliers)
        print(f"Original data len: {len(dist_list)}")
        print(f"No.of Outliers (more than 2 standard deviations): {len(outliers)}")
        print(f"Outliers (more than 2 standard deviations): {outliers}")
        sum_dist += np.sum(squareform(
            pdist(np.vstack([dat_zip.loc[np.where(dat_zip == i)[0], ["lat", "lon"]].to_numpy()[0], dat.loc[np.where(df_array == i)[0], ["lat", "lon"]].to_numpy()])))[0, 1:])
        region_dists.append(sum_dist)
        
    rounded_list_2 = round_list_values(region_dists)
    rounded_list_2 = np.array(rounded_list_2)
    print("rounded_list:",rounded_list)
    print("rounded_list_2:",rounded_list_2)


    region_dists_avg_2 = []
    for i in n_lvl5_unq:
        sum_dist_avg = 0.0
        sum_dist_avg += np.mean(squareform(
            pdist(np.vstack([dat_zip.loc[np.where(dat_zip == i)[0], ["lat", "lon"]].to_numpy()[0], dat.loc[np.where(column_array == i)[0], ["lat", "lon"]].to_numpy()])))[0, 1:])
        region_dists_avg_2.append(sum_dist_avg)
    
    rounded_list_avg_2 = round_list_values(region_dists_avg_2)
    rounded_list_avg_2 = [x * 100 for x in rounded_list_avg_2]
    rounded_list_avg_2 = np.array(rounded_list_avg_2)

    avg_diff_curr_prop = compare_lists(rounded_list_avg,rounded_list_avg_2)
    
    greater_elements = rounded_list_2[rounded_list_2 > rounded_list]
    f_list = compare_lists(rounded_list_2,rounded_list)
    result = count_greater_than(f_list,rounded_list_avg_10,avg_diff_curr_prop,0)
    print("new pred more than 10 miles:", result)
    summaf = result*1000000
    print("greater_elements:",greater_elements)
    summaf2 = outliers_total*1000000
    # if len(greater_elements) > 0:
        # summaf = len(greater_elements)*1000
    # else:
        # summaf = 0

    print(region_dists)
    summa1 = np.std(region_revs)/10000
    print(summa1)
    summa2 = (np.mean(region_dists))*1000
    print(summa2)
    summa3 = (np.mean(actual_array != df_array))*100
    print(summa3)
    # summa4 = scores_vp_rg
    # summa5 = scores_sig_rg
    total_error -= np.std(region_revs)/10000
    total_error -= (np.mean(region_dists))*1000
    # total_error -= (sum(region_revs[0:len(region_revs)]))/(sum(region_dists[0:len(region_dists)]))
    # total_error -= summa4
    # total_error -= summa5
    total_error -= summaf
    total_error -= summaf2
    print(total_error)
    # if len(actual_array) == len(df_array):
    #     align_error = (np.mean(actual_array != df_array))*100
    # else:
    #     print("Length Diff")
    # total_error -= align_error

    return summa1,summa2,summa3,total_error,region_dists


Individual.set_fitness_function(fitness_function)

## test_ind = Individual.generate_random()

with open('C:/Users/praven.kumar/OneDrive - Cardinal Health/Desktop/HMM/SCSE_Run_2_V3/Model_Files/best_ga_model_forced_crossover_v3_cf_update_v2_tot_pop_28102025_v2_sample.pkl', 'rb') as f:
     first_population=pickle.load(f)

# first_population = [Individual.generate_random_2() for _ in range(POPULATION_SIZE)]
## first_population[0] = Individual.generate_fixed()
first_ind = random.choice(first_population)
best_ind = random.choice(first_population)
fit_avg = []
fit_best = []
prev_best = 0
prev_avg = 0
same_best = 0
avg_stat = 0
generation_num = 0
population = first_population.copy()

time1 = time.time()

# log_file = pd.read_excel("C:/Users/praven.kumar/OneDrive - Cardinal Health/Desktop/HMM/19112024_100_Alignment.xlsx", index_col=False)

while generation_num < MAX_GENERATIONS and best_ind.fitness[3] != 0:
    generation_num += 1
    print("********************************* Generation",generation_num,"*********************************")
    offspring, val = select(population)
    for i in range(0,100):
        print(offspring[i].fitness[3])
    
    if generation_num == 10 or generation_num % 110 == 0 and abs(round((gen_avg_stat - avg_stat),2)) < abs(round((0.001*(gen_avg_stat)),2)):
    # if generation_num == 1 or generation_num % 100 == 0:
        print("Forced Crossover Happening")
        temp = [Individual.generate_random_2()]
        crossed_offspring = crossover_operation_2(offspring, temp, crossover_2, CROSSOVER_PROBABILITY)
        print(len(crossed_offspring))
    else:
        crossed_offspring = crossover_operation(offspring, crossover, CROSSOVER_PROBABILITY)
    mutated_offspring = mutation_operation(crossed_offspring, mutate, MUTATION_PROBABILITY)
    population = mutated_offspring.copy()
    best_ind, fit_avg, fit_best = stats(population, best_ind, fit_avg, fit_best)
    best_of_generation = max(population, key=lambda ind: ind.fitness[3])
    rounded_list = round_list_values(best_of_generation.fitness[4])
    # print("This is to check: ",best_of_generation.fitness[4])
    print(["region_revs - ", best_ind.fitness[0], "region_dists - ", best_ind.fitness[1], "align_error - ", best_ind.fitness[2], "total_error - ", best_ind.fitness[3]])
    print(["best_of_generation_region_revs - ", best_of_generation.fitness[0], "best_of_generation_region_dists - ", best_of_generation.fitness[1], "best_of_generation_align_error - ", best_of_generation.fitness[2],"best_of_generation_total_error - ", best_of_generation.fitness[3]])
    best_stat = fit_best[-1]
    prev_avg = avg_stat
    # print(fit_avg[-1])
    avg_stat = fit_avg[-1]
    if prev_best == best_stat:
        same_best += 1
    else:
        same_best = 0
    if generation_num == 10 or generation_num % 110 == 0:
        with open('C:/Users/praven.kumar/OneDrive - Cardinal Health/Desktop/HMM/SCSE_Run_2_V3/Model_Files/best_ga_model_forced_crossover_v3_cf_update_v2_tot_pop_28102025_v2_sample.pkl', 'wb') as file:
            pickle.dump(population, file)
        gen_avg_stat = fit_avg[-1]
    if generation_num % 10 == 0 or generation_num == 1:
        print([round(avg_stat, 2), round(best_stat, 2), generation_num])
    if same_best > 108:
        generation_num = MAX_GENERATIONS

    if generation_num % 75 == 0:
        predicted = pd.DataFrame(np.array(best_ind.gene_list))

        predicted.columns = ["predicted"]

        predicted["ShipTo"] = dat["shipto"]

        predicted.to_csv('C:/Users/praven.kumar/OneDrive - Cardinal Health/Desktop/HMM/SCSE_Run_2_V3/data/dat_random2_shipto_50_28102025_predicted_v2.csv',index = False)
        log_file = pd.read_excel("C:/Users/praven.kumar/OneDrive - Cardinal Health/Desktop/HMM/SCSE_Run_2_V3/data/28102025_100_Alignment.xlsx", index_col=False)
        log_file = log_file._append({'Generation': generation_num, 'Std. Deviation - Revenue':round(best_of_generation.fitness[0]), 'Avg Distance (Miles)':round(best_of_generation.fitness[1]), 'VP Error':0, 'Total Error':round(best_of_generation.fitness[3])},ignore_index=True)
        log_file.to_excel("C:/Users/praven.kumar/OneDrive - Cardinal Health/Desktop/HMM/SCSE_Run_2_V3/data/28102025_100_Alignment.xlsx", index=False)
    # if same_best > 400 and abs(round((gen_avg_stat - avg_stat),2)) < abs(round((0.001*(gen_avg_stat)),2)):
    #     generation_num = MAX_GENERATIONS
    # elif same_best > 400 and abs(round((gen_avg_stat - avg_stat),2)) > abs(round((0.001*(gen_avg_stat)),2)):
    #     best_of_generation = max(population, key=lambda ind: ind.fitness[3])
    #     best_ind = best_of_generation

gc.collect()

plot_stats(fit_avg, fit_best, "Error")

print(f'Total Number of Individuals: {Individual.counter}')

# dat.to_csv("C:/Users/praven.kumar/OneDrive - Cardinal Health/Desktop/HMM/M2/dat_random2_shipto_50_18112024.csv", index=False)

predicted = pd.DataFrame(np.array(best_ind.gene_list))

predicted.columns = ["predicted"]

predicted["ShipTo"] = dat["shipto"]

predicted.to_csv('C:/Users/praven.kumar/OneDrive - Cardinal Health/Desktop/HMM/SCSE_Run_2_V3/data/dat_random2_shipto_50_28102025_predicted_v2.csv',index = False)

# best_ind.to_csv("C:/Users/praven.kumar/OneDrive - Cardinal Health/Desktop/HMM/best_ind.csv")

# best_ind.plot_grid()

print(time.time() - time1)
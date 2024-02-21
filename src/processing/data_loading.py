import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns
import pickle as pkl
from Bio import SeqIO


def load_volkova():
    xls = pd.ExcelFile('../../data/1_Volkova2021.xlsx')
    volkova_df1 = pd.read_excel(xls, 'Mutation counts for all samples', header= 1)
    volkova_df2 = pd.read_excel(xls, 'Sample_description', header= 1)
    volkova_df1.rename(columns = {"Unnamed: 0":"Sample"}, inplace=True)

    file_name = "../../CLEAN/mutationtypes.pkl"
    open_file = open(file_name, "wb")
    pkl.dump(list(volkova_df1.columns)[1:97], open_file)
    open_file.close()
    open_file = open(file_name, "rb")
    loaded_list = pkl.load(open_file)
    open_file.close()
    print(loaded_list)

    volkova_df1.to_pickle('volkova.pkl')
    volkova_df2.to_pickle('volkova2.pkl')

def load_zou_2():
    '''9 KOs, parent cell line and sum of 7 child cell lines'''
    # file = p
    meier_df = pd.read_csv('../../data/3_Zou2018.txt', sep='\t')
    muttypes = []
    for i in ['C','T']:
        for j in ['A','C','G','T']:
            if i != j:
                muttypes.append('[{}>{}]'.format(i,j))
    counts = np.zeros((18,6))
    samples = []
    ko = -1
    last = ''
    for index, row in meier_df.iterrows():
        if row['knockout'] != last:
            samples.append(row['knockout'])
            ko += 1
            # break
        last = row['knockout']
        mut = '[{}>{}]'.format(row['Ref'], row['Alt'])
        for idx, i in enumerate(muttypes):
            if mut == i:
                counts[ko,idx] += 1
    df = pd.DataFrame(counts, columns = muttypes, index = samples)
    df.to_pickle('zou_2.pkl')
    pass


def load_zou2018():
    '''9 KOs, parent cell line and sum of 7 child cell lines'''
    # file = p
    dna = {}
    for i in range(24):
        if i == 22:
            f = open('../../data/dna/Homo_sapiens.GRCh37.dna.chromosome.X.fa', 'r')
        elif i == 23:
            f = open('../../data/dna/Homo_sapiens.GRCh37.dna.chromosome.Y.fa', 'r')
        else:
            f = open('data/dna/Homo_sapiens.GRCh37.dna.chromosome.{}.fa'.format(i+1), 'r')
        content = f.read()
        seq = ''.join(content.split()[4:])
        dna[i+1] = seq

    zou2018_df = pd.read_csv('../../data/3_Zou2018.txt', sep='\t')

    open_file = open('../../CLEAN/mutationtypes.pkl', 'rb')
    muttypes = pkl.load(open_file)
    open_file.close()

    counts = np.zeros((18,96))
    samples = []
    ko = -1
    last = ''
    for index, row in zou2018_df.iterrows():
        if row['knockout'] != last:
            samples.append(row['knockout'])
            ko += 1
            # break
            last = row['knockout']
        up, down = flanking_bases(dna,row['Chrom'], row['Pos']-1, row['Ref'])
        mut = '{}[{}>{}]{}'.format(up,row['Ref'], row['Alt'],down)
        for idx, i in enumerate(muttypes):
            if mut == i:
                counts[ko,idx] += 1
    df = pd.DataFrame(counts, columns = muttypes, index = samples)
    df.to_pickle('zou_2.pkl')
    pass

def load_zou2018_sample():
    '''9 KOs, parent cell line and sum of 7 child cell lines'''
    # file = p
    dna = {}
    for i in range(24):
        if i == 22:
            f = open('../../data/dna/Homo_sapiens.GRCh37.dna.chromosome.X.fa', 'r')
        elif i == 23:
            f = open('../../data/dna/Homo_sapiens.GRCh37.dna.chromosome.Y.fa', 'r')
        else:
            f = open('data/dna/Homo_sapiens.GRCh37.dna.chromosome.{}.fa'.format(i+1), 'r')
        content = f.read()
        seq = ''.join(content.split()[4:])
        dna[i+1] = seq

    zou2018_df = pd.read_csv('../../data/3_Zou2018.txt', sep='\t')
    df = pd.read_pickle('../../CLEAN/zou_2.pkl')

    open_file = open('../../CLEAN/mutationtypes.pkl', 'rb')
    muttypes = pkl.load(open_file)
    open_file.close()

    # counts = np.zeros((18,96))
    parent = np.zeros((9,96))
    child = np.zeros((9,96,7))
    samples = []
    ko = -1
    last = ''
    KOs = {}

    samp_num = {}
    for i in range(9):
        samp_num[i] = []

    for index, row in zou2018_df.iterrows():
        if row['knockout'] not in KOs:
            ko += 1
            KOs[row['knockout']] = ko
            samples.append(row['knockout'])
        else:
            ko = KOs[row['knockout']]

        up, down = flanking_bases(dna,row['Chrom'], row['Pos']-1, row['Ref'])
        mut = '{}[{}>{}]{}'.format(up,row['Ref'], row['Alt'],down)
        sample = row['Sample'].split('-')[-1]
        if isint(sample):
            sample = int(sample)
            if sample in samp_num[ko]:
                num = samp_num[ko].index(sample)
            else:
                num = len(samp_num[ko])
                samp_num[ko].append(sample)
            for idx, i in enumerate(muttypes):
                if mut == i:
                    child[ko, idx, num] += 1
        else:
            for idx, i in enumerate(muttypes):
                if mut == i:
                    parent[ko, idx] += 1

    child_ko = np.zeros= np.zeros((9,96,7))
    for i in range(8):
        child_ko[i,:,:] = (child[i,:,:].transpose() - parent[i,:]).transpose()

    child_ko = np.clip(child_ko, 0, None)

    child_u = child.mean(axis=2)
    child_sd = child.std(axis=2)
    child_sum = child.sum(axis=2)
    # child_df = pd.DataFrame(child, columns = muttypes, index = samples)
    # child_df.to_pickle('data/zou2018/zou2018_child.pkl')
    child_sum_df = pd.DataFrame(child_sum, columns=muttypes, index=samples)
    child_sum_df.to_pickle('data/zou2018/zou2018_child_sum.pkl')
    child_u_df = pd.DataFrame(child_u, columns = muttypes, index = samples)
    child_u_df.to_pickle('data/zou2018/zou2018_child_u.pkl')
    child_sd_df = pd.DataFrame(child_sd, columns = muttypes, index = samples)
    child_sd_df.to_pickle('data/zou2018/zou2018_child_sd.pkl')
    parent_df = pd.DataFrame(parent, columns = muttypes, index = samples)
    parent_df.to_pickle('data/zou2018/zou2018_parent.pkl')
    pass

def load_zou2021_sample():
    '''42 KOs, parent cell line and sum of 7 child cell lines'''
    # file = p
    dna = {}
    for i in range(24):
        if i == 22:
            f = open('../../data/dna/Homo_sapiens.GRCh37.dna.chromosome.X.fa', 'r')
        elif i == 23:
            f = open('../../data/dna/Homo_sapiens.GRCh37.dna.chromosome.Y.fa', 'r')
        else:
            f = open('data/dna/Homo_sapiens.GRCh37.dna.chromosome.{}.fa'.format(i+1), 'r')
        content = f.read()
        seq = ''.join(content.split()[4:])
        dna[i+1] = seq

    zou2021_df = pd.read_csv('../../data/volkova2021/2_zou2021.txt', sep='\t')
    df = pd.read_pickle('../../CLEAN/zou_2.pkl')

    open_file = open('../../CLEAN/mutationtypes.pkl', 'rb')
    muttypes = pkl.load(open_file)
    open_file.close()

    # counts = np.zeros((18,96))
    parent = np.zeros((9,96))
    child = np.zeros((9,96,7))
    samples = []
    ko = -1
    last = ''
    KOs = {}

    samp_num = {}
    for i in range(9):
        samp_num[i] = []

    for index, row in zou2018_df.iterrows():
        if row['knockout'] not in KOs:
            ko += 1
            KOs[row['knockout']] = ko
            samples.append(row['knockout'])
        else:
            ko = KOs[row['knockout']]

        up, down = flanking_bases(dna,row['Chrom'], row['Pos']-1, row['Ref'])
        mut = '{}[{}>{}]{}'.format(up,row['Ref'], row['Alt'],down)
        sample = row['Sample'].split('-')[-1]
        if isint(sample):
            sample = int(sample)
            if sample in samp_num[ko]:
                num = samp_num[ko].index(sample)
            else:
                num = len(samp_num[ko])
                samp_num[ko].append(sample)
            for idx, i in enumerate(muttypes):
                if mut == i:
                    child[ko, idx, num] += 1
        else:
            for idx, i in enumerate(muttypes):
                if mut == i:
                    parent[ko, idx] += 1

    child_ko = np.zeros= np.zeros((9,96,7))
    for i in range(8):
        child_ko[i,:,:] = (child[i,:,:].transpose() - parent[i,:]).transpose()

    child_ko = np.clip(child_ko, 0, None)

    child_u = child.mean(axis=2)
    child_sd = child.std(axis=2)
    child_sum = child.sum(axis=2)
    # child_df = pd.DataFrame(child, columns = muttypes, index = samples)
    # child_df.to_pickle('data/zou2018/zou2018_child.pkl')
    child_sum_df = pd.DataFrame(child_sum, columns=muttypes, index=samples)
    child_sum_df.to_pickle('data/zou2018/zou2018_child_sum.pkl')
    child_u_df = pd.DataFrame(child_u, columns = muttypes, index = samples)
    child_u_df.to_pickle('data/zou2018/zou2018_child_u.pkl')
    child_sd_df = pd.DataFrame(child_sd, columns = muttypes, index = samples)
    child_sd_df.to_pickle('data/zou2018/zou2018_child_sd.pkl')
    parent_df = pd.DataFrame(parent, columns = muttypes, index = samples)
    parent_df.to_pickle('data/zou2018/zou2018_parent.pkl')
    pass


def load_zou2021():
    xls = pd.ExcelFile('../../data/zou2021/Supplementary_tables.xlsx')
    # S3 = Mutational profiles per Sample  (96 x 173)
    profiles = pd.read_excel(xls, 'TableS3', index_col= 0,header= 0)
    # S1 = Gene - Pathway mapping   (43 KOs
    KO_pathway = pd.read_excel(xls, 'TableS1', index_col= 0, header= 0)
    # zou2021.rename(columns = {"Unnamed: 0":"Sample"}, inplace=True)

    # To look-up tables Gene <-> Pathway
    pathways = {}
    genes = {}
    for index, row in KO_pathway.iterrows():
        p = row['Subpathway_KO'].split('/')
        for i in p:
            if i in pathways:
                pathways[i].append(index)
            else:
                pathways[i] = [index]
            if index in genes:
                genes[index].append(i)
            else:
                genes[index] = [i]
    pass





def isint(value):
    try:
        int(value)
        return True
    except ValueError:
        return False

def flanking_bases(dna, chrom, pos, ref):
    '''dna = chromosome
        pos = position mutation in reference genome
        ref = reference nucleotide'''
    # print(dna[chrom][pos-5:pos+5])
    # print(ref)
    if dna[chrom][pos] == ref:
        return dna[chrom][pos-1], dna[chrom][pos+1]
    else:
        print('Reference error')
        return 'A','A'

def load_zou():
    dna = {}
    for i in range(24):
        if i == 22:
            f = open('../../data/dna/Homo_sapiens.GRCh37.dna.chromosome.X.fa', 'r')
        elif i == 23:
            f = open('../../data/dna/Homo_sapiens.GRCh37.dna.chromosome.Y.fa', 'r')
        else:
            f = open('data/dna/Homo_sapiens.GRCh37.dna.chromosome.{}.fa'.format(i+1), 'r')
        content = f.read()
        seq = ''.join(content.split()[4:])
        dna[i+1] = seq

    raw_sbs = pd.read_csv('../../data/zou2021/denovo_subs_43genes.txt', sep='\t')
    raw_dbs = pd.read_csv('../../data/zou2021/denovo_doublesub_43genes.txt', sep='\t')
    raw_indel = pd.read_csv('../../data/zou2021/denovo_indels_43genes.txt', sep='\t')
    raw_rgs = pd.read_csv('../../data/zou2021/denovo_rgs_43genes.txt', sep='\t')

    open_file = open('../../CLEAN/mutationtypes.pkl', 'rb')
    muttypes = pkl.load(open_file)
    open_file.close()

    # counts = np.zeros((18,96))
    count = np.zeros((173,96))
    # child = np.zeros((9,96,7))
    samples = []
    ko = -1
    last = ''
    KOs = {}

    samp_num = {}

    # SUBSTITUTION
    for index, row in raw_sbs.iterrows():
        if row['knockout'] not in KOs:
            ko += 1
            KOs[row['knockout']] = ko
            samples.append(row['knockout'])
        else:
            ko = KOs[row['knockout']]

        up, down = flanking_bases(dna,row['Chrom'], row['Pos']-1, row['Ref'])
        mut = '{}[{}>{}]{}'.format(up,row['Ref'], row['Alt'],down)
        sample = row['Sample'].split('-')[-1]


        if isint(sample):
            sample = int(sample)
            if sample in samp_num[ko]:
                num = samp_num[ko].index(sample)
            else:
                num = len(samp_num[ko])
                samp_num[ko].append(sample)
            for idx, i in enumerate(muttypes):
                if mut == i:
                    count[ko, idx, num] += 1
        else:
            for idx, i in enumerate(muttypes):
                if mut == i:
                    parent[ko, idx] += 1



# load_volkova()
# load_meier()
load_zou2021()
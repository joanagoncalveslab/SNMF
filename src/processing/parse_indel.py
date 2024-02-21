import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.patches import Patch
import matplotlib.image as mpimg
from matplotlib import rcParams

import sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import sigProfilerPlotting as sigPlt
import umap.umap_ as umap

# %matplotlib inline

def load_zou():
    '''
    Generate dataframe:
    173 samples
    96 SBSs ; gene_KO ; Count ; Protein_KO ; Subpathway_KO
    '''
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

    features = list(profiles.index)

    profiles = profiles.drop(columns=['Mutation'])
    samples = list(profiles.columns)
    count = profiles.sum(axis=0)
    profiles = profiles.loc[:,:].div(profiles.sum(axis=0), axis=1)
    profiles = profiles.transpose()
    gene_KO = []
    for s in samples:
        gene_KO.append(s.split('_')[0])
    profiles['Gene_KO'] = gene_KO
    profiles['Count'] = count
    joined = pd.merge(left=profiles, right=KO_pathway, how= 'left', left_on='Gene_KO', right_index = True)

    x = profiles.loc[:,features].values  # Separating out the target
    y = profiles.loc[:, 'Gene_KO'].values  # Standardizing the features

    # Add columns for Indels
    indel_features = []
    for i in range(1, 84):
        indel_features.append(str(i))
    joined = joined.assign(**dict.fromkeys(indel_features, 0))

    return joined


def load_ref():
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
    return dna



# TODO - lookup table: SAMPLE -> idx
# TODO - lookup table/ data frame: Idx -> Del/ins ; Basepair

def reps(dna, ref, alt, length, chrom, pos):
    #
    # CHECK
    if dna[chrom][pos] != ref[0]:
        print('NOT ALIGNED!!!')

    rep_size = 0
    up = 1
    up_pos = pos+1
    down = 1
    down_pos = pos

    if length > 0:
        # deletion
        indel = ref[1:]  # NOT Sure....
    else:
        # insertion
        indel = alt[1:]
        length = -length


    while up == 1 or down == 1:
        if up == 1:
            if dna[chrom][up_pos: up_pos + length] != indel:
                up = 0
            up_pos += length

        if down == 1:
            if dna[chrom][down_pos: down_pos + length] != indel:
                down = 0
            else:
                print('Downstream Rep !!!')
            down_pos -= length

        rep_size += up + down
        if rep_size >= 6:
            break
    print(rep_size)
    return rep_size



def mh_len(ref, alt, length, chrom, pos):
    # Return MH-lenght (max = deletion length)

    up_length = 0
    while up_length < length:
        if dna[chrom][pos+1 + up_length] == dna[chrom][pos+1 + length + up_length]:
            up_length += 1
        else:
            break

    down_length = 0
    while down_length < length:
        if dna[chrom][pos + length - down_length] == dna[chrom][pos - down_length]:
            down_length += 1
        else:
            break
    print(dna[chrom][pos: pos + length+10])
    return max(up_length, down_length)


raw_indel = pd.read_csv('../../data/zou2021/denovo_indels_43genes.txt', sep='\t')
raw_indel
# # 173 Samples - 15 Indel catagories
# indels = np.zeros((173, 15))
zou = load_zou()
dna = load_ref()

for index, row in raw_indel.iterrows():
    ref = row['Ref']
    alt = row['Alt']
    pos = int(row['Pos'])-1
    chrom =row['Chrom']
    if chrom == 'X':
        chrom = 23
    else:
        chrom = int(chrom)

    length = len(ref) - len(alt)
    deletion = False

    print(dna[chrom][pos-10:pos+10])
    print(ref)
    print(alt)

    if length > 0:
        # DELETION
        deletion = True

        if length == 1:
            # 1bp Deletion
            rep_size = reps(dna, ref, alt, length, chrom, pos)
            idx = 1+ ( min(rep_size, 6)-1 ) * 2
            if ref[1] in ('C', 'G'):
                idx += 1

        else:
            mh_length = mh_len(ref, alt, length, chrom, pos)

            if mh_length == 0:
                # >1bp Deleltion with repeat size 0
                idx = 25 + (min(length,5) - 2) * 6

            elif mh_length == length:
                # >1bp Deletion with repeat size >= 1
                rep_size = reps(dna, ref, alt, length, chrom, pos)
                idx = 24 + (min(length, 5) - 2)*6 + (min(rep_size, 5))



            else:
                # Microhomology
                idx = 73
                if length == 2:
                    idx = 72 + mh_length
                    # CHECK if 73
                elif length == 3:
                    idx = 73 + mh_length
                elif length == 4:
                    idx = 75 + mh_length
                else:
                    idx = 78 + min(mh_length, 5)


    else:
        # INSERTION
        if length == -1:
            # 1bp Insertion
            rep_size = reps(dna, ref, alt, length, chrom, pos)
            idx = 13 + min(rep_size, 5) * 2
            if alt[1] in ('C', 'G'):
                idx += 1

        else:
            # >1bp Insertion
            rep_size = reps(dna, ref, alt, length, chrom, pos)
            # Often rep_size 1 --> other ref genome?
            idx = 49 + (min(-length, 5) - 2) * 6 + min(rep_size, 5)

    print(idx)
    sample = row['Ko_gene'] + '_' + row['Sample'].split('.')[1]
    print(sample)
    zou.loc[sample, str(idx)] += 1


# zou[indel_count] = zou[indel_features].sum(axis=1)
# zou[indel_count] = zou.loc[:,indel_features].sum(axis=1)

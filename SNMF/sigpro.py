#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 13:39:29 2018

@author: S M Ashiqul Islam (Mishu)


    ##########################################
    SigProfilerExtractor (``sigproextractor``)
    ##########################################
    
    SigProfilerExtractor allows de novo extraction of mutational signatures from data 
    generated in a matrix format. The tool identifies the number of operative mutational 
    signatures, their activities in each sample, and the probability for each signature to 
    cause a specific mutation type in a cancer sample. The tool makes use of SigProfilerMatrixGenerator 
    and SigProfilerPlotting. 

"""
import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 

import matplotlib.pyplot as plt
plt.switch_backend('agg')

import scipy
from scipy import io as sio
import sklearn
import numpy as np
import pandas as pd
import time
import shutil
import platform
import datetime
import psutil
import copy
import sigProfilerPlotting 
import multiprocessing
from SNMF import subroutines as sub
import SigProfilerMatrixGenerator
from SigProfilerMatrixGenerator.scripts import SigProfilerMatrixGeneratorFunc as datadump   
import multiprocessing as mp
import SNMF as cosmic
from SNMF import single_sample as ss
from numpy.random import Generator, PCG64DXSM, SeedSequence
from sklearn import metrics
from sklearn.metrics import f1_score
import statistics
from SNMF import plotActivity as plot_ac


import pdb
def memory_usage():
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse1 = py.memory_info()[0]/2.**30  # memory use in GB...I think
    print('\n************** Reported Current Memory Use: '+ str(round(memoryUse1,2))+" GB *****************\n")

def importdata(datatype="matrix"):
    
    """
    Imports the path of example data.
    
    parameters
    ----------
    
    datatype: A string. Type of data. The type of data should be one of the following:
            - "vcf": used for vcf format data.
            - "matrix": used for text format data. This format represents the catalog of mutations seperated by tab. 
            - "matobj": used for matlab object format data.
            
    
    
    Returns:
    -------

    The path of the example data.

    Example: 
    -------
    >>> from src.models.S_NMF.SigProfilerExtractor import sigpro as sig
    >>> data = sig.importdata("table")
    
    This "data" variable can be used as a parameter of the "project" argument of the sigProfilerExtractor function
        
    """
    
    paths = cosmic.__path__[0]
    if datatype=="matobj":
        data = paths+"/data/21_breast_WGS_substitutions.mat"
    elif datatype=="text" or datatype=="table" or datatype=="matrix":
        data = paths+"/data/Samples.txt"
    elif datatype=="csv":
        data = paths+"/data/csvexample.csv"
    elif datatype=="vcf":
        directory = os.getcwd()
        dataold = paths+"/data/vcftest"
        datanew = directory+"/vcftest"
        if not os.path.exists(datanew):
            shutil.copytree(dataold , datanew)
        data="vcftest"
    return data
    
def record_parameters(sysdata, execution_parameters, start_time):
    """
    Extracts mutational signatures from an array of samples.
    
    """
    sysdata.write("\n--------------EXECUTION PARAMETERS--------------\n")
    sysdata.write("INPUT DATA\n")
    sysdata.write("\tinput_type: {}\n".format(execution_parameters["input_type"]))
    sysdata.write("\toutput: {}\n".format(execution_parameters["output"]))
    sysdata.write("\tinput_data: {}\n".format(execution_parameters["input_data"]))
    sysdata.write("\treference_genome: {}\n".format(execution_parameters["reference_genome"]))
    sysdata.write("\tcontext_types: {}\n".format(execution_parameters["context_type"]))
    sysdata.write("\texome: {}\n".format(execution_parameters["exome"]))
    sysdata.write("NMF REPLICATES\n")
    sysdata.write("\tminimum_signatures: {}\n".format(execution_parameters["minimum_signatures"]))
    sysdata.write("\tmaximum_signatures: {}\n".format(execution_parameters["maximum_signatures"]))
    sysdata.write("\tNMF_replicates: {}\n".format(execution_parameters["NMF_replicates"]))
    sysdata.write("NMF ENGINE\n")
    sysdata.write("\tNMF_init: {}\n".format(execution_parameters["NMF_init"]))
    sysdata.write("\tprecision: {}\n".format(execution_parameters["precision"]))
    sysdata.write("\tmatrix_normalization: {}\n".format(execution_parameters["matrix_normalization"]))
    sysdata.write("\tresample: {}\n".format(execution_parameters["resample"]))
    sysdata.write("\tseeds: {}\n".format(execution_parameters["seeds"]))
    sysdata.write("\tmin_NMF_iterations: {}\n".format(format(execution_parameters["min_NMF_iterations"],',d')))
    sysdata.write("\tmax_NMF_iterations: {}\n".format(format(execution_parameters["max_NMF_iterations"], ',d')))
    sysdata.write("\tNMF_test_conv: {}\n".format(format(execution_parameters["NMF_test_conv"],',d')))
    sysdata.write("\tNMF_tolerance: {}\n".format(execution_parameters["NMF_tolerance"]))
    sysdata.write("CLUSTERING\n")
    sysdata.write("\tclustering_distance: {}\n".format(execution_parameters["dist"]))

    sysdata.write("EXECUTION\n")
    if execution_parameters["cpu"]==-1:
        sysdata.write("\tcpu: {}; Maximum number of CPU is {}\n".format(multiprocessing.cpu_count(), multiprocessing.cpu_count()))
    else:
        sysdata.write("\tcpu: {}; Maximum number of CPU is {}\n".format(execution_parameters["cpu"], multiprocessing.cpu_count()))

    sysdata.write("\tgpu: {}\n".format(execution_parameters["gpu"]))
    sysdata.write("Solution Estimation\n")
    sysdata.write("\tstability: {}\n".format(execution_parameters["stability"]))
    sysdata.write("\tmin_stability: {}\n".format(execution_parameters["min_stability"]))
    sysdata.write("\tcombined_stability: {}\n".format(execution_parameters["combined_stability"]))
    
    sysdata.write("COSMIC MATCH\n")
    sysdata.write("\topportunity_genome: {}\n".format(execution_parameters["opportunity_genome"]))
    sysdata.write("\cosmic_version: {}\n".format(execution_parameters["cosmic_version"]))
    sysdata.write("\tnnls_add_penalty: {}\n".format(execution_parameters["nnls_add_penalty"]))
    sysdata.write("\tnnls_remove_penalty: {}\n".format(execution_parameters["nnls_remove_penalty"]))
    sysdata.write("\tinitial_remove_penalty: {}\n".format(execution_parameters["initial_remove_penalty"]))
    sysdata.write("\tde_novo_fit_penalty: {}\n".format(execution_parameters["de_novo_fit_penalty"])) 
    sysdata.write("\trefit_denovo_signatures: {}\n".format(execution_parameters["refit_denovo_signatures"]))
    sysdata.write("\tcollapse_to_SBS96: {}\n".format(execution_parameters["collapse_to_SBS96"]))
    
    sysdata.write("\n-------Analysis Progress------- \n")
    sysdata.write("[{}] Analysis started: \n".format(str(start_time).split(".")[0]))
            
def sigProfilerExtractor(input_type, 
                         output, 
                         input_data,
                         input_label,
                         reference_genome="GRCh37", 
                         opportunity_genome = "GRCh37", 
                         cosmic_version=3.1,
                         context_type = "default", 
                         exome = False, 
                         minimum_signatures=1,
                         maximum_signatures=25,  
                         nmf_replicates=500, 
                         resample = False,
                         batch_size=1, 
                         cpu=-1, 
                         gpu=False, 
                         nmf_init="random", 
                         precision= "single", 
                         matrix_normalization= "gmm", 
                         seeds= "random", 
                         min_nmf_iterations= 10000, 
                         max_nmf_iterations=1000000, 
                         nmf_test_conv= 5000,
                         # nmf_tolerance= 1e-15,
                         # nnls_add_penalty=0.05,
                         # nnls_remove_penalty=0.01,
                         # de_novo_fit_penalty=0.02,
                         # initial_remove_penalty=0.05,
                         nmf_tolerance=1e-18,
                         nnls_add_penalty=0.05,
                         nnls_remove_penalty=0.01,
                         de_novo_fit_penalty=0.0002,
                         initial_remove_penalty=0.0005,
                         # refit_denovo_signatures=True,
                         refit_denovo_signatures=False,
                         collapse_to_SBS96=True,
                         clustering_distance="cosine",
                         export_probabilities=True,
                         make_decomposition_plots=True,
                         stability=0.8, 
                         min_stability=0.2, 
                         combined_stability=1.0,
                         get_all_signature_matrices= False,
                         lambda_c = 1e-40,
                         lr = 0.001,
                         lambda_p = 0.5):
    """
    Extracts mutational signatures from an array of samples.
    
    
    Parameters
    ----------
    
    INPUT DATA:-
    
    input_type: A string. Type of input. The type of input should be one of the following:
            - "vcf": used for vcf format inputs.
            - "matrix": used for table format inputs using a tab seperated file.
             
        
    output: A string. The name of the output folder. The output folder will be generated in the current working directory. 
            
    input_data: A string. Name of the input folder (in case of "vcf" type input) or the input file (in case of "table"  type input). The project file or folder should be inside the current working directory. For the "vcf" type input,the project has to be a folder which will contain the vcf files in vcf format or text formats. The "text"type projects have to be a file.   
            
    reference_genome: A string, optional. The name of the reference genome. The default reference genome is "GRCh37". This parameter is applicable only if the input_type is "vcf".
       
    opportunity_genome: The build or version of the reference signatures for the reference genome. The default opportunity genome is GRCh37. If the input_type is "vcf", the genome_build automatically matches the input reference genome value.    
     
    context_type: A list of strings, optional. The items in the list defines the mutational contexts to be considered to extract the signatures. The default value is "SBS96,DBS78,ID83". 
    
    exome: Boolean, optional. Defines if the exomes will be extracted. The default value is "False".
    
    
    NMF RUNS:-
    
    minimum_signature: A positive integer, optional. The minimum number of signatures to be extracted. The default value is 1 
    
    maximum_signatures: A positive integer, optional. The maximum number of signatures to be extracted. The default value is 10
    
    nmf_replicates: A positive integer, optional. The number of iteration to be performed to extract each number signature. The default value is 100
    
    resample: Boolean, optional. Default is True. If True, add poisson noise to samples by resampling.  
    
    seeds: Boolean. Default is "random". If random, then the seeds for resampling will be random for different analysis.
                  If not random, then seeds will be obtained from a given path of a .txt file that contains a list of seed. 
    
    NMF RUNS:-
    
    matrix_normalization: A string. Method of normalizing the genome matrix before it is analyzed by NMF. Default is "log2". Other options are "gmm", "100X" or "no_normalization".         
    
    nmf_init: A String. The initialization algorithm for W and H matrix of NMF. Options are 'random', 'nndsvd', 'nndsvda', 'nndsvdar' and 'nndsvd_min'
              Default is 'nndsvd_min'.
    
    precision: A string. Values should be single or double. Default is single.
    
    min_nmf_iterations: An integer. Value defines the minimum number of iterations to be completed before NMF converges. Default is 2000.
    
    max_nmf_iterations: An integer. Value defines the maximum number of iterations to be completed before NMF converges. Default is 200000
    
    nmf_test_conv: An integer. Value definer the number number of iterations to done between checking next convergence.
            
    nmf_tolerance: A float. Value defines the tolerance to achieve to converge. 
    
    
    EXECUTION:-
    
    cpu: An integer, optional. The number of processors to be used to extract the signatures. The default value is -1 which will use all available        processors. 
    
    gpu:Boolean, optional. Defines if the GPU resource will used if available. Default is False. If True, the GPU resource 
        will be used in the computation.

    batch_size: An integer. Will be effective only if the GPU is used. Defines the number of NMF replicates to be performed
              by each CPU during the parallel processing. Default is 1.
              
    
    SOLUTION ESTIMATION THRESH-HOLDS:-

    stability: A float. Default is 0.8. The cutoff thresh-hold of the average stability. Solutions with average stabilities below this thresh-hold will not be considered. 

    min_stability: A float. Default is 0.2. The cutoff thresh-hold of the minimum stability. Solutions with minimum stabilities below this thresh-hold will not be considered. 

    combined_stability: A float. Default is 1.0. The cutoff thresh-hold of the combined stability (sum of average and minimum stability). Solutions with combined stabilities below this thresh-hold will not be considered.            
    
    
    DECOMPOSITION:-
    
    de_novo_fit_penalty: Float, optional. Takes any positive float. Default is 0.02. Defines the weak (remove) thresh-hold cutoff to be assigned denovo signatures to a sample. 
    
    nnls_add_penalty: Float, optional. Takes any positive float. Default is 0.05. Defines the strong (add) thresh-hold cutoff to be assigned COSMIC signatures to a sample. 
    
    nnls_remove_penalty: Float, optional. Takes any positive float. Default is 0.01. Defines the weak (remove) thresh-hold cutoff to be assigned COSMIC signatures to a sample.
     
    initial_remove_penalty: Float, optional. Takes any positive float. Default is 0.05. Defines the initial weak (remove) thresh-hold cutoff to be COSMIC assigned signatures to a sample.
    
    refit_denovo_signatures: Boolean, optional. Default is False. If True, then refit the denovo signatures with nnls.
    
    make_decomposition_plots: Boolean, optional. Defualt is True. If True, Denovo to Cosmic sigantures decompostion plots will be created as a part the results.

    
    OTHERS:-
    
    get_all_signature_matrices: A Boolean. If true, the Ws and Hs from all the NMF iterations are generated in the output.
    
    export_probabilities: A Boolean. Defualt is True. If False, then doesn't create the probability matrix.
    

    
    Returns
    -------
    To learn about the output, please visit https://osf.io/t6j7u/wiki/home/
    
    
    Examples
    --------
    
    Examples
    --------

    >>> from src.models.S_NMF.SigProfilerExtractor import sigpro as sig
    
    # to get input from vcf files
    >>> path_to_example_folder_containing_vcf_files = sig.importdata("vcf")
    >>> data = path_to_example_folder_containing_vcf_files # you can put the path to your folder containing the vcf samples
    >>> sig.sigProfilerExtractor("vcf", "example_output", data, minimum_signatures=1, maximum_signatures=3)
    
    Wait untill the excecution is finished. The process may a couple of hours based on the size of the data.
    Check the current working directory for the "example_output" folder.
    
    # to get input from table format (mutation catalog matrix)
    >>> path_to_example_table = sig.importdata("matrix")
    >>> data = path_to_example_table # you can put the path to your tab delimited file containing the mutational catalog         matrix/table
    >>> sig.sigProfilerExtractor("matrix", "example_output", data, opportunity_genome="GRCh38", minimum_signatures=1, maximum_signatures=3)
    
    Wait untill the excecution is finished. The process may a couple of hours based on the size of the data.
    Check the results in the "example_output" folder.
    """
    memory_usage()
    #record the start time
    start_time = datetime.datetime.now()
    
    #set the output variable
    out_put = output; 
    
    if gpu == True:
        import torch
    
        if gpu and (torch.cuda.device_count() == 0):
            raise RuntimeError("GPU not available!")
    
    
    #################################### At first create the system data file ####################################
    if not os.path.exists(out_put):
        os.makedirs(out_put)
    sysdata = open(out_put+"/JOB_METADATA.txt", "w")
    sysdata.write("THIS FILE CONTAINS THE METADATA ABOUT SYSTEM AND RUNTIME\n\n\n")
    sysdata.write("-------System Info-------\n")
    sysdata.write("Operating System Name: "+ platform.uname()[0]+"\n"+"Nodename: "+platform.uname()[1]+"\n"+"Release: "+platform.uname()[2]+"\n"+"Version: "+platform.uname()[3]+"\n")
    sysdata.write("\n-------Python and Package Versions------- \n")
    sysdata.write("Python Version: "+str(platform.sys.version_info.major)+"."+str(platform.sys.version_info.minor)+"."+str(platform.sys.version_info.micro)+"\n")
    sysdata.write("SigProfilerExtractor Version: "+cosmic.__version__+"\n")
    sysdata.write("SigProfilerPlotting Version: "+sigProfilerPlotting.__version__+"\n")
    sysdata.write("SigProfilerMatrixGenerator Version: "+SigProfilerMatrixGenerator.__version__+"\n")
    sysdata.write("Pandas version: "+pd.__version__+"\n")
    sysdata.write("Numpy version: "+np.__version__+"\n")
    sysdata.write("Scipy version: "+scipy.__version__+"\n")
    sysdata.write("Scikit-learn version: "+sklearn.__version__+"\n")

    #format the project_name first:
    project = input_data  #will use this variable as the parameter for project argument in SigprofilerMatrixGenerator
    project_label = input_label
    try:
        if project[-1] != "/":
            project_name = project.split("/")[-1]   #will use this variable as the parameter for project_name argument in SigprofilerMatrixGenerator
        else:
            project_name = project.split("/")[-2]
    except:
        project_name = "Input from DataFrame"
    try:
        if project_label[-1] != "/":
            project_label_name = project_label.split("/")[-1]   #will use this variable as the parameter for project_name argument in SigprofilerMatrixGenerator
        else:
            project_label_name = project_label.split("/")[-2]
    except:
        project_name = "Input from DataFrame"

    execution_parameters= {"input_type":input_type, 
                        "output":output, 
                        "input_data":input_data, 
                        "reference_genome":reference_genome, 
                        "opportunity_genome":opportunity_genome, 
                        "cosmic_version":cosmic_version,
                        "context_type":context_type,
                        "exome":exome,
                        "minimum_signatures":minimum_signatures, 
                        "maximum_signatures":maximum_signatures, 
                        "NMF_replicates":nmf_replicates, 
                        "cpu":cpu, 
                        "gpu":gpu, 
                        "batch_size":batch_size, 
                        "NMF_init":nmf_init,
                        "precision":precision,
                        "matrix_normalization":matrix_normalization,
                        "resample":resample, 
                        "seeds":seeds,
                        "min_NMF_iterations":min_nmf_iterations,
                        "max_NMF_iterations":max_nmf_iterations,
                        "NMF_test_conv": nmf_test_conv,
                        "NMF_tolerance": nmf_tolerance,
                        "nnls_add_penalty":nnls_add_penalty,
                        "nnls_remove_penalty":nnls_remove_penalty,
                        "initial_remove_penalty":initial_remove_penalty,
                        "de_novo_fit_penalty":de_novo_fit_penalty,
                        "refit_denovo_signatures":refit_denovo_signatures,
                        "collapse_to_SBS96":collapse_to_SBS96,
                        "dist":clustering_distance,
                        "export_probabilities":export_probabilities,
                        "make_decompostion_plots":make_decomposition_plots,
                        "stability":stability, 
                        "min_stability":min_stability, 
                        "combined_stability":combined_stability,
                        "get_all_signature_matrices":get_all_signature_matrices,
                        "Lambda_c":lambda_c,
                        "lr":lr,
                        "lambda_p":lambda_p}
                        
    ################################ take the inputs from the general optional arguments ####################################
    startProcess  = minimum_signatures
    endProcess    = maximum_signatures
    mtype         = context_type
    wall          = get_all_signature_matrices
    add_penalty   = nnls_add_penalty
    remove_penalty= nnls_remove_penalty
    genome_build  = opportunity_genome
    refgen        = reference_genome


    #set the squence type ("genome" or "exome") for the tmb plot inside the make_final_solution function
    if exome==False:
        sequence="genome"
    if exome==True:
        sequence="exome"
    
    # Use a SeedSequence to create generators for random number generation
    if seeds=="random":
        execution_parameters["seeds"] = seeds
        tmp_seed = SeedSequence().entropy
        seed     = np.array(tmp_seed)
        seeds    = pd.DataFrame([tmp_seed], columns=["Seed"])
        seeds.to_csv(out_put+"/Seeds.txt", sep="\t", quoting=None)
    else:
        try:
            execution_parameters["seeds"] = seeds
            seeds = pd.read_csv(seeds,sep="\t", index_col=0)
            seeds.to_csv(out_put+"/Seeds.txt", sep="\t")
            seed  = np.array(seeds["Seed"])

        except:
            "Please set valid seeds"
    
    if input_type=="text" or input_type =="table" or input_type=="matrix":
        
        ################################### For text input files ######################################################
        text_file = project
        text_file_label = project_label
        title = "" # set the title for plotting 
            
        if type(text_file)!=str:
            data=text_file
            execution_parameters["input_data"]="Matrix["+str(data.shape[0])+" rows X "+str(data.shape[1])+ " columns]"
        else:
            data = pd.read_csv(text_file, sep="\t").iloc[:,:]
            label = pd.read_csv(text_file_label, sep="\t").iloc[:,:]
        
        data = data.dropna(axis=1, inplace=False)
        data = data.loc[:, (data != 0).any(axis=0)]
        genomes = data.iloc[:,1:]
        genomes = np.array(genomes)
        allgenomes = genomes.copy()  # save the allgenomes for the final results 
        #Contruct the indeces of the matrix
        #setting index and columns names of processAvg and exposureAvg
        index = data.iloc[:,0]
        colnames  = data.columns[1:]
        allcolnames = colnames.copy() # save the allcolnames for the final results

        label = label.dropna(axis=1, inplace=False)
        label = label.loc[:, (label != 0).any(axis=0)]
        Y = label.iloc[:,1:]
        Y = np.array(Y)
        classnames = label.columns[1:]


        #creating list of mutational type to sync with the vcf type input
        mtypes = [str(genomes.shape[0])]
        if mtypes[0] == "78":
            mtypes = ["DBS78"]
        elif mtypes[0] == "83":
            mtypes = ["ID83"]
        elif mtypes[0] == "48":
            mtypes = ["CNV48"]
        elif mtypes[0]=="32":
            mtypes = ["SV32"]
        elif mtypes[0]=="96" or "288" or "384" or "1536":
            mtypes = ["SBS"+mtypes[0]]
        else:
            mtypes = ["CH"+mtypes[0]]

    #recording context types
    execution_parameters["context_type"]=",".join(mtypes) 
    record_parameters(sysdata, execution_parameters, start_time)
    sysdata.close()      
    ###########################################################################################################################################################################################                  
    
    for m in mtypes:
        mutation_context = m
        # we need to rename the m because users input could be SBS96, SBS1536, DBS78, ID83 etc
        if m.startswith("SBS"):
            m = m[3:] #removing "SBS"
        elif m.startswith("DBS"):
            m = "DINUC"
        elif m.startswith("ID"):
            m = "ID"
        elif m.startswith("CNV"):
            m="CNV"
        elif m.startswith("SV"):
            m="SV"
        
        # Determine the types of mutation which will be needed for exporting and copying the files
        if not (m=="DINUC" or m.startswith("DBS") or m.startswith("ID") or m.startswith("CNV") or m.startswith("SV")):
            
            if m.startswith("SBS"):
                mutation_type = m
            elif m in ["96","288","384","1536"]:
                mutation_type="SBS"+m
            elif m.startswith("78"): 
                mutation_type="DBS78"
            elif m.startswith("83"):
                mutation_type="ID83"
            elif m.startswith("48"):
                mutation_type="CNV48"
            elif m.startswith("32"):
                mutation_type="SV32"
            else:
                mutation_type = "CH"+m
            
        else:
            if m == "DINUC" or m.startswith("DBS"):
                mutation_type = "DBS78"
            elif m== "ID" or m.startswith("ID"):
                mutation_type = "ID83"
            elif m== "CNV" or m.startswith("CNV"):
                mutation_type = "CNV48"
            elif m== "SV" or m.startswith("SV"):
                mutation_type = "SV32"

        #check if start and end processes are bigger than the number of samples
        startProcess = min(startProcess, genomes.shape[1])
        endProcess = min(endProcess, genomes.shape[1])   
        
        #in the plotting funciton "ID" is used as "INDEL"
        if m=="ID":
            m="INDEL" #for plotting     
            
        #create output directories to store all the results 
        output = out_put+"/"+mutation_type
        est_genomes = np.zeros([1,1])
        genomes = np.array(genomes)

        layer_directory = output
        try:
            if not os.path.exists(layer_directory):
                os.makedirs(layer_directory)
        except: 
            print ("The {} folder could not be created".format("output"))
        
        fh = open(layer_directory+"/All_solutions_stat.csv", "w")   
        fh.write("Total Signatures,Stability,Matrix Frobenius%,avgStability\n") 
        fh.close()

        fh = open(layer_directory+"/All_solutions_stat_filter.csv", "w")
        fh.write("Total Signatures,Stability,Matrix Frobenius%,avgStability\n")
        fh.close()
        # The following for loop operates to extract data from each number of signature
        
        all_similirities_list = [] #this list is going to store the dataframes of different similirieties as items
        minimum_stabilities   = []

        # get the cutoff for normatization to handle the hypermutators 
        
        normalization_cutoff = sub.get_normalization_cutoff(genomes, manual_cutoff=100*genomes.shape[0])
        execution_parameters["normalization_cutoff"] = normalization_cutoff
        
        #pass the seed values to inner funtions:
        execution_parameters["seeds"] = seed

        if genomes.shape[1]<endProcess:
            endProcess=genomes.shape[1]
        
        #report the notmatlization criteria
        sysdata = open(out_put+"/JOB_METADATA.txt", "a")
        context_start_time=datetime.datetime.now()
        sysdata.write("\n##################################\n")
        sysdata.write("\n[{}] Analysis started for {}. Matrix size [{} rows x {} columns]\n".format(str(context_start_time).split(".")[0],mutation_type,genomes.shape[0],genomes.shape[1])) 
        if execution_parameters["matrix_normalization"]=="gmm":
                sysdata.write("\n[{}] Normalization GMM with cutoff value set at {}\n". \
                              format(str(datetime.datetime.now()).split(".")[0], normalization_cutoff)) 
        elif execution_parameters["matrix_normalization"]=="100X":
                sysdata.write("\n[{}] Normalization 100X with cutoff value set at {}\n". \
                              format(str(datetime.datetime.now()).split(".")[0],(genomes.shape[0]*100)))
        elif execution_parameters["matrix_normalization"]=="log2":
            sysdata.write("\n[{}] Normalization Log2\n". \
                              format(str(datetime.datetime.now()).split(".")[0]))
        elif execution_parameters["matrix_normalization"]=="none":
            sysdata.write("\n[{}] Analysis is proceeding without normalization\n". \
                          format(str(datetime.datetime.now()).split(".")[0]))
        else:
            sysdata.write("\n[{}] Normalization Custom with cutoff value set at {}\n". \
                              format(str(datetime.datetime.now()).split(".")[0],execution_parameters["matrix_normalization"]))

        sysdata.close()     

        # Create list of pairs (x,y) where x is poisson generator (will be used to create the same noise at each rank)
        # and y is a random generator. The pair will be used to spawn more generators.
        # Note: Poisson seed will be same in each pair, but random generator will be different.

        # initialize root seed sequence with seed
        seed_seq     = SeedSequence(int(execution_parameters["seeds"]))
        poisson_seed = seed_seq.spawn(1)

        # create num rank copies of the poisson seed so that noise is consistent across ranks for same replicate number
        poisson_list = [copy.deepcopy(poisson_seed) for x in range(startProcess, endProcess+1)]
        replicate_generators = seed_seq.spawn(endProcess + 1 - startProcess)
        cluster_generators   = seed_seq.spawn(endProcess + 1 - startProcess)
        noise_rep_pair       = []

        for i, j, k in zip(poisson_list, replicate_generators, cluster_generators):
            noise_rep_pair.append([i,j,k])




        for num_sigs in range(startProcess,endProcess+1):
            current_time_start = datetime.datetime.now()
            processAvg, \
            exposureAvg, \
            processStd, \
            exposureStd, \
            avgSilhouetteCoefficients, \
            clusterSilhouetteCoefficients, \
            finalgenomeErrors, \
            finalgenomesReconstructed, \
            finalWall, \
            finalHall, \
            finalBall, \
            converge_information, \
            reconstruction_error, \
            processes, \
            neg_mag,    \
            learningcurve, \
            acc, \
            f1, \
            Lrec, \
            Lce, \
            Ltot, \
            epochs, \
            n_filter = sub.decipher_signatures(execution_parameters,
                                                genomes= genomes,
                                                Y= Y,
                                                mut_context=m,
                                                i = num_sigs,
                                                lambda_c= lambda_c,
                                                lambda_p= lambda_p,
                                                lr= lr,
                                                noise_rep_pair=noise_rep_pair[num_sigs - startProcess],
                                                path = layer_directory)
            filt = len(processAvg)
            print(filt)
            information = []

            for idx in range(filt):
                stic = time.time()

                information.append([])

                if avgSilhouetteCoefficients[idx] > -1.0:
                    # stic = time.time()

                    replicate_generators = noise_rep_pair[num_sigs-startProcess][:2]

                    rep_generator = replicate_generators[1]

                    # rep_generator = batch_generator_pair[1][1]
                    rand_rng = Generator(PCG64DXSM(rep_generator))

                    bootstrapGenomes = genomes
                    # bootstrapGenomes[bootstrapGenomes<0.0001]= 0.0001
                    bootstrapGenomes[bootstrapGenomes == 0.0] = 1e-30
                    # normalize the samples to handle the hypermutators
                    totalMutations = np.sum(bootstrapGenomes, axis=0)
                    # print(normalization_cutoff)
                    norm = 'log2'
                    bootstrapGenomes = sub.normalize_samples(bootstrapGenomes, totalMutations, norm=norm,
                                                         normalization_cutoff=normalization_cutoff)
                    bootstrapGenomes = np.array(bootstrapGenomes)


                    # REFIT
                    H, B, kl, neg_mag_refit, acc_refit, f1_refit, Lrec_refit, Lce_refit, Ltot_refit, epochs_refit = sub.nnmf_cpu_refit(genomes, Y, processAvg[idx], lambda_c, lambda_p, lr,\
                                      processAvg[idx].shape[1], execution_parameters=execution_parameters, generator=rand_rng)  #uses custom function nnmf


                    exposureAvg[idx] = np.array(H)
                    weightAvg = np.array(B)
                    total = processAvg[idx].sum(axis=0)[np.newaxis]
                    processAvg[idx] = processAvg[idx] / total
                    exposureAvg[idx] = exposureAvg[idx] * total.T



                    for i in range(exposureAvg[idx].shape[1]):
                        genome = genomes[:, i]
                        genome = np.array(genome)
                        maxmutation = round(np.sum(genome))

                        est_genome = np.dot(processAvg[idx], exposureAvg[idx][:,i])
                        normalised_weights = exposureAvg[idx][:,i] / sum(exposureAvg[idx][:,i])
                        solution = normalised_weights * sum(genome)
                        # print(W1)
                        # convert the newExposure vector into list type structure
                        newExposure = list(solution)

                        maxcoef = max(newExposure)
                        idxmaxcoef = newExposure.index(maxcoef)
                        # We may need to tweak the maximum value of the new exposure to keep the total number of mutation equal to the original mutations in a genome
                        if np.sum(newExposure) != maxmutation:
                            # newExposure[idxmaxcoef] = round(newExposure[idxmaxcoef])+maxmutation-sum(newExposure)
                            newExposure[idxmaxcoef] = newExposure[idxmaxcoef] + maxmutation - sum(newExposure)
                        exposureAvg[idx][:,i] = newExposure


                    stoc = time.time()
                    print("Optimization time is {} seconds".format(stoc - stic))

                #Get total mutationation for each signature in reverse order and order the signatures from high to low mutation barden
                    signature_total_mutations = np.sum(exposureAvg[idx], axis =1).astype(int)
                    sorted_idx = np.argsort(-signature_total_mutations)
                    processAvg[idx] = np.take(processAvg[idx], sorted_idx, axis=1)
                    exposureAvg[idx] = np.take(exposureAvg[idx], sorted_idx, axis=0)
                    weightAvg = np.take(weightAvg, sorted_idx, axis = 1)
                    signature_total_mutations = np.sum(exposureAvg[idx], axis =1).astype(int)
                    processStd[idx]=np.take(processStd[idx], sorted_idx, axis=1)
                    exposureStd[idx]=np.take(exposureStd[idx], sorted_idx, axis=0)
                    clusterSilhouetteCoefficients[idx]=np.take(clusterSilhouetteCoefficients[idx], sorted_idx, axis=0)
                    signature_stats = pd.DataFrame({"Stability": clusterSilhouetteCoefficients[idx], "Total Mutations": signature_total_mutations})
                    minimum_stabilities.append([round(np.mean(clusterSilhouetteCoefficients[idx]),2)]) #here minimum stability is the average stability !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    # Compute the estimated genome from the processAvg and exposureAvg
                    est_genomes = np.dot(processAvg[idx], exposureAvg[idx])
                    #check the similarities between the original and estimated genome for each number of signatures
                    all_similarities, cosine_similarities = sub.calculate_similarities(genomes, est_genomes, colnames)
                    ##########################################################################################################################################################################
                    # store the resutls of the loop.  Here,  processStd and exposureStd are standard Errors, NOT STANDARD DEVIATIONS.
                    loopResults = [genomes, processAvg[idx], exposureAvg[idx], processStd[idx], exposureStd[idx], avgSilhouetteCoefficients[idx], clusterSilhouetteCoefficients[idx], signature_total_mutations, all_similarities, signature_stats, reconstruction_error[idx], finalgenomeErrors, finalgenomesReconstructed, converge_information, learningcurve, weightAvg, neg_mag, finalBall, finalWall, finalHall,  processes]

                    ################################# Export the results ###########################################################
                    #TODO: do not export?! in CV
                    if idx == 0:
                        rec, stability_mean, stability_min = sub.export_information(loopResults, m, layer_directory, index, colnames, wall=wall, sequence=sequence)
                    elif idx ==1:
                        rec, stability_mean, stability_min = sub.export_information(loopResults, m, layer_directory, index, colnames, wall=wall, sequence=sequence, filter = True)


                    all_similirities_list.append([all_similarities])
                    current_time_end = datetime.datetime.now()

                    information[idx].append([processAvg[idx], exposureAvg[idx], processStd[idx], exposureStd[idx], clusterSilhouetteCoefficients[idx],
                                        signature_total_mutations, signature_stats, all_similarities, acc_refit, f1_refit,
                                        Lrec_refit, Lce_refit, Ltot_refit, epochs_refit, rec, stability_mean,
                                        stability_min, acc, f1, Lrec, Lce, Ltot, epochs, neg_mag, weightAvg, learningcurve])  # Will be used during hierarchycal approach


            sysdata = open(out_put+"/JOB_METADATA.txt", "a")
            sysdata.write("\n[{}] {} de novo extraction completed for a total of {} signatures! \nExecution time:{}\n". \
                          format(str(datetime.datetime.now()).split(".")[0],mutation_type,processes,str(current_time_end-current_time_start).split(".")[0], current_time_end))
            sysdata.close()

        stability_min_2 = [None]*(2)
        stability_mean_2 = [None]*(2)
        rec_2 = [None]*(2)

        epochs_refit_2 = [None]*(2)
        Ltot_refit_2 = [None]*(2)
        Lce_refit_2 = [None]*(2)
        Lrec_refit_2 = [None]*(2)
        f1_refit_2 = [None]*(2)
        acc_refit_2 = [None]*(2)

        for j in range(filt):
            ########################################## Plot Stabiltity vs Reconstruction Error #############################
            # Print the Stabiltity vs Reconstruction Error as get the solution as well
            if j == 0:
                all_sol = "/All_solutions_stat.csv"
            elif j ==1:
                all_sol = "/All_solutions_stat_filter.csv"
            solution, all_stats = sub.stabVsRError(layer_directory+all_sol, layer_directory, title, all_similirities_list[j], mtype=mutation_type, stability=stability, min_stability=min_stability, combined_stability=combined_stability)
            all_stats.insert(1, 'Stability (Avg Silhouette)', minimum_stabilities[j]) #!!!!!!!!!!!!!!!!1 here minimum stability is avg stability
            all_stats=all_stats.set_index(["Signatures"])
            all_stats.to_csv(layer_directory+all_sol, sep = ",")

            # write the name of Samples and Matrix participating in each Layer.
            layer_genome = pd.DataFrame(genomes)
            layer_genome = layer_genome.set_index(index)
            layer_genome.columns = colnames
            layer_genome = layer_genome.rename_axis("Mutation Types", axis="columns")

            # record the samples
            layer_genome.to_csv(output+"/Samples.txt", sep = "\t", index_label=[layer_genome.columns.name])
            #similarity_dataframe.to_csv(data_stat_folder+"/Similatiry_Data_All_Sigs"+str(H_iteration)+".text", sep = "\t")
            del layer_genome
            ################################### Decompose the new signatures into global signatures   #########################
            processAvg = information[j][solution-startProcess][0]
            exposureAvg = information[j][solution-startProcess][1]
            processSTE = information[j][solution-startProcess][2]
            signature_stabilities = information[j][solution-startProcess][4]
            signature_total_mutations = information[j][solution-startProcess][5]
            signature_stats = information[j][solution-startProcess][6]
            all_similarities = information[j][solution-startProcess][7]

            learningcurve_final = information[j][solution - startProcess][-1]
            weightAvg = information[j][solution-startProcess][-2]
            neg_mag_final = information[j][solution-startProcess][-3]
            epochs = information[j][solution-startProcess][-4]
            Ltot = information[j][solution-startProcess][-5]
            Lce = information[j][solution-startProcess][-6]
            Lrec = information[j][solution-startProcess][-7]
            f1 = information[j][solution-startProcess][-8]
            acc = information[j][solution-startProcess][-9]
            stability_min_2[j] = information[j][solution-startProcess][-10]
            stability_mean_2[j] = information[j][solution-startProcess][-11]
            rec_2[j] = information[j][solution-startProcess][-12]

            epochs_refit_2[j] = information[j][solution-startProcess][-13]
            Ltot_refit_2[j] = information[j][solution-startProcess][-14]
            Lce_refit_2[j] = information[j][solution-startProcess][-15]
            Lrec_refit_2[j] = information[j][solution-startProcess][-16]
            f1_refit_2[j] = information[j][solution-startProcess][-17]
            acc_refit_2[j] = information[j][solution-startProcess][-18]


            # create the folder for the final solution/ De Novo Solution
            if j == 0:
                layer_directory1 = output+"/Suggested_Solution/"+mutation_type+"_De-Novo_Solution"
            elif j == 1:
                layer_directory1 = output+"/Suggested_Solution/"+mutation_type+"_De-Novo_Solution_filter"

            try:
                if not os.path.exists(layer_directory1):
                    os.makedirs(layer_directory1)
            except:
                print ("The {} folder could not be created".format("output"))

            # make the texts for signature plotting
            signature_stabilities = sub.signature_plotting_text(signature_stabilities, "Stability", "float")
            signature_total_mutations = sub.signature_plotting_text(signature_total_mutations, "Total Mutations", "integer")
            listOfSignatures = sub.make_letter_ids(idlenth = processAvg.shape[1], mtype=mutation_context)
            allgenomes = pd.DataFrame(allgenomes)

            #TODO: For original and filtered?
            exposureAvg = sub.make_final_solution(processAvg, allgenomes, listOfSignatures, layer_directory1, m, index, \
                           allcolnames, process_std_error = processSTE, signature_stabilities = signature_stabilities, \
                           signature_total_mutations = signature_total_mutations,denovo_exposureAvg  = exposureAvg, \
                           signature_stats = signature_stats, add_penalty=add_penalty, remove_penalty=remove_penalty, \
                           initial_remove_penalty=initial_remove_penalty, refit_denovo_signatures=refit_denovo_signatures, \
                           de_novo_fit_penalty=de_novo_fit_penalty, sequence=sequence, weights = weightAvg, neg_mag = neg_mag_final, learningcurve = learningcurve_final)

            layer_directory2 = output+"/Suggested_Solution/COSMIC_"+mutation_type+"_Decomposed_Solution"
            try:
                if not os.path.exists(layer_directory2):
                    os.makedirs(layer_directory2)
            except:
                print ("The {} folder could not be created".format("output"))

    sysdata = open(out_put+"/JOB_METADATA.txt", "a")
    end_time = datetime.datetime.now()
    sysdata.write("\n[{}] Analysis ended: \n".format(str(end_time).split(".")[0]))
    sysdata.write("\n-------Job Status------- \n")
    sysdata.write("Analysis of mutational signatures completed successfully! \nTotal execution time: "+str(end_time-start_time).split(".")[0]+" \nResults can be found in: "+" "+out_put+ " " +" folder")
    sysdata.close()

    print("\n\n \nYour Job Is Successfully Completed! Thank You For Using SigProfiler Extractor.\n ")


    return np.mean(acc), np.mean(f1), np.mean(Lrec), np.mean(Lce), np.mean(Ltot), np.mean(epochs), n_filter,    \
           stability_mean_2[0], stability_min_2[0], rec_2[0], acc_refit_2[0], f1_refit_2[0], Lrec_refit_2[0], Lce_refit_2[0], Ltot_refit_2[0], epochs_refit_2[0],\
           stability_mean_2[1], stability_min_2[1], rec_2[1], acc_refit_2[1], f1_refit_2[1], Lrec_refit_2[1], Lce_refit_2[1], Ltot_refit_2[1], epochs_refit_2[1]

    # Acc_train ; F1_train ; Rec_train ; Lce ; Ltot ; Stability_avg ; Stability_min ; Epochs    Acc_refit ; F1_refit ; Rec_refit ; Epoch_refit ;




import numpy as np
import pandas as pd


def test_sigProfilerExtractor(input_type,
                         output,
                         test_data,
                         test_label,
                         reference_genome="GRCh37",
                         opportunity_genome = "GRCh37",
                         cosmic_version=3.1,
                         context_type = "default",
                         exome = False,
                         minimum_signatures=1,
                         maximum_signatures=25,
                         nmf_replicates=500,
                         resample = False,
                         batch_size=1,
                         cpu=-1,
                         gpu=False,
                         nmf_init="random",
                         precision= "single",
                         matrix_normalization= "gmm",
                         seeds= "random",
                         min_nmf_iterations= 10000,
                         max_nmf_iterations=1000000,
                         nmf_test_conv= 10000,
                         # nmf_tolerance= 1e-15,
                         # nnls_add_penalty=0.05,
                         # nnls_remove_penalty=0.01,
                         # de_novo_fit_penalty=0.02,
                         # initial_remove_penalty=0.05,
                         nmf_tolerance=1e-18,
                         nnls_add_penalty=0.05,
                         nnls_remove_penalty=0.01,
                         de_novo_fit_penalty=0.0002,
                         initial_remove_penalty=0.0005,
                         refit_denovo_signatures=True,
                         collapse_to_SBS96=True,
                         clustering_distance="cosine",
                         export_probabilities=True,
                         make_decomposition_plots=True,
                         stability=0.8,
                         min_stability=0.2,
                         combined_stability=1.0,
                         get_all_signature_matrices= False,
                         lambda_c = 1e-40,
                         lr = 0.001,
                         lambda_p = 0.5,
                         filter = False,
                         model_path = '',
                         cancer = '_',
                         cosmic = '_'):

    data = pd.read_csv(test_data, sep="\t").iloc[:, :]
    label = pd.read_csv(test_label, sep="\t").iloc[:, :]
    data = data.dropna(axis=1, inplace=False)
    data = data.loc[:, (data != 0).any(axis=0)]
    X_test = data.iloc[:, 1:]
    X_test = np.array(X_test)
    allgenomes = X_test.copy()  # save the allgenomes for the final results
    # Contruct the indeces of the matrix
    # setting index and columns names of processAvg and exposureAvg
    index = data.iloc[:, 0]
    colnames = data.columns[1:]
    allcolnames = colnames.copy()  # save the allcolnames for the final results

    label = label.dropna(axis=1, inplace=False)
    label = label.loc[:, (label != 0).any(axis=0)]
    Y_test = label.iloc[:, 1:]
    Y_test = np.array(Y_test)
    classnames = label.columns[1:]

    # n  Number of samples
    n = X_test.shape[1]

    if model_path != '':
        model_path = model_path + '/SBS96/Suggested_Solution/SBS96_De-Novo_Solution/Signatures/'
        signatures = pd.read_csv(model_path + 'SBS96_De-Novo_Signatures.txt', sep="\t").iloc[:, 1:]
        # signatures
        signatures = pd.read_csv( model_path + 'SBS96_De-Novo_Signatures.txt', sep="\t").iloc[:, 1:]
        # classification weights
        weights = pd.read_csv( model_path + 'SBS96_De-Novo_Weights.txt', sep="\t").iloc[:, 1:]
        output_path = output

    else:
        if filter is False:
            output_path = output + '/SBS96/Suggested_Solution/SBS96_De-Novo_Solution/Signatures/'
        elif filter is True:
            output_path = output+ '/SBS96/Suggested_Solution/SBS96_De-Novo_Solution_filter/Signatures/'
        # signatures
        signatures = pd.read_csv( output_path + 'SBS96_De-Novo_Signatures.txt', sep="\t").iloc[:, 1:]
        # classification weights
        weights = pd.read_csv( output_path + 'SBS96_De-Novo_Weights.txt', sep="\t").iloc[:, 1:]

    # k  Number of signatures
    k = signatures.shape[1]

    exposureTest = np.zeros((k,n))
    similarityTest = np.zeros((1,n))

    stic = time.time()
    pool = mp.Pool()
    results = [pool.apply_async(ss.fit_signatures_pool, args=(X_test,signatures,x,)) for x in range(n)]
    pooloutput = [p.get() for p in results]
    pool.close()


    for i in range(len(pooloutput)):
        exposureTest[:,i]=pooloutput[i][0]
        similarityTest[:,i]=pooloutput[i][1]
    stoc = time.time()
    print("Optimization time is {} seconds".format(stoc-stic))

    Z = np.dot(weights, exposureTest)

    e_x = np.exp(Z - np.max(Z))
    Y_hat = e_x / np.sum(np.exp(Z- np.max(Z)), axis= 0)
    Y_pred = np.argmax(Y_hat,axis=0)
    Y_test = np.argmax(Y_test, axis=0)

    exposures = pd.DataFrame(exposureTest.T, index = data.columns[1:] , columns = signatures.columns)
    similarities = pd.DataFrame(similarityTest.T, index = data.columns[1:] , columns = ['similarity'])

    ## Final test (Seeds_3.txt)
    # if lambda_c == 0:
    #     exposures = exposures.rename(columns={"SBS96A": "SBS96E",
    #                                           "SBS96B": "SBS96A",
    #                                           "SBS96C": "SBS96B",
    #                                           "SBS96D": "SBS96D",
    #                                           "SBS96E": "SBS96C"})
    #
    # exposures = exposures.rename(columns={"SBS96A": "SBS96A_HR",
    #                                       "SBS96B": "SBS96B_Control",
    #                                       "SBS96C": "SBS96C_MMR",
    #                                       "SBS96D": "SBS96D_BER_UNG",
    #                                       "SBS96E": "SBS96E_BER_OGG1"})
    #
    # exposures = exposures[['SBS96A_HR','SBS96B_Control','SBS96C_MMR','SBS96D_BER_UNG','SBS96E_BER_OGG1']]

    exposures.to_csv(output_path + '/Exposures_test.txt', "\t", index_label=[exposures.columns.name])
    similarities.to_csv(output_path + '/Similarities_test.txt', "\t", index_label=[similarities.columns.name])
    plot_ac.plotActivity(output_path + '/Exposures_test.txt',
                              output_file= output_path + 'SBS96_Exposures_test.pdf', bin_size=50, log=False, lambda_c=lambda_c)

    # Store Yhat
    np.save(output_path + 'Y_hat_N{}'.format(n), Y_hat)
    Yhat_df = pd.DataFrame(data= Y_hat, columns= allcolnames, index = ['Control', 'MMR','HR', 'BER'])
    Yhat_df.to_csv(output + '/SBS96/Yhat_NMF.text', sep='\t')

    acc = np.sum(Y_pred == Y_test)/Y_test.shape[0]
    f1 = metrics.f1_score(Y_test, Y_pred, average='macro')
    print(n)
    print("accuracy = ", acc)
    print(f1)
    Y_pred_1h = np.zeros((Y_pred.size, Y_pred.max() + 1))
    Y_pred_1h[np.arange(Y_pred.size), Y_pred] = 1
    np.save(output_path + 'Y_pred_N{}.txt'.format(n), Y_pred_1h)


    rec = np.linalg.norm(X_test - signatures @ exposureTest, ord='fro')
    rec2 = np.linalg.norm(X_test - signatures @ exposureTest)
    return acc , f1 , rec

    pass

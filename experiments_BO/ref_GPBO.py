import os
import uuid


import torch
from torch.quasirandom import SobolEngine
import time





# botorch.settings.debug(True)

import warnings
warnings.filterwarnings("ignore")


from _utils import *
from _GPBO_utils import *


device = torch.device("cpu")
dtype = torch.double
tkwargs = {"device": device, "dtype": dtype}

SMOKE_TEST = os.environ.get("SMOKE_TEST")


def GP_BO(Function,  
          SEED,
           Trail_N = 0,
           N_iterations = 100, 
           Acquisition = 'EI', 
           INITIAL_DIR = None,        # None for testing
           SAVE_DIR = None,           # None for testing
           N_PENDING = 1000,          # Xpen has 1000 samples
           N_SEARCH_CANDIDATES = 1,          # Pend 1 for each iteration
           DEVICE='cpu',
           EXPERIMENT=None, # 'CONCATE'
          ):

    # with torch.no_grad():
    tkwargs = {"device": DEVICE, "dtype": dtype}
    print(f'Compute Setting: {DEVICE}')

    # Fix seed
    np.random.seed(SEED)  # Choose any integer seed value you want
    torch.manual_seed(SEED)  # For CPU
    # torch.cuda.manual_seed(SEED)  # For single GPU
    # torch.cuda.manual_seed_all(SEED)  # For all GPUs

    
    # Get dimension:
    DIM = Function.dim

    
    # Setup initial samples file Directory
    if INITIAL_DIR != None and EXPERIMENT==None:
        trained_X = torch.load(f'{INITIAL_DIR}/_trial_{Trail_N}.pt')
    elif INITIAL_DIR != None and EXPERIMENT=='CONCATE':
        dt = torch.load(f'{INITIAL_DIR}')
        trained_X = dt['trained_X'][:820,:]
        print(f'concate mode x: {trained_X.shape}')
        del dt
    else: 
        trained_X = torch.rand(50,DIM)


    # Get how many init sample I have
    N_INIT = trained_X.shape[0]
    
    # Evaluate
    GX, trained_Y = Function.evaluate(trained_X)

    # Feb 24 2025
    INIT_LOC =  trained_X.shape[0]
    trained_X = torch.cat([trained_X, torch.zeros(N_iterations*N_SEARCH_CANDIDATES, DIM)])
    trained_Y = torch.cat([trained_Y, torch.zeros(N_iterations*N_SEARCH_CANDIDATES, 1)])
    if GX is not None:
        GX = torch.cat([GX, torch.zeros(N_iterations*N_SEARCH_CANDIDATES, GX.shape[1])])
    ITER_IND_LOC = INIT_LOC
    
    # Things to store
    TIME_ARR = torch.zeros(N_iterations,)
    MAX_ARR = torch.zeros(N_iterations,)
    TOTAL_TIME = 0
    
    if Acquisition == 'TurBO' or Acquisition == 'SCBO' or 'TurBO' in Acquisition:
        TR_LB_List = torch.zeros(N_iterations, DIM)
        TR_UB_List = torch.zeros(N_iterations, DIM)
        
        batch_size = N_SEARCH_CANDIDATES
        state = Unified_TS_State(Function.dim, batch_size=batch_size)
        
        sobol = SobolEngine(Function.dim, scramble=True, seed=SEED)
        weights = torch.ones(1, Function.dim)
        
    # Reset the GPU device (more aggressive)
    # torch.cuda.reset_peak_memory_stats()
    # torch.cuda.empty_cache()
    # Optimization Loop
    
    for iter_ in range(N_iterations):
        
        trained_X = trained_X.to(**tkwargs)
        trained_Y = trained_Y.to(**tkwargs)
        
        # Start the time
        start_time = time.monotonic()


        # (1) Confirm the best so far
        if GX != None:
            sorted_ind = get_sorted_indices(trained_Y[:ITER_IND_LOC,:], GX[:ITER_IND_LOC,:])
        else:
            sorted_ind = get_sorted_indices(trained_Y[:ITER_IND_LOC,:], GX)
        BEST_F = trained_Y[sorted_ind[0],:]

        # (2) Fit the Gaussian Process and Calculate CEI
        if Acquisition == 'EI':
            best_candidate = get_next_candidates_EI(trained_X[:ITER_IND_LOC,:], trained_Y[:ITER_IND_LOC,:], BEST_F)
            
        elif Acquisition == 'CEI':
            best_candidate = get_next_candidates_CEI(trained_X[:ITER_IND_LOC,:], GX[:ITER_IND_LOC] if GX is not None else None, trained_Y[:ITER_IND_LOC,:], BEST_F)
            
        elif 'TurBO' in Acquisition:
            
            batch_size = N_SEARCH_CANDIDATES
            N_PENDING = min(5000, max(2000, 200 * Function.dim)) if not SMOKE_TEST else 4
            NUM_RESTARTS = 10 if not SMOKE_TEST else 2
            RAW_SAMPLES = 512 if not SMOKE_TEST else 4

            if Acquisition == 'TurBO_EI':
                (best_candidate,
                 tr_lb, tr_ub) = get_next_candidates_TurBO(trained_X[:ITER_IND_LOC,:], 
                                                              trained_Y[:ITER_IND_LOC,:], 
                                                              state,
                                                              batch_size,
                                                              N_PENDING,
                                                              NUM_RESTARTS,
                                                              RAW_SAMPLES,
                                                              DEVICE,
                                                               ACQF="ei",
                                                              )
            elif Acquisition == 'TurBO_TS' or Acquisition == 'TurBO':
                (best_candidate,
                 tr_lb, tr_ub) = get_next_candidates_TurBO(trained_X[:ITER_IND_LOC,:], 
                                                              trained_Y[:ITER_IND_LOC,:], 
                                                              state,
                                                              batch_size,
                                                              N_PENDING,
                                                              NUM_RESTARTS,
                                                              RAW_SAMPLES,
                                                              DEVICE,
                                                               ACQF="ts",
                                                              )
            
        elif Acquisition == 'SCBO':
            
            batch_size = N_SEARCH_CANDIDATES
            N_PENDING = min(5000, max(2000, 200 * Function.dim)) if not SMOKE_TEST else 4
            NUM_RESTARTS = 10 if not SMOKE_TEST else 2
            RAW_SAMPLES = 512 if not SMOKE_TEST else 4
            max_cholesky_size = float("inf")

            (best_candidate,
             tr_lb, tr_ub) = get_next_candidates_SCBO(trained_X[:ITER_IND_LOC,:], 
                                                     trained_Y[:ITER_IND_LOC,:], 
                                                     GX[:ITER_IND_LOC] if GX is not None else None,
                                                      state,
                                                      batch_size,
                                                      N_PENDING,
                                                      NUM_RESTARTS,
                                                      RAW_SAMPLES,
                                                      max_cholesky_size,
                                                      sobol,
                                                      DEVICE,
                                                      )
        

        # Timed before pending
        TOTAL_TIME += time.monotonic() - start_time
        TIME_ARR[iter_] = TOTAL_TIME

        # Feb 24 2025
        ITER_IND_LOC = INIT_LOC + N_SEARCH_CANDIDATES #*(iter_+1)
        trained_X[INIT_LOC:ITER_IND_LOC,:] = best_candidate
        g_, y_ = Function.evaluate(best_candidate)
        trained_Y[INIT_LOC:ITER_IND_LOC,:] = y_
        if GX is not None:
            GX[INIT_LOC:ITER_IND_LOC,:] = g_
        INIT_LOC = ITER_IND_LOC
        
        # # Append to trained_X
        # # print(f'best_candidate.shape: {best_candidate.shape}')
        # trained_X = torch.cat([trained_X, best_candidate])
        
        # # Evaluate the new X
        # GX, trained_Y = Function.evaluate(trained_X)
        # trained_Y = trained_Y.to(**tkwargs)
        # if GX!=None:
        #     GX = GX.to(**tkwargs)

        # UPDATE!!!
        if Acquisition == 'TurBO' or Acquisition == 'SCBO':
            state = update_ts_state(state, trained_Y, GX)
            TR_LB_List[iter_,:] = tr_lb
            TR_UB_List[iter_,:] = tr_ub

        # TOTAL_TIME += time.monotonic() - start_time
        # TIME_ARR[iter_] = TOTAL_TIME
        
        # Feb 24 2025
        # print(f'is GX None:{GX == None}')
        if GX != None:
            sorted_ind = get_sorted_indices(trained_Y[:ITER_IND_LOC,:], GX[:ITER_IND_LOC,:])
        else:
            sorted_ind = get_sorted_indices(trained_Y[:ITER_IND_LOC,:], GX)
        MAX_ARR[iter_] = trained_Y[:ITER_IND_LOC,:][sorted_ind[0],:]
        

        if SAVE_DIR == None:
            print(f'GP {Acquisition} Iter {iter_}) Opt: {MAX_ARR[iter_]}')
            if GX != None:
                print(f'is_feas: {(GX <= 0).all(dim=-1).any()}')

            


    GX, Y = Function.evaluate(trained_X)
    if GX != None:
        Y[(GX<=0).all(dim=1),0] = -1e10
        MAX_ARR = torch.cummax(Y, dim=0)[0]
        MAX_ARR = MAX_ARR[N_INIT:,:]
    else:
        MAX_ARR = torch.cummax(Y, dim=0)[0]
        MAX_ARR = MAX_ARR[N_INIT:,:]
    
    if SAVE_DIR == None:
        print(f'GP {Acquisition} Opt: {MAX_ARR[-1]}')
    else:
        save_folder = f"{SAVE_DIR}/GP_{Acquisition}/{Function.__class__.__name__}_DIM_{DIM}/"
        
        # Setup output file Directory
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
            
        save_file_name = f"{save_folder}/_trial_{Trail_N}_iters_{N_iterations}_{uuid.uuid4()}.pt"

        if Acquisition == 'TurBO' or Acquisition == 'SCBO':
            torch.save({
                'TIME_ARR': TIME_ARR,
                'MAX_ARR': MAX_ARR,
                'trained_X': trained_X,
                'TR_LB_List': TR_LB_List,
                'TR_UB_List': TR_UB_List,
            }, save_file_name)
        else:
            torch.save({
                'TIME_ARR': TIME_ARR,
                'MAX_ARR': MAX_ARR,
                'trained_X': trained_X,
            }, save_file_name)
            
        print(f'Save GPBO file at {save_file_name}')
        

    return trained_X, MAX_ARR



        




















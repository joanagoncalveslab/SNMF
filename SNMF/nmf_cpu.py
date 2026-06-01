#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 13:14:06 2020

@author: mishugeb
"""
"""
Implementation of non-negative matrix factorization for GPU
"""

from datetime import datetime

from nimfa.methods.seeding import nndsvd
import numpy as np
import torch
import torch.nn
from torch import nn
from sklearn import metrics



class NMF:
    def __init__(self, V, Y, rank, lambda_c = 1e-1, lambda_p = .5, lr = 0.003, report_loss = 1000, max_iterations=200000, tolerance=1e-8, tolerance_ce=1e-2, test_conv=1000, gpu_id=0, generator=None,
                 init_method='nndsvd', floating_point_precision='double', min_iterations=2000):

        """
        Run non-negative matrix factorisation using GPU. Uses beta-divergence.

        Args:
          V: Matrix to be factorised
          rank: (int) number of latent dimensnions to use in factorisation
          max_iterations: (int) Maximum number of update iterations to use during fitting
          tolerance: tolerance to use in convergence tests. Lower numbers give longer times to convergence
          test_conv: (int) How often to test for convergnce
          gpu_id: (int) Which GPU device to use
          generator: random generator, if None (default) datetime is used
          init_method: how to initialise basis and coefficient matrices, options are:
            - random (will always be the same if set generator != None)
            - NNDSVD
            - NNDSVDa (fill in the zero elements with the average),
            - NNDSVDar (fill in the zero elements with random values in the space [0:average/100]).
          floating_point_precision: (string or type). Can be `double`, `float` or any type/string which
              torch can interpret.
          min_iterations: the minimum number of iterations to execute before termination. Useful when using
              fp32 tensors as convergence can happen too early.
        """
        #torch.cuda.set_device(gpu_id)
        floating_point_precision = 'double'
        if floating_point_precision == 'single':
            self._tensor_type = torch.FloatTensor
            self._np_dtype = np.float32
        elif floating_point_precision == 'double':
            self._tensor_type = torch.DoubleTensor
            self._np_dtype = np.float64
        else:
            raise ValueError("Precision needs to be either 'single' or 'double'." )

        self.max_iterations = max_iterations
        self.min_iterations = min_iterations

        # If V is not in a batch, put it in a batch of 1
        if len(V.shape) == 2:
            V = V[None, :, :]

        if len(Y.shape) == 2:
            Y = Y[None, :, :]

        self._V = V.type(self._tensor_type)
        self._Y = Y.type(self._tensor_type)
        self._Y_max = torch.argmax(self._Y, dim= 1)
        self._fix_neg = nn.Threshold(0., 1e-20)
        # self._tolerance = tolerance
        self._tolerance = 1e-4
        self._tolerance_ce = tolerance_ce
        self._prev_loss = None
        self._prev_ce_loss = None
        self._kl_converged = False
        self._ce_converged = False
        self._iter = 0
        self._test_conv = test_conv
        #self._gpu_id = gpu_id
        self._rank = rank
        self._lambda_c = lambda_c
        self._lambda_p = lambda_p
        self._lr = lr
        self._report_loss = report_loss
        self._generator = generator
        # print(init_method)
        self._W, self._H, self._B= self._initialise_wh(init_method)
        self._neg_mag = {}
        self._results = {}
        self._hard_stop = 100000


    def _initialise_wh(self, init_method):
        """
        Initialise basis and coefficient matrices according to `init_method`
        """
        # print(init_method)
        init_method = 'random'
        if init_method == 'random':
            W = torch.unsqueeze(torch.from_numpy(self._generator.random((self._V.shape[1],self._rank), dtype=np.float64)),0)
            H = torch.unsqueeze(torch.from_numpy(self._generator.random((self._rank, self._V.shape[2]), dtype=np.float64)),0)
            B = torch.unsqueeze(torch.from_numpy(self._generator.random((self._Y.shape[1], self._rank), dtype=np.float64)), 0)
            if self._np_dtype is np.float32:
                W = W.float()
                H = H.float()
                B = B.float()

            return W, H, B

        # elif init_method == 'nndsvd':
        #     W = np.zeros([self._V.shape[0], self._V.shape[1], self._rank])
        #     H = np.zeros([self._V.shape[0], self._rank, self._V.shape[2]])
        #     nv = nndsvd.Nndsvd()
        #     for i in range(self._V.shape[0]):
        #         vin = np.mat(self._V.cpu().numpy()[i])
        #         W[i,:,:], H[i,:,:] = nv.initialize(vin, self._rank, options={'flag': 0})
        
        # elif init_method == 'nndsvda':
        #     W = np.zeros([self._V.shape[0], self._V.shape[1], self._rank])
        #     H = np.zeros([self._V.shape[0], self._rank, self._V.shape[2]])
        #     nv = nndsvd.Nndsvd()
        #     for i in range(self._V.shape[0]):
        #         vin = np.mat(self._V.cpu().numpy()[i])
        #         W[i,:,:], H[i,:,:] = nv.initialize(vin, self._rank, options={'flag': 1})
        
        # elif init_method == 'nndsvdar':
        #     W = np.zeros([self._V.shape[0], self._V.shape[1], self._rank])
        #     H = np.zeros([self._V.shape[0], self._rank, self._V.shape[2]])
        #     nv = nndsvd.Nndsvd()
        #     for i in range(self._V.shape[0]):
        #         vin = np.mat(self._V.cpu().numpy()[i])
        #         W[i,:,:], H[i,:,:] = nv.initialize(vin, self._rank, options={'flag': 2})
        # elif init_method =='nndsvd_min':
        #    W = np.zeros([self._V.shape[0], self._V.shape[1], self._rank])
        #    H = np.zeros([self._V.shape[0], self._rank, self._V.shape[2]])
        #    nv = nndsvd.Nndsvd()
        #    for i in range(self._V.shape[0]):
        #        vin = np.mat(self._V.cpu().numpy()[i])
        #        w, h = nv.initialize(vin, self._rank, options={'flag': 2})
        #        min_X = np.min(vin[vin>0])
        #        h[h <= min_X] = min_X
        #        w[w <= min_X] = min_X
        #        #W= np.expand_dims(W, axis=0)
        #        #H = np.expand_dims(H, axis=0)
        #        W[i,:,:]=w
        #        H[i,:,:]=h
        # W,H= initialize_nm(vin, nfactors, init=init, eps=1e-6,random_state=None)
        # W = torch.from_numpy(W).type(self._tensor_type)
        # H = torch.from_numpy(H).type(self._tensor_type)
        # return W, H

    @property
    def reconstruction(self):
        return self.W @ self.H

    @property
    def W(self):
        # Signatures
        # S -> (N, 96)
        return self._W

    @property
    def H(self):
        # Exposure
        # E -> (N, K)
        return self._H

    @property
    def B(self):
        # Weights of Logistic Regression
        # W -> (K, C)
        return self._B

    # @property
    # def Y_hat(self):
    #     # Y_hat = softmax(W E)
    #     # Y_hat -> (C, N)
    #     Z = self.B @ self.H
    #     # Z = torch.clamp(self.B @ self.H, -50, 50)
    #     # add max for stability
    #     # Stable softmax (Torch version)
    #     Z = Z - torch.amax(Z, dim=1, keepdim=True)  # shift by max for numerical stability
    #     e_x = torch.exp(Z)
    #     sum_e_x = torch.sum(e_x, dim=1, keepdim=True).clamp(min=1e-30)
    #     return e_x / sum_e_x
    @property
    def Y_hat(self):
        Z = self.B @ self.H                      # (1,C,N)
        Z = Z.clamp(min=-50.0, max=50.0)         # <- key line; 50..100 works
        Z = Z - torch.amax(Z, dim=1, keepdim=True)
        e_x = torch.exp(Z)
        return e_x / torch.sum(e_x, dim=1, keepdim=True).clamp(min=1e-30)



    @property
    def conv(self):
        try:
            return self._conv
        except: 
            return 0

    @property
    def _tot_loss(self):
        return self._fro_loss + self._lambda_c * self._ce_loss


    @property
    def _kl_loss(self):
        """
        KL(V || WH) = sum_{ij} V_ij * log(V_ij / (WH)_ij) - V_ij + (WH)_ij
        computed in a numerically safe way (avoids 0 * log(0) -> NaN).
        """
        V = self._V
        R = self.reconstruction.clamp(min=1e-25)

        # Hard fail fast if V is invalid
        if torch.isnan(V).any() or torch.isinf(V).any():
            print("[KL] V has NaN/Inf")
            return torch.tensor(float("nan"), dtype=torch.float64, device=V.device)
        if (V < 0).any():
            print("[KL] V has negative entries")
            return torch.tensor(float("nan"), dtype=torch.float64, device=V.device)

        # Mask zeros: only compute V*log(V/R) where V>0
        mask = V > 0
        Vp = V[mask]
        Rp = R[mask]

        term = (Vp * torch.log(Vp / Rp)).sum(dtype=torch.float64)
        kl = term - V.sum(dtype=torch.float64) + R.sum(dtype=torch.float64)

        if torch.isnan(kl) or torch.isinf(kl):
            print("[KL] NaN/Inf detected")
            print("  V: min/max =", float(V.min()), float(V.max()), " zeros =", int((V == 0).sum().item()))
            print("  R: min/max =", float(R.min()), float(R.max()))
        return kl


    # @property
    # def _kl_loss(self):
    #     # calculate kl_loss in double precision for better convergence criteria
    #     R_safe = self.reconstruction.clamp(min=1e-25) # avoid division by zero
    #     # kl_loss = (self._V * (self._V / self.reconstruction).log()).sum(dtype=torch.float64) - self._V.sum(dtype=torch.float64) + self.reconstruction.sum(dtype=torch.float64)
    #     kl_loss = (self._V * (self._V / R_safe).log()).sum(dtype=torch.float64) - self._V.sum(dtype=torch.float64) + R_safe.sum(dtype=torch.float64)
    #     if torch.isinf(kl_loss):
    #         print('nan in kl_loss')
    #         print(self._V)
    #         print(self.reconstruction)
    #         print(self._V / self.reconstruction)
    #         print((self._V * (self._V / R_safe).log()).sum(dtype=torch.float64))
    #         print((self._V * (self._V / self.reconstruction).log()).sum(dtype=torch.float64))
    #         print(self._V.sum(dtype=torch.float64))
    #         print(self.reconstruction.sum(dtype=torch.float64))
    #     return kl_loss

    @property
    def _fro_loss(self):
        # calculate - normalized - frobenius reconstruction error
        Vnorm = torch.norm(self._V, p='fro', dim=(1, 2)).clamp(min=1e-30)
        RecErr = torch.norm(self._V - self.reconstruction, p='fro', dim=(1, 2))
        return (RecErr / Vnorm)[0]
        # return (torch.norm(self._V - self.reconstruction, p='fro', dim=(1,2)))[0]


    # @property ---> _ce_loss uses np.log on a torch tensor → this can yield NaNs/Infs.
    # def _ce_loss(self):
    #     '''
    #     Calculate Cross Entropy loss from one-hot encoded input
    #      L  = -1/n * Y log(Y_hat) + lam/2 * w^2
    #     '''
    #     # n = self._V.shape[2]
    #     # loss = (-1/n) *  torch.sum(self._Y * np.log(self.Y_hat))
    #     return - torch.sum(self._Y * np.log(self.Y_hat))
   
    @property
    def _ce_loss(self):
        """Stable cross-entropy loss normalized by number of samples."""
        P = torch.clamp(self.Y_hat, min=1e-30)
        N = self._Y.shape[2]  # number of samples
        return - torch.sum(self._Y * torch.log(P)) / N



    @property
    def _acc(self):
        Y_pred = torch.argmax(self.Y_hat, dim = 1)
        return torch.sum(Y_pred == self._Y_max)/self._Y_max.shape[1]

    # @property
    # def _acc(self, Y_hat, Y_true):
    #     Y_pred = torch.argmax(Y_hat, dim = 1)
    #     return torch.sum(Y_pred == Y_true)/Y_true.shape[1]

    @property
    def _f1(self):
        Y_true = np.argmax(self._Y[0,:,:].numpy(), axis=0)
        Y_pred = np.argmax(self.Y_hat[0,:,:].numpy(), axis=0)
        return metrics.f1_score(Y_true, Y_pred, average='macro')



    @property
    def _rec(self):
        return round(np.linalg.norm(self._V - self.reconstruction, 'fro') / np.linalg.norm(self._V, 'fro'), 4)

    @property
    def generator(self):
        return self._generator

    @property
    def _loss_converged(self):
        """
        Check if loss has converged
        """
        if not self._iter:
            self._loss_init = self._kl_loss
        elif ((self._prev_loss - self._kl_loss) / self._loss_init) < self._tolerance:
            return True
        self._prev_loss = self._kl_loss
        return False

    @staticmethod
    def safe_div(num, den, eps=1e-25):
        # avoids divide-by-zero without hard clipping outputs
        return num / (den + eps)




    @property
    def _super_loss_converged(self):
        """
        Check convergence of Frobenius (reconstruction) + CE.
        IMPORTANT: compute losses once per call to avoid inconsistencies.
        """
        # --- compute once ---
        # kl = self._kl_loss  # (COMMENTED OUT) tensor/float64
        fro = self._fro_loss  # tensor (normalized Frobenius recon error)
        ce = self._ce_loss    # tensor (normalized CE)

        # kl_val = float(kl.item()) if hasattr(kl, "item") else float(kl)  # (COMMENTED OUT)
        fro_val = float(fro.item()) if hasattr(fro, "item") else float(fro)
        ce_val = float(ce.item()) if hasattr(ce, "item") else float(ce)

        # --- init ---
        if self._iter == 0:
            self._loss_init = fro_val          # was KL init
            self._ce_loss_init = ce_val
            self._prev_loss = fro_val          # was prev KL
            self._prev_ce_loss = ce_val
            self._kl_converged = False         # keep attribute for compatibility
            self._ce_converged = False
            self._fro_converged = False        # new flag
            return False

        # --- deltas (use current prev values, BEFORE updating them) ---
        fro_test_val = (abs(self._prev_loss - fro_val) / (abs(self._loss_init) + 1e-30))
        fro_pass = bool(fro_test_val < self._tolerance)

        ce_test_val = (abs(self._prev_ce_loss - ce_val) / (self._ce_loss_init + 1e-30))
        ce_pass = bool(ce_test_val < self._tolerance_ce)

        # update flags
        if (not getattr(self, "_fro_converged", False)) and fro_pass:
            self._fro_converged = True
        if (not self._ce_converged) and ce_pass:
            self._ce_converged = True

        converged = bool(self._fro_converged and self._ce_converged)

        # --- DEBUG PRINT: only when you would check stopping anyway ---
        if (self._iter % self._test_conv == 0):
            # keep your lambda_c effect debug
            N = self._V.shape[2]  # number of samples
            A = self.W.transpose(1, 2) @ self._V
            S = (self._lambda_c / 2) * self.B.transpose(1, 2) @ (self.Y_hat - self._Y)

            a = torch.norm(A).item()
            s = torch.norm(S).item()
            mx = (torch.max(torch.abs(S)) / (torch.max(torch.abs(A)) + 1e-30)).item()
            # print(f"[lambda_c effect] ||S||/||A||={s/(a+1e-30):.3e}   max|S|/max|A|={mx:.3e}")

            # print(
            #     f"[convchk iter={self._iter}] "
            #     f"FRO={fro_val:.6g} CE={ce_val:.6g} "
            #     f"fro_test=(|Δ|/init)={fro_test_val:.3e} tol={self._tolerance:.1e} "
            #     f"ce_test=(|Δ|/init)={ce_test_val:.3e} tol_ce={self._tolerance_ce:.1e} "
            #     f"flags: fro={self._fro_converged} ce={self._ce_converged} -> converged={converged}"
            # )

            # if not self._fro_converged:
            #     print("  blocked by: FRO criterion (fro_test >= tol or flag never set)")
            # if not self._ce_converged:
            #     print("  blocked by: CE criterion (ce_test >= tol_ce or flag never set)")

        # --- update prev AFTER debug and tests ---
        self._prev_loss = fro_val
        self._prev_ce_loss = ce_val

        return converged


    # @property
    # def _super_loss_converged(self):
    #     """
    #     Check convergence of KL + CE with explicit debug showing which criterion fails.
    #     IMPORTANT: compute losses once per call to avoid inconsistencies.
    #     """
    #     # --- compute once ---
    #     kl = self._kl_loss          # tensor/float64
    #     ce = self._ce_loss          # tensor
    #     kl_val = float(kl.item()) if hasattr(kl, "item") else float(kl)
    #     ce_val = float(ce.item()) if hasattr(ce, "item") else float(ce)

    #     # --- init ---
    #     if self._iter == 0:
    #         self._loss_init = kl_val
    #         self._ce_loss_init = ce_val
    #         self._prev_loss = kl_val
    #         self._prev_ce_loss = ce_val
    #         self._kl_converged = False
    #         self._ce_converged = False
    #         return False

    #     # --- deltas (use current prev values, BEFORE updating them) ---
    #     # your original KL criterion (one-sided):
    #     kl_test_val = (abs(self._prev_loss - kl_val) / (abs(self._loss_init) + 1e-30))
    #     kl_pass = bool(kl_test_val < self._tolerance)

    #     ce_test_val = (abs(self._prev_ce_loss - ce_val) / (self._ce_loss_init + 1e-30))
    #     ce_pass = bool(ce_test_val < self._tolerance_ce)

    #     # update flags (same logic as you have)
    #     if (not self._kl_converged) and kl_pass:
    #         self._kl_converged = True
    #     if (not self._ce_converged) and ce_pass:
    #         self._ce_converged = True

    #     converged = bool(self._kl_converged and self._ce_converged)

    #     # --- DEBUG PRINT: only when you would check stopping anyway ---
    #     # (otherwise this prints every iter and slows things down)
    #     if (self._iter % self._test_conv == 0):
    #         if self._iter % self._test_conv == 0:
    #             N = self._V.shape[2]  # number of samples

    #             A = self.W.transpose(1, 2) @ self._V                      # (1,K,N)
    #             # S = (self._lambda_c / 2) * self.B.transpose(1, 2) @ ((self.Y_hat - self._Y) / N)  # (1,K,N)
    #             S = (self._lambda_c / 2) * self.B.transpose(1, 2) @ ((self.Y_hat - self._Y))  # (1,K,N)

    #             a = torch.norm(A).item()
    #             s = torch.norm(S).item()
    #             mx = (torch.max(torch.abs(S)) / (torch.max(torch.abs(A)) + 1e-30)).item()

    #             print(f"[lambda_c effect] ||S||/||A||={s/(a+1e-30):.3e}   max|S|/max|A|={mx:.3e}")

    #         print(
    #             f"[convchk iter={self._iter}] "
    #             f"KL={kl_val:.6g} CE={ce_val:.6g} "
    #             f"kl_test=((prev-cur)/init)={kl_test_val:.3e} tol={self._tolerance:.1e} "
    #             f"ce_test=(|Δ|/init)={ce_test_val:.3e} tol_ce={self._tolerance_ce:.1e} "
    #             f"flags: kl={self._kl_converged} ce={self._ce_converged} -> converged={converged}"
    #         )

    #         # show WHICH thing blocks convergence
    #         if not self._kl_converged:
    #             print("  blocked by: KL criterion (kl_test >= tol or flag never set)")
    #         if not self._ce_converged:
    #             print("  blocked by: CE criterion (ce_test >= tol_ce or flag never set)")

    #     # --- update prev AFTER debug and tests ---
    #     self._prev_loss = kl_val
    #     self._prev_ce_loss = ce_val

    #     return converged



    @property
    def neg_mag(self):
        return self._neg_mag


    @property
    def results(self):
        return self._results



    def _debug_stats(self, tag="", max_classes=5):
        """Lightweight debug stats: exposure scale, B scale, logits / confidence."""
        with torch.no_grad():
            eps = 1e-30

            # H: exposures (1, K, N)
            H = self.H
            N = H.shape[2]

            # per-sample total exposure: sum over K -> (1, N)
            H_sum_per_sample = H.sum(dim=1).squeeze(0)  # (N,)
            H_sum_mean = H_sum_per_sample.mean().item()
            H_sum_med  = H_sum_per_sample.median().item()
            H_sum_min  = H_sum_per_sample.min().item()
            H_sum_max  = H_sum_per_sample.max().item()

            # also useful: mean H value
            H_mean = H.mean().item()

            # B: (1, C, K) in your code
            B = self.B
            B_abs_mean = B.abs().mean().item()
            B_norm = torch.norm(B, p="fro").item()

            # logits and confidence
            Z = (self.B @ self.H)  # (1, C, N)
            zmin = Z.amin().item()
            zmax = Z.amax().item()

            P = self.Y_hat  # (1, C, N)
            pmax = P.max(dim=1).values.squeeze(0)  # (N,)
            pmax_mean = pmax.mean().item()

            # class balance (pred histogram) quick view
            ypred = torch.argmax(P, dim=1).squeeze(0)  # (N,)
            # show up to first `max_classes` counts (assumes C small)
            C = P.shape[1]
            counts = torch.bincount(ypred, minlength=C).float()
            frac = (counts / (counts.sum() + eps)).cpu().numpy()

            print(
                f"[dbg{tag}] "
                f"H_sum(mean/med/min/max)={H_sum_mean:.3g}/{H_sum_med:.3g}/{H_sum_min:.3g}/{H_sum_max:.3g} "
                f"H_mean={H_mean:.3g} "
                f"|B|(mean)={B_abs_mean:.3g} ||B||F={B_norm:.3g} "
                f"logits[min,max]={zmin:.3g},{zmax:.3g} "
                f"pmax_mean={pmax_mean:.3g} "
                f"pred_frac={np.array2string(frac[:max_classes], precision=2, separator=',')}"
            )






    def fit(self, beta=2, supervised = True):
        """
        Fit the basis (W) and coefficient (H) matrices to the input matrix (V) using multiplicative updates and
            beta divergence
        Args:
          beta: value to use for generalised beta divergence. Default is 1 for KL divergence
            beta == 2 => Euclidean updates
            beta == 1 => Generalised Kullback-Leibler updates
            beta == 0 => Itakura-Saito updates
        """
        # print('lr = ',self._lr)
        # print('lam_c = ',self._lambda_c)
        # print('l2 = ',self._lambda_p)

        #
        # results_u = {'epoch': [], 'Ltot':[], 'Lrec':[], 'Lce':[], 'acc':[], 'rec':[]}
        # results_u = {'epoch': [], 'Ltot':[], 'Lrec':[], 'Lce':[], 'acc':[], 'rec':[]}


        with torch.no_grad():
            # Optional warm start (e.g., from unsupervised NMF)
            if hasattr(self, "_W") and hasattr(self, "_H") and hasattr(self, "_B"):
                self._W = self._W.clone()
                self._H = self._H.clone()
                self._B = self._B.clone()

            def stop_iterations():
                # hard cap
                if self._iter >= self._hard_stop:
                    return [True, self._iter]

                # your existing convergence logic
                stop = (self._V.shape[0] == 1) and \
                    (self._iter % self._test_conv == 0) and \
                    self._super_loss_converged and \
                    (self._iter > self.min_iterations)

                return [stop, self._iter]


            if (beta == 2 and not supervised):
                for self._iter in range(self.max_iterations):
                    self._H = self.H * (self.W.transpose(1, 2) @ self._V) / (self.W.transpose(1, 2) @ (self.W @ self.H))
                    self._W = self.W * (self._V @ self.H.transpose(1, 2)) / (self.W @ (self.H @ self.H.transpose(1, 2)))

                    if stop_iterations()[0]:
                        self._conv=stop_iterations()[1]
                        break

            elif (beta == 2 and supervised):
                for self._iter in range(self.max_iterations):

                    N = self._V.shape[2]  # number of samples

                    WH = self.W @ self.H
                    denH = self.W.transpose(1, 2) @ WH
                    numH = (self.W.transpose(1, 2) @ self._V) - (self._lambda_c/2) * self.B.transpose(1,2) @ (self.Y_hat - self._Y)
                    # numH = (self.W.transpose(1, 2) @ self._V) - (self._lambda_c / 2) * self.B.transpose(1, 2) @ ((self.Y_hat - self._Y) / N)

                    self._H_new = self.H * self.safe_div(numH, denH)



                    # self._H_new = self.H * ((self.W.transpose(1, 2) @ self._V) - (self._lambda_c/2) * self.B.transpose(1,2) @ (self.Y_hat - self._Y) ) / (self.W.transpose(1, 2) @ (self.W @ self.H))

                    #TODO: store magnitudes:
                    self._neg_H = self._H_new < 0
                    self._H_new[self._neg_H] = 1e-25

                    temp = []


                    # # self._H = self._H_new
                    # div = (self.W @ (self.H @ self.H.transpose(1, 2)))
                    # div[div == 0] = 1e-25
                    # self._W_new = self.W * (self._V @ self.H.transpose(1, 2)) / div
                    # # self._W_new[self._W_new < 0] = 1e-6
                    # W_sum = torch.sum(self._W_new, 1).unsqueeze(dim=1)
                    # W_sum[W_sum == 0] = 1e-25
                    # self._W_new = self._W_new / W_sum


                    EPS = 1e-25  # keep consistent with your other eps
                    # --- W update (same logic, but numerically safe) ---
                    denW = self.W @ (self.H @ self.H.transpose(1, 2))     # (1, 96, K)
                    numW = self._V @ self.H.transpose(1, 2)               # (1, 96, K)

                    # safe multiplicative ratio (prevents NaN/inf from tiny denoms)
                    # ratioW = safe_div(numW, denW, eps=EPS)
                    # self._W_new = self.W * ratioW
                    self._W_new = self.W * self.safe_div(numW, denW)


                    # maintain non-negativity floor (rare, but safe if safe_div returns 0s)
                    self._W_new = torch.clamp(self._W_new, min=EPS)

                    # preserve your normalization (columns/signatures sum to 1 across mutation types)
                    W_sum = torch.sum(self._W_new, dim=1, keepdim=True)   # (1, 1, K)
                    W_sum = torch.clamp(W_sum, min=EPS)
                    self._W_new = self._W_new / W_sum




                    # self._W = self._W_new

                    #avoid sum to 0
                    W_sum = torch.sum(self._W, dim=1, keepdim=True).clamp_min(1e-25)
                    self._W = (self._W / W_sum).clamp_min(1e-25)
                    N = self._V.shape[2]
                    # self._B_new = (1 - 2 * self._lambda_p) * self.B - self._lr * (self.Y_hat - self._Y) @ self.H.transpose(1,2)
                    # self._B_new = (1 - 2 * self._lambda_p) * self.B - self._lr * ((self.Y_hat - self._Y) / N) @ self.H.transpose(1, 2)
                    self._B_new = self.B - self._lr * ((self.Y_hat - self._Y)  @ self.H.transpose(1,2) + 2 * self._lambda_p * self.B)

# #! DEBUG: cap magnitude of B to avoid overflow
#                     Zcap = 50.0
#                     Z = self._B_new @ self.H
#                     zmax = torch.max(torch.abs(Z)).clamp(min=1e-30)
#                     if zmax > Zcap:
#                         self._B_new = self._B_new * (Zcap / zmax)





                    if torch.isnan(self._H_new).any():
                        print('nan in H')
                        print(self.H)
                        print(self.W.transpose(1, 2) @ (self.W @ self.H))

                    if torch.isnan(self._W_new).any():
                        print('nan in W')
                        print(self.W)
                        print(self.W @ (self.H @ self.H.transpose(1, 2)))


                    if torch.isnan(self.Y_hat).any():
                        print('nan in Y_hat')

                        print(self.B)
                        # print(self.Y_hat)
                        # print(np.log(self.Y_hat))
                        # print(self.Y_hat)
                        # print(np.log(self.Y_hat))
                        break

                    if torch.isnan(self._ce_loss):
                        print('nan in ce loss')
                        print(self._ce_loss)
                        print(self.Y_hat)
                        break

                    self._H = self._H_new
                    self._W = self._W_new
                    self._B = self._B_new


                    # TODO: learning curve
                    # call acc with validation data
                    if (self._iter % self._report_loss == 0 or self._iter in [100, 200, 300, 400, 500]):
                        # print('Epoch: {:.0f} ; Ltot = {:.2f} ;  Lrec = {:.2f} ; Lce = {:.2f} ; acc = {:.3f}'.format(self._iter, self._tot_loss,  self._fro_loss, self._ce_loss, self._acc))
                        print('Epoch: {:.0f} ; Ltot = {:.2f} ;  Lrec = {:.2f} ; Lce = {:.2f} ; acc = {:.3f}'.format(
                                self._iter, self._tot_loss, self._fro_loss, self._ce_loss, self._acc) )
                        # self._debug_stats(tag=" fit")

                        if self._iter not in self._results.keys():
                            self._results[self._iter] = [0.]*5
                        self._results[self._iter][0] += self._tot_loss.item()
                        self._results[self._iter][1] += self._fro_loss.item()
                        self._results[self._iter][2] += self._ce_loss.item()
                        self._results[self._iter][3] += self._acc.item()

#! DEBUG:
                        # with torch.no_grad():
                        #      # quantile vector must match dtype/device of the tensor you query
                        #     q = torch.tensor([0.0, 0.5, 1.0], dtype=self.W.dtype, device=self.W.device)

                        #     # W column-sums over mutation types (dim=1 is mut-type axis in your batch format)
                        #     W_colsum = self.W.sum(dim=1)  # shape: (1, K)
                        #     w_q = torch.quantile(W_colsum, q, dim=1).squeeze(0).cpu().numpy()
                        #     print(f"W colsum (min/med/max): {w_q}")

                        #     # H distribution (exposures)
                        #     h_q = torch.quantile(self.H, q).cpu().numpy()
                        #     print(f"H (min/med/max): {h_q}")

                        #     # Norm of B (LR weights)
                        #     print(f"||B||_F: {torch.norm(self.B, p='fro').item():.6g}")

                        #     # Logits range (pre-softmax)
                        #     Z = self.B @ self.H  # (1, C, N)
                        #     print(f"logit range: {Z.amin().item():.6g} .. {Z.amax().item():.6g}")

                        #     # Optional: sanity checks for NaN/Inf (prints once per report interval)
                        #     if torch.isnan(self.W).any() or torch.isnan(self.H).any() or torch.isnan(self.B).any():
                        #         print("[warn] NaN detected in W/H/B")
                        #     if torch.isinf(self.W).any() or torch.isinf(self.H).any() or torch.isinf(self.B).any():
                        #         print("[warn] Inf detected in W/H/B")



                    #
                    # if (torch.any(torch.isnan(self.B))):
                    #     self._conv=stop_iterations()[1]
                    #     break
                    if stop_iterations()[0]:
                        self._conv=stop_iterations()[1]
                        break

            #
            # # Optimisations for the (common) beta=1 (KL) case.
            # elif beta == 1:
            #     ones = torch.ones(self._V.shape).type(self._tensor_type)
            #     for self._iter in range(self.max_iterations):
            #         ht = self.H.transpose(1, 2)
            #         numerator = (self._V / (self.W @ self.H)) @ ht
            #
            #         denomenator = ones @ ht
            #         self._W *= numerator / denomenator
            #
            #         wt = self.W.transpose(1, 2)
            #         numerator = wt @ (self._V / (self.W @ self.H))
            #         denomenator = wt @ ones
            #         self._H *= numerator / denomenator
            #         if stop_iterations()[0]:
            #             self._conv=stop_iterations()[1]
            #             break
            #
            # else:
            #     for self._iter in range(self.max_iterations):
            #         self._H = self.H * ((self.W.transpose(1, 2) @ (((self.W @ self.H) ** (beta - 2)) * self._V)) /
            #                            (self.W.transpose(1, 2) @ ((self.W @ self.H)**(beta-1))))
            #         self._W = self.W * ((((self.W@self.H)**(beta-2) * self._V) @ self.H.transpose(1, 2)) /
            #                            (((self.W @ self.H) ** (beta - 1)) @ self.H.transpose(1, 2)))
            #         if stop_iterations()[0]:
            #             self._conv=stop_iterations()[1]
            #             break
            #

    def refit(self, W, beta=2, supervised = True):
        """
        Refit the coefficient (H) and weight (B) matrices to the input matrix (V) given fixed basis (W) matrix
        using multiplicative updates and beta divergence
        Args:
          beta: value to use for generalised beta divergence. Default is 1 for KL divergence
            beta == 2 => Euclidean updates
            beta == 1 => Generalised Kullback-Leibler updates
            beta == 0 => Itakura-Saito updates
        """
        # --- reset convergence bookkeeping for a fresh run ---
        self._iter = 0
        self._prev_loss = None
        self._prev_ce_loss = None
        self._kl_converged = False
        self._ce_converged = False

        # refit needs looser tolerances with counts
        tol_old = self._tolerance
        tolce_old = self._tolerance_ce

        self._tolerance = 1e-4        # instead of 1e-8
        self._tolerance_ce = 1e-2     # instead of 1e-2?

        # (optional but helps) clear any cached initial losses
        if hasattr(self, "_loss_init"): del self._loss_init
        if hasattr(self, "_ce_loss_init"): del self._ce_loss_init

        self._W[0, :, :] = torch.tensor(W)
        self._lambda_c = 0.
        N = self._V.shape[2]  # number of samples
        print('lr = ',self._lr)
        print('lam_c = ',self._lambda_c)
        print('l2 = ',self._lambda_p)






        with torch.no_grad():
            def stop_iterations():
                # --- hard stop after 100k epochs ---
                if self._iter >= 100_000:
                    print(f"[hard stop] reached max_iter={self._iter}")
                    return [True, self._iter]

                # --- existing convergence logic ---
                stop = (self._V.shape[0] == 1) and \
                    (self._iter % self._test_conv == 0) and \
                    self._super_loss_converged and \
                    (self._iter > self.min_iterations)

                return [stop, self._iter]

            
            # def stop_iterations_refit():
            #     # Only use reconstruction convergence when lambda_c == 0
            #     if self._lambda_c == 0.0:
            #         converged = self._loss_converged   # KL-based criterion you already have
            #     else:
            #         converged = self._super_loss_converged

            #     stop = (self._V.shape[0] == 1) and \
            #         (self._iter % self._test_conv == 0) and \
            #         converged and \
            #         (self._iter > self.min_iterations)
            #     return [stop, self._iter]


            if (beta == 2 and not supervised):
                # for self._iter in range(self.max_iterations):
                #     self._H = self.H * (self.W.transpose(1, 2) @ self._V) / (self.W.transpose(1, 2) @ (self.W @ self.H))
                #     self._W = self.W * (self._V @ self.H.transpose(1, 2)) / (self.W @ (self.H @ self.H.transpose(1, 2)))
                #     if stop_iterations()[0]:
                #         self._conv=stop_iterations()[1]
                #         break
                pass

            elif (beta == 2 and supervised):
                for self._iter in range(self.max_iterations):
                    # self._H_new = self.H * ((self.W.transpose(1, 2) @ self._V) - (self._lambda_c/2) * self.B.transpose(1,2) @ ((self.Y_hat - self._Y) / N) ) / (self.W.transpose(1, 2) @ (self.W @ self.H))
                    # self._H_new = self.H * ((self.W.transpose(1, 2) @ self._V) - (self._lambda_c/2) * self.B.transpose(1,2) @ ((self.Y_hat - self._Y)) ) / (self.W.transpose(1, 2) @ (self.W @ self.H))
                    numH = (self.W.transpose(1,2) @ self._V)
                    denH = (self.W.transpose(1,2) @ (self.W @ self.H))
                    self._H_new = self.H * self.safe_div(numH, denH)

                    #TODO: store magnitudes:
                    self._neg_H = self._H_new < 0

                    temp = []
                    temp_i = np.where(self._neg_H)[1]
                    temp_j = np.where(self._neg_H)[2]
                    # if len(temp_i > 0):
                    #     # print(len(temp_i))
                    #     for n in range(len(temp_i)):
                    #         temp.append(self._H_new[0, temp_i[n], temp_j[n]].item())
                    # self._neg_mag[self._iter] = temp

                    self._H_new[self._neg_H] = 1e-25

                    # self._H = self._H_new

                    if self._iter == 0:
                        w_sum = torch.sum(self._W, dim=1, keepdim=True).clamp(min=1e-25)
                        self._W = self._W / w_sum
                        # self._H_new = self._H_new * w_sum



                    # self._B = self.B - self._lr * (self.Y_hat - self._Y) @ self.H.transpose(1,2) + self._lambda_p * self.B
                    # self._B_new = (1 - 2 * self._lambda_p) * self.B - self._lr * ((self.Y_hat - self._Y) / N)  @ self.H.transpose(1,2)
                    # self._B_new = (1 - 2 * self._lambda_p) * self.B - self._lr * ((self.Y_hat - self._Y))  @ self.H.transpose(1,2)
                    self._B_new = self.B - self._lr * ((self.Y_hat - self._Y)  @ self.H.transpose(1,2) + 2 * self._lambda_p * self.B)

#! DEBUG: cap magnitude of B to avoid overflow
                    Zcap = 50.0
                    Z = self._B_new @ self.H
                    zmax = torch.max(torch.abs(Z)).clamp(min=1e-30)
                    if zmax > Zcap:
                        self._B_new = self._B_new * (Zcap / zmax)


                    self._H = self._H_new
                    self._B = self._B_new






                    if (self._iter % self._report_loss == 0):
                        print(
                            f"Epoch {self._iter}  "
                            f"Lrec={float(self._fro_loss):.5f}  "
                            f"Lce={float(self._ce_loss):.5f}  "
                            f"Ltot={float(self._tot_loss):.5f}  "
                            f"acc={float(self._acc):.3f}"
                        )
                        # self._debug_stats(tag=" refit")


                    # if (self._iter % self._report_loss == 0):
                    #     # print('Epoch: {:.0f} ; Ltot = {:.2f} ;  Lrec = {:.2f} ; Lce = {:.2f} ; acc = {:.3f}'.format(self._iter, self._tot_loss,  self._fro_loss, self._ce_loss, self._acc))
                    #     print(f"Epoch {self._iter}  Lrec={float(self._fro_loss):.5f}  Lce={float(self._ce_loss):.5f}  Ltot={float(self._tot_loss):.5f}")

                    if torch.isnan(self._H_new).any():
                        print('nan in H')


                    if torch.isnan(self.Y_hat).any():
                        print('nan in Y_hat')
                        print(self.W)
                        print(self.H)
                        print(self.B)
                        print(self.Y_hat)
                        print(torch.log(torch.clamp(self.Y_hat, min=1e-30)))
                        print(self.Y_hat)
                        print(np.log(self.Y_hat))
                        break

                    if torch.isnan(self._ce_loss):
                        print(self.W)
                        print(self.H)
                        print(self.B)
                        print(self.Y_hat)
                        print(torch.log(torch.clamp(self.Y_hat, min=1e-30)))
                        break
                    #
                    # if (torch.any(torch.isnan(self.B))):
                    #     self._conv=stop_iterations()[1]
                    #     break
                    if stop_iterations()[0]:
                        self._conv=stop_iterations()[1]
                        break

                    # if stop_iterations_refit()[0]:
                    #     self._conv = stop_iterations_refit()[1]
                    #     break



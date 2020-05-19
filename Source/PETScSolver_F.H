#ifndef PETScSolver_F_H_
#define PETScSolver_F_H_

#include <AMReX_BLFort.H>

#ifdef __cplusplus
extern "C" {
#endif
            
    void petsc_fillmatbox (const int* lo, const int* hi,
                           const void* nrows,
                           void* ncolsnE, void* ncolsphiV,
                           void* rowsnE, void* rowsphiV,
                           void* colsnE, amrex_real* matnE,
                           void* colsphiV, amrex_real* matphiV,
                           const void* cell_id_ne, const int* clo, const int* chi,
                           const void* cell_id_phiV, const int* cplo, const int* cphi,
                           const void* cell_id_ne_begin, const void* cell_id_phiV_begin,
                           amrex_real* diaginv, const int* dlo, const int* dhi,
                           const amrex_real* Dex, const int* Dexlo, const int* Dexhi,
#if (AMREX_SPACEDIM >= 2)
                           const amrex_real* Dey, const int* Deylo, const int* Deyhi,
#if (AMREX_SPACEDIM == 3)
                           const amrex_real* Dez, const int* Dezlo, const int* Dezhi,
#endif
#endif
                           const amrex_real* Ueffx, const int* Uxlo, const int* Uxhi,
#if (AMREX_SPACEDIM >= 2)
                           const amrex_real* Ueffy, const int* Uylo, const int* Uyhi,
#if (AMREX_SPACEDIM == 3)
                           const amrex_real* Ueffz, const int* Uzlo, const int* Uzhi,
#endif
#endif
                           const amrex_real* neKex, const int* nexlo, const int* nexhi,
#if (AMREX_SPACEDIM >= 2)
                           const amrex_real* neKey, const int* neylo, const int* neyhi,
#if (AMREX_SPACEDIM == 3)
                           const amrex_real* neKez, const int* nezlo, const int* nezhi,
#endif
#endif
                           const amrex_real* lapl_fac,
                           const int* bct_ne, const amrex_real* bcl_ne, 
                           const int* bct_phiV, const amrex_real* bcl_phiV,
                           const amrex_real* s_dtDiffI, const amrex_real* s_dtDrift,
                           const amrex_real* s_Ie, const amrex_real* s_L, const amrex_real* dx,
                           const amrex_real* dt);

#ifdef __cplusplus
};
#endif

#endif

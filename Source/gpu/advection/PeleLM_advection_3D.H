#ifndef PELELM_ADVECTION_3D_H
#define PELELM_ADVECTION_3D_H
#include <AMReX_Gpu.H> 
#include <AMReX_FArrayBox.H>
#include <AMReX_BCRec.H> 
#include <AMReX_BC_TYPES.H>
#include <AMReX_Array.H>
#include <PeleLM_index_macros.H>

/* This header file contains the inlined __host__ __device__ functions required for 
   the scalar advection routines for PeleLM 3D. It also contains fuction declarations 
   for controlling host functions.  */


AMREX_GPU_HOST_DEVICE 
inline amrex::Real vanLeer(const amrex::Real a, const amrex::Real b, const amrex::Real c)
{    
    amrex::Real dsc = 0.5e0*(b - c); 
    amrex::Real dsl = 2.e0*(a - c); 
    amrex::Real dsr = 2.e0*(b - a); 
    return std::copysign(1, dsc)*amrex::min(std::abs(dsc), 
                                 amrex::min(std::abs(dsl), std::abs(dsr))); 
}

AMREX_GPU_HOST_DEVICE
inline void PeleLM_ppm_xbc(const int i, const int j, const int k, const int n, 
                           amrex::Real &sm, amrex::Real &sp, 
                           amrex::Real &sedge,
                           const amrex::Array4<const amrex::Real> &s, 
                           const int bclo, const int bchi, 
                           const int domlo, const int domhi)
{
        if(bclo == EXT_DIR || bclo == HOEXTRAP){
            if(i == domlo+1){
                sedge = -0.2*s(i-2, j, k, n) + 0.75*s(i-1,j, k, n) 
                       +  0.5*s(i,j,k,n) - 0.05*s(i+1,j,k,n); 
                sedge = amrex::max(sedge, amrex::min(s(i,j,k,n), s(i+1,j,k,n))); 
                sedge = amrex::min(sedge, amrex::max(s(i,j,k,n), s(i+1,j,k,n))); 

                sp = sedge; 
                sm = sedge; 
                if( (sp - s(i,j,k,n))*(s(i,j,k,n) - sm) <= 0.){
                    sp = s(i,j,k,n); 
                    sm = s(i,j,k,n); 
                }
                else if(std::abs(sp - s(i,j,k,n)) >= 2.*std::abs(sm - s(i,j,k,n))){
                    sp = 3.*s(i,j,k,n) - 2.*sm; 
                }
                else if(std::abs(sm - s(i,j,k,n)) >= 2.*std::abs(sp - s(i,j,k,n))){
                    sm = 3.*s(i,j,k,n) - 2.*sp; 
                } 
            }

        }

      if(bchi == EXT_DIR || bchi == HOEXTRAP){
          if(i == domhi - 1){ 
                sedge = -0.2*s(i+2, j, k, n) + 0.75*s(i+1,j, k, n) 
                       +  0.5*s(i,j,k,n) - 0.05*s(i-1,j,k,n); 
                sedge = amrex::max(sedge, amrex::min(s(i+1,j,k,n), s(i,j,k,n))); 
                sedge = amrex::min(sedge, amrex::max(s(i+1,j,k,n), s(i,j,k,n))); 

                sp = sedge; 
                sm = sedge; 
                if( (sp - s(i,j,k,n))*(s(i,j,k,n) - sm) <= 0.){
                    sp = s(i,j,k,n); 
                    sm = s(i,j,k,n); 
                }
                else if(std::abs(sp - s(i,j,k,n)) >= 2.*std::abs(sm - s(i,j,k,n))){
                    sp = 3.*s(i,j,k,n) - 2.*sm; 
                }
                else if(std::abs(sm - s(i,j,k,n)) >= 2.*std::abs(sp - s(i,j,k,n))){
                    sm = 3.*s(i,j,k,n) - 2.*sp; 
                } 
    
           }
      }
} 

AMREX_GPU_HOST_DEVICE
inline void PeleLM_ppm_ybc(const int i, const int j, const int k, const int n, 
                           amrex::Real &sm, amrex::Real &sp, 
                           amrex::Real &sedge,
                           const amrex::Array4<const amrex::Real> &s, 
                           const int bclo, const int bchi, 
                           const int domlo, const int domhi)
{
        if(bclo == EXT_DIR || bclo == HOEXTRAP){
            if(j == domlo+1){
                sedge = -0.2*s(i, j-2, k, n) + 0.75*s(i,j-1, k, n) 
                       +  0.5*s(i,j,k,n) - 0.05*s(i,j+1,k,n); 
                sedge = amrex::max(sedge, amrex::min(s(i,j,k,n), s(i,j+1,k,n))); 
                sedge = amrex::min(sedge, amrex::max(s(i,j,k,n), s(i,j+1,k,n))); 

                sp = sedge; 
                sm = sedge; 
                if( (sp - s(i,j,k,n))*(s(i,j,k,n) - sm) <= 0.){
                    sp = s(i,j,k,n); 
                    sm = s(i,j,k,n); 
                }
                else if(std::abs(sp - s(i,j,k,n)) >= 2.*std::abs(sm - s(i,j,k,n))){
                    sp = 3.*s(i,j,k,n) - 2.*sm; 
                }
                else if(std::abs(sm - s(i,j,k,n)) >= 2.*std::abs(sp - s(i,j,k,n))){
                    sm = 3.*s(i,j,k,n) - 2.*sp; 
                } 
            }

        }

      if(bchi == EXT_DIR || bchi == HOEXTRAP){
          if(j == domhi - 1){ 
                sedge = -0.2*s(i, j+2, k, n) + 0.75*s(i,j+2, k, n) 
                       +  0.5*s(i,j,k,n) - 0.05*s(i,j-1,k,n); 
                sedge = amrex::max(sedge, amrex::min(s(i,j+1,k,n), s(i,j,k,n))); 
                sedge = amrex::min(sedge, amrex::max(s(i,j+1,k,n), s(i,j,k,n))); 

                sp = sedge; 
                sm = sedge; 
                if( (sp - s(i,j,k,n))*(s(i,j,k,n) - sm) <= 0.){
                    sp = s(i,j,k,n); 
                    sm = s(i,j,k,n); 
                }
                else if(std::abs(sp - s(i,j,k,n)) >= 2.*std::abs(sm - s(i,j,k,n))){
                    sp = 3.*s(i,j,k,n) - 2.*sm; 
                }
                else if(std::abs(sm - s(i,j,k,n)) >= 2.*std::abs(sp - s(i,j,k,n))){
                    sm = 3.*s(i,j,k,n) - 2.*sp; 
                } 
    
           }
      }
} 

AMREX_GPU_HOST_DEVICE
inline void PeleLM_ppm_zbc(const int i, const int j, const int k, const int n, 
                           amrex::Real &sm, amrex::Real &sp, 
                           amrex::Real &sedge,
                           const amrex::Array4<const amrex::Real> &s, 
                           const int bclo, const int bchi, 
                           const int domlo, const int domhi)
{
        if(bclo == EXT_DIR || bclo == HOEXTRAP){
            if(k == domlo+1){
                sedge = -0.2*s(i, j, k-2, n) + 0.75*s(i,j, k-2, n) 
                       +  0.5*s(i,j,k,n) - 0.05*s(i,j,k+1,n); 
                sedge = amrex::max(sedge, amrex::min(s(i,j,k,n), s(i,j,k+1,n))); 
                sedge = amrex::min(sedge, amrex::max(s(i,j,k,n), s(i,j,k+1,n))); 

                sp = sedge; 
                sm = sedge; 
                if( (sp - s(i,j,k,n))*(s(i,j,k,n) - sm) <= 0.){
                    sp = s(i,j,k,n); 
                    sm = s(i,j,k,n); 
                }
                else if(std::abs(sp - s(i,j,k,n)) >= 2.*std::abs(sm - s(i,j,k,n))){
                    sp = 3.*s(i,j,k,n) - 2.*sm; 
                }
                else if(std::abs(sm - s(i,j,k,n)) >= 2.*std::abs(sp - s(i,j,k,n))){
                    sm = 3.*s(i,j,k,n) - 2.*sp; 
                } 
            }

        }

      if(bchi == EXT_DIR || bchi == HOEXTRAP){
          if(k == domhi - 1){ 
                sedge = -0.2*s(i, j, k+2, n) + 0.75*s(i,j, k+1, n) 
                       +  0.5*s(i,j,k,n) - 0.05*s(i,j,k-1,n); 
                sedge = amrex::max(sedge, amrex::min(s(i,j,k,n), s(i,j,k+1,n))); 
                sedge = amrex::min(sedge, amrex::max(s(i,j,k,n), s(i,j,k+1,n))); 

                sp = sedge; 
                sm = sedge; 
                if( (sp - s(i,j,k,n))*(s(i,j,k,n) - sm) <= 0.){
                    sp = s(i,j,k,n); 
                    sm = s(i,j,k,n); 
                }
                else if(std::abs(sp - s(i,j,k,n)) >= 2.*std::abs(sm - s(i,j,k,n))){
                    sp = 3.*s(i,j,k,n) - 2.*sm; 
                }
                else if(std::abs(sm - s(i,j,k,n)) >= 2.*std::abs(sp - s(i,j,k,n))){
                    sm = 3.*s(i,j,k,n) - 2.*sp; 
                } 
    
           }
      }
} 


// Right now only ppm type 1 is supported on GPU
AMREX_GPU_HOST_DEVICE
inline void PeleLM_ppm(const int i, const int j, const int k, const int n, 
                       const amrex::Real dt, const amrex::Real dx, 
                       const amrex::Array4<const amrex::Real> &S, 
                       const amrex::Array4<const amrex::Real> &vel_edge, 
                       const amrex::Array4<amrex::Real> &Im, 
                       const amrex::Array4<amrex::Real> &Ip,
                       const amrex::BCRec bc, const int domlo, const int domhi,                  
                       const int dir)
{
    amrex::Real sedge1, sedge2, d1, d2, s2, s3, sp, sm; 
    amrex::Print()<< "n" << NCOMP<< std::endl;  
    amrex::Real s1 = S(i,j,k,n);     

    //X 
    if(dir==0){
        amrex::Real s2 = S(i-1,j,k,n); 
        amrex::Real d1 = vanLeer(s1, S(i+1,j,k,n), s2); 
        amrex::Real d2 = vanLeer(s2, s1, S(i-2,j,k,n)); 
        sedge1 = 0.5e0*(s1 + s2) - 1.e0/6.e0*(d1 - d2); 
        sedge1 = amrex::max(sedge1, amrex::min(s1, s2)); 
        sedge1 = amrex::min(sedge1, amrex::max(s1, s2)); 

        amrex::Real s3 = s2;
        s2 = s1; 
        s1 = S(i+1,j,k,n); 
        
        d1 = vanLeer(s1, S(i+2, j, k,n), s2); 
        d2 = vanLeer(s2, s1, s3);
        sedge2 = 0.5e0*(s1 + s2) - 1.e0/6.e0*(d1 - d2); 
        sedge2 = amrex::max(sedge2, amrex::min(s1, s2)); 
        sedge2 = amrex::min(sedge2, amrex::max(s1, s2)); 
        
        sm = sedge1; 
        sp = sedge2;         
        //in the following statements sedge2 = sp, sedg1 = sm, s2 = s(i,j,k)
        //This is done to avoid unnecessary reads from global mem 
        if ((sedge2-s2)*(s2-sedge1) < 0.e0){
            sp = s2;
            sm = s2;
        }
        else if (std::abs(sedge2-s2) >= 2.e0*std::abs(sedge1-s2))
            sp = 3.e0*s2 - 2.e0*sedge1;
        else if (std::abs(sedge1-s2) >=  2.e0*std::abs(sedge2-s2)) 
            sm = 3.e0*s2 - 2.e0*sedge2;
        
        PeleLM_ppm_xbc(i, j, k, n, sm, sp, sedge1, S, bc.lo(0), bc.hi(0), domlo, domhi); 
        amrex::Real s6 = 6.e0*s2- 3.e0*(sm + sp); 
        amrex::Real sigmap = std::abs(vel_edge(i+1,j,k))*dx/dt; 
        amrex::Real sigmam = std::abs(vel_edge(i,j,k))*dx/dt;

        if(vel_edge(i+1,j,k) > 1.e-10)
            Ip(i,j,k,n) = sp - (0.5*sigmap)*((sp - sm) -
                                             (1.e0 -2.e0/3.e0*sigmap)*s6); 
        else
            Ip(i,j,k,n) = s2; 

        if(vel_edge(i,j,k) < -1.e-10)
            Im(i,j,k,n) = sm + (0.5*sigmam)*((sp-sm) - 
                                            (1.e0 - 2.e0/3.e0*sigmam)*s6); 
        else
            Im(i,j,k,n) = s2; 
         

    }
    else if(dir==1){ // Y
        amrex::Real s2 = S(i,j-1,k,n); 
        amrex::Real d1 = vanLeer(s1, S(i,j+1,k,n), s2); 
        amrex::Real d2 = vanLeer(s2, s1, S(i,j-2,k,n)); 
        sedge1 = 0.5e0*(s1 + s2) - 1.e0/6.e0*(d1 - d2); 
        sedge1 = amrex::max(sedge1, amrex::min(s1, s2)); 
        sedge1 = amrex::min(sedge1, amrex::max(s1, s2)); 

        amrex::Real s3 = s2;
        s2 = s1; 
        s1 = S(i,j+1,k,n); 
        
        d1 = vanLeer(s1, S(i, j+2, k,n), s2); 
        d2 = vanLeer(s2, s1, s3);
        sedge2 = 0.5e0*(s1 + s2) - 1.e0/6.e0*(d1 - d2); 
        sedge2 = amrex::max(sedge2, amrex::min(s1, s2)); 
        sedge2 = amrex::min(sedge2, amrex::max(s1, s2)); 
        
        sm = sedge1; 
        sp = sedge2;         
        //in the following statements sedge2 = sp, sedg1 = sm, s2 = s(i,j,k)
        //This is done to avoid unnecessary reads from global mem 
        if ((sedge2-s2)*(s2-sedge1) < 0.e0){
            sp = s2;
            sm = s2;
        }
        else if (std::abs(sedge2-s2) >= 2.e0*std::abs(sedge1-s2))
            sp = 3.e0*s2 - 2.e0*sedge1;
        else if (std::abs(sedge1-s2) >=  2.e0*std::abs(sedge2-s2)) 
            sm = 3.e0*s2 - 2.e0*sedge2;
        PeleLM_ppm_ybc(i, j, k, n, sm, sp, sedge1, S, bc.lo(1), bc.hi(1), domlo, domhi); 

        amrex::Real s6 = 6.e0*s2- 3.e0*(sm + sp); 
        amrex::Real sigmap = std::abs(vel_edge(i,j+1,k))*dx/dt; 
        amrex::Real sigmam = std::abs(vel_edge(i,j,k))*dx/dt;
        if(vel_edge(i,j+1,k) > 1.e-10)
            Ip(i,j,k,n) = sp - (0.5*sigmap)*((sp - sm) -
                                             (1.e0 -2.e0/3.e0*sigmap)*s6); 
        else
            Ip(i,j,k,n) = s2; 

        if(vel_edge(i,j,k) < -1.e-10)
            Im(i,j,k,n) = sm + (0.5*sigmam)*((sp-sm) - 
                                            (1.e0 - 2.e0/3.e0*sigmam)*s6); 
        else
            Im(i,j,k,n) = s2; 

    }    
    else{ // Z
        amrex::Real s2 = S(i,j,k-1,n); 
        amrex::Real d1 = vanLeer(s1, S(i,j,k+1,n), s2); 
        amrex::Real d2 = vanLeer(s2, s1, S(i,j,k-2,n)); 
        sedge1 = 0.5e0*(s1 + s2) - 1.e0/6.e0*(d1 - d2); 
        sedge1 = amrex::max(sedge1, amrex::min(s1, s2)); 
        sedge1 = amrex::min(sedge1, amrex::max(s1, s2)); 

        amrex::Real s3 = s2;
        s2 = s1; 
        s1 = S(i,j,k+1,n); 
        
        d1 = vanLeer(s1, S(i, j, k+2,n), s2); 
        d2 = vanLeer(s2, s1, s3);
        sedge2 = 0.5e0*(s1 + s2) - 1.e0/6.e0*(d1 - d2); 
        sedge2 = amrex::max(sedge2, amrex::min(s1, s2)); 
        sedge2 = amrex::min(sedge2, amrex::max(s1, s2)); 
        
        sm = sedge1; 
        sp = sedge2;         
        //in the following statements sedge2 = sp, sedg1 = sm, s2 = s(i,j,k)
        //This is done to avoid unnecessary reads from global mem 
        if ((sedge2-s2)*(s2-sedge1) < 0.e0){
            sp = s2;
            sm = s2;
        }
        else if (std::abs(sedge2-s2) >= 2.e0*std::abs(sedge1-s2))
            sp = 3.e0*s2 - 2.e0*sedge1;
        else if (std::abs(sedge1-s2) >=  2.e0*std::abs(sedge2-s2)) 
            sm = 3.e0*s2 - 2.e0*sedge2;
        PeleLM_ppm_zbc(i, j, k, n, sm, sp, sedge1,S, bc.lo(2), bc.hi(2), domlo, domhi); 

        amrex::Real s6 = 6.e0*s2- 3.e0*(sm + sp); 
        amrex::Real sigmap = std::abs(vel_edge(i,j,k+1))*dx/dt; 
        amrex::Real sigmam = std::abs(vel_edge(i,j,k))*dx/dt;
        if(vel_edge(i,j,k+1) > 1.e-10)
            Ip(i,j,k,n) = sp - (0.5*sigmap)*((sp - sm) -
                                             (1.e0 -2.e0/3.e0*sigmap)*s6); 
        else
            Ip(i,j,k,n) = s2; 

        if(vel_edge(i,j,k) < -1.e-10)
            Im(i,j,k,n) = sm + (0.5*sigmam)*((sp-sm) - 
                                            (1.e0 - 2.e0/3.e0*sigmam)*s6); 
        else
            Im(i,j,k,n) = s2; 

    }
}


AMREX_GPU_HOST_DEVICE
inline void PeleLM_trans_xbc(const int i, const int j, const int k, const int n, 
                             const amrex::Array4<const amrex::Real> &s, 
                             amrex::Real &lo, amrex::Real &hi, 
                             const amrex::Array4< const amrex::Real> &uedge, 
                             const int bclo, const int bchi, 
                             const int domlox, const int domhix)
{
// Lower X
    if(i == domlox){ 
        if(bclo==EXT_DIR){
            if(n == XVEL){
                if(uedge(i,j,k) >= 0.e0){
                    lo = s(i-1, j, k, n); 
                    hi = s(i-1, j, k, n); 
                }
                else
                    lo = s(i, j, k, n); 
            }
            else{
                amrex::Real st = (uedge(i,j,k) <= 1e-10) ? hi : s(i-1,j,k,n); 
                lo = st; 
                hi = st; 
            }
        }
        else if(bclo == FOEXTRAP || bclo == HOEXTRAP || bclo == REFLECT_EVEN)
        {
            lo = hi; 
        }
        else if(bclo == REFLECT_ODD)
        {
            hi = 0.; 
            lo = 0.; 
        }
    }
//Upper X 
    else if(i == domhix){
        if(bchi==EXT_DIR){
            if(n == XVEL){
                if(uedge(i,j,k) <=0.){
                    lo = s(i,j,k,n); 
                    hi = s(i,j,k,n);
                }
                else hi = lo; 
            }
            else{
                amrex::Real st = (uedge(i,j,k) >= -1e-10)? lo : s(i,j,k,n); 
                lo = st; 
                hi = st; 
            }
        }
        else if (bchi == FOEXTRAP || bchi == HOEXTRAP || bchi == REFLECT_EVEN){
            hi = lo; 
        }
        else if(bchi == REFLECT_ODD){
            hi = 0.e0; 
            lo = 0.e0; 
        }
    }
}

AMREX_GPU_HOST_DEVICE
inline void PeleLM_trans_ybc(const int i, const int j, const int k, const int n, 
                             const amrex::Array4<const amrex::Real> &s, 
                             amrex::Real &lo, amrex::Real hi, 
                             const amrex::Array4< const amrex::Real> &vedge, 
                             const int bclo, const int bchi, 
                             const int domloy, const int domhiy)
{
// Lower X 
    if(j == domloy){ 
        if(bclo==EXT_DIR){
            if(n == YVEL){
                if(vedge(i,j,k) >= 0.e0){
                    lo = s(i, j-1, k, n); 
                    hi = s(i, j-1, k, n); 
                }
                else
                    lo = s(i, j, k, n); 
            }
            else{
                amrex::Real st = (vedge(i,j,k) <= 1e-10) ? hi : s(i,j-1,k,n); 
                lo = st; 
                hi = st; 
            }
        }
        else if(bclo == FOEXTRAP || bclo == HOEXTRAP || bclo == REFLECT_EVEN)
        {
            lo = hi; 
        }
        else if(bclo == REFLECT_ODD)
        {
            hi = 0.; 
            lo = 0.; 
        }
    }
//Upper Y
    else if(j == domhiy){
        if(bchi==EXT_DIR){
            if(n == YVEL){
                if(vedge(i,j,k) <=0.){
                    lo = s(i,j,k,n); 
                    hi = s(i,j,k,n);
                }
                else hi = lo; 
            }
            else{
                amrex::Real st = (vedge(i,j,k) >= -1e-10)? lo : s(i,j,k,n); 
                lo = st; 
                hi = st; 
            }
        }
        else if (bchi == FOEXTRAP || bchi == HOEXTRAP || bchi == REFLECT_EVEN){
            hi = lo; 
        }
        else if(bchi == REFLECT_ODD){
            hi = 0.e0; 
            lo = 0.e0; 
        }
    }
}

AMREX_GPU_HOST_DEVICE
inline void PeleLM_trans_zbc(const int i, const int j, const int k, const int n, 
                             const amrex::Array4<const amrex::Real> &s, 
                             amrex::Real &lo, amrex::Real hi, 
                             const amrex::Array4< const amrex::Real> &wedge, 
                             const int bclo, const int bchi, 
                             const int domloz, const int domhiz)
{
// Lower Z
    if(k == domloz){ 
        if(bclo==EXT_DIR){
            if(n == ZVEL){
                if(wedge(i,j,k) >= 0.e0){
                    lo = s(i-1, j, k, n); 
                    hi = s(i-1, j, k, n); 
                }
                else
                    lo = s(i, j, k, n); 
            }
            else{
                amrex::Real st = (wedge(i,j,k) <= 1e-10) ? hi : s(i-1,j,k,n); 
                lo = st; 
                hi = st; 
            }
        }
        else if(bclo == FOEXTRAP || bclo == HOEXTRAP || bclo == REFLECT_EVEN)
        {
            lo = hi; 
        }
        else if(bclo == REFLECT_ODD)
        {
            hi = 0.; 
            lo = 0.; 
        }
    }
//Upper Z 
    else if(k == domhiz){
        if(bchi==EXT_DIR){
            if(n == ZVEL){
                if(wedge(i,j,k) <=0.){
                    lo = s(i+1,j,k,n); 
                    hi = s(i+1,j,k,n);
                }
                else hi = lo; 
            }
            else{
                amrex::Real st = (wedge(i,j,k) >= -1e-10)? lo : s(i,j,k,n); 
                lo = st; 
                hi = st; 
            }
        }
        else if (bchi == FOEXTRAP || bchi == HOEXTRAP || bchi == REFLECT_EVEN){
            hi = lo; 
        }
        else if(bchi == REFLECT_ODD){
            hi = 0.e0; 
            lo = 0.e0; 
        }
    }
}

AMREX_GPU_HOST_DEVICE
inline void PeleLM_corner_couple(const int i, const int j, const int k, const int n, 
                                 const amrex::Real dt, const amrex::Real dx1, 
                                 const amrex::Real dx2, const int iconserv[], 
                                 const amrex::Array4<const amrex::Real> &lo, 
                                 const amrex::Array4<const amrex::Real> &hi,  
                                 const amrex::Array4<const amrex::Real> &s, 
                                 const amrex::Array4<const amrex::Real> &divu, 
                                 const amrex::Array4<const amrex::Real> &mac1,         
                                 const amrex::Array4<const amrex::Real> &mac2,
                                 const amrex::Array4<const amrex::Real> &state1, 
                                 const amrex::Array4<const amrex::Real> &state2, 
                                 const amrex::Array4<amrex::Real> &lo1, 
                                 const amrex::Array4<amrex::Real> &hi1,
                                 const amrex::Array4<amrex::Real> &lo2, 
                                 const amrex::Array4<amrex::Real> &hi2,                  
                                 const int dir)
{
//=========================== X ====================================================
    if(dir ==0){
        if(iconserv[n]){
        //xylo xyhi 
                lo1(i,j,k,n)  = lo(i, j, k, n)
                              - dt/(3.e0*dx1)*(state1(i-1,j+1,k,n)*mac1(i-1,j+1,k)
                              - state1(i-1,j,k,n)*mac1(i-1,j,k))
                              - dt/3.e0*s(i-1,j,k,n)*divu(i-1,j,k)
                              + dt/(3.e0*dx1)*s(i-1,j,k,n)*(mac1(i-1,j+1,k)-mac1(i-1,j,k));

                hi1(i,j,k,n)  = hi(i, j, k, n)
                              - dt/(3.e0*dx1)*(state1(i,j+1,k,n)*mac1(i,j+1,k)
                              - state1(i,j,k,n)*mac1(i,j,k))
                              - dt/3.e0*s(i,j,k,n)*divu(i,j,k)
                              + dt/(3.e0*dx1)*s(i,j,k,n)*(mac1(i,j+1,k)-mac1(i,j,k));

        //xzlo xzhi 
                lo2(i,j,k,n)  = lo(i,j,k,n)
                              - dt/(3.e0*dx2)*(state2(i-1,j,k+1,n)*mac2(i-1,j,k+1)
                              - state2(i-1,j,k,n)*mac2(i-1,j,k))
                              - dt/(3.e0)*s(i-1,j,k,n)*divu(i-1,j,k)
                              + dt/(3.e0*dx2)*s(i-1,j,k,n)*(mac2(i-1,j,k+1) - mac2(i-1,j,k));

                hi2(i,j,k,n)  = hi(i,j,k,n)
                              - dt/(3.e0*dx2)*(state2(i,j,k+1,n)*mac2(i,j,k+1)
                              - state2(i,j,k,n)*mac2(i,j,k))
                              - dt/(3.e0)*s(i,j,k,n)*divu(i,j,k)
                              + dt/(3.e0*dx2)*s(i,j,k,n)*(mac2(i,j,k+1) - mac2(i,j,k));
            }
        else{
               lo1(i,j,k,n)  = lo(i,j,k,n)
                             - dt/(6.e0*dx1)*(mac1(i-1,j+1,k) + mac1(i-1,j,k))
                             * (state1(i-1,j+1,k,n) - state1(i-1,j,k,n));
               hi1(i,j,k,n)  = hi(i,j,k,n)
                             - dt/(6.e0*dx1)*(mac1(i,j+1,k) + mac1(i,j,k))
                             * (state1(i, j+1,k, n) - state1(i, j, k, n));

               lo2(i,j,k,n)  = lo(i,j,k,n)
                             - dt/(6.e0*dx2)*(mac2(i-1,j,k+1) + mac2(i-1,j,k))
                             * (state2(i-1,j,k+1,n) - state2(i-1,j,k,n)); 
                
               hi2(i,j,k,n)  = hi(i,j,k,n) 
                             - dt/(6.e0*dx2)*(mac2(i,j,k+1) + mac2(i,j,k))
                             * (state2(i,j,k+1,n) - state2(i,j,k,n)); 
        }
    }
//=========================== Y ========================================================
    if(dir==1){
        if(iconserv[n]){
        //yxlo yxhi 
                lo1(i,j,k,n)  = lo(i,j,k,n) 
                              - dt/(3.e0*dx1)*(state1(i+1,j-1,k,n)*mac1(i+1,j-1,k)
                              - state1(i,j-1,k,n)*mac1(i,j-1,k))
                              - dt/(3.e0)*s(i,j-1,k,n)*divu(i,j-1,k)
                              + dt/(3.e0*dx1)*s(i,j-1,k,n)*(mac1(i+1,j-1,k)-mac1(i,j-1,k));

                hi1(i,j,k,n)  = hi(i,j,k,n) 
                              - dt/(3.e0*dx1)*(state1(i+1,j,k,n)*mac1(i+1,j,k)
                              - state1(i,j,k,n)*mac1(i,j,k))
                              - dt/(3.e0)*s(i,j,k,n)*divu(i,j,k) 
                              + dt/(3.e0*dx1)*s(i,j,k,n)*(mac1(i+1,j,k) - mac1(i,j,k));

        //yzlo yzhi         
                lo2(i,j,k,n)  = lo(i,j,k,n) 
                              - dt/(3.e0*dx2)*(state2(i,j-1,k+1,n)*mac2(i,j-1,k+1)
                              - state2(i,j-1,k,n)*mac2(i,j-1,k))
                              - dt/(3.e0)*s(i,j-1,k,n)*divu(i,j-1,k)
                              + dt/(3.e0*dx2)*s(i,j-1,k,n)*(mac2(i,j-1,k+1)-mac2(i,j-1,k));

                hi2(i,j,k,n)  = hi(i,j,k,n) 
                              - dt/(3.e0*dx2)*(state2(i,j,k+1,n)*mac2(i,j,k+1)
                              - state2(i,j,k,n)*mac2(i,j,k))
                              - dt/(3.e0)*s(i,j,k,n)*divu(i,j,k) 
                              + dt/(3.e0*dx2)*s(i,j,k,n)*(mac2(i,j,k+1) - mac2(i,j,k));
        }
        else{
               lo1(i,j,k,n)  = lo(i,j,k,n)
                             - dt/(6.e0*dx1)*(mac1(i+1,j-1,k) + mac1(i,j-1,k))
                             * (state1(i+1,j-1,k,n) - state1(i,j-1,k,n));
               hi1(i,j,k,n)  = hi(i,j,k,n)
                             - dt/(6.e0*dx1)*(mac1(i+1,j,k) + mac1(i,j,k))
                             * (state1(i+1, j,k, n) - state1(i, j, k, n));

               lo2(i,j,k,n)  = lo(i,j,k,n)
                             - dt/(6.e0*dx2)*(mac2(i,j-1,k+1) + mac2(i,j-1,k))
                             * (state2(i,j-1,k+1,n) - state2(i,j-1,k,n)); 
                
               hi2(i,j,k,n)  = hi(i,j,k,n) 
                             - dt/(6.e0*dx2)*(mac2(i,j,k+1) + mac2(i,j,k))
                             * (state2(i,j,k+1,n) - state2(i,j,k,n)); 
        }
    }
//=========================== Z ======================================================    
    if(dir==2){
        if(iconserv[n]){
        //zxlo zxhi 
                lo1(i,j,k,n)  = lo(i,j,k,n) 
                              - dt/(3.e0*dx1)*(state1(i+1,j,k-1,n)*mac1(i+1,j,k-1)
                              - state1(i,j,k-1,n)*mac1(i,j,k-1))
                              - dt/(3.e0)*s(i,j,k-1,n)*divu(i,j,k-1)
                              + dt/(3.e0*dx1)*s(i,j,k-1,n)*(mac1(i+1,j,k-1)-mac1(i,j,k-1));

                hi1(i,j,k,n)  = hi(i,j,k,n) 
                              - dt/(3.e0*dx1)*(state1(i+1,j,k,n)*mac1(i+1,j,k)
                              - state1(i,j,k,n)*mac1(i,j,k))
                              - dt/(3.e0)*s(i,j,k,n)*divu(i,j,k) 
                              + dt/(3.e0*dx1)*s(i,j,k,n)*(mac1(i+1,j,k) - mac1(i,j,k));

        //zylo zyhi         
                lo2(i,j,k,n)  = lo(i,j,k,n) 
                              - dt/(3.e0*dx2)*(state2(i,j+1,k-1,n)*mac2(i,j+1,k-1)
                              - state2(i,j,k-1,n)*mac2(i,j,k-1))
                              - dt/(3.e0)*s(i,j,k-1,n)*divu(i,j,k-1)
                              + dt/(3.e0*dx2)*s(i,j,k-1,n)*(mac2(i,j+1,k-1)-mac2(i,j,k-1));

                hi2(i,j,k,n)  = hi(i,j,k,n) 
                              - dt/(3.e0*dx2)*(state2(i,j+1,k,n)*mac2(i,j+1,k)
                              - state2(i,j,k,n)*mac2(i,j,k))
                              - dt/(3.e0)*s(i,j,k,n)*divu(i,j,k) 
                              + dt/(3.e0*dx2)*s(i,j,k,n)*(mac2(i,j+1,k) - mac2(i,j,k));
        }
        else{
               lo1(i,j,k,n)  = lo(i,j,k,n)
                             - dt/(6.e0*dx1)*(mac1(i+1,j,k-1) + mac1(i,j,k-1))
                             * (state1(i+1,j,k-1,n) - state1(i,j,k-1,n));
               hi1(i,j,k,n)  = hi(i,j,k,n)
                             - dt/(6.e0*dx1)*(mac1(i+1,j,k) + mac1(i,j,k))
                             * (state1(i+1, j,k, n) - state1(i, j, k, n));

               lo2(i,j,k,n)  = lo(i,j,k,n)
                             - dt/(6.e0*dx2)*(mac2(i,j+1,k-1) + mac2(i,j,k-1))
                             * (state2(i,j+1,k-1,n) - state2(i,j,k-1,n)); 
                
               hi2(i,j,k,n)  = hi(i,j,k,n) 
                             - dt/(6.e0*dx2)*(mac2(i,j+1,k) + mac2(i,j,k))
                             * (state2(i,j+1,k,n) - state2(i,j,k,n)); 
        }
    }

}

                    
void PeleLM_estate_fpu(const amrex::Box bx,
                       const amrex::Array4<const amrex::Real> &s,
                       const amrex::Array4<const amrex::Real> &tf,
                       const amrex::Array4<const amrex::Real> &divu,
                       const amrex::Array4<const amrex::Real> &umac,
                       const amrex::Array4<const amrex::Real> &vmac,
                       const amrex::Array4<const amrex::Real> &wmac,
                       const amrex::Array4<amrex::Real> &xstate,
                       const amrex::Array4<amrex::Real> &ystate,
                       const amrex::Array4<amrex::Real> &zstate,
                       const amrex::Real dt, const amrex::Real del[],
                       const amrex::GpuArray<amrex::BCRec, NCOMP> &BCs, 
                       const int domlo[], const int domhi[], const int iconserv[]);

#endif 

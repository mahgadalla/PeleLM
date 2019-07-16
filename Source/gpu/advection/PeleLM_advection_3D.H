#ifndef PELELM_ADVECTION_3D_H
#define PELELM_ADVECTION_3D_H
#include <AMReX_Gpu.H> 
#include <AMReX_FArrayBox.H> 


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


// Right now only ppm type 1 is supported on GPU
AMREX_GPU_HOST_DEVICE
inline void PeleLM_ppm(const int i, const int j, const int k, const int n, 
                       const amrex::Real dt, const amrex::Real dx, 
                       const amrex::Array4<const amrex::Real> &S, 
                       const amrex::Array4<const amrex::Real> &vel_edge, 
                       const amrex::Array4<amrex::Real> &Im, 
                       const amrex::Array4<amrex::Real> &Ip,                 
                       const int dir)
{
    amrex::Real sedge1, sedge2, d1, d2, s2, s3, sp, sm;  
    amrex::Real s1 = S(i,j,k,n);     

    //X 
    if(dir==0){
        amrex::Real s2 = S(i-1,j,k,n); 
        amrex::Real d1 = vanLeer(s1, S(i+1,j,k,n), s2); 
        amrex::Real d2 = vanLeer(s2, s1, S(i-2,j,k,n)); 
        sedge1 = 0.5e0*(s1 + s2) - 1.e0/6.e0*(d1 - d2); 
        sedge1 = amrex::max(sedge1, amrex::min(s1, s2); 
        sedge1 = amrex::min(sedge1, amrex::max(s1, s2); 

        amrex::Real s3 = s2;
        s2 = s1; 
        s1 = S(i+1,j,k,n); 
        
        d1 = vanLeer(s1, S(i+2, j, k,n), s2); 
        d2 = vanleer(s2, s1, s3);
        sedge2 = 0.5e0*(s1 + s2) - 1.e0/6.e0*(d1 - d2); 
        sedge2 = amrex::max(sedge2, amrex::min(s1, s2); 
        sedge2 = amrex::min(sedge2, amrex::max(s1, s2); 
        
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
        sedge1 = amrex::max(sedge1, amrex::min(s1, s2); 
        sedge1 = amrex::min(sedge1, amrex::max(s1, s2); 

        amrex::Real s3 = s2;
        s2 = s1; 
        s1 = S(i,j+1,k,n); 
        
        d1 = vanLeer(s1, S(i, j+2, k,n), s2); 
        d2 = vanleer(s2, s1, s3);
        sedge2 = 0.5e0*(s1 + s2) - 1.e0/6.e0*(d1 - d2); 
        sedge2 = amrex::max(sedge2, amrex::min(s1, s2); 
        sedge2 = amrex::min(sedge2, amrex::max(s1, s2); 
        
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
        sedge1 = amrex::max(sedge1, amrex::min(s1, s2); 
        sedge1 = amrex::min(sedge1, amrex::max(s1, s2); 

        amrex::Real s3 = s2;
        s2 = s1; 
        s1 = S(i,j,k+1,n); 
        
        d1 = vanLeer(s1, S(i, j, k+2,n), s2); 
        d2 = vanleer(s2, s1, s3);
        sedge2 = 0.5e0*(s1 + s2) - 1.e0/6.e0*(d1 - d2); 
        sedge2 = amrex::max(sedge2, amrex::min(s1, s2); 
        sedge2 = amrex::min(sedge2, amrex::max(s1, s2); 
        
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
    //TODO Deal with BCs look in IAMR Godunov for this ppm_()dir functions  
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
                       const amrex::Real dt, const amrex::Real dx[], 
                       const int iconserv[NVAR]);

#endif 

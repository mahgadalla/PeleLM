#include "PeleLM_advection_3D.H" 


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
                       const int iconserv[NVAR]) 
{
    

    
    AMREX_PARALLEL_FOR_3D(bx, i, j, k, {

        PeleLM_ppm(); 

    }); 




}

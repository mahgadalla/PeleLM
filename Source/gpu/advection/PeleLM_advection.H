#ifndef PELELM_ADVECTION_H
#define PELELM_ADVECTION_H
#include <AMReX_BCRec.H> 
#include <AMReX_Array.H>
#include <PeleLM_index_macros.H>
#include <Godunov.H>

void PeleLM_AdvectScalars(const amrex::Box bx, const amrex::Real del[],
                          const amrex::Real dt,
                   D_DECL(const amrex::Array4<const amrex::Real> &Ax,
                          const amrex::Array4<const amrex::Real> &Ay,
                          const amrex::Array4<const amrex::Real> &Az),
                   D_DECL(const amrex::Array4<const amrex::Real> &umac, 
                          const amrex::Array4<const amrex::Real> &vmac,
                          const amrex::Array4<const amrex::Real> &wmac), 
                   D_DECL(const amrex::Array4<amrex::Real> &xflx,
                          const amrex::Array4<amrex::Real> &yflx,  
                          const amrex::Array4<amrex::Real> &zflx),
                   D_DECL(const amrex::Array4<amrex::Real> &xstate, 
                          const amrex::Array4<amrex::Real> &ystate,
                          const amrex::Array4<amrex::Real> &zstate), 
                          const amrex::Array4<const amrex::Real> &s,
                          const amrex::Array4<const amrex::Real> &forces, 
                          const amrex::Array4<const amrex::Real> &divu,
                          const amrex::Array4<amrex::Real> &aofs, 
                          const amrex::Vector<AdvectionForm> &advectionType,
                          const amrex::GpuArray<amrex::BCRec, NCOMP> &BCs,
                          const amrex::Array4<const amrex::Real> &V);
#endif

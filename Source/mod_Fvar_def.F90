#include <AMReX_REAL.H>

module mod_Fvar_def

  implicit none
  
  
  ! From visc.H
  
  logical :: LeEQ1

  REAL_T  :: Pr, Sc
  REAL_T  :: thickFac
  
  ! From htdata.H
  
  REAL_T  :: pamb, dpdt_factor
  integer :: closed_chamber
  
  integer :: Density, Temp, RhoH, Trac, FirstSpec, LastSpec
  
  ! From timedata.H
  
  integer :: iteration
  REAL_T  :: time
  
  ! From probdata.H
  integer :: bathID, fuelID, oxidID, prodID
  
  integer         , save :: f_flag_active_control
  ! dimension information
  integer         , save :: dim
  ! geometry information
  double precision, save :: domnlo(3), domnhi(3)

  ! From cdwrk.H
  integer, parameter :: maxspec  = 200
  integer, parameter :: maxspnml =  16
  
  ! From bc.H
  integer, parameter :: MAXPNTS = 50
  REAL_T :: time_points(0:MAXPNTS),vel_points(0:MAXPNTS),cntl_points(0:MAXPNTS)
      
  character(50) :: ac_hist_file
  REAL_T :: tau_control, cfix, coft_old, sest, V_in, V_in_old, corr, &
          changeMax_control, tbase_control, dV_control, scale_control, &
          zbase_control, h_control, controlVelMax
  integer :: navg_pnts, flame_dir, pseudo_gravity
  
#ifdef USE_EFIELD
   integer :: nE, PhiV                                  ! nE and phiV index 
   integer :: iE_sp                                     ! electron index in species list
   integer :: iH3Op                                     ! main anion idx in species list
   REAL_T, PARAMETER :: Na = 6.022d23                   ! Avogadro's number
   REAL_T, PARAMETER :: CperECharge = 1.60217662d-19    ! Coulomb per charge
   REAL_T, PARAMETER :: e0 = 8.854187817d-12            ! Free space permittivity (C/(V.m))
   REAL_T, PARAMETER :: er = 1.d0                       ! Relative permittivity of air
   REAL_T, DIMENSION(:), ALLOCATABLE :: zk
   REAL_T, DIMENSION(:), ALLOCATABLE :: invmwt
   integer, DIMENSION(:), ALLOCATABLE :: spec_charge
#endif		

  
end module mod_Fvar_def

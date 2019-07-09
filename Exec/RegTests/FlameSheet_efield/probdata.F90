#include <AMReX_REAL.H>


module probdata_module

  use mod_Fvar_def, only: maxspec

  implicit none

  ! from probdata.H
  REAL_T :: standoff
  REAL_T :: pertmag
  REAL_T :: splity
    
  ! from bc.H
  
  logical :: bcinit
  
  REAL_T :: u_bc(2), v_bc(2), w_bc(2)
  REAL_T :: Y_bc(0:maxspec-1,2), T_bc(2), h_bc(2), rho_bc(2)
  REAL_T :: ne_bc(2), phiV_bc(2)
    
  integer, parameter :: flame_dir = 2
  
contains

!subroutines here

end module probdata_module

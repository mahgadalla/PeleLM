module PETScSolver_F

   use, intrinsic :: iso_c_binding
   use amrex_petsc_fort_module, only : it=>petsc_int
!   use amrex_hypre_fort_module, only : hypre_int
   use amrex_fort_module, only : rt => amrex_real
   use amrex_error_module, only : amrex_error
   use amrex_lo_bctypes_module, only : amrex_lo_dirichlet, amrex_lo_neumann
   use amrex_constants_module, only : zero, one, half, three
   implicit none

   real(rt), parameter :: eps = 1.0e-6;

contains

   subroutine petsc_fillmatbox ( lo, hi, &
                                 nrows, &
                                 ncolsnE, ncolsphiV, &
                                 rowsnE, rowsphiV, &
                                 colsnE, matnE, &
                                 colsphiV, matphiV, &
                                 cell_ne_id, clo, chi, &
                                 cell_phiV_id, cplo, cphi, &
                                 cell_id_ne_begin, cell_id_phiV_begin, &
                                 diag, dlo, dhi, &
                                 Dex, Dexlo, Dexhi, &
                                 Dey, Deylo, Deyhi, &
#if (AMREX_SPACEDIM == 3)
                                 Dez, Dezlo, Dezhi, &
#endif
                                 Ueffx, uxlo, uxhi, &
                                 Ueffy, uylo, uyhi, &
#if (AMREX_SPACEDIM == 3)
                                 Ueffz, uzlo, uzhi, &
#endif
                                 neKex, nexlo, nexhi, &
                                 neKey, neylo, neyhi, &
#if (AMREX_SPACEDIM == 3)
                                 neKez, nezlo, nezhi, &
#endif
                                 lapl_fac, &
                                 bct_ne, bcl_ne, bct_phiV, bcl_phiV, &
                                 s_dtDiffI, s_dtDrift, s_Ie, s_L, dx, dt ) bind(c,name='petsc_fillmatbox')

      implicit none                        
      integer, dimension(2), intent(in) :: lo, hi
      integer(it), intent(in) :: nrows
      integer(it), dimension(0:nrows-1), intent(out) :: rowsnE, rowsPhiV
      integer(it), dimension(0:nrows-1), intent(out) :: ncolsnE, ncolsphiV
      integer(it), dimension(0:nrows*10-1), intent(out) :: colsnE
      integer(it), dimension(0:nrows*10-1), intent(out) :: colsphiV
      real(rt)   , dimension(0:nrows*10-1), intent(out) :: matnE
      real(rt)   , dimension(0:nrows*10-1), intent(out) :: matphiV

      integer, dimension(2), intent(in) :: clo, chi, cplo, cphi
      integer, dimension(2), intent(in) :: dlo, dhi
      integer, dimension(2), intent(in) :: Dexlo, Dexhi, Deylo, Deyhi, uxlo, uxhi, uylo, uyhi, nexlo, nexhi, neylo, neyhi
      integer(it), intent(in) :: cell_ne_id(clo(1):chi(1),clo(2):chi(2))
      integer(it), intent(in) :: cell_phiV_id(cplo(1):cphi(1),cplo(2):cphi(2))
      integer(it), intent(in) :: cell_id_ne_begin, cell_id_phiV_begin
      real(rt)   , intent(inout)::  diag(dlo(1):dhi(1),dlo(2):dhi(2),2)
      real(rt)   , intent(in)::     Dex(Dexlo(1):Dexhi(1),Dexlo(2):Dexhi(2))
      real(rt)   , intent(in)::     Dey(Deylo(1):Deyhi(1),Deylo(2):Deyhi(2))
      real(rt)   , intent(in)::     Ueffx(uxlo(1):uxhi(1),uxlo(2):uxhi(2))
      real(rt)   , intent(in)::     Ueffy(uylo(1):uyhi(1),uylo(2):uyhi(2))
      real(rt)   , intent(in)::     neKex(nexlo(1):nexhi(1),nexlo(2):nexhi(2))
      real(rt)   , intent(in)::     neKey(neylo(1):neyhi(1),neylo(2):neyhi(2))
      real(rt)   , intent(in):: lapl_fac
      integer, intent(in) :: bct_ne(0:3), bct_phiV(0:3)
      real(rt), intent(in) :: bcl_ne(0:3), bcl_phiV(0:3)
      real(rt), intent(in) :: s_dtDiffI, s_dtDrift, s_Ie, s_L, dt, dx(2)

      real(rt) :: dxsqinv(2), dxinv(2), h, h2, h3
      real(rt) :: bf1_ne(0:3), bf2_ne(0:3), bf1_phiV(0:3), bf2_phiV(0:3), bf1_phiV_HO(0:3), bf2_phiV_HO(0:3)
      real(rt) :: bf1_dxne(0:3), bf2_dxne(0:3)
      real(rt) :: matnE_tmp(0:9), matphiV_tmp(0:5)
      integer(it) :: colsnE_tmp(0:9), colsphiV_tmp(0:5)
      integer :: cdir, idim, irow, imatnE, imatphiV
      integer :: i, j, ic

      ncolsnE(:) = 0
      ncolsphiV(:) = 0
      rowsnE(:) = 0
      rowsphiV(:) = 0
      colsnE(:) = 0
      colsphiV(:) = 0
      matnE(:) = 0.0_rt
      matphiV(:) = 0.0_rt

      diag(:,:,:) = 0.0_rt

      dxsqinv(:) = 1.0d0/dx(:)**2.0d0
      dxinv(:) = 1.0d0/dx(:)

      do cdir = 0, 3
         if (cdir == 0 .OR. cdir == 2) then
            idim = 1
         else
            idim = 2
         end if
         h = dx(idim)
         if (bct_ne(cdir) == amrex_lo_dirichlet) then
            h2 = half * h
            h3 = three * h2
            bf1_ne(cdir) = - dxsqinv(idim) * ((h3 - bcl_ne(cdir)) / (bcl_ne(cdir) + h2) - one)
            bf2_ne(cdir) = - dxsqinv(idim) * (bcl_ne(cdir) - h2) / (bcl_ne(cdir) + h3)
            bf1_dxne(cdir) = - dxinv(idim) * ((h3 - bcl_ne(cdir)) / (bcl_ne(cdir) + h2) - one)
            bf2_dxne(cdir) = - dxinv(idim) * (bcl_ne(cdir) - h2) / (bcl_ne(cdir) + h3)
!            bf1_ne(cdir) = - dxsqinv(idim) * (h / (bcl_ne(cdir) + h2) - one)
!            bf2_ne(cdir) = zero 
         else if (bct_ne(cdir) == amrex_lo_neumann) then
            bf1_ne(cdir) = dxsqinv(idim)
            bf2_ne(cdir) = zero
            bf1_dxne(cdir) = dxsqinv(idim)
            bf2_dxne(cdir) = zero
         endif
         if (bct_phiV(cdir) == amrex_lo_dirichlet) then
            h2 = half * h
            h3 = three * h2
            bf1_phiV_HO(cdir) = - dxsqinv(idim) * ((h3 - bcl_phiV(cdir)) / (bcl_phiV(cdir) + h2) - one)
            bf2_phiV_HO(cdir) = - dxsqinv(idim) * (bcl_phiV(cdir) - h2) / (bcl_phiV(cdir) + h3)
            bf1_phiV(cdir) = - dxsqinv(idim) * (h / (bcl_phiV(cdir) + h2) - one)
            bf2_phiV(cdir) = zero
         else if (bct_phiV(cdir) == amrex_lo_neumann) then
            bf1_phiV_HO(cdir) = dxsqinv(idim)
            bf2_phiV_HO(cdir) = zero
            bf1_phiV(cdir) = dxsqinv(idim)
            bf2_phiV(cdir) = zero
         endif
      enddo

      irow = 0 
      imatnE = 0    
      imatphiV = 0
      do    j = lo(2), hi(2)
         do i = lo(1), hi(1)

            rowsnE(irow) = cell_ne_id(i,j)
            rowsphiV(irow) = cell_phiV_id(i,j)

            ncolsnE(irow) = 0
            ncolsphiV(irow) = 0

            !-----------------------------------
            ! nE rows
            !-----------------------------------

            ! dtDiff-I
            colsnE_tmp(0) = cell_ne_id(i,j)
            matnE_tmp(0)  = - s_dtDiffI &
                            - s_dtDiffI * dt * ( dxsqinv(1) * ( Dex(i,j) + Dex(i+1,j) ) + &
                                                 dxsqinv(2) * ( Dey(i,j) + Dey(i,j+1) ) ) &
                            - s_dtDiffI * dt * ( dxinv(1) * getCellUpwindCoeff(Ueffx(i,j),Ueffx(i+1,j)) + &
                                                 dxinv(2) * getCellUpwindCoeff(Ueffy(i,j),Ueffy(i,j+1)) )                   

            colsnE_tmp(1) = cell_ne_id(i-1,j)
            matnE_tmp(1)  = s_dtDiffI * dt * (  dxsqinv(1) * Dex(i,j) &
                                              - dxinv(1) * getLeftCellUpwindCoeff(Ueffx(i,j),Ueffx(i+1,j)) ) 
            colsnE_tmp(2) = cell_ne_id(i+1,j)
            matnE_tmp(2)  = s_dtDiffI * dt * (  dxsqinv(1) * Dex(i+1,j) &
                                              - dxinv(1) * getRightCellUpwindCoeff(Ueffx(i,j),Ueffx(i+1,j)) )
            colsnE_tmp(3) = cell_ne_id(i,j-1)
            matnE_tmp(3)  = s_dtDiffI * dt * (  dxsqinv(2) * Dey(i,j) &
                                              - dxinv(2) * getLeftCellUpwindCoeff(Ueffy(i,j),Ueffy(i,j+1)) )
            colsnE_tmp(4) = cell_ne_id(i,j+1)
            matnE_tmp(4)  = s_dtDiffI * dt * (  dxsqinv(2) * Dey(i,j+1) &
                                              - dxinv(2) * getRightCellUpwindCoeff(Ueffy(i,j),Ueffy(i,j+1)) )

            ! dtDr
            colsnE_tmp(5) = cell_phiV_id(i,j)
            matnE_tmp(5)  = + s_dtDrift * dt * ( dxsqinv(1) * ( neKex(i,j) + neKex(i+1,j) ) + &
                                                 dxsqinv(2) * ( neKey(i,j) + neKey(i,j+1) ) )
            colsnE_tmp(6) = cell_phiV_id(i-1,j)
            matnE_tmp(6)  = - s_dtDrift * dt * dxsqinv(1) * neKex(i,j)
            colsnE_tmp(7) = cell_phiV_id(i+1,j)
            matnE_tmp(7)  = - s_dtDrift * dt * dxsqinv(1) * neKex(i+1,j)
            colsnE_tmp(8) = cell_phiV_id(i,j-1)
            matnE_tmp(8)  = - s_dtDrift * dt * dxsqinv(2) * neKey(i,j)
            colsnE_tmp(9) = cell_phiV_id(i,j+1)
            matnE_tmp(9)  = - s_dtDrift * dt * dxsqinv(2) * neKey(i,j+1)

            !-----------------------------------
            ! phiV rows
            !-----------------------------------

            ! First term is Ie
            colsphiV_tmp(0) = cell_ne_id(i,j)         
            matphiV_tmp(0)  = - s_Ie

            ! Laplacian of phiV
            colsphiV_tmp(1) = cell_phiV_id(i,j)
            matphiV_tmp(1)  = -2.0d0 * s_L * lapl_fac * dxsqinv(1) &
                              -2.0d0 * s_L * lapl_fac * dxsqinv(2) 

            colsphiV_tmp(2) = cell_phiV_id(i-1,j)
            matphiV_tmp(2)  = s_L * lapl_fac * dxsqinv(1)
            colsphiV_tmp(3) = cell_phiV_id(i+1,j)
            matphiV_tmp(3)  = s_L * lapl_fac * dxsqinv(1)
            colsphiV_tmp(4) = cell_phiV_id(i,j-1)
            matphiV_tmp(4)  = s_L * lapl_fac * dxsqinv(2)
            colsphiV_tmp(5) = cell_phiV_id(i,j+1)
            matphiV_tmp(5)  = s_L * lapl_fac * dxsqinv(2)

            ! Check on cell_id_ne only since cell_id_phiv is offsetted by ncells
            if (i.eq.lo(1) .and. cell_ne_id(i-1,j).lt.0) then
               cdir = 0
               matnE_tmp(0)   = matnE_tmp(0)   + s_dtDiffI * dt * bf1_ne(cdir) * Dex(i,j)
               matnE_tmp(2)   = matnE_tmp(2)   + s_dtDiffI * dt * bf2_ne(cdir) * Dex(i,j)
               matnE_tmp(5)   = matnE_tmp(5)   - s_dtDrift * dt * bf1_phiV(cdir) * neKex(i,j) 
               matnE_tmp(7)   = matnE_tmp(7)   - s_dtDrift * dt * bf2_phiV(cdir) * neKex(i,j)
               matphiV_tmp(1) = matphiV_tmp(1) + s_L * lapl_fac * bf1_phiV(cdir)
               matphiV_tmp(3) = matphiV_tmp(3) + s_L * lapl_fac * bf2_phiV(cdir)
            end if

            if (i.eq.hi(1) .and. cell_ne_id(i+1,j).lt.0) then
               cdir = 2
               matnE_tmp(0)   = matnE_tmp(0)   + s_dtDiffI * dt * bf1_ne(cdir) * Dex(i+1,j)
               matnE_tmp(1)   = matnE_tmp(1)   + s_dtDiffI * dt * bf2_ne(cdir) * Dex(i+1,j)
               matnE_tmp(5)   = matnE_tmp(5)   - s_dtDrift * dt * bf1_phiV(cdir) * neKex(i+1,j) 
               matnE_tmp(6)   = matnE_tmp(6)   - s_dtDrift * dt * bf2_phiV(cdir) * neKex(i+1,j)
               matphiV_tmp(1) = matphiV_tmp(1) + s_L * lapl_fac * bf1_phiV(cdir)
               matphiV_tmp(2) = matphiV_tmp(2) + s_L * lapl_fac * bf2_phiV(cdir)
            end if

            if (j.eq.lo(2) .and. cell_ne_id(i,j-1).lt.0) then
               cdir = 1
               matnE_tmp(0)   = matnE_tmp(0)   + s_dtDiffI * dt * bf1_ne(cdir) * Dey(i,j)
               matnE_tmp(4)   = matnE_tmp(4)   + s_dtDiffI * dt * bf2_ne(cdir) * Dey(i,j)
               matnE_tmp(5)   = matnE_tmp(5)   - s_dtDrift * dt * bf1_phiV(cdir) * neKey(i,j) 
               matnE_tmp(9)   = matnE_tmp(9)   - s_dtDrift * dt * bf2_phiV(cdir) * neKey(i,j)
               matphiV_tmp(1) = matphiV_tmp(1) + s_L * lapl_fac * bf1_phiV(cdir)
               matphiV_tmp(5) = matphiV_tmp(5) + s_L * lapl_fac * bf2_phiV(cdir)
            end if

            if (j.eq.hi(2) .and. cell_ne_id(i,j+1).lt.0) then
               cdir = 3
               matnE_tmp(0)   = matnE_tmp(0)   + s_dtDiffI * dt * bf1_ne(cdir) * Dey(i,j+1)
               matnE_tmp(3)   = matnE_tmp(3)   + s_dtDiffI * dt * bf2_ne(cdir) * Dey(i,j+1)
!               matnE_tmp(0)   = matnE_tmp(0)   - s_dtDiffI * dt * bf1_dxne(cdir) * &
!                                                 getRightCellUpwindCoeff(Ueffy(i,j),Ueffy(i,j+1))  
!               matnE_tmp(3)   = matnE_tmp(3)   - s_dtDiffI * dt * bf2_dxne(cdir) * &
!                                                 getRightCellUpwindCoeff(Ueffy(i,j),Ueffy(i,j+1))
               matnE_tmp(5)   = matnE_tmp(5)   - s_dtDrift * dt * bf1_phiV(cdir) * neKey(i,j+1) 
               matnE_tmp(8)   = matnE_tmp(8)   - s_dtDrift * dt * bf2_phiV(cdir) * neKey(i,j+1)
               matphiV_tmp(1) = matphiV_tmp(1) + s_L * lapl_fac * bf1_phiV(cdir)
               matphiV_tmp(4) = matphiV_tmp(4) + s_L * lapl_fac * bf2_phiV(cdir)
            end if

            ! Store diagonal scale
            diag(i,j,1) = 1.0_rt/matnE_tmp(0)
            diag(i,j,2) = 1.0_rt/matphiV_tmp(1)

            ! Gather ne row : 2*(2*DIM+1)
            do ic = 0, 9
               if ( colsnE_tmp(ic) >= 0 ) then
                  ncolsnE(irow)  = ncolsnE(irow) + 1
                  colsnE(imatnE) = colsnE_tmp(ic)
                  matnE(imatnE)  = matnE_tmp(ic) * diag(i,j,1) 
                  imatnE = imatnE + 1
               end if
            end do
            ! Gather phiv row : 2*DIM+1 + 1
            do ic = 0, 5
               if ( colsphiV_tmp(ic) >= 0 ) then
                  ncolsphiV(irow)    = ncolsphiV(irow) + 1
                  colsphiV(imatphiV) = colsphiV_tmp(ic)
                  matphiV(imatphiV)  = matphiV_tmp(ic) * diag(i,j,2) 
                  imatphiV = imatphiV + 1
               end if
            end do

            irow = irow + 1

         end do
      end do

   end subroutine petsc_fillmatbox

   function getCellUpwindCoeff ( leftedgeVel, rightedgeVel ) RESULT(coeff)
      implicit none
      real(rt), intent(in) :: leftedgeVel, rightedgeVel
      real(rt) :: coeff
      coeff = 0.0;
      if ( leftedgeVel < -eps ) then
         coeff = coeff - leftedgeVel;
      else if ( abs(leftedgeVel) <= eps ) then
         coeff = coeff - 0.5*leftedgeVel
      endif
      if ( rightedgeVel > eps ) then
         coeff = coeff + rightedgeVel
      else if ( abs(rightedgeVel) <= eps ) then
         coeff = coeff + 0.5*rightedgeVel
      endif
   end function getCellUpwindCoeff

   function getLeftCellUpwindCoeff ( leftedgeVel, rightedgeVel ) RESULT(coeff)
      implicit none
      real(rt), intent(in) :: leftedgeVel, rightedgeVel
      real(rt) :: coeff
      coeff = 0.0;
      if ( leftedgeVel > eps ) then
         coeff = coeff - leftedgeVel;
      else if ( abs(leftedgeVel) <= eps ) then
         coeff = coeff - 0.5*leftedgeVel
      endif
   end function getLeftCellUpwindCoeff

   function getRightCellUpwindCoeff ( leftedgeVel, rightedgeVel ) RESULT(coeff)
      implicit none
      real(rt), intent(in) :: leftedgeVel, rightedgeVel
      real(rt) :: coeff
      coeff = 0.0;
      if ( rightedgeVel < -eps ) then
         coeff = coeff + rightedgeVel;
      else if ( abs(rightedgeVel) <= eps ) then
         coeff = coeff + 0.5*rightedgeVel
      endif
   end function getRightCellUpwindCoeff

end module PETScSolver_F

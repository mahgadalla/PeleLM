#ifndef _PETScSolver_H_
#define _PETScSolver_H_

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_LayoutData.H>
#include <AMReX_BndryData.H>

namespace amrex {

class PETScSolver
{
public:

   PETScSolver (const BoxArray& grids, const DistributionMapping& dmap,
                const Geometry& geom_, MPI_Comm comm_);

   ~PETScSolver();   

   void solve (MultiFab& soln, const MultiFab& rhs, Real rel_tol, Real abs_tol, 
               const BndryData& bndry_ne, const BndryData& bndry_phiV, int max_iter, int max_bndry_order);
       
   void setScalars(Real scal_ne_ne, Real scal_ne_phiV, Real scal_phiV_ne, Real scal_phiV_phiV) {
      s_dtDiffI = scal_ne_ne;
      s_dtDr = scal_ne_phiV;
      s_Ie = scal_phiV_ne;
      s_L = scal_phiV_phiV;
   };

   void setLaplacianFactor(Real lap_scal) {
      lapl_fac = lap_scal;
   };

   void setdt(Real dt) {
      m_dt = dt;
   };

   void setDriftFaceCoeffs(const Array<const MultiFab*,AMREX_SPACEDIM>& neKe_ec_upwind);
   void setDiffFaceCoeffs(const Array<const MultiFab*,AMREX_SPACEDIM>& De_ec);
   void setUeffFaceCoeffs(const Array<const MultiFab*,AMREX_SPACEDIM>& Ueff_ec_upwind);

private:
   MPI_Comm comm = MPI_COMM_NULL;
   Geometry geom;

   int verbose = 0;

// Linear system data:   
// ---                  ---
// | --     --    --   -- | 
// | |       |    |     | | 
// | |dtDiffI|    |dtDr | | 
// | |       |    |     | | 
// | --     --    --   -- | 
// | --   --      --   -- | 
// | |     |      |     | | 
// | | Ie  |      |  L  | | 
// | |     |      |     | | 
// | --   --      --   -- | 
// ---                  ---
 
// Scaling of each submatrix   
   Real s_dtDiffI;
   Real s_dtDr;
   Real s_Ie;
   Real s_L;

// Data dtDiffI
   Array<MultiFab,AMREX_SPACEDIM> De_coefs;
   Array<MultiFab,AMREX_SPACEDIM> effvel_coefs;
 
// Data dtDr
   Array<MultiFab,AMREX_SPACEDIM> neKeupwind_coefs;

// Data L
   Real lapl_fac;

// Internal storage of dt
   Real m_dt;

   MultiFab diaginv;

   KSP solver = nullptr;
   Mat A = nullptr;
   Vec b = nullptr;
   Vec x = nullptr;

   FabFactory<FArrayBox> const* m_factory = nullptr;
   BndryData const* m_bndry_ne = nullptr;
   BndryData const* m_bndry_phiV = nullptr;
   int m_maxorder = -1;

   LayoutData<PetscInt> ncells_grid;
   LayoutData<Vector<PetscInt> > cell_id_ne_vec;
   LayoutData<Vector<PetscInt> > cell_id_phiV_vec;
   FabArray<BaseFab<PetscInt> > cell_id_ne;
   FabArray<BaseFab<PetscInt> > cell_id_phiV;

   void loadVectors (MultiFab& soln, const MultiFab& rhs);
   void getSolution (MultiFab& soln);
   void prepareSolver();

};

}

#endif /*_PETScSolver_H_*/

#ifndef _PETSCBNDRY_H_
#define _PETSCBNDRY_H_

#include <AMReX_InterpBndryData.H>
#include <AMReX_BC_TYPES.H>

class PETScBndry
    :
    public amrex::InterpBndryData
{
public:
    PETScBndry ();

    PETScBndry (const amrex::BoxArray& _grids,
                const amrex::DistributionMapping& _dmap,
                int             _ncomp,
                const amrex::Geometry& _geom);

    virtual ~PETScBndry () override;

    virtual void setBndryConds (const amrex::BCRec& phys_bc,
                                int          ratio) override;

    virtual void setBndryConds (const amrex::BCRec&   phys_bc,
                                const amrex::IntVect& ratio,
				                    int   comp=0) override;

    void setnEBndryConds (const amrex::BCRec& phys_bc,
                          int          ratio);

    void setnEBndryConds (const amrex::BCRec&   phys_bc,
                          const amrex::IntVect& ratio,
				              int   comp=0);

    void setphiVBndryConds (const amrex::BCRec& phys_bc,
                           int          ratio);

    void setphiVBndryConds (const amrex::BCRec&   phys_bc,
                            const amrex::IntVect& ratio,
				                int   comp=0);

    void setHomogValues (const amrex::BCRec&   bc,
                         const amrex::IntVect& ratio);
};

#endif /*_PETSCBNDRY_H_*/

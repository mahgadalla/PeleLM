#ifndef PELELM_INDEX_MACROS_H
#define PELELM_INDEX_MACROS_H
#include <mechanism.h> 


#ifdef BL_RHORT_IN_TRACER
#define NTHERM 5
#else
#define NTHERM 4
#endif

#define NVAR NUM_SPECIES + NTHERM 

#endif

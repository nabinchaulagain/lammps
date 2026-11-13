// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   This software is distributed under the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Axel Kohlmeyer (Temple U)
------------------------------------------------------------------------- */

#include "pair_lj_long_tip4p_long_omp.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "ewald_const.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "suffix.h"

#include <cmath>

#include "omp_compat.h"

using namespace LAMMPS_NS;
using namespace EwaldConst;

// bits in ewald_order / ewald_off flag which interaction order is treated
// with the long-range (k-space) solver: order 1 = Coulomb (1/r),
// order 6 = LJ dispersion (1/r^6).

enum { EWALD_COUL = 1 << 1, EWALD_DISP = 1 << 6 };

/* ---------------------------------------------------------------------- */

PairLJLongTIP4PLongOMP::PairLJLongTIP4PLongOMP(LAMMPS *lmp) :
  PairLJLongTIP4PLong(lmp), ThrOMP(lmp, THR_PAIR)
{
  suffix_flag |= Suffix::OMP;
  respa_enable = 1;
  cut_respa = nullptr;
  newsite_thr = nullptr;
  hneigh_thr = nullptr;
  tip4pflag = dispersionflag = 1;
  no_virial_fdotr_compute = 1;
  single_enable = 0;
  nmax = 0;
}

/* ---------------------------------------------------------------------- */

PairLJLongTIP4PLongOMP::~PairLJLongTIP4PLongOMP()
{
  memory->destroy(hneigh_thr);
  memory->destroy(newsite_thr);
}

/* ---------------------------------------------------------------------- */

void PairLJLongTIP4PLongOMP::compute(int eflag, int vflag)
{
  ev_init(eflag,vflag);

  // reallocate hneigh_thr & newsite_thr if necessary
  // initialize hneigh_thr[0] to -1 on steps when reneighboring occurred
  // initialize hneigh_thr[2] to 0 every step
  const int nlocal = atom->nlocal;
  const int nall = nlocal + atom->nghost;

  if (atom->nmax > nmax) {
    nmax = atom->nmax;
    memory->destroy(hneigh_thr);
    memory->create(hneigh_thr,nmax,"pair:hneigh_thr");
    memory->destroy(newsite_thr);
    memory->create(newsite_thr,nmax,"pair:newsite_thr");
  }

  int i;
  // tag entire list as completely invalid after a neighbor
  // list update, since that can change the order of atoms.
  if (neighbor->ago == 0)
    for (i = 0; i < nall; i++) hneigh_thr[i].a = -1;

  // indicate that the coordinates for the M point need to
  // be updated. this needs to be done in every step.
  for (i = 0; i < nall; i++) hneigh_thr[i].t = 0;

  const int order1 = ewald_order & EWALD_COUL;
  const int order6 = ewald_order & EWALD_DISP;

  const int nthreads = comm->nthreads;
  const int inum = list->inum;

#if defined(_OPENMP)
#pragma omp parallel LMP_DEFAULT_NONE LMP_SHARED(eflag,vflag)
#endif
  {
    int ifrom, ito, tid;

    loop_setup_thr(ifrom, ito, tid, inum, nthreads);
    ThrData *thr = fix->get_thr(tid);
    thr->timer(Timer::START);
    ev_setup_thr(eflag, vflag, nall, eatom, vatom, nullptr, thr);

    if (order6) {
      if (order1) {
        if (!ndisptablebits) {
          if (!ncoultablebits) {
            if (evflag) {
              if (eflag) {
                if (force->newton_pair) eval<1,1,1,0,0,1,1>(ifrom, ito, thr);
                else eval<1,1,0,0,0,1,1>(ifrom, ito, thr);
              } else {
                if (force->newton_pair) eval<1,0,1,0,0,1,1>(ifrom, ito, thr);
                else eval<1,0,0,0,0,1,1>(ifrom, ito, thr);
              }
            } else {
              if (force->newton_pair) eval<0,0,1,0,0,1,1>(ifrom, ito, thr);
              else eval<0,0,0,0,0,1,1>(ifrom, ito, thr);
            }
          } else {
            if (evflag) {
              if (eflag) {
                if (force->newton_pair) eval<1,1,1,1,0,1,1>(ifrom, ito, thr);
                else eval<1,1,0,1,0,1,1>(ifrom, ito, thr);
              } else {
                if (force->newton_pair) eval<1,0,1,1,0,1,1>(ifrom, ito, thr);
                else eval<1,0,0,1,0,1,1>(ifrom, ito, thr);
              }
            } else {
              if (force->newton_pair) eval<0,0,1,1,0,1,1>(ifrom, ito, thr);
              else eval<0,0,0,1,0,1,1>(ifrom, ito, thr);
            }
          }
        } else {
          if (!ncoultablebits) {
            if (evflag) {
              if (eflag) {
                if (force->newton_pair) eval<1,1,1,0,1,1,1>(ifrom, ito, thr);
                else eval<1,1,0,0,1,1,1>(ifrom, ito, thr);
              } else {
                if (force->newton_pair) eval<1,0,1,0,1,1,1>(ifrom, ito, thr);
                else eval<1,0,0,0,1,1,1>(ifrom, ito, thr);
              }
            } else {
              if (force->newton_pair) eval<0,0,1,0,1,1,1>(ifrom, ito, thr);
              else eval<0,0,0,0,1,1,1>(ifrom, ito, thr);
            }
          } else {
            if (evflag) {
              if (eflag) {
                if (force->newton_pair) eval<1,1,1,1,1,1,1>(ifrom, ito, thr);
                else eval<1,1,0,1,1,1,1>(ifrom, ito, thr);
              } else {
                if (force->newton_pair) eval<1,0,1,1,1,1,1>(ifrom, ito, thr);
                else eval<1,0,0,1,1,1,1>(ifrom, ito, thr);
              }
            } else {
              if (force->newton_pair) eval<0,0,1,1,1,1,1>(ifrom, ito, thr);
              else eval<0,0,0,1,1,1,1>(ifrom, ito, thr);
            }
          }
        }
      } else {
        if (!ndisptablebits) {
          if (!ncoultablebits) {
            if (evflag) {
              if (eflag) {
                if (force->newton_pair) eval<1,1,1,0,0,0,1>(ifrom, ito, thr);
                else eval<1,1,0,0,0,0,1>(ifrom, ito, thr);
              } else {
                if (force->newton_pair) eval<1,0,1,0,0,0,1>(ifrom, ito, thr);
                else eval<1,0,0,0,0,0,1>(ifrom, ito, thr);
              }
            } else {
              if (force->newton_pair) eval<0,0,1,0,0,0,1>(ifrom, ito, thr);
              else eval<0,0,0,0,0,0,1>(ifrom, ito, thr);
            }
          } else {
            if (evflag) {
              if (eflag) {
                if (force->newton_pair) eval<1,1,1,1,0,0,1>(ifrom, ito, thr);
                else eval<1,1,0,1,0,0,1>(ifrom, ito, thr);
              } else {
                if (force->newton_pair) eval<1,0,1,1,0,0,1>(ifrom, ito, thr);
                else eval<1,0,0,1,0,0,1>(ifrom, ito, thr);
              }
            } else {
              if (force->newton_pair) eval<0,0,1,1,0,0,1>(ifrom, ito, thr);
              else eval<0,0,0,1,0,0,1>(ifrom, ito, thr);
            }
          }
        } else {
          if (!ncoultablebits) {
            if (evflag) {
              if (eflag) {
                if (force->newton_pair) eval<1,1,1,0,1,0,1>(ifrom, ito, thr);
                else eval<1,1,0,0,1,0,1>(ifrom, ito, thr);
              } else {
                if (force->newton_pair) eval<1,0,1,0,1,0,1>(ifrom, ito, thr);
                else eval<1,0,0,0,1,0,1>(ifrom, ito, thr);
              }
            } else {
              if (force->newton_pair) eval<0,0,1,0,1,0,1>(ifrom, ito, thr);
              else eval<0,0,0,0,1,0,1>(ifrom, ito, thr);
            }
          } else {
            if (evflag) {
              if (eflag) {
                if (force->newton_pair) eval<1,1,1,1,1,0,1>(ifrom, ito, thr);
                else eval<1,1,0,1,1,0,1>(ifrom, ito, thr);
              } else {
                if (force->newton_pair) eval<1,0,1,1,1,0,1>(ifrom, ito, thr);
                else eval<1,0,0,1,1,0,1>(ifrom, ito, thr);
              }
            } else {
              if (force->newton_pair) eval<0,0,1,1,1,0,1>(ifrom, ito, thr);
              else eval<0,0,0,1,1,0,1>(ifrom, ito, thr);
            }
          }
        }
      }
    } else {
      if (order1) {
        if (!ndisptablebits) {
          if (!ncoultablebits) {
            if (evflag) {
              if (eflag) {
                if (force->newton_pair) eval<1,1,1,0,0,1,0>(ifrom, ito, thr);
                else eval<1,1,0,0,0,1,0>(ifrom, ito, thr);
              } else {
                if (force->newton_pair) eval<1,0,1,0,0,1,0>(ifrom, ito, thr);
                else eval<1,0,0,0,0,1,0>(ifrom, ito, thr);
              }
            } else {
              if (force->newton_pair) eval<0,0,1,0,0,1,0>(ifrom, ito, thr);
              else eval<0,0,0,0,0,1,0>(ifrom, ito, thr);
            }
          } else {
            if (evflag) {
              if (eflag) {
                if (force->newton_pair) eval<1,1,1,1,0,1,0>(ifrom, ito, thr);
                else eval<1,1,0,1,0,1,0>(ifrom, ito, thr);
              } else {
                if (force->newton_pair) eval<1,0,1,1,0,1,0>(ifrom, ito, thr);
                else eval<1,0,0,1,0,1,0>(ifrom, ito, thr);
              }
            } else {
              if (force->newton_pair) eval<0,0,1,1,0,1,0>(ifrom, ito, thr);
              else eval<0,0,0,1,0,1,0>(ifrom, ito, thr);
            }
          }
        } else {
          if (!ncoultablebits) {
            if (evflag) {
              if (eflag) {
                if (force->newton_pair) eval<1,1,1,0,1,1,0>(ifrom, ito, thr);
                else eval<1,1,0,0,1,1,0>(ifrom, ito, thr);
              } else {
                if (force->newton_pair) eval<1,0,1,0,1,1,0>(ifrom, ito, thr);
                else eval<1,0,0,0,1,1,0>(ifrom, ito, thr);
              }
            } else {
              if (force->newton_pair) eval<0,0,1,0,1,1,0>(ifrom, ito, thr);
              else eval<0,0,0,0,1,1,0>(ifrom, ito, thr);
            }
          } else {
            if (evflag) {
              if (eflag) {
                if (force->newton_pair) eval<1,1,1,1,1,1,0>(ifrom, ito, thr);
                else eval<1,1,0,1,1,1,0>(ifrom, ito, thr);
              } else {
                if (force->newton_pair) eval<1,0,1,1,1,1,0>(ifrom, ito, thr);
                else eval<1,0,0,1,1,1,0>(ifrom, ito, thr);
              }
            } else {
              if (force->newton_pair) eval<0,0,1,1,1,1,0>(ifrom, ito, thr);
              else eval<0,0,0,1,1,1,0>(ifrom, ito, thr);
            }
          }
        }
      } else {
        if (!ndisptablebits) {
          if (!ncoultablebits) {
            if (evflag) {
              if (eflag) {
                if (force->newton_pair) eval<1,1,1,0,0,0,0>(ifrom, ito, thr);
                else eval<1,1,0,0,0,0,0>(ifrom, ito, thr);
              } else {
                if (force->newton_pair) eval<1,0,1,0,0,0,0>(ifrom, ito, thr);
                else eval<1,0,0,0,0,0,0>(ifrom, ito, thr);
              }
            } else {
              if (force->newton_pair) eval<0,0,1,0,0,0,0>(ifrom, ito, thr);
              else eval<0,0,0,0,0,0,0>(ifrom, ito, thr);
            }
          } else {
            if (evflag) {
              if (eflag) {
                if (force->newton_pair) eval<1,1,1,1,0,0,0>(ifrom, ito, thr);
                else eval<1,1,0,1,0,0,0>(ifrom, ito, thr);
              } else {
                if (force->newton_pair) eval<1,0,1,1,0,0,0>(ifrom, ito, thr);
                else eval<1,0,0,1,0,0,0>(ifrom, ito, thr);
              }
            } else {
              if (force->newton_pair) eval<0,0,1,1,0,0,0>(ifrom, ito, thr);
              else eval<0,0,0,1,0,0,0>(ifrom, ito, thr);
            }
          }
        } else {
          if (!ncoultablebits) {
            if (evflag) {
              if (eflag) {
                if (force->newton_pair) eval<1,1,1,0,1,0,0>(ifrom, ito, thr);
                else eval<1,1,0,0,1,0,0>(ifrom, ito, thr);
              } else {
                if (force->newton_pair) eval<1,0,1,0,1,0,0>(ifrom, ito, thr);
                else eval<1,0,0,0,1,0,0>(ifrom, ito, thr);
              }
            } else {
            if (force->newton_pair) eval<0,0,1,0,1,0,0>(ifrom, ito, thr);
              else eval<0,0,0,0,1,0,0>(ifrom, ito, thr);
            }
          } else {
            if (evflag) {
              if (eflag) {
                if (force->newton_pair) eval<1,1,1,1,1,0,0>(ifrom, ito, thr);
                else eval<1,1,0,1,1,0,0>(ifrom, ito, thr);
              } else {
                if (force->newton_pair) eval<1,0,1,1,1,0,0>(ifrom, ito, thr);
                else eval<1,0,0,1,1,0,0>(ifrom, ito, thr);
              }
            } else {
              if (force->newton_pair) eval<0,0,1,1,1,0,0>(ifrom, ito, thr);
              else eval<0,0,0,1,1,0,0>(ifrom, ito, thr);
            }
          }
        }
      }
    }

    thr->timer(Timer::PAIR);
    reduce_thr(this, eflag, vflag, thr);
  } // end of omp parallel region
}

/* ---------------------------------------------------------------------- */

void PairLJLongTIP4PLongOMP::compute_inner()
{
   // reallocate hneigh_thr & newsite_thr if necessary
  // initialize hneigh_thr[0] to -1 on steps when reneighboring occurred
  // initialize hneigh_thr[2] to 0 every step
  const int nall = atom->nlocal + atom->nghost;

  if (atom->nmax > nmax) {
    nmax = atom->nmax;
    memory->destroy(hneigh_thr);
    memory->create(hneigh_thr,nmax,"pair:hneigh_thr");
    memory->destroy(newsite_thr);
    memory->create(newsite_thr,nmax,"pair:newsite_thr");
  }

  int i;
  // tag entire list as completely invalid after a neighbor
  // list update, since that can change the order of atoms.
  if (neighbor->ago == 0)
    for (i = 0; i < nall; i++) hneigh_thr[i].a = -1;

  // indicate that the coordinates for the M point need to
  // be updated. this needs to be done in every step.
  for (i = 0; i < nall; i++) hneigh_thr[i].t = 0;

  const int nthreads = comm->nthreads;
  const int inum = list->inum_inner;
#if defined(_OPENMP)
#pragma omp parallel LMP_DEFAULT_NONE
#endif
  {
    int ifrom, ito, tid;

    loop_setup_thr(ifrom, ito, tid, inum, nthreads);
    ThrData *thr = fix->get_thr(tid);
    thr->timer(Timer::START);
    ev_setup_thr(0, 0, nall, nullptr, nullptr, nullptr, thr);
    eval_inner(ifrom, ito, thr);
    thr->timer(Timer::PAIR);

  }  // end of omp parallel region
}

/* ---------------------------------------------------------------------- */

void PairLJLongTIP4PLongOMP::compute_middle()
{

  const int nall = atom->nlocal + atom->nghost;
  const int nthreads = comm->nthreads;
  const int inum = list->inum_middle;

#if defined(_OPENMP)
#pragma omp parallel LMP_DEFAULT_NONE
#endif
  {
    int ifrom, ito, tid;

    loop_setup_thr(ifrom, ito, tid, inum, nthreads);
    ThrData *thr = fix->get_thr(tid);
    thr->timer(Timer::START);
    ev_setup_thr(0, 0, nall, nullptr, nullptr, nullptr, thr);
    eval_middle(ifrom, ito, thr);
    thr->timer(Timer::PAIR);

  }  // end of omp parallel region
}

/* ---------------------------------------------------------------------- */

void PairLJLongTIP4PLongOMP::compute_outer(int eflag, int vflag)
{
  ev_init(eflag,vflag);
  const int order1 = ewald_order & EWALD_COUL;
  const int order6 = ewald_order & EWALD_DISP;

  const int nall = atom->nlocal + atom->nghost;

  // reallocate hneigh_thr & newsite_thr if necessary
  // initialize hneigh_thr[0] to -1 on steps when reneighboring occurred
  // initialize hneigh_thr[2] to 0 every step

  if (atom->nmax > nmax) {
    nmax = atom->nmax;
    memory->destroy(hneigh_thr);
    memory->create(hneigh_thr,nmax,"pair:hneigh_thr");
    memory->destroy(newsite_thr);
    memory->create(newsite_thr,nmax,"pair:newsite_thr");
  }

  int i;
  // tag entire list as completely invalid after a neighbor
  // list update, since that can change the order of atoms.
  if (neighbor->ago == 0) {
    for (i = 0; i < nall; i++) hneigh_thr[i].a = -1;
    // indicate that the coordinates for the M point need to
    // be updated. this needs to be done only if neighbor list
    // has been updated in compute_outer
    for (i = 0; i < nall; i++) hneigh_thr[i].t = 0;
  }

  const int nthreads = comm->nthreads;
  const int inum = list->inum;

#if defined(_OPENMP)
#pragma omp parallel LMP_DEFAULT_NONE LMP_SHARED(eflag,vflag)
#endif
  {
    int ifrom, ito, tid;

    loop_setup_thr(ifrom, ito, tid, inum, nthreads);
    ThrData *thr = fix->get_thr(tid);
    thr->timer(Timer::START);
    ev_setup_thr(eflag, vflag, nall, eatom, vatom, nullptr, thr);

    if (order6) {
      if (order1) {
        if (!ndisptablebits) {
          if (!ncoultablebits) {
            if (evflag) {
              if (eflag) {
                if (force->newton_pair) eval_outer<1,1,1,0,0,1,1>(ifrom, ito, thr);
                else eval_outer<1,1,0,0,0,1,1>(ifrom, ito, thr);
              } else {
                if (force->newton_pair) eval_outer<1,0,1,0,0,1,1>(ifrom, ito, thr);
                else eval_outer<1,0,0,0,0,1,1>(ifrom, ito, thr);
              }
            } else {
              if (force->newton_pair) eval_outer<0,0,1,0,0,1,1>(ifrom, ito, thr);
              else eval_outer<0,0,0,0,0,1,1>(ifrom, ito, thr);
            }
          } else {
            if (evflag) {
              if (eflag) {
                if (force->newton_pair) eval_outer<1,1,1,1,0,1,1>(ifrom, ito, thr);
                else eval_outer<1,1,0,1,0,1,1>(ifrom, ito, thr);
              } else {
                if (force->newton_pair) eval_outer<1,0,1,1,0,1,1>(ifrom, ito, thr);
                else eval_outer<1,0,0,1,0,1,1>(ifrom, ito, thr);
              }
            } else {
              if (force->newton_pair) eval_outer<0,0,1,1,0,1,1>(ifrom, ito, thr);
              else eval_outer<0,0,0,1,0,1,1>(ifrom, ito, thr);
            }
          }
        } else {
          if (!ncoultablebits) {
            if (evflag) {
              if (eflag) {
                if (force->newton_pair) eval_outer<1,1,1,0,1,1,1>(ifrom, ito, thr);
                else eval_outer<1,1,0,0,1,1,1>(ifrom, ito, thr);
              } else {
                if (force->newton_pair) eval_outer<1,0,1,0,1,1,1>(ifrom, ito, thr);
                else eval_outer<1,0,0,0,1,1,1>(ifrom, ito, thr);
              }
            } else {
              if (force->newton_pair) eval_outer<0,0,1,0,1,1,1>(ifrom, ito, thr);
              else eval_outer<0,0,0,0,1,1,1>(ifrom, ito, thr);
            }
          } else {
            if (evflag) {
              if (eflag) {
                if (force->newton_pair) eval_outer<1,1,1,1,1,1,1>(ifrom, ito, thr);
                else eval_outer<1,1,0,1,1,1,1>(ifrom, ito, thr);
              } else {
                if (force->newton_pair) eval_outer<1,0,1,1,1,1,1>(ifrom, ito, thr);
                else eval_outer<1,0,0,1,1,1,1>(ifrom, ito, thr);
              }
            } else {
              if (force->newton_pair) eval_outer<0,0,1,1,1,1,1>(ifrom, ito, thr);
              else eval_outer<0,0,0,1,1,1,1>(ifrom, ito, thr);
            }
          }
        }
      } else {
        if (!ndisptablebits) {
          if (!ncoultablebits) {
            if (evflag) {
              if (eflag) {
                if (force->newton_pair) eval_outer<1,1,1,0,0,0,1>(ifrom, ito, thr);
                else eval_outer<1,1,0,0,0,0,1>(ifrom, ito, thr);
              } else {
                if (force->newton_pair) eval_outer<1,0,1,0,0,0,1>(ifrom, ito, thr);
                else eval_outer<1,0,0,0,0,0,1>(ifrom, ito, thr);
              }
            } else {
              if (force->newton_pair) eval_outer<0,0,1,0,0,0,1>(ifrom, ito, thr);
              else eval_outer<0,0,0,0,0,0,1>(ifrom, ito, thr);
            }
          } else {
            if (evflag) {
              if (eflag) {
                if (force->newton_pair) eval_outer<1,1,1,1,0,0,1>(ifrom, ito, thr);
                else eval_outer<1,1,0,1,0,0,1>(ifrom, ito, thr);
              } else {
                if (force->newton_pair) eval_outer<1,0,1,1,0,0,1>(ifrom, ito, thr);
                else eval_outer<1,0,0,1,0,0,1>(ifrom, ito, thr);
              }
            } else {
              if (force->newton_pair) eval_outer<0,0,1,1,0,0,1>(ifrom, ito, thr);
              else eval_outer<0,0,0,1,0,0,1>(ifrom, ito, thr);
            }
          }
        } else {
          if (!ncoultablebits) {
            if (evflag) {
              if (eflag) {
                if (force->newton_pair) eval_outer<1,1,1,0,1,0,1>(ifrom, ito, thr);
                else eval_outer<1,1,0,0,1,0,1>(ifrom, ito, thr);
              } else {
                if (force->newton_pair) eval_outer<1,0,1,0,1,0,1>(ifrom, ito, thr);
                else eval_outer<1,0,0,0,1,0,1>(ifrom, ito, thr);
              }
            } else {
              if (force->newton_pair) eval_outer<0,0,1,0,1,0,1>(ifrom, ito, thr);
              else eval_outer<0,0,0,0,1,0,1>(ifrom, ito, thr);
            }
          } else {
            if (evflag) {
              if (eflag) {
                if (force->newton_pair) eval_outer<1,1,1,1,1,0,1>(ifrom, ito, thr);
                else eval_outer<1,1,0,1,1,0,1>(ifrom, ito, thr);
              } else {
                if (force->newton_pair) eval_outer<1,0,1,1,1,0,1>(ifrom, ito, thr);
                else eval_outer<1,0,0,1,1,0,1>(ifrom, ito, thr);
              }
            } else {
              if (force->newton_pair) eval_outer<0,0,1,1,1,0,1>(ifrom, ito, thr);
              else eval_outer<0,0,0,1,1,0,1>(ifrom, ito, thr);
            }
          }
        }
      }
    } else {
      if (order1) {
        if (!ndisptablebits) {
          if (!ncoultablebits) {
            if (evflag) {
              if (eflag) {
                if (force->newton_pair) eval_outer<1,1,1,0,0,1,0>(ifrom, ito, thr);
                else eval_outer<1,1,0,0,0,1,0>(ifrom, ito, thr);
              } else {
                if (force->newton_pair) eval_outer<1,0,1,0,0,1,0>(ifrom, ito, thr);
                else eval_outer<1,0,0,0,0,1,0>(ifrom, ito, thr);
              }
            } else {
              if (force->newton_pair) eval_outer<0,0,1,0,0,1,0>(ifrom, ito, thr);
              else eval_outer<0,0,0,0,0,1,0>(ifrom, ito, thr);
            }
          } else {
            if (evflag) {
              if (eflag) {
                if (force->newton_pair) eval_outer<1,1,1,1,0,1,0>(ifrom, ito, thr);
                else eval_outer<1,1,0,1,0,1,0>(ifrom, ito, thr);
              } else {
                if (force->newton_pair) eval_outer<1,0,1,1,0,1,0>(ifrom, ito, thr);
                else eval_outer<1,0,0,1,0,1,0>(ifrom, ito, thr);
              }
            } else {
              if (force->newton_pair) eval_outer<0,0,1,1,0,1,0>(ifrom, ito, thr);
              else eval_outer<0,0,0,1,0,1,0>(ifrom, ito, thr);
            }
          }
        } else {
          if (!ncoultablebits) {
            if (evflag) {
              if (eflag) {
                if (force->newton_pair) eval_outer<1,1,1,0,1,1,0>(ifrom, ito, thr);
                else eval_outer<1,1,0,0,1,1,0>(ifrom, ito, thr);
              } else {
                if (force->newton_pair) eval_outer<1,0,1,0,1,1,0>(ifrom, ito, thr);
                else eval_outer<1,0,0,0,1,1,0>(ifrom, ito, thr);
              }
            } else {
              if (force->newton_pair) eval_outer<0,0,1,0,1,1,0>(ifrom, ito, thr);
              else eval_outer<0,0,0,0,1,1,0>(ifrom, ito, thr);
            }
          } else {
            if (evflag) {
              if (eflag) {
                if (force->newton_pair) eval_outer<1,1,1,1,1,1,0>(ifrom, ito, thr);
                else eval_outer<1,1,0,1,1,1,0>(ifrom, ito, thr);
              } else {
                if (force->newton_pair) eval_outer<1,0,1,1,1,1,0>(ifrom, ito, thr);
                else eval_outer<1,0,0,1,1,1,0>(ifrom, ito, thr);
              }
            } else {
              if (force->newton_pair) eval_outer<0,0,1,1,1,1,0>(ifrom, ito, thr);
              else eval_outer<0,0,0,1,1,1,0>(ifrom, ito, thr);
            }
          }
        }
      } else {
        if (!ndisptablebits) {
          if (!ncoultablebits) {
            if (evflag) {
              if (eflag) {
                if (force->newton_pair) eval_outer<1,1,1,0,0,0,0>(ifrom, ito, thr);
                else eval_outer<1,1,0,0,0,0,0>(ifrom, ito, thr);
              } else {
                if (force->newton_pair) eval_outer<1,0,1,0,0,0,0>(ifrom, ito, thr);
                else eval_outer<1,0,0,0,0,0,0>(ifrom, ito, thr);
              }
            } else {
              if (force->newton_pair) eval_outer<0,0,1,0,0,0,0>(ifrom, ito, thr);
              else eval_outer<0,0,0,0,0,0,0>(ifrom, ito, thr);
            }
          } else {
            if (evflag) {
              if (eflag) {
                if (force->newton_pair) eval_outer<1,1,1,1,0,0,0>(ifrom, ito, thr);
                else eval_outer<1,1,0,1,0,0,0>(ifrom, ito, thr);
              } else {
                if (force->newton_pair) eval_outer<1,0,1,1,0,0,0>(ifrom, ito, thr);
                else eval_outer<1,0,0,1,0,0,0>(ifrom, ito, thr);
              }
            } else {
              if (force->newton_pair) eval_outer<0,0,1,1,0,0,0>(ifrom, ito, thr);
              else eval_outer<0,0,0,1,0,0,0>(ifrom, ito, thr);
            }
          }
        } else {
          if (!ncoultablebits) {
            if (evflag) {
              if (eflag) {
                if (force->newton_pair) eval_outer<1,1,1,0,1,0,0>(ifrom, ito, thr);
                else eval_outer<1,1,0,0,1,0,0>(ifrom, ito, thr);
              } else {
                if (force->newton_pair) eval_outer<1,0,1,0,1,0,0>(ifrom, ito, thr);
                else eval_outer<1,0,0,0,1,0,0>(ifrom, ito, thr);
              }
            } else {
            if (force->newton_pair) eval_outer<0,0,1,0,1,0,0>(ifrom, ito, thr);
              else eval_outer<0,0,0,0,1,0,0>(ifrom, ito, thr);
            }
          } else {
            if (evflag) {
              if (eflag) {
                if (force->newton_pair) eval_outer<1,1,1,1,1,0,0>(ifrom, ito, thr);
                else eval_outer<1,1,0,1,1,0,0>(ifrom, ito, thr);
              } else {
                if (force->newton_pair) eval_outer<1,0,1,1,1,0,0>(ifrom, ito, thr);
                else eval_outer<1,0,0,1,1,0,0>(ifrom, ito, thr);
              }
            } else {
              if (force->newton_pair) eval_outer<0,0,1,1,1,0,0>(ifrom, ito, thr);
              else eval_outer<0,0,0,1,1,0,0>(ifrom, ito, thr);
            }
          }
        }
      }
    }

    thr->timer(Timer::PAIR);
    reduce_thr(this, eflag, vflag, thr);
  } // end of omp parallel region
}

/* ---------------------------------------------------------------------- */

template < const int EVFLAG, const int EFLAG,
           const int NEWTON_PAIR, const int CTABLE, const int LJTABLE, const int ORDER1, const int ORDER6 >
void PairLJLongTIP4PLongOMP::eval(int iifrom, int iito, ThrData * const thr)
{
  const auto * _noalias const x = (dbl3_t *) atom->x[0];
  auto * _noalias const f = (dbl3_t *) thr->get_f()[0];
  const double * _noalias const q = atom->q;
  const int * _noalias const type = atom->type;
  const tagint * _noalias const tag = atom->tag;
  const int nlocal = atom->nlocal;
  const double * _noalias const special_coul = force->special_coul;
  const double * _noalias const special_lj = force->special_lj;
  const double qqrd2e = force->qqrd2e;
  const double cut_coulsqplus = (cut_coul+2.0*qdist)*(cut_coul+2.0*qdist);
  const int vflag = vflag_global || vflag_atom;

  const double g2 = g_ewald_6*g_ewald_6, g6 = g2*g2*g2, g8 = g6*g2;

  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;

  auto compute_tip4p_site = [&](int idx, dbl3_t &site_out, int &H1_out, int &H2_out) {
    if (type[idx] == typeO) {
      if (hneigh_thr[idx].a < 0) {
        int H1 = atom->map(tag[idx] + 1);
        int H2 = atom->map(tag[idx] + 2);
        if (H1 == -1 || H2 == -1)
          error->one(FLERR, "TIP4P hydrogen is missing");
        if (type[H1] != typeH || type[H2] != typeH)
          error->one(FLERR, "TIP4P hydrogen has incorrect atom type");
        H1 = domain->closest_image(idx, H1);
        H2 = domain->closest_image(idx, H2);
        compute_newsite_thr(x[idx], x[H1], x[H2], newsite_thr[idx]);
        hneigh_thr[idx].t = 1;
        hneigh_thr[idx].b = H2;
        hneigh_thr[idx].a = H1;
        H1_out = H1;
        H2_out = H2;
      } else {
        H1_out = hneigh_thr[idx].a;
        H2_out = hneigh_thr[idx].b;
        if (hneigh_thr[idx].t == 0) {
          compute_newsite_thr(x[idx], x[H1_out], x[H2_out], newsite_thr[idx]);
          hneigh_thr[idx].t = 1;
        }
      }
      site_out = newsite_thr[idx];
    } else {
      site_out = x[idx];
      H1_out = -1;
      H2_out = -1;
    }
  };

  auto apply_tip4p_force_and_virial = [&](int idx, int itype, int H1, int H2, 
                                          double delx, double dely, double delz, 
                                          double cforce, double sign, 
                                          double &fx_acc, double &fy_acc, double &fz_acc,
                                          int &n, int &key, int *vlist, double *v) {
    if (itype != typeO) {
      fx_acc += sign * delx * cforce;
      fy_acc += sign * dely * cforce;
      fz_acc += sign * delz * cforce;

      if constexpr (EVFLAG) {
        if (vflag) {
          v[0] += sign * x[idx].x * delx * cforce;
          v[1] += sign * x[idx].y * dely * cforce;
          v[2] += sign * x[idx].z * delz * cforce;
          v[3] += sign * x[idx].x * dely * cforce;
          v[4] += sign * x[idx].x * delz * cforce;
          v[5] += sign * x[idx].y * delz * cforce;
          vlist[n++] = idx;
        }
      }
    } else {
      if constexpr (EVFLAG) {
        if (vflag) key += (sign > 0 ? 1 : 2);
      }
      
      double fdx = sign * delx * cforce;
      double fdy = sign * dely * cforce;
      double fdz = sign * delz * cforce;

      double fOx = fdx * (1.0 - alpha);
      double fOy = fdy * (1.0 - alpha);
      double fOz = fdz * (1.0 - alpha);

      double fHx = 0.5 * alpha * fdx;
      double fHy = 0.5 * alpha * fdy;
      double fHz = 0.5 * alpha * fdz;

      fx_acc += fOx;
      fy_acc += fOy;
      fz_acc += fOz;

      f[H1].x += fHx;
      f[H1].y += fHy;
      f[H1].z += fHz;

      f[H2].x += fHx;
      f[H2].y += fHy;
      f[H2].z += fHz;

      if constexpr (EVFLAG) {
        if (vflag) {
          dbl3_t xH1 = x[H1];
          dbl3_t xH2 = x[H2];
          v[0] += x[idx].x * fOx + xH1.x * fHx + xH2.x * fHx;
          v[1] += x[idx].y * fOy + xH1.y * fHy + xH2.y * fHy;
          v[2] += x[idx].z * fOz + xH1.z * fHz + xH2.z * fHz;
          v[3] += x[idx].x * fOy + xH1.x * fHy + xH2.x * fHy;
          v[4] += x[idx].x * fOz + xH1.x * fHz + xH2.x * fHz;
          v[5] += x[idx].y * fOz + xH1.y * fHz + xH2.y * fHz;
          vlist[n++] = idx;
          vlist[n++] = H1;
          vlist[n++] = H2;
        }
      }
    }
  };

  for (int ii = iifrom; ii < iito; ++ii) {
    int i = ilist[ii];
    double qtmp = q[i];
    double xtmp = x[i].x;
    double ytmp = x[i].y;
    double ztmp = x[i].z;
    int itype = type[i];
    
    dbl3_t x1;
    int iH1, iH2;
    compute_tip4p_site(i, x1, iH1, iH2);

    int *jlist = firstneigh[i];
    int jnum = numneigh[i];
    double fxtmp = 0.0, fytmp = 0.0, fztmp = 0.0;
    
    double *offseti = offset[itype];
    double *lj1i = lj1[itype]; 
    double *lj2i = lj2[itype]; 
    double *lj3i = lj3[itype]; 
    double *lj4i = lj4[itype];

    for (int jj = 0; jj < jnum; jj++) {
      int j = jlist[jj];
      int ni = sbmask(j);
      double factor_coul = special_coul[ni];
      j &= NEIGHMASK;

      double delx = xtmp - x[j].x;
      double dely = ytmp - x[j].y;
      double delz = ztmp - x[j].z;
      double rsq = delx*delx + dely*dely + delz*delz;
      int jtype = type[j];

      if (rsq < cut_ljsq[itype][jtype]) {           // lj
        double r2inv = 1.0/rsq;
        double forcelj = 0.0;
        double evdwl = 0.0;

        if constexpr (ORDER6) {                   // long-range lj
          if constexpr (!LJTABLE) {
            double rn = r2inv*r2inv*r2inv;
            double x2 = g2*rsq, a2 = 1.0/x2;
            x2 = a2*exp(-x2)*lj4i[jtype];
            if (ni == 0) {
              forcelj = (rn*=rn)*lj1i[jtype]-g8*(((6.0*a2+6.0)*a2+3.0)*a2+1.0)*x2*rsq;
              if constexpr (EFLAG) evdwl = rn*lj3i[jtype]-g6*((a2+1.0)*a2+0.5)*x2;
            } else {                  // special case
              double f = special_lj[ni], t = rn*(1.0-f);
              forcelj = f*(rn *= rn)*lj1i[jtype]-g8*(((6.0*a2+6.0)*a2+3.0)*a2+1.0)*x2*rsq+t*lj2i[jtype];
              if constexpr (EFLAG) evdwl = f*rn*lj3i[jtype]-g6*((a2+1.0)*a2+0.5)*x2+t*lj4i[jtype];
            }
          } else {
            if (rsq <= tabinnerdispsq) {
              double rn = r2inv*r2inv*r2inv;
              double x2 = g2*rsq, a2 = 1.0/x2;
              x2 = a2*exp(-x2)*lj4i[jtype];
              if (ni == 0) {
                forcelj = (rn*=rn)*lj1i[jtype]-g8*(((6.0*a2+6.0)*a2+3.0)*a2+1.0)*x2*rsq;
                if constexpr (EFLAG) evdwl = rn*lj3i[jtype]-g6*((a2+1.0)*a2+0.5)*x2;
              } else {                  // special case
                double f = special_lj[ni], t = rn*(1.0-f);
                forcelj = f*(rn *= rn)*lj1i[jtype]-g8*(((6.0*a2+6.0)*a2+3.0)*a2+1.0)*x2*rsq+t*lj2i[jtype];
                if constexpr (EFLAG) evdwl = f*rn*lj3i[jtype]-g6*((a2+1.0)*a2+0.5)*x2+t*lj4i[jtype];
              }
            } else {                                        // table real space
              union_int_float_t disp_t;
              disp_t.f = rsq;
              const int disp_k = (disp_t.i & ndispmask)>>ndispshiftbits;
              double f_disp = (rsq-rdisptable[disp_k])*drdisptable[disp_k];
              double rn = r2inv*r2inv*r2inv;
              if (ni == 0) {
                forcelj = (rn*=rn)*lj1i[jtype]-(fdisptable[disp_k]+f_disp*dfdisptable[disp_k])*lj4i[jtype];
                if constexpr (EFLAG) evdwl = rn*lj3i[jtype]-(edisptable[disp_k]+f_disp*dedisptable[disp_k])*lj4i[jtype];
              } else {                  // special case
                double f = special_lj[ni], t = rn*(1.0-f);
                forcelj = f*(rn *= rn)*lj1i[jtype]-(fdisptable[disp_k]+f_disp*dfdisptable[disp_k])*lj4i[jtype]+t*lj2i[jtype];
                if constexpr (EFLAG) evdwl = f*rn*lj3i[jtype]-(edisptable[disp_k]+f_disp*dedisptable[disp_k])*lj4i[jtype]+t*lj4i[jtype];
              }
            }
          }
        } else {                      // cut lj
          double rn = r2inv*r2inv*r2inv;
          if (ni == 0) {
            forcelj = rn*(rn*lj1i[jtype]-lj2i[jtype]);
            if constexpr (EFLAG) evdwl = rn*(rn*lj3i[jtype]-lj4i[jtype])-offseti[jtype];
          } else {                    // special case
            double f = special_lj[ni];
            forcelj = f*rn*(rn*lj1i[jtype]-lj2i[jtype]);
            if constexpr (EFLAG) evdwl = f * (rn*(rn*lj3i[jtype]-lj4i[jtype])-offseti[jtype]);
          }
        }

        forcelj *= r2inv;
        fxtmp += delx*forcelj;
        fytmp += dely*forcelj;
        fztmp += delz*forcelj;
        f[j].x -= delx*forcelj;
        f[j].y -= dely*forcelj;
        f[j].z -= delz*forcelj;

        if constexpr (EVFLAG) {
          ev_tally_thr(this, i, j, nlocal, /* newton_pair = */ 1,
                       evdwl, 0.0, forcelj, delx, dely, delz, thr);
        }
      }

      // adjust rsq and delxyz for off-site O charge(s)
      if (rsq < cut_coulsqplus) {
        dbl3_t x2;
        int jH1, jH2;
        if (itype == typeO || jtype == typeO) {
          if (jtype == typeO) {
            compute_tip4p_site(j, x2, jH1, jH2);
          } else {
            x2 = x[j];
          }
          delx = x1.x - x2.x;
          dely = x1.y - x2.y;
          delz = x1.z - x2.z;
          rsq = delx*delx + dely*dely + delz*delz;
        }

        // test current rsq against cutoff and compute Coulombic force
        if (rsq < cut_coulsq && ORDER1) {
          double r2inv = 1.0 / rsq;
          double forcecoul = 0.0;
          double prefactor = 0.0;
          double ecoul = 0.0;
          int itable = 0;
          double fraction = 0.0;
          double erfc = 0.0;

          if constexpr (!CTABLE) {
            double r = sqrt(rsq);
            double grij = g_ewald * r;
            double expm2 = exp(-grij*grij);
            double t = 1.0 / (1.0 + EWALD_P*grij);
            erfc = t * (A1+t*(A2+t*(A3+t*(A4+t*A5)))) * expm2;
            prefactor = qqrd2e * qtmp*q[j]/r;
            forcecoul = prefactor * (erfc + EWALD_F*grij*expm2);
            if (factor_coul < 1.0) forcecoul -= (1.0-factor_coul)*prefactor;
          } else {
            if (rsq <= tabinnersq) {
              double r = sqrt(rsq);
              double grij = g_ewald * r;
              double expm2 = exp(-grij*grij);
              double t = 1.0 / (1.0 + EWALD_P*grij);
              erfc = t * (A1+t*(A2+t*(A3+t*(A4+t*A5)))) * expm2;
              prefactor = qqrd2e * qtmp*q[j]/r;
              forcecoul = prefactor * (erfc + EWALD_F*grij*expm2);
              if (factor_coul < 1.0) forcecoul -= (1.0-factor_coul)*prefactor;
            } else {
              union_int_float_t rsq_lookup;
              rsq_lookup.f = rsq;
              itable = rsq_lookup.i & ncoulmask;
              itable >>= ncoulshiftbits;
              fraction = ((double) rsq_lookup.f - rtable[itable]) * drtable[itable];
              double table = ftable[itable] + fraction*dftable[itable];
              forcecoul = qtmp*q[j] * table;
              if (factor_coul < 1.0) {
                table = ctable[itable] + fraction*dctable[itable];
                prefactor = qtmp*q[j] * table;
                forcecoul -= (1.0-factor_coul)*prefactor;
              }
            }
          }

          double cforce = forcecoul * r2inv;

          if constexpr (EFLAG) {
            if constexpr (!CTABLE) {
              ecoul = prefactor*erfc;
            } else {
              if (rsq <= tabinnersq) {
                ecoul = prefactor*erfc;
              } else {
                double table = etable[itable] + fraction*detable[itable];
                ecoul = qtmp*q[j] * table;
              }
            }
            if (factor_coul < 1.0) ecoul -= (1.0-factor_coul)*prefactor;
          }

          int vlist[6];
          double v[6] = {};
          int n = 0;
          int key = 0;

          apply_tip4p_force_and_virial(i, itype, iH1, iH2, delx, dely, delz, cforce, 1.0, fxtmp, fytmp, fztmp, n, key, vlist, v);
          apply_tip4p_force_and_virial(j, jtype, jH1, jH2, delx, dely, delz, cforce, -1.0, f[j].x, f[j].y, f[j].z, n, key, vlist, v);

          if constexpr (EVFLAG) {
            ev_tally_list_thr(this, key, vlist, v, ecoul, alpha, thr);
          }
        }
      }
    }
    f[i].x += fxtmp;
    f[i].y += fytmp;
    f[i].z += fztmp;
  }
}

/* ---------------------------------------------------------------------- */

void PairLJLongTIP4PLongOMP::eval_inner(int iifrom, int iito, ThrData * const thr)
{
  double rsq, r2inv, forcecoul = 0.0, forcelj, cforce;

  const auto * _noalias const x = (dbl3_t *) atom->x[0];
  auto * _noalias const f = (dbl3_t *) thr->get_f()[0];
  const double * _noalias const q = atom->q;
  const int * _noalias const type = atom->type;
  const tagint * _noalias const tag = atom->tag;
  const double * _noalias const special_coul = force->special_coul;
  const double * _noalias const special_lj = force->special_lj;
  const double qqrd2e = force->qqrd2e;
  const double cut_coulsqplus = (cut_coul+2.0*qdist)*(cut_coul+2.0*qdist);

  const double cut_out_on = cut_respa[0];
  const double cut_out_off = cut_respa[1];

  const double cut_out_diff = cut_out_off - cut_out_on;
  const double cut_out_on_sq = cut_out_on*cut_out_on;
  const double cut_out_off_sq = cut_out_off*cut_out_off;

  int ni;
  const int order1 = (ewald_order | ~ewald_off) & EWALD_COUL;
  double qri;

  int i,j,ii,jj,jnum,itype,jtype;
  int iH1,iH2,jH1,jH2;
  double qtmp,xtmp,ytmp,ztmp,delx,dely,delz,fxtmp,fytmp,fztmp;
  double fOx,fOy,fOz,fHx,fHy,fHz,fdx,fdy,fdz;
  dbl3_t x1,x2;
  int *ilist,*jlist,*numneigh,**firstneigh;

  double *lj1i, *lj2i;

  ilist = list->ilist_inner;
  numneigh = list->numneigh_inner;
  firstneigh = list->firstneigh_inner;

  // loop over neighbors of my atoms

  for (ii = iifrom; ii < iito; ++ii) {
    i = ilist[ii];
    qtmp = q[i];
    xtmp = x[i].x;
    ytmp = x[i].y;
    ztmp = x[i].z;
    itype = type[i];

    // if atom I = water O, set x1 = offset charge site
    // else x1 = x of atom I
    // NOTE: to make this part thread safe, we need to
    // make sure that the hneigh_thr[][] entries only get
    // updated, when all data is in place. worst case,
    // some calculation is repeated, but since the results
    // will be the same, there is no race condition.
    if (itype == typeO) {
      if (hneigh_thr[i].a < 0) {
        iH1 = atom->map(tag[i] + 1);
        iH2 = atom->map(tag[i] + 2);
        if (iH1 == -1 || iH2 == -1)
          error->one(FLERR,"TIP4P hydrogen is missing");
        if (type[iH1] != typeH || type[iH2] != typeH)
          error->one(FLERR,"TIP4P hydrogen has incorrect atom type");
        // set iH1,iH2 to index of closest image to O
        iH1 = domain->closest_image(i,iH1);
        iH2 = domain->closest_image(i,iH2);
        compute_newsite_thr(x[i],x[iH1],x[iH2],newsite_thr[i]);
        hneigh_thr[i].t = 1;
        hneigh_thr[i].b = iH2;
        hneigh_thr[i].a = iH1;
      } else {
        iH1 = hneigh_thr[i].a;
        iH2 = hneigh_thr[i].b;
        if (hneigh_thr[i].t == 0) {
          compute_newsite_thr(x[i],x[iH1],x[iH2],newsite_thr[i]);
          hneigh_thr[i].t = 1;
        }
      }
      x1 = newsite_thr[i];
    } else x1 = x[i];

    jlist = firstneigh[i];
    jnum = numneigh[i];
    lj1i = lj1[itype]; lj2i = lj2[itype];
    fxtmp=fytmp=fztmp=0.0;

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      ni = sbmask(j);
      j &= NEIGHMASK;

      delx = xtmp - x[j].x;
      dely = ytmp - x[j].y;
      delz = ztmp - x[j].z;
      rsq = delx*delx + dely*dely + delz*delz;
      jtype = type[j];

      if (rsq < cut_ljsq[itype][jtype] && rsq < cut_out_off_sq) {  // lj
        r2inv = 1.0/rsq;
        double r6inv = r2inv*r2inv*r2inv;
        if (ni == 0) forcelj = r6inv*(r6inv*lj1i[jtype]-lj2i[jtype]);
        else {                  // special case
          double factor = special_lj[ni];
          forcelj = factor*r6inv*(r6inv*lj1i[jtype]-lj2i[jtype]);
        }

        if (rsq > cut_out_on_sq) {                        // switching
          double rsw = (sqrt(rsq) - cut_out_on)/cut_out_diff;
          forcelj  *= 1.0 + rsw*rsw*(2.0*rsw-3.0);
        }

        forcelj *= r2inv;
        fxtmp += delx*forcelj;
        fytmp += dely*forcelj;
        fztmp += delz*forcelj;
        f[j].x -= delx*forcelj;
        f[j].y -= dely*forcelj;
        f[j].z -= delz*forcelj;
      }


      // adjust rsq and delxyz for off-site O charge(s)
      // ADDITIONAL REQEUST REQUIRED HERE!!!!!

      if (rsq < cut_coulsqplus && order1) {
        if (itype == typeO || jtype == typeO) {
          if (jtype == typeO) {
            if (hneigh_thr[j].a < 0) {
              jH1 = atom->map(tag[j] + 1);
              jH2 = atom->map(tag[j] + 2);
              if (jH1 == -1 || jH2 == -1)
                error->one(FLERR,"TIP4P hydrogen is missing");
              if (type[jH1] != typeH || type[jH2] != typeH)
                error->one(FLERR,"TIP4P hydrogen has incorrect atom type");
              // set jH1,jH2 to closest image to O
              jH1 = domain->closest_image(j,jH1);
              jH2 = domain->closest_image(j,jH2);
              compute_newsite_thr(x[j],x[jH1],x[jH2],newsite_thr[j]);
              hneigh_thr[j].t = 1;
              hneigh_thr[j].b = jH2;
              hneigh_thr[j].a = jH1;
            } else {
              jH1 = hneigh_thr[j].a;
              jH2 = hneigh_thr[j].b;
              if (hneigh_thr[j].t == 0) {
                compute_newsite_thr(x[j],x[jH1],x[jH2],newsite_thr[j]);
                hneigh_thr[j].t = 1;
              }
            }
            x2 = newsite_thr[j];
          } else x2 = x[j];
          delx = x1.x - x2.x;
          dely = x1.y - x2.y;
          delz = x1.z - x2.z;
          rsq = delx*delx + dely*dely + delz*delz;
        }

        // test current rsq against cutoff and compute Coulombic force

        if (rsq < cut_coulsq && rsq < cut_out_off_sq) {
          r2inv = 1.0 / rsq;
          qri = qqrd2e*qtmp;
          if (ni == 0) forcecoul = qri*q[j]*sqrt(r2inv);
          else {
            forcecoul = qri*q[j]*sqrt(r2inv)*special_coul[ni];
          }

          if (rsq > cut_out_on_sq) {                        // switching
            double rsw = (sqrt(rsq) - cut_out_on)/cut_out_diff;
            forcecoul  *= 1.0 + rsw*rsw*(2.0*rsw-3.0);
          }

          cforce = forcecoul * r2inv;

          //if (evflag) ev_tally(i,j,nlocal,newton_pair,
          //               evdwl,0.0,cforce,delx,dely,delz);

          // if i,j are not O atoms, force is applied directly
          // if i or j are O atoms, force is on fictitious atom & partitioned
          // force partitioning due to Feenstra, J Comp Chem, 20, 786 (1999)
          // f_f = fictitious force, fO = f_f (1 - 2 alpha), fH = alpha f_f
          // preserves total force and torque on water molecule
          // virial = sum(r x F) where each water's atoms are near xi and xj
          // vlist stores 2,4,6 atoms whose forces contribute to virial

          if (itype != typeO) {
            fxtmp += delx * cforce;
            fytmp += dely * cforce;
            fztmp += delz * cforce;

          } else {
            fdx = delx*cforce;
            fdy = dely*cforce;
            fdz = delz*cforce;

            fOx = fdx*(1 - alpha);
            fOy = fdy*(1 - alpha);
            fOz = fdz*(1 - alpha);

            fHx = 0.5 * alpha * fdx;
            fHy = 0.5 * alpha * fdy;
            fHz = 0.5 * alpha * fdz;

            fxtmp += fOx;
            fytmp += fOy;
            fztmp += fOz;

            f[iH1].x += fHx;
            f[iH1].y += fHy;
            f[iH1].z += fHz;

            f[iH2].x += fHx;
            f[iH2].y += fHy;
            f[iH2].z += fHz;
          }

          if (jtype != typeO) {
            f[j].x -= delx * cforce;
            f[j].y -= dely * cforce;
            f[j].z -= delz * cforce;

          } else {
            fdx = -delx*cforce;
            fdy = -dely*cforce;
            fdz = -delz*cforce;

            fOx = fdx*(1 - alpha);
            fOy = fdy*(1 - alpha);
            fOz = fdz*(1 - alpha);

            fHx = 0.5 * alpha * fdx;
            fHy = 0.5 * alpha * fdy;
            fHz = 0.5 * alpha * fdz;

            f[j].x += fOx;
            f[j].y += fOy;
            f[j].z += fOz;

            f[jH1].x += fHx;
            f[jH1].y += fHy;
            f[jH1].z += fHz;

            f[jH2].x += fHx;
            f[jH2].y += fHy;
            f[jH2].z += fHz;
          }
        }
      }
    }
    f[i].x += fxtmp;
    f[i].y += fytmp;
    f[i].z += fztmp;
  }
}

/* ---------------------------------------------------------------------- */

void PairLJLongTIP4PLongOMP::eval_middle(int iifrom, int iito, ThrData * const thr)
{
  double rsq, r2inv, forcecoul,forcelj, cforce;

  const auto * _noalias const x = (dbl3_t *) atom->x[0];
  auto * _noalias const f = (dbl3_t *) thr->get_f()[0];
  const double * _noalias const q = atom->q;
  const int * _noalias const type = atom->type;
  const tagint * _noalias const tag = atom->tag;
  const double * _noalias const special_coul = force->special_coul;
  const double * _noalias const special_lj = force->special_lj;
  const double qqrd2e = force->qqrd2e;

  const double cut_coulsqplus = (cut_coul+2.0*qdist)*(cut_coul+2.0*qdist);
  const int order1 = (ewald_order | ~ewald_off) & EWALD_COUL;

  const double cut_in_off = cut_respa[0];
  const double cut_in_on = cut_respa[1];
  const double cut_out_on = cut_respa[2];
  const double cut_out_off = cut_respa[3];

  const double cut_in_diff = cut_in_on - cut_in_off;
  const double cut_out_diff = cut_out_off - cut_out_on;
  const double cut_in_off_sq = cut_in_off*cut_in_off;
  const double cut_in_on_sq = cut_in_on*cut_in_on;
  const double cut_out_on_sq = cut_out_on*cut_out_on;
  const double cut_out_off_sq = cut_out_off*cut_out_off;

  int i,j,ii,jj,jnum,itype,jtype;
  int iH1,iH2,jH1,jH2;
  double qtmp,xtmp,ytmp,ztmp,delx,dely,delz,fxtmp,fytmp,fztmp;
  double fOx,fOy,fOz,fHx,fHy,fHz,fdx,fdy,fdz;
  dbl3_t x1,x2;
  int *ilist,*jlist,*numneigh,**firstneigh;
  double qri;

  int ni;
  double *lj1i, *lj2i;

  ilist = list->ilist_middle;
  numneigh = list->numneigh_middle;
  firstneigh = list->firstneigh_middle;

  // loop over neighbors of my atoms

  for (ii = iifrom; ii < iito; ++ii) {
    i = ilist[ii];
    qtmp = q[i];
    xtmp = x[i].x;
    ytmp = x[i].y;
    ztmp = x[i].z;
    itype = type[i];
    if (itype == typeO) {
      if (hneigh_thr[i].a < 0) {
        iH1 = atom->map(tag[i] + 1);
        iH2 = atom->map(tag[i] + 2);
        if (iH1 == -1 || iH2 == -1)
          error->one(FLERR,"TIP4P hydrogen is missing");
        if (type[iH1] != typeH || type[iH2] != typeH)
          error->one(FLERR,"TIP4P hydrogen has incorrect atom type");
        // set iH1,iH2 to index of closest image to O
        iH1 = domain->closest_image(i,iH1);
        iH2 = domain->closest_image(i,iH2);
        compute_newsite_thr(x[i],x[iH1],x[iH2],newsite_thr[i]);
        hneigh_thr[i].t = 1;
        hneigh_thr[i].b = iH2;
        hneigh_thr[i].a = iH1;
      } else {
        iH1 = hneigh_thr[i].a;
        iH2 = hneigh_thr[i].b;
        if (hneigh_thr[i].t == 0) {
          compute_newsite_thr(x[i],x[iH1],x[iH2],newsite_thr[i]);
          hneigh_thr[i].t = 1;
        }
      }
      x1 = newsite_thr[i];
    } else x1 = x[i];

    jlist = firstneigh[i];
    jnum = numneigh[i];
    lj1i = lj1[itype]; lj2i = lj2[itype];
    fxtmp = fytmp = fztmp = 0.0;

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      ni = sbmask(j);
      j &= NEIGHMASK;

      delx = xtmp - x[j].x;
      dely = ytmp - x[j].y;
      delz = ztmp - x[j].z;
      rsq = delx*delx + dely*dely + delz*delz;
      jtype = type[j];

      if (rsq < cut_ljsq[itype][jtype] && rsq >= cut_in_off_sq && rsq <= cut_out_off_sq) {  // lj
        r2inv = 1.0/rsq;
        double r6inv = r2inv*r2inv*r2inv;
        if (ni == 0) forcelj = r6inv*(r6inv*lj1i[jtype]-lj2i[jtype]);
        else {                  // special case
          double factor = special_lj[ni];
          forcelj = factor*r6inv*(r6inv*lj1i[jtype]-lj2i[jtype]);
        }

        if (rsq < cut_in_on_sq) {                                // switching
          double rsw = (sqrt(rsq) - cut_in_off)/cut_in_diff;
          forcelj  *= rsw*rsw*(3.0 - 2.0*rsw);
        }
        if (rsq > cut_out_on_sq) {
          double rsw = (sqrt(rsq) - cut_out_on)/cut_out_diff;
          forcelj  *= 1.0 + rsw*rsw*(2.0*rsw-3.0);
        }

        forcelj *= r2inv;
        fxtmp += delx*forcelj;
        fytmp += dely*forcelj;
        fztmp += delz*forcelj;
        f[j].x -= delx*forcelj;
        f[j].y -= dely*forcelj;
        f[j].z -= delz*forcelj;
      }


      // adjust rsq and delxyz for off-site O charge(s)
      // ADDITIONAL REQEUST REQUIRED HERE!!!!!

      if (rsq < cut_coulsqplus && order1) {
        if (itype == typeO || jtype == typeO) {
          if (jtype == typeO) {
            if (hneigh_thr[j].a < 0) {
              jH1 = atom->map(tag[j] + 1);
              jH2 = atom->map(tag[j] + 2);
              if (jH1 == -1 || jH2 == -1)
                error->one(FLERR,"TIP4P hydrogen is missing");
              if (type[jH1] != typeH || type[jH2] != typeH)
                error->one(FLERR,"TIP4P hydrogen has incorrect atom type");
              // set jH1,jH2 to closest image to O
              jH1 = domain->closest_image(j,jH1);
              jH2 = domain->closest_image(j,jH2);
              compute_newsite_thr(x[j],x[jH1],x[jH2],newsite_thr[j]);
              hneigh_thr[j].t = 1;
              hneigh_thr[j].b = jH2;
              hneigh_thr[j].a = jH1;
            } else {
              jH1 = hneigh_thr[j].a;
              jH2 = hneigh_thr[j].b;
              if (hneigh_thr[j].t == 0) {
                compute_newsite_thr(x[j],x[jH1],x[jH2],newsite_thr[j]);
                hneigh_thr[j].t = 1;
              }
            }
            x2 = newsite_thr[j];
          } else x2 = x[j];
          delx = x1.x - x2.x;
          dely = x1.y - x2.y;
          delz = x1.z - x2.z;
          rsq = delx*delx + dely*dely + delz*delz;
        }

        // test current rsq against cutoff and compute Coulombic force

        if (rsq < cut_coulsq &&  rsq >= cut_in_off_sq && rsq <= cut_out_off_sq) {
          r2inv = 1.0 / rsq;
          qri = qqrd2e*qtmp;
          if (ni == 0) forcecoul = qri*q[j]*sqrt(r2inv);
          else {
            forcecoul = qri*q[j]*sqrt(r2inv)*special_coul[ni];
          }

          if (rsq < cut_in_on_sq) {                                // switching
            double rsw = (sqrt(rsq) - cut_in_off)/cut_in_diff;
            forcecoul  *= rsw*rsw*(3.0 - 2.0*rsw);
          }
          if (rsq > cut_out_on_sq) {
            double rsw = (sqrt(rsq) - cut_out_on)/cut_out_diff;
            forcecoul  *= 1.0 + rsw*rsw*(2.0*rsw-3.0);
          }

          cforce = forcecoul * r2inv;

          //if (evflag) ev_tally(i,j,nlocal,newton_pair,
          //               evdwl,0.0,cforce,delx,dely,delz);

          // if i,j are not O atoms, force is applied directly
          // if i or j are O atoms, force is on fictitious atom & partitioned
          // force partitioning due to Feenstra, J Comp Chem, 20, 786 (1999)
          // f_f = fictitious force, fO = f_f (1 - 2 alpha), fH = alpha f_f
          // preserves total force and torque on water molecule
          // virial = sum(r x F) where each water's atoms are near xi and xj
          // vlist stores 2,4,6 atoms whose forces contribute to virial

          if (itype != typeO) {
            fxtmp += delx * cforce;
            fytmp += dely * cforce;
            fztmp += delz * cforce;

          } else {
            fdx = delx*cforce;
            fdy = dely*cforce;
            fdz = delz*cforce;

            fOx = fdx*(1 - alpha);
            fOy = fdy*(1 - alpha);
            fOz = fdz*(1 - alpha);

            fHx = 0.5 * alpha * fdx;
            fHy = 0.5 * alpha * fdy;
            fHz = 0.5 * alpha * fdz;

            fxtmp += fOx;
            fytmp += fOy;
            fztmp += fOz;

            f[iH1].x += fHx;
            f[iH1].y += fHy;
            f[iH1].z += fHz;

            f[iH2].x += fHx;
            f[iH2].y += fHy;
            f[iH2].z += fHz;
          }

          if (jtype != typeO) {
            f[j].x -= delx * cforce;
            f[j].y -= dely * cforce;
            f[j].z -= delz * cforce;

          } else {
            fdx = -delx*cforce;
            fdy = -dely*cforce;
            fdz = -delz*cforce;

            fOx = fdx*(1 - alpha);
            fOy = fdy*(1 - alpha);
            fOz = fdz*(1 - alpha);

            fHx = 0.5 * alpha * fdx;
            fHy = 0.5 * alpha * fdy;
            fHz = 0.5 * alpha * fdz;

            f[j].x += fOx;
            f[j].y += fOy;
            f[j].z += fOz;

            f[jH1].x += fHx;
            f[jH1].y += fHy;
            f[jH1].z += fHz;

            f[jH2].x += fHx;
            f[jH2].y += fHy;
            f[jH2].z += fHz;
          }
        }
      }
    }
    f[i].x += fxtmp;
    f[i].y += fytmp;
    f[i].z += fztmp;
  }
}

/* ---------------------------------------------------------------------- */

template < const int EVFLAG, const int EFLAG,
           const int NEWTON_PAIR, const int CTABLE, const int LJTABLE, const int ORDER1, const int ORDER6 >
void PairLJLongTIP4PLongOMP::eval_outer(int iifrom, int iito, ThrData * const thr)
{
  double evdwl,ecoul,fvirial;
  evdwl = ecoul = 0.0;
  double qtmp,xtmp,ytmp,ztmp,delx,dely,delz;
  double r2inv,forcecoul,forcelj,cforce, respa_coul, respa_lj, frespa;
  double fdx,fdy,fdz,fOx,fOy,fOz,fHx,fHy,fHz;
  double v[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  dbl3_t x1,x2,xH1,xH2;

  const auto * _noalias const x = (dbl3_t *) atom->x[0];
  auto * _noalias const f = (dbl3_t *) thr->get_f()[0];
  const double * _noalias const q = atom->q;
  const int * _noalias const type = atom->type;
  const tagint * _noalias const tag = atom->tag;
  const int nlocal = atom->nlocal;
  const double * _noalias const special_coul = force->special_coul;
  const double * _noalias const special_lj = force->special_lj;
  const double qqrd2e = force->qqrd2e;
  const double cut_coulsqplus = (cut_coul+2.0*qdist)*(cut_coul+2.0*qdist);
  const int vflag = vflag_atom || vflag_global;

  int i,j,ii,jj,jnum,itype,jtype;
  int n,vlist[6];
  int key;
  int iH1,iH2,jH1,jH2;
  int *ilist,*jlist,*numneigh,**firstneigh;
  double rsq,qri;
  int respa_flag;

  int ni;
  double *lj1i, *lj2i, *lj3i, *lj4i, *offseti;
  double g2 = g_ewald_6*g_ewald_6, g6 = g2*g2*g2, g8 = g6*g2;

  const double cut_in_off = cut_respa[2];
  const double cut_in_on = cut_respa[3];

  const double cut_in_diff = cut_in_on - cut_in_off;
  const double cut_in_off_sq = cut_in_off*cut_in_off;
  const double cut_in_on_sq = cut_in_on*cut_in_on;

  double fxtmp,fytmp,fztmp;

  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // loop over neighbors of my atoms

  for (ii = iifrom; ii < iito; ++ii) {
    i = ilist[ii];
    qtmp = q[i];
    qri = qtmp*qqrd2e;
    xtmp = x[i].x;
    ytmp = x[i].y;
    ztmp = x[i].z;
    itype = type[i];
    if (itype == typeO) {
      if (hneigh_thr[i].a < 0) {
        iH1 = atom->map(tag[i] + 1);
        iH2 = atom->map(tag[i] + 2);
        if (iH1 == -1 || iH2 == -1)
          error->one(FLERR,"TIP4P hydrogen is missing");
        if (type[iH1] != typeH || type[iH2] != typeH)
          error->one(FLERR,"TIP4P hydrogen has incorrect atom type");
        // set iH1,iH2 to closest image to O
        iH1 = domain->closest_image(i,iH1);
        iH2 = domain->closest_image(i,iH2);
        hneigh_thr[i].t = 1;
        hneigh_thr[i].b = iH2;
        hneigh_thr[i].a = iH1;
        compute_newsite_thr(x[i],x[iH1],x[iH2],newsite_thr[i]);
      } else {
        iH1 = hneigh_thr[i].a;
        iH2 = hneigh_thr[i].b;
        if (hneigh_thr[i].t == 0) {
          compute_newsite_thr(x[i],x[iH1],x[iH2],newsite_thr[i]);
          hneigh_thr[i].t = 1;
        }
      }
      x1 = newsite_thr[i];
    } else x1 = x[i];

    jlist = firstneigh[i];
    jnum = numneigh[i];
    offseti = offset[itype];
    lj1i = lj1[itype]; lj2i = lj2[itype]; lj3i = lj3[itype]; lj4i = lj4[itype];
    fxtmp = fytmp = fztmp = 0.0;

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      ni = sbmask(j);
      j &= NEIGHMASK;

      delx = xtmp - x[j].x;
      dely = ytmp - x[j].y;
      delz = ztmp - x[j].z;
      rsq = delx*delx + dely*dely + delz*delz;
      jtype = type[j];

      respa_coul = 0;
      respa_lj = 0;
      if (rsq < cut_ljsq[itype][jtype]) {           // lj
        frespa = 1.0;                                       // check whether and how to compute respa corrections
        respa_flag = rsq < cut_in_on_sq ? 1 : 0;
        if (respa_flag && (rsq > cut_in_off_sq)) {
          double rsw = (sqrt(rsq)-cut_in_off)/cut_in_diff;
          frespa = 1-rsw*rsw*(3.0-2.0*rsw);
        }

        r2inv = 1.0/rsq;
        double r6inv = r2inv*r2inv*r2inv;
        if (respa_flag) respa_lj = ni == 0 ?                 // correct for respa
                          frespa*r6inv*(r6inv*lj1i[jtype]-lj2i[jtype]) :
                          frespa*r6inv*(r6inv*lj1i[jtype]-lj2i[jtype])*special_lj[ni];
        if (ORDER6) {                                        // long-range form
          if (!ndisptablebits || rsq <= tabinnerdispsq) {
            double r12inv = r6inv*r6inv;
            double gr2 = g2*rsq, a2 = 1.0/gr2;
            double expterm = a2*exp(-gr2)*lj4i[jtype];
            double g6term = g6*((a2+1.0)*a2+0.5)*expterm;
            double g8term = g8*(((6.0*a2+6.0)*a2+3.0)*a2+1.0)*expterm*rsq;
            if (ni == 0) {
              forcelj = r12inv*lj1i[jtype]-g8term-respa_lj;
              if (EFLAG) evdwl = r12inv*lj3i[jtype]-g6term;
            } else {                                        // correct for special
              double factor = special_lj[ni], t = r6inv*(1.0-factor);
              forcelj = factor*r12inv*lj1i[jtype]-g8term+t*lj2i[jtype]-respa_lj;
              if (EFLAG) evdwl = factor*r12inv*lj3i[jtype]-g6term+t*lj4i[jtype];
            }
          } else {                        // table real space
            union_int_float_t rsq_lookup;
            rsq_lookup.f = rsq;
            const int disp_k = (rsq_lookup.i & ndispmask)>>ndispshiftbits;
            double f_disp = (rsq-rdisptable[disp_k])*drdisptable[disp_k];
            double ftable_disp = fdisptable[disp_k]+f_disp*dfdisptable[disp_k];
            double etable_disp = edisptable[disp_k]+f_disp*dedisptable[disp_k];
            double r12inv = r6inv*r6inv;
            if (ni == 0) {
              forcelj = r12inv*lj1i[jtype]-ftable_disp*lj4i[jtype]-respa_lj;
              if (EFLAG) evdwl = r12inv*lj3i[jtype]-etable_disp*lj4i[jtype];
            } else {                  // special case
              double factor = special_lj[ni], t = r6inv*(1.0-factor);
              forcelj = factor*r12inv*lj1i[jtype]-ftable_disp*lj4i[jtype]+t*lj2i[jtype]-respa_lj;
              if (EFLAG) evdwl = factor*r12inv*lj3i[jtype]-etable_disp*lj4i[jtype]+t*lj4i[jtype];
            }
          }
        } else {                                                // cut form
          if (ni == 0) {
            forcelj = r6inv*(r6inv*lj1i[jtype]-lj2i[jtype])-respa_lj;
            if (EFLAG) evdwl = r6inv*(r6inv*lj3i[jtype]-lj4i[jtype])-offseti[jtype];
          } else {                                        // correct for special
            double factor = special_lj[ni];
            forcelj = factor*r6inv*(r6inv*lj1i[jtype]-lj2i[jtype])-respa_lj;
            if (EFLAG)
              evdwl = factor*(r6inv*(r6inv*lj3i[jtype]-lj4i[jtype])-offseti[jtype]);
          }
        }

        forcelj *= r2inv;
        fxtmp += delx*forcelj;
        fytmp += dely*forcelj;
        fztmp += delz*forcelj;
        f[j].x -= delx*forcelj;
        f[j].y -= dely*forcelj;
        f[j].z -= delz*forcelj;

        if (EVFLAG) {
          fvirial = forcelj + respa_lj*r2inv;
          ev_tally_thr(this,i,j,nlocal,/*newton_pair = */ 1,
                       evdwl,0.0,fvirial,delx,dely,delz, thr);
        }
      }


      // adjust rsq and delxyz for off-site O charge(s)
      // ADDITIONAL REQEUST REQUIRED HERE!!!!!

      if (rsq < cut_coulsqplus) {
        if (itype == typeO || jtype == typeO) {
          if (jtype == typeO) {
            if (hneigh_thr[j].a < 0) {
              jH1 = atom->map(tag[j] + 1);
              jH2 = atom->map(tag[j] + 2);
              if (jH1 == -1 || jH2 == -1)
                error->one(FLERR,"TIP4P hydrogen is missing");
              if (type[jH1] != typeH || type[jH2] != typeH)
                error->one(FLERR,"TIP4P hydrogen has incorrect atom type");
              // set jH1,jH2 to closest image to O
              jH1 = domain->closest_image(j,jH1);
              jH2 = domain->closest_image(j,jH2);
              compute_newsite_thr(x[j],x[jH1],x[jH2],newsite_thr[j]);
              hneigh_thr[j].t = 1;
              hneigh_thr[j].b = jH2;
              hneigh_thr[j].a = jH1;
            } else {
              jH1 = hneigh_thr[j].a;
              jH2 = hneigh_thr[j].b;
              if (hneigh_thr[j].t == 0) {
                compute_newsite_thr(x[j],x[jH1],x[jH2],newsite_thr[j]);
                hneigh_thr[j].t = 1;
              }
            }
            x2 = newsite_thr[j];
          } else x2 = x[j];
          delx = x1.x - x2.x;
          dely = x1.y - x2.y;
          delz = x1.z - x2.z;
          rsq = delx*delx + dely*dely + delz*delz;
        }

        // test current rsq against cutoff and compute Coulombic force
        if ((rsq < cut_coulsq) && ORDER1) {

          frespa = 1.0;                                       // check whether and how to compute respa corrections
          respa_flag = rsq < cut_in_on_sq ? 1 : 0;
          if (respa_flag && (rsq > cut_in_off_sq)) {
            double rsw = (sqrt(rsq)-cut_in_off)/cut_in_diff;
            frespa = 1-rsw*rsw*(3.0-2.0*rsw);
          }

          r2inv = 1.0 / rsq;
          if (!CTABLE || rsq <= tabinnersq) {        // series real space
            double r = sqrt(rsq), s = qri*q[j];
            if (respa_flag)                                // correct for respa
              respa_coul = ni == 0 ? frespa*s/r : frespa*s/r*special_coul[ni];
            double x = g_ewald*r, t = 1.0/(1.0+EWALD_P*x);
            if (ni == 0) {
              s *= g_ewald*exp(-x*x);
              forcecoul = (t *= ((((t*A5+A4)*t+A3)*t+A2)*t+A1)*s/x)+EWALD_F*s-respa_coul;
              if (EFLAG) ecoul = t;
            } else {                                        // correct for special
              r = s*(1.0-special_coul[ni])/r; s *= g_ewald*exp(-x*x);
              forcecoul = (t *= ((((t*A5+A4)*t+A3)*t+A2)*t+A1)*s/x)+EWALD_F*s-r-respa_coul;
              if (EFLAG) ecoul = t-r;
            }
          } else {                                             // table real space
            if (respa_flag) {
              double r = sqrt(rsq), s = qri*q[j];
              respa_coul = ni == 0 ? frespa*s/r : frespa*s/r*special_coul[ni];
            }
            union_int_float_t t;
            t.f = rsq;
            const int k = (t.i & ncoulmask) >> ncoulshiftbits;
            double f = ((double)t.f-rtable[k])*drtable[k], qiqj = qtmp*q[j];
            if (ni == 0) {
              forcecoul = qiqj*(ftable[k]+f*dftable[k]);
              if (EFLAG) ecoul = qiqj*(etable[k]+f*detable[k]);
            } else {                                        // correct for special
              t.f = (1.0-special_coul[ni])*(ctable[k]+f*dctable[k]);
              forcecoul = qiqj*(ftable[k]+f*dftable[k]-(double)t.f);
              if (EFLAG) {
                t.f = (1.0-special_coul[ni])*(ptable[k]+f*dptable[k]);
                ecoul = qiqj*(etable[k]+f*detable[k]-(double)t.f);
              }
            }
          }

          cforce = forcecoul * r2inv;
          fvirial = (forcecoul + respa_coul) * r2inv;

          // if i,j are not O atoms, force is applied directly
          // if i or j are O atoms, force is on fictitious atom & partitioned
          // force partitioning due to Feenstra, J Comp Chem, 20, 786 (1999)
          // f_f = fictitious force, fO = f_f (1 - 2 alpha), fH = alpha f_f
          // preserves total force and torque on water molecule
          // virial = sum(r x F) where each water's atoms are near xi and xj
          // vlist stores 2,4,6 atoms whose forces contribute to virial

          if (EVFLAG && vflag) {
            n = 0;
            key = 0;
          }

          if (itype != typeO) {
            fxtmp += delx * cforce;
            fytmp += dely * cforce;
            fztmp += delz * cforce;

            if (EVFLAG && vflag) {
              v[0] = x[i].x * delx * fvirial;
              v[1] = x[i].y * dely * fvirial;
              v[2] = x[i].z * delz * fvirial;
              v[3] = x[i].x * dely * fvirial;
              v[4] = x[i].x * delz * fvirial;
              v[5] = x[i].y * delz * fvirial;
              vlist[n++] = i;
            }

          } else {
            if (EVFLAG && vflag) key += 1;

            fdx = delx*cforce;
            fdy = dely*cforce;
            fdz = delz*cforce;

            fOx = fdx*(1 - alpha);
            fOy = fdy*(1 - alpha);
            fOz = fdz*(1 - alpha);

            fHx = 0.5*alpha * fdx;
            fHy = 0.5*alpha * fdy;
            fHz = 0.5*alpha * fdz;

            fxtmp += fOx;
            fytmp += fOy;
            fztmp += fOz;

            f[iH1].x += fHx;
            f[iH1].y += fHy;
            f[iH1].z += fHz;

            f[iH2].x += fHx;
            f[iH2].y += fHy;
            f[iH2].z += fHz;

            if (EVFLAG && vflag) {
              xH1 = x[iH1];
              xH2 = x[iH2];

              fdx = delx*fvirial;
              fdy = dely*fvirial;
              fdz = delz*fvirial;

              fOx = fdx*(1 - alpha);
              fOy = fdy*(1 - alpha);
              fOz = fdz*(1 - alpha);

              fHx = 0.5 * alpha * fdx;
              fHy = 0.5 * alpha * fdy;
              fHz = 0.5 * alpha * fdz;

              v[0] = x[i].x*fOx + xH1.x*fHx + xH2.x*fHx;
              v[1] = x[i].y*fOy + xH1.y*fHy + xH2.y*fHy;
              v[2] = x[i].z*fOz + xH1.z*fHz + xH2.z*fHz;
              v[3] = x[i].x*fOy + xH1.x*fHy + xH2.x*fHy;
              v[4] = x[i].x*fOz + xH1.x*fHz + xH2.x*fHz;
              v[5] = x[i].y*fOz + xH1.y*fHz + xH2.y*fHz;
              vlist[n++] = i;
              vlist[n++] = iH1;
              vlist[n++] = iH2;
            }
          }

          if (jtype != typeO) {
            f[j].x -= delx * cforce;
            f[j].y -= dely * cforce;
            f[j].z -= delz * cforce;

            if (EVFLAG && vflag) {
              v[0] -= x[j].x * delx * fvirial;
              v[1] -= x[j].y * dely * fvirial;
              v[2] -= x[j].z * delz * fvirial;
              v[3] -= x[j].x * dely * fvirial;
              v[4] -= x[j].x * delz * fvirial;
              v[5] -= x[j].y * delz * fvirial;
              vlist[n++] = j;
            }

          } else {
            if (EVFLAG && vflag) key += 2;

            fdx = -delx*cforce;
            fdy = -dely*cforce;
            fdz = -delz*cforce;

            fOx = fdx*(1 - alpha);
            fOy = fdy*(1 - alpha);
            fOz = fdz*(1 - alpha);

            fHx = 0.5 * alpha * fdx;
            fHy = 0.5 * alpha * fdy;
            fHz = 0.5 * alpha * fdz;

            f[j].x += fOx;
            f[j].y += fOy;
            f[j].z += fOz;

            f[jH1].x += fHx;
            f[jH1].y += fHy;
            f[jH1].z += fHz;

            f[jH2].x += fHx;
            f[jH2].y += fHy;
            f[jH2].z += fHz;

            if (EVFLAG && vflag) {
              xH1 = x[jH1];
              xH2 = x[jH2];

              fdx = -delx*fvirial;
              fdy = -dely*fvirial;
              fdz = -delz*fvirial;

              fOx = fdx*(1 - alpha);
              fOy = fdy*(1 - alpha);
              fOz = fdz*(1 - alpha);

              fHx = 0.5 * alpha * fdx;
              fHy = 0.5 * alpha * fdy;
              fHz = 0.5 * alpha * fdz;

              v[0] += x[j].x*fOx + xH1.x*fHx + xH2.x*fHx;
              v[1] += x[j].y*fOy + xH1.y*fHy + xH2.y*fHy;
              v[2] += x[j].z*fOz + xH1.z*fHz + xH2.z*fHz;
              v[3] += x[j].x*fOy + xH1.x*fHy + xH2.x*fHy;
              v[4] += x[j].x*fOz + xH1.x*fHz + xH2.x*fHz;
              v[5] += x[j].y*fOz + xH1.y*fHz + xH2.y*fHz;
              vlist[n++] = j;
              vlist[n++] = jH1;
              vlist[n++] = jH2;
            }
          }

          if (EVFLAG) ev_tally_list_thr(this,key,vlist,v,ecoul,alpha,thr);
        }
      }
    }
    f[i].x += fxtmp;
    f[i].y += fytmp;
    f[i].z += fztmp;
  }
}


/* ----------------------------------------------------------------------
  compute position xM of fictitious charge site for O atom and 2 H atoms
  return it as xM
------------------------------------------------------------------------- */

void PairLJLongTIP4PLongOMP::compute_newsite_thr(const dbl3_t &xO,
                                                const dbl3_t &xH1,
                                                const dbl3_t &xH2,
                                                dbl3_t &xM) const
{
  double delx1 = xH1.x - xO.x;
  double dely1 = xH1.y - xO.y;
  double delz1 = xH1.z - xO.z;

  double delx2 = xH2.x - xO.x;
  double dely2 = xH2.y - xO.y;
  double delz2 = xH2.z - xO.z;

  const double prefac = alpha * 0.5;
  xM.x = xO.x + prefac * (delx1 + delx2);
  xM.y = xO.y + prefac * (dely1 + dely2);
  xM.z = xO.z + prefac * (delz1 + delz2);
}

/* ---------------------------------------------------------------------- */

double PairLJLongTIP4PLongOMP::memory_usage()
{
  double bytes = memory_usage_thr();
  bytes += PairLJLongTIP4PLong::memory_usage();
  return bytes;
}

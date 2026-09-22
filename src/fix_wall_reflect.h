/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(wall/reflect,FixWallReflect);
// clang-format on
#else

#ifndef LMP_FIX_WALL_REFLECT_H
#define LMP_FIX_WALL_REFLECT_H

#include "fix.h"

namespace LAMMPS_NS {

class FixWallReflect : public Fix {
 public:
  enum { XLO = 0, XHI = 1, YLO = 2, YHI = 3, ZLO = 4, ZHI = 5 };
  enum { NONE = 0, EDGE, CONSTANT, VARIABLE };
  // imgparms indices for CYLINDER graphics objects (2D walls)
  enum {
    IMG_TYPE = 0,
    IMG_X1 = 1,
    IMG_Y1 = 2,
    IMG_Z1 = 3,
    IMG_X2 = 4,
    IMG_Y2 = 5,
    IMG_Z2 = 6,
    IMG_DIAM = 7
  };

  FixWallReflect(class LAMMPS *, int, char **);
  ~FixWallReflect() override;
  int setmask() override;
  void init() override;
  void post_integrate() override;

  int image(int *&, double **&) override;

 protected:
  int nwall;
  int wallwhich[6], wallstyle[6];
  double coord0[6];
  char *varstr[6];
  int varindex[6];
  int varflag;
  double xscale, yscale, zscale;
  int *imgobjs;
  double **imgparms;

  virtual void wall_particle(int m, int which, double coord);
  void wall_update_objs(int m, int which, double coord);
};

}    // namespace LAMMPS_NS

#endif
#endif

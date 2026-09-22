// clang-format off
 /* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "fix_wall_reflect.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "dump_image.h"
#include "error.h"
#include "input.h"
#include "lattice.h"
#include "memory.h"
#include "modify.h"
#include "update.h"
#include "variable.h"

#include <cstring>

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixWallReflect::FixWallReflect(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg), nwall(0), varflag(0), imgobjs(nullptr), imgparms(nullptr)
{
  if (narg < 4) utils::missing_cmd_args(FLERR, "fix wall/reflect", error);

  // let child class process all args

  if (strcmp(arg[2],"wall/reflect/stochastic") == 0) return;

  dynamic_group_allow = 1;

  // parse args

  nwall = 0;
  int scaleflag = 1;

  int iarg = 3;
  while (iarg < narg) {
    if ((strcmp(arg[iarg],"xlo") == 0) || (strcmp(arg[iarg],"xhi") == 0) ||
        (strcmp(arg[iarg],"ylo") == 0) || (strcmp(arg[iarg],"yhi") == 0) ||
        (strcmp(arg[iarg],"zlo") == 0) || (strcmp(arg[iarg],"zhi") == 0)) {
      if (iarg+2 > narg) error->all(FLERR, "Illegal fix wall/reflect {} command: missing argument(s)", arg[iarg]);

      int newwall;
      if (strcmp(arg[iarg],"xlo") == 0) newwall = XLO;
      else if (strcmp(arg[iarg],"xhi") == 0) newwall = XHI;
      else if (strcmp(arg[iarg],"ylo") == 0) newwall = YLO;
      else if (strcmp(arg[iarg],"yhi") == 0) newwall = YHI;
      else if (strcmp(arg[iarg],"zlo") == 0) newwall = ZLO;
      else if (strcmp(arg[iarg],"zhi") == 0) newwall = ZHI;

      for (int m = 0; (m < nwall) && (m < 6); m++)
        if (newwall == wallwhich[m])
          error->all(FLERR,"Wall defined twice in fix wall/reflect command");

      wallwhich[nwall] = newwall;
      if (strcmp(arg[iarg+1],"EDGE") == 0) {
        wallstyle[nwall] = EDGE;
        int dim = wallwhich[nwall] / 2;
        int side = wallwhich[nwall] % 2;
        if (side == 0) coord0[nwall] = domain->boxlo[dim];
        else coord0[nwall] = domain->boxhi[dim];
      } else if (utils::strmatch(arg[iarg+1],"^v_")) {
        wallstyle[nwall] = VARIABLE;
        varstr[nwall] = utils::strdup(arg[iarg+1]+2);
      } else {
        wallstyle[nwall] = CONSTANT;
        coord0[nwall] = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      }

      nwall++;
      iarg += 2;

    } else if (strcmp(arg[iarg],"units") == 0) {
      if (iarg+2 > narg) utils::missing_cmd_args(FLERR, "fix wall/reflect units", error);
      if (strcmp(arg[iarg+1],"box") == 0) scaleflag = 0;
      else if (strcmp(arg[iarg+1],"lattice") == 0) scaleflag = 1;
      else error->all(FLERR,"Unknown fix wall/reflect units argument: {}", arg[iarg+1]);
      iarg += 2;

    } else error->all(FLERR,"Unknown fix wall/reflect keyword: {}", arg[iarg]);
  }

  // error check

  if (nwall == 0) utils::missing_cmd_args(FLERR, "fix wall/reflect", error);

  for (int m = 0; m < nwall; m++) {
    if ((wallwhich[m] == XLO || wallwhich[m] == XHI) && domain->xperiodic)
      error->all(FLERR,"Cannot use fix wall/reflect in periodic dimension x");
    if ((wallwhich[m] == YLO || wallwhich[m] == YHI) && domain->yperiodic)
      error->all(FLERR,"Cannot use fix wall/reflect in periodic dimension y");
    if ((wallwhich[m] == ZLO || wallwhich[m] == ZHI) && domain->zperiodic)
      error->all(FLERR,"Cannot use fix wall/reflect in periodic dimension z");
  }

  for (int m = 0; m < nwall; m++)
    if ((wallwhich[m] == ZLO || wallwhich[m] == ZHI) && domain->dimension == 2)
      error->all(FLERR,
                 "Cannot use fix wall/reflect zlo/zhi for a 2d simulation");

  // scale factors for CONSTANT and VARIABLE walls

  int flag = 0;
  for (int m = 0; m < nwall; m++)
    if (wallstyle[m] != EDGE) flag = 1;

  if (flag) {
    if (scaleflag) {
      xscale = domain->lattice->xlattice;
      yscale = domain->lattice->ylattice;
      zscale = domain->lattice->zlattice;
    }
    else xscale = yscale = zscale = 1.0;

    for (int m = 0; m < nwall; m++) {
      if (wallstyle[m] != CONSTANT) continue;
      if (wallwhich[m] < YLO) coord0[m] *= xscale;
      else if (wallwhich[m] < ZLO) coord0[m] *= yscale;
      else coord0[m] *= zscale;
    }
  }

  // set varflag if any wall positions are variable

  varflag = 0;
  for (int m = 0; m < nwall; m++)
    if (wallstyle[m] == VARIABLE) varflag = 1;

  // for rendering walls with dump image.
  if (domain->dimension == 2) {
    // one cylinder object per wall to draw in 2d
    memory->create(imgobjs, nwall, "fix_indent:imgobjs");
    memory->create(imgparms, nwall, 8, "fix_indent:imgparms");
    for (int m = 0; m < nwall; ++m) {
      imgobjs[m] = DumpImage::CYLINDER;
      imgparms[m][0] = 1;    // use color of first atom type by default
    }
  } else {
    // two triangle objects per wall to draw in 3d
    memory->create(imgobjs, 2 * nwall, "fix_indent:imgobjs");
    memory->create(imgparms, 2 * nwall, 10, "fix_indent:imgparms");
    for (int m = 0; m < nwall; ++m) {
      imgobjs[2 * m] = DumpImage::TRIANGLE;
      imgobjs[2 * m + 1] = DumpImage::TRIANGLE;
      imgparms[2 * m][0] = 1;        // use color of first atom type by default
      imgparms[2 * m + 1][0] = 1;    // use color of first atom type by default
    }
  }
}

/* ---------------------------------------------------------------------- */

FixWallReflect::~FixWallReflect()
{
  if (copymode) return;

  for (int m = 0; m < nwall; m++)
    if (wallstyle[m] == VARIABLE) delete [] varstr[m];

  memory->destroy(imgobjs);
  memory->destroy(imgparms);
}

/* ---------------------------------------------------------------------- */

int FixWallReflect::setmask()
{
  int mask = 0;
  mask |= POST_INTEGRATE;
  mask |= POST_INTEGRATE_RESPA;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixWallReflect::init()
{
  for (int m = 0; m < nwall; m++) {
    if (wallstyle[m] != VARIABLE) continue;
    varindex[m] = input->variable->find(varstr[m]);
    if (varindex[m] < 0)
      error->all(FLERR,"Variable {} for fix wall/reflect does not exist", varstr[m]);
    if (!input->variable->equalstyle(varindex[m]))
      error->all(FLERR,"Variable {} for fix wall/reflect is invalid style", varstr[m]);
  }

  int nrigid = 0;
  for (const auto &ifix : modify->get_fix_list())
    if (ifix->rigid_flag) nrigid++;

  if (nrigid && (comm->me == 0))
    error->warning(FLERR,"Should not use reflecting walls with rigid bodies");
}

/* ---------------------------------------------------------------------- */

void FixWallReflect::post_integrate()
{
  double coord;

  // coord = current position of wall
  // evaluate variable if necessary, wrap with clear/add

  if (varflag) modify->clearstep_compute();

  for (int m = 0; m < nwall; m++) {
    if (wallstyle[m] == VARIABLE) {
      coord = input->variable->compute_equal(varindex[m]);
      if (wallwhich[m] < YLO) coord *= xscale;
      else if (wallwhich[m] < ZLO) coord *= yscale;
      else coord *= zscale;
    } else coord = coord0[m];

    wall_particle(m,wallwhich[m],coord);

    // record wall graphics objects for dump image
    wall_update_objs(m,wallwhich[m],coord);
  }

  if (varflag) modify->addstep_compute(update->ntimestep + 1);
}

/* ----------------------------------------------------------------------
   this method may be overwritten by a child class
------------------------------------------------------------------------- */

void FixWallReflect::wall_particle(int /* m */, int which, double coord)
{
  int i,dim,side;

  double **x = atom->x;
  double **v = atom->v;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  dim = which / 2;
  side = which % 2;

  for (i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      if (side == 0) {
        if (x[i][dim] < coord) {
          x[i][dim] = coord + (coord - x[i][dim]);
          v[i][dim] = -v[i][dim];
        }
      } else {
        if (x[i][dim] > coord) {
          x[i][dim] = coord - (x[i][dim] - coord);
          v[i][dim] = -v[i][dim];
        }
      }
    }
  }
}

/* ----------------------------------------------------------------------
   update wall graphics object info for dump image
------------------------------------------------------------------------- */

void FixWallReflect::wall_update_objs(int m, int which, double coord)
{
  if (domain->dimension == 2) {
    // one cylinder for 2d. diameter is zero and can be set with fparam2
    switch (which) {
    case XLO:    // fallthrough
    case XHI:
      imgparms[m][IMG_X1] = coord;
      imgparms[m][IMG_Y1] = domain->boxlo[1];
      imgparms[m][IMG_Z1] = 0.0;
      imgparms[m][IMG_X2] = coord;
      imgparms[m][IMG_Y2] = domain->boxhi[1];
      imgparms[m][IMG_Z2] = 0.0;
      imgparms[m][IMG_DIAM] = 0.0;
      break;
    case YLO:    // fallthrough
    case YHI:
      imgparms[m][IMG_X1] = domain->boxlo[0];
      imgparms[m][IMG_Y1] = coord;
      imgparms[m][IMG_Z1] = 0.0;
      imgparms[m][IMG_X2] = domain->boxhi[0];
      imgparms[m][IMG_Y2] = coord;
      imgparms[m][IMG_Z2] = 0.0;
      imgparms[m][IMG_DIAM] = 0.0;
      break;
    case ZLO:     // fallthrough
    case ZHI:;    // no wall in z-direction allowed for 2d systems
      break;
    }
  } else {
    // two triangles for 3d
    switch (which) {
    case XLO:    // fallthrough
    case XHI:
      imgparms[2 * m][1] = coord;
      imgparms[2 * m][2] = domain->boxlo[1];
      imgparms[2 * m][3] = domain->boxlo[2];
      imgparms[2 * m][4] = coord;
      imgparms[2 * m][5] = domain->boxhi[1];
      imgparms[2 * m][6] = domain->boxlo[2];
      imgparms[2 * m][7] = coord;
      imgparms[2 * m][8] = domain->boxlo[1];
      imgparms[2 * m][9] = domain->boxhi[2];
      imgparms[2 * m + 1][1] = coord;
      imgparms[2 * m + 1][2] = domain->boxhi[1];
      imgparms[2 * m + 1][3] = domain->boxhi[2];
      imgparms[2 * m + 1][4] = coord;
      imgparms[2 * m + 1][5] = domain->boxlo[1];
      imgparms[2 * m + 1][6] = domain->boxhi[2];
      imgparms[2 * m + 1][7] = coord;
      imgparms[2 * m + 1][8] = domain->boxhi[1];
      imgparms[2 * m + 1][9] = domain->boxlo[2];
      break;
    case YLO:    // fallthrough
    case YHI:
      imgparms[2 * m][1] = domain->boxlo[0];
      imgparms[2 * m][2] = coord;
      imgparms[2 * m][3] = domain->boxlo[2];
      imgparms[2 * m][4] = domain->boxhi[0];
      imgparms[2 * m][5] = coord;
      imgparms[2 * m][6] = domain->boxlo[2];
      imgparms[2 * m][7] = domain->boxlo[0];
      imgparms[2 * m][8] = coord;
      imgparms[2 * m][9] = domain->boxhi[2];
      imgparms[2 * m + 1][1] = domain->boxhi[0];
      imgparms[2 * m + 1][2] = coord;
      imgparms[2 * m + 1][3] = domain->boxhi[2];
      imgparms[2 * m + 1][4] = domain->boxlo[0];
      imgparms[2 * m + 1][5] = coord;
      imgparms[2 * m + 1][6] = domain->boxhi[2];
      imgparms[2 * m + 1][7] = domain->boxhi[0];
      imgparms[2 * m + 1][8] = coord;
      imgparms[2 * m + 1][9] = domain->boxlo[2];
      break;
    case ZLO:    // fallthrough
    case ZHI:
      imgparms[2 * m][1] = domain->boxlo[0];
      imgparms[2 * m][2] = domain->boxlo[1];
      imgparms[2 * m][3] = coord;
      imgparms[2 * m][4] = domain->boxhi[0];
      imgparms[2 * m][5] = domain->boxlo[1];
      imgparms[2 * m][6] = coord;
      imgparms[2 * m][7] = domain->boxlo[0];
      imgparms[2 * m][8] = domain->boxhi[1];
      imgparms[2 * m][9] = coord;
      imgparms[2 * m + 1][1] = domain->boxhi[0];
      imgparms[2 * m + 1][2] = domain->boxhi[1];
      imgparms[2 * m + 1][3] = coord;
      imgparms[2 * m + 1][4] = domain->boxlo[0];
      imgparms[2 * m + 1][5] = domain->boxhi[1];
      imgparms[2 * m + 1][6] = coord;
      imgparms[2 * m + 1][7] = domain->boxhi[0];
      imgparms[2 * m + 1][8] = domain->boxlo[1];
      imgparms[2 * m + 1][9] = coord;
      break;
    }
  }
}

/* ----------------------------------------------------------------------
   provide graphics information to dump image to render wall as plane
   data has been copied to dedicated storage during fix indent execution
------------------------------------------------------------------------- */

int FixWallReflect::image(int *&objs, double **&parms)
{
  objs = imgobjs;
  parms = imgparms;
  if (domain->dimension == 2) {
    return nwall;
  } else {
    return 2 * nwall;
  }
}

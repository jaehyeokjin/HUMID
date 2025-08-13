/* ----------------------------------------------------------------------
  Copyright @ Voth Group
  Written by Yuxing Peng
------------------------------------------------------------------------- */

#ifdef COMMAND_CLASS

CommandStyle(evb_checkout,EVB_Checkout)

#else

#ifndef LMP_EVB_CHECKOUT_H
#define LMP_EVB_CHECKOUT_H

#include "pointers.h"

namespace LAMMPS_NS {

class EVB_Checkout : protected Pointers 
{
 public:
  
  EVB_Checkout(class LAMMPS *);
  
  void command(int, char **);
  
  void init(char*);
  void run();
  void exit();

  void write2bin();
  void write2txt();
  
  bool allatom;
  bool write_types;
  bool write_types_mol;
};

}

#endif
#endif

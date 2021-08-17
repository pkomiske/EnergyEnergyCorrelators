#include <cstdlib>

#include "NPZEventProducer.hh"

#ifndef EEC_HIST_FORMATTED_OUTPUT
#define EEC_HIST_FORMATTED_OUTPUT
#endif

#include "EEC.hh"

// without these lines, `eec::` should be prefixed with `fastjet::contrib`
using namespace fastjet;
using namespace fastjet::contrib;


template<class T>
void run_eec_comp(T & eec, EventProducer * evp) {

  std::cout << eec.description() << std::endl;

  // loop over events
  evp->reset();
  while (evp->next())
    eec.compute(evp->particles());

  // uncomment to output axis and histogram
  //eec.output(std::cout);

  // check total
  std::pair<std::vector<double>, std::vector<double>> hist_errs(eec.get_hist_errs());
  double total(0);
  for (double h : hist_errs.first)
    total += h;
  std::cout << "Hist total: "
            << std::setprecision(16) << total << std::endl;
}

EventProducer * load_events(int argc, char** argv) {

  // get number of events from command line
  long num_events(1000);
  EventType evtype(All);
  if (argc >= 2)
    num_events = atol(argv[1]);
  if (argc >= 3)
    evtype = atoi(argv[2]) == 1 ? Quark : Gluon;

  // get energyflow samples
  const char * home(std::getenv("HOME"));
  if (home == NULL)
    throw std::invalid_argument("Error: cannot get HOME environment variable");

  // form path
  std::string filepath(home);
  filepath += "/.energyflow/datasets/QG_jets.npz";
  std::cout << "Filepath: " << filepath << '\n';

  // open file
  NPZEventProducer * npz(nullptr);
  try {
    npz = new NPZEventProducer(filepath, num_events, evtype);
  }
  catch (std::exception & e) {
    std::cerr << "Error: cannot open file " << filepath << ", try running "
              << "`python3 -c \"import energyflow as ef; ef.qg_jets.load()\"`\n";
    return nullptr;
  }

  return npz;
}

int main(int argc, char** argv) {

  // load events
  EventProducer * evp(load_events(argc, argv));
  if (evp == nullptr)
    return 1;

  // specify EECs
  eec::EECLongestSideLog eec_longestside(75, 1e-5, 1, 6, true);
  eec::EECTriangleOPELogLogId eec_triangleope(50, 1e-4, 1,
                                              50, 1e-4, 1,
                                              25, 0, eec::PI/2,
                                              true);
  run_eec_comp(eec_longestside, evp);
  run_eec_comp(eec_triangleope, evp);

  return 0;
}

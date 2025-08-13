# Python

import sys
import numpy as np
import pandas as pd

from rmdtoolkit2 import DumpParser
from rmdtoolkit2 import RDFAnalysis


def pbc_dist(d, box_len):
    return abs(d) if abs(d) <= box_len/2 else box_len-abs(d)


v_pbc_dist = np.vectorize(pbc_dist)


TrajPath = '255wat1hyd_cg.lammpstrj'
DumpFile = open(TrajPath, 'r')
DParser = DumpParser(DumpFile, tell_time=True)
iframe = 0

OH_OW_RDF = RDFAnalysis(hist_range=(0., 20.), hist_bins=400)
OW_OW_RDF = RDFAnalysis(hist_range=(0., 20.), hist_bins=400)

while True:
    iframe += 1
    print(iframe)
    sys.stdout.flush()
    status = DParser.read_frame()
    if status == 1:
        break
    DParser.parse_frame()
    DJson = DParser.frame_json
    DTime = DParser.frame_time
    if iframe == 1:
        Bounds = DJson['bounds']
        BoxLen = np.average(Bounds[:, 1] - Bounds[:, 0])
        OH_OW_RDF.bounds = Bounds
        OW_OW_RDF.bounds = Bounds

    AtomInfo = DJson["atom_info"]
    AtomTypes = AtomInfo[:, 1]
    AtomMol = AtomInfo[:, 2]
    Coords = np.array(AtomInfo[:, 2:5])
    OW_Coords = Coords[AtomTypes == 1]
    OH_Coords = Coords[AtomTypes == 2]

    OH_OW_RDF.rdf(OH_Coords, OW_Coords)
    OW_OW_RDF.rdf(OW_Coords, OW_Coords)

    if iframe % 5000 == 0:
        bin_edges, oh_ow_hist = OH_OW_RDF.normalize_rdf(update=False)
        _, ow_ow_hist = OW_OW_RDF.normalize_rdf(update=False)
        data_out = np.vstack([bin_edges, oh_ow_hist, ow_ow_hist]).T
        df_out = pd.DataFrame(data=data_out, columns=['bins', '2_1', '1_1'])
        df_out.to_csv('rdf_cg.csv', index=False)

    # if iframe % 5000 == 0:
    #     OH_OW_RDF.normalize_rdf()
    #     OW_OW_RDF.normalize_rdf()
    #     data_out = np.vstack([OH_OW_RDF.bin_edges, OH_OW_RDF.hist, OW_OW_RDF.hist]).T
    #     df_out = pd.DataFrame(data=data_out, columns=['bins', '2_1', '1_1'])
    #     df_out.to_csv('rdf_cg.csv', index=False)



# Python

import sys
import numpy as np
import pandas as pd

from rmdtoolkit2 import DumpParser
from rmdtoolkit2 import RDFAnalysis


def pbc_dist(d, box_len):
    return abs(d) if abs(d) <= box_len/2 else box_len-abs(d)


v_pbc_dist = np.vectorize(pbc_dist)


TrajPath = '255wat1hyd_fg.lammpstrj'
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
    Coords = np.array(AtomInfo[:, 4:7])
    OCoords = Coords[((AtomTypes == 1).astype(int) + (AtomTypes == 3).astype(int)).astype(bool)]
    HCoords = Coords[((AtomTypes == 2).astype(int) + (AtomTypes == 4).astype(int)).astype(bool)]
    NumO = len(OCoords)
    NumH = len(HCoords)
    Ox = OCoords[:, 0]
    Oy = OCoords[:, 1]
    Oz = OCoords[:, 2]
    Hx = HCoords[:, 0]
    Hy = HCoords[:, 1]
    Hz = HCoords[:, 2]

    dOHx = v_pbc_dist(Ox[..., np.newaxis] - Hx[np.newaxis, ...], 19.73)
    dOHy = v_pbc_dist(Oy[..., np.newaxis] - Hy[np.newaxis, ...], 19.73)
    dOHz = v_pbc_dist(Oz[..., np.newaxis] - Hz[np.newaxis, ...], 19.73)
    dOH = np.array([dOHx, dOHy, dOHz])
    dOH = np.linalg.norm(dOH, axis=0)
    VisitedH = set()
    ArgH2O = list()
    for dOHLine in dOH:
        H1, H2, *_ = np.argpartition(dOHLine, 1)
        VisitedH.update([H1, H2])
        ArgH2O.append(np.array([H1, H2]))
    UnvisitedH = set(np.arange(NumH)) - VisitedH
    for iH in UnvisitedH:
        iO = np.argmin(dOH.T[iH])
        ArgH2O[iO] = np.append(ArgH2O[iO], [iH])
    ArgH2O = np.array(ArgH2O)
    HydO = np.where(np.array(list(map(len, ArgH2O))) == 3)[0][0]

    OH_Coords = [OCoords[HydO]]
    OW_Coords = np.delete(OCoords, HydO, axis=0)

    OH_OW_RDF.rdf(OH_Coords, OW_Coords)
    OW_OW_RDF.rdf(OW_Coords, OW_Coords)

    if iframe % 5000 == 0:
        bin_edges, oh_ow_hist = OH_OW_RDF.normalize_rdf(update=False)
        _, ow_ow_hist = OW_OW_RDF.normalize_rdf(update=False)
        data_out = np.vstack([bin_edges, oh_ow_hist, ow_ow_hist]).T
        df_out = pd.DataFrame(data=data_out, columns=['bins', '2_1', '1_1'])
        df_out.to_csv('rdf_fg.csv', index=False)

    # if iframe % 5000 == 0:
    #     OH_OW_RDF.normalize_rdf()
    #     OW_OW_RDF.normalize_rdf()
    #     data_out = np.vstack([OH_OW_RDF.bin_edges, OH_OW_RDF.hist, OW_OW_RDF.hist]).T
    #     df_out = pd.DataFrame(data=data_out, columns=['bins', '2_1', '1_1'])
    #     df_out.to_csv('rdf_cg.csv', index=False)



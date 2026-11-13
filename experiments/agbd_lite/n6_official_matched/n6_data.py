"""N1-compatible inputs for all official Train rows and original Validation rows."""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
N1_DIR=HERE.parent/'n1_niconet_mrf'
sys.path.insert(0,str(N1_DIR))
import run_n1 as N1

N_TRAIN=371849
N_VAL=103550


class OfficialTrainData(N1.Data):
    def __init__(self):
        super().__init__()
        self.all_rows=np.arange(N_TRAIN,dtype=np.int64)
        self.q_official=np.quantile(self.y,[.5,.9,.95,.99])
        assert len(self.all_rows)==N_TRAIN


class OfficialValData:
    """Construct Validation S2 from its H5 with the locked Train-only P95 stats.

    This class is instantiated only after M0/M1 full training has finished.
    """
    def __init__(self,train:OfficialTrainData):
        self.stats=train.stats
        self.eye=train.eye
        self.lut=train.lut
        self.h5=h5py.File(ROOT/'data/agbd_lite/AGBD-Lite-val.h5','r')
        assert self.h5['S2_bands'].shape==(N_VAL,25,25,12)
        assert self.h5['S2_bands'].dtype==np.uint16
        self.s2=self.h5['S2_bands']
        self.boa=self.h5['Sentinel_metadata/S2_boa_offset']
        self.dem=np.memmap(ROOT/'experiments/agbd_lite/p19_2_dem_pilot/run_aw3d30/val_dem.i16',
                           mode='r',dtype=np.int16,shape=(N_VAL,25,25))
        self.lc=np.memmap(ROOT/'experiments/agbd_lite/p48_lc_multimodal_pilot/val_lc.u8',
                          mode='r',dtype=np.uint8,shape=(N_VAL,25,25,2))
        self.meta=np.memmap(ROOT/'experiments/agbd_lite/p54_final_feature_alignment/val_features_raw.f32',
                            mode='r',dtype=np.float32,shape=(N_VAL,8))
        self.y=np.memmap(ROOT/'data/agbd_lite/cache_s2only/val_y.f32',
                         mode='r',dtype=np.float32,shape=(N_VAL,))
        check=np.array([0,117,103549],dtype=np.int64)
        assert np.array_equal(self.h5['GEDI/agbd'][check].astype(np.float32),
                              np.asarray(self.y[check]))
        self.p1=np.asarray(self.stats['s2_p1'],np.float32)
        self.p99=np.asarray(self.stats['s2_p99'],np.float32)

    def batch(self,start:int,stop:int):
        assert 0<=start<stop<=N_VAL
        dn=self.s2[start:stop]
        boa=self.boa[start:stop].astype(np.float32)
        sr=(dn.astype(np.float32)-boa[:,None,None,None]*1000.0)/10000.0
        sr[dn==0]=0
        np.maximum(sr,0,out=sr)
        sr=np.clip((sr-self.p1)/(self.p99-self.p1),0,1).astype(np.float32)
        sr[dn==0]=0
        s2=np.transpose(sr,(0,3,1,2))
        dsm=((np.asarray(self.dem[start:stop],np.float32)-self.stats['dem_pixel_mean'])
             /self.stats['dem_pixel_std'])[:,None]
        raw=np.asarray(self.lc[start:stop]);cls,prob=raw[...,0],raw[...,1]
        valid_cls,valid_prob=cls!=255,prob!=255
        onehot=self.eye[self.lut[cls]].transpose(0,3,1,2)
        lc=np.concatenate((onehot,valid_cls[:,None].astype(np.float32),
             (np.where(valid_prob,prob,0).astype(np.float32)/100)[:,None],
             valid_prob[:,None].astype(np.float32)),1)
        meta=np.asarray(self.meta[start:stop],np.float32).copy()
        meta[:,2:]=((meta[:,2:]-self.stats['metadata_mean_columns_2_7'])
                    /self.stats['metadata_std_columns_2_7'])
        y=np.asarray(self.y[start:stop],np.float32)
        return tuple(torch.from_numpy(np.ascontiguousarray(v)).to('cuda')
                     for v in (s2,dsm,lc,meta,y))

    def close(self):
        self.h5.close()

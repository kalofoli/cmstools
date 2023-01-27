'''
Created on Jan 17, 2023

@author: janis
'''

import itertools
import typing

import numpy as np
import pandas as pd

class CMSError(Exception): pass
class ParseError(Exception):
    def __init__(self, what):
        super().__init__(what + ' This could be due to a change in the CMS website or another parsing error. Please contact developers.')
class LoginError(Exception): pass

class GradingError(Exception): pass
class GradingWarning(Warning): pass


def memoised_property(fn):
    name = '_'+fn.__name__
    @property
    def decorated(self):
        if not hasattr(self, name) or getattr(self, name) is None:
            setattr(self, name, fn(self))
        return getattr(self, name)
    return decorated





class GatheredGradeSelector:
    def __init__(self, df):
        self._df = df
        
    def idl(self, kind, source, partner):
        dct_fn = {
            'nz':lambda x:x!=0,
            'nn':lambda x:~x.isna(),
            'ne':lambda x:x!='',
            'ok':lambda x:~x.isna() & (x!='')
        }
        if kind not in dct_fn:
            raise AttributeError()
        fn = dct_fn[kind]
        
        fn_key = lambda s,p: self._df[f'Points{"Partner" if p else ""}{s.upper()}']
        if source not in ('any','all','cms','gs'): raise AttributeError()
        if source in ('any','all'):
            d = np.c_[[fn(x) for x in (fn_key('cms',partner),fn_key('cms',partner))]]
            r = getattr(np, source)(d, axis=0)
        else:
            r = fn(fn_key(source, partner))
        return r
    @property
    def df(self): return self._df.copy()
    
    def __getattr__(self, attr):
        try:
            parts = attr.split('_')
            if parts[0] == 'df':
                idl = getattr(self, '_'.join(['idl']+parts[1:]))
                return self.df[idl]
            elif parts[0] == 'gs':
                df = getattr(self, '_'.join(['df']+parts[1:]))
                return self.__class__(df)
            elif parts[0]!='idl':
                raise AttributeError()
            parts = parts[1:]
            if len(parts) == 3:
                if parts[2]!='p': raise AttributeError()
                partner = True
            else:
                partner = False
            kind,source = parts[:2]
            return self.idl(kind, source, partner)
        except AttributeError as e:
            raise AttributeError(f'{type(self).__name__} object has no attribute {attr}')
    
    def __hasattr__(self, attr):
        try:
            parts = attr.split('_')
            if parts[0] in ['df','gs']:
                idl = getattr(self, '_'.join(['idl']+parts[1:]))
                return True
            elif parts[0]!='idl':
                return False
            parts = parts[1:]
            if len(parts) == 3:
                if parts[2]!='p': raise AttributeError()
                partner = True
            else:
                partner = False
            kind,source = parts[:2]
            return self.idl(kind, source, partner)
        except AttributeError as e:
            return False
    
    @property
    def idl_has_pair(self): return ~pd.isna(self._df.PairMN)
    @property
    def idl_err_nosub(self): return pd.isna(self._df.SubId) & self.idl_ok_any
    @property
    def idl_miss_cms(self): return ~self.idl_nn_cms & (self.idl_ok_gs | self.idl_ok_gs_p | self.idl_ok_cms_p) 
    @property
    def idl_miss_gs(self): return ~self.idl_nn_gs & (self.idl_ok_cms | self.idl_ok_cms_p | self.idl_ok_gs_p) 
    @property
    def idl_err_collide_src(self): return (self.df.PointsCMS != self.df.PointsGS) & self.idl_ok_all
    @property
    def idl_err_collide_teamcms(self): return (self._df.PointsCMS != self._df.PointsPartnerCMS) & (self._df.IsSub==False) & (self.idl_ok_cms | self.idl_ok_cms_p)
    @property
    def idl_err_collide_teamgs(self): return (self._df.PointsGS != self._df.PointsPartnerGS) & self.idl_ok_gs & self.idl_ok_gs_p
    @property
    def idl_err_missing_teamcms(self): return ~self.idl_nn_cms & self.idl_ok_cms_p & (self._df.IsSub==False)
    @property
    def idl_err_missing_teamgs(self): return ~self.idl_nn_gs & self.idl_ok_gs_p & (self._df.IsSub==False)
    @property
    def idl_err_nofbk(self): return self._df.FeedbackSol != 'yes'
    
    def __dir__(self):
        keys = [f'idl_{k}_{s}{"_p" if p else ""}' for k,s,p in itertools.product(['nz','ok','nn','ne'],['cms','gs','all','any'],[True,False])] + list(self.__class__.__dict__.keys())
        df_keys = ['df'+key[3:] for key in keys if key[:3] == 'idl']
        gs_keys = ['gs'+key[3:] for key in keys if key[:3] == 'idl']
        return keys + df_keys + gs_keys
    
    def assignment(self, assignment):
        if not isinstance(assignment, typing.Sequence):
            assignment = [assignment]
        df = self.df
        df = df[df.Assignment.isin(assignment)]
        return self.__class__(df)
    def problem(self, problem):
        if not isinstance(problem, typing.Sequence):
            problem = [problem]
        problem = [str(p) for p in problem]
        df = self.df
        df = df[df.Problem.isin(problem)]
        return self.__class__(df)
        
    def __repr__(self):
        return f'<{type(self).__name__} with {self.idl_nn_cms.sum()}(CMS),{self.idl_nn_gs.sum()}(GS)/{self._df.shape[0]}(total) grades for assignment(s) {",".join(map(str,self._df.Assignment.unique()))}.>'
    
    def reset(self, what):
        if isinstance(what, (np.ndarray, pd.Series)):
            if what.dtype == bool:
                df = self.df[what]
        elif isinstance(what, pd.DataFrame):
            df = what
        else:
            raise TypeError(f'Cannot reset {type(self).__name__} from value of type {type(what).__name__}.')
        self._df = df
        return self
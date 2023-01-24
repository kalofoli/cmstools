'''
Created on Jan 19, 2023

@author: janis
'''



import typing
import logging
import itertools
from types import SimpleNamespace

import numpy as np
import pandas as pd

import gspread

from cmstools.parsers import TeamParser, SubmissionItemParser, SubmissionListParser, TestingParser, StudentParser
from cmstools.cms import CMSSession
from cmstools.common import memoised_property
from cmstools.reports import HTMLReport
from cmstools.sheets import GradesView, TableView


logging.basicConfig()
log = logging.getLogger('cmstools')



def multiple_index_getter(default_from, with_coc=True, name=None):
    def decorator(fn):
        from functools import wraps
        @wraps(fn)
        def decorated(self, indices=None):
            if indices is None:
                df = self.submissions
                if not with_coc:
                    df = df.loc[df.Problem != 'coc']
                indices = df[default_from].unique()
            if isinstance(indices, typing.Iterable) and not isinstance(indices, str):
                s = [fn(self, i) for i in indices]
                res = pd.concat(s, axis=0)
            else:
                res = fn(self, indices)
            return res
        return decorated
    
    return decorator


class CMSController:
    def __init__(self, session):
        self.cms = CMSSession(session) if isinstance(session, str) else session
        self._submissions = None
        self._teams = None
    @multiple_index_getter(default_from='Assignment')
    def get_team(self, index):
        log.info(f'Fetching team {index}')
        root = self.cms.fetch_team(index)[1]
        return TeamParser(root).data.assign(Assignment=index)
    
    @multiple_index_getter(default_from='Id',with_coc=False)
    def get_submission_item(self, index):
        log.info(f'Fetching submission item {index}')
        root = self.cms.fetch_sub_item(index)[1]
        return SubmissionItemParser(root).data.assign(SubId=index)

    @memoised_property
    def students(self):
        root = self.cms.fetch_students()[1]
        return StudentParser(root).data.set_index('MN')
    @memoised_property
    def teams(self):
        return self.get_team()
    @memoised_property
    def submissions(self):
        root = self.cms.fetch_submissions()[1]
        return SubmissionListParser(root).data
    @memoised_property
    def submission_items(self):
        return self.get_submission_item()
    @property
    def submissions_graded(self):
        df = self.submissions
        return df.loc[df.Problem != 'coc']
    
    
    def get_testing(self, indices=None, tutorials=None, named_columns=True, name_with_testings=True, only_grades=False, empty=False):
        if indices is None:
            indices = self.submissions.TestingId.unique()
            indices = indices[~pd.isna(indices)]
        if tutorials is None:
            tutorials = ["1","2"]
        params = dict(named_columns=named_columns, name_with_testings=name_with_testings,only_grades=only_grades)
        if isinstance(tutorials, str):
            root = self.cms.fetch_testings(tutorials, indices)[1]
            tp = TestingParser(root, **params)
            df = tp.data
            if not empty:
                df_g = df[tp.grade_columns]
                idl_null = (df_g.isnull() | (df_g=='')).all(axis=1)
                df = df.loc[~idl_null,:]
            if not only_grades:
                df = df.assign(tutorial=tutorials)
            return df
        else:
            dfs = [self.get_testing(indices=indices, tutorials=t, empty=empty, **params) for t in tutorials]
            return pd.concat(dfs, axis=0).reset_index(drop=True)
    def find_submission(self, assignment=None, problem=None, multiple=False, only_grades=False):
        df = self.submissions
        idl = np.ones(df.shape[0],bool)
        if assignment is not None:
            if np.isscalar(assignment): assignment=[assignment]
            idl = idl & df.Assignment.isin([int(a) for a in assignment])
        if problem is not None:
            if not isinstance(problem, typing.Sequence) or isinstance(problem, str):
                problem = [str(problem)]
            else:
                problem = list(map(str, problem))
            idl &= df.Problem.isin(problem)
        if only_grades:
            idl &= df.Problem != 'coc'
        df_s = df.loc[idl]
        if multiple:
            return df_s
        else:
            assert df_s.shape[0] == 1,f'Search found {df_s.shape[0]} matches but exactly one was expected.'
            return df_s.iloc[0,:]

    _column_sets = dict(
        single=['MN','TeamId','FounderMN','PartnerMN','MNs'],
        compact=['MN','Assignment','Problem','TeamId','SubId','FeedbackSol','PairMN','MNs'],
        compact_teams=['MN','Assignment','Problem','TeamId','SubId','FeedbackSol','FounderMN','PartnerMN','PairMN','MNs'],
        named=['MN','StudentName','# Members','TeamId','FounderName','FounderMN','PartnerName','PartnerMN','MNs']
    )
    def find_submission_teams(self, assignment, problem, only_columns='compact', extra=False, explode=False):
        df_sub = self.find_submission(assignment, problem, multiple=True, only_grades=True)
        return self.get_submission_teams(df_sub, only_columns=only_columns, extra=extra, explode=explode)
    
    def find_grades(self, assignment=None, problem=None, multiple=None, only_grades=False, tutorials=None, multi_index=False, compact_names=False, empty=False):
        '''
        @param assignment: The assignment. Can be None, int, str, Sequence[int], str, Sequence[str]
        @param problem:    The problem. Can be None, int, str, Sequence[int], str, Sequence[str]
        @param tutorials   Only restrict answer to this tutorial group.
        @param only_grades: only return grade columns
        @param multi_index: return a dataframe with a multi index.
        @param compact_names: If True, the grade columns only contain an abbreviation of the assignmnt/problem.
                            If false, the testing id is also prepended.
        @return pandas.DataFrame
        '''
        if multiple is None:
            multiple = multi_index
        assert not multi_index or multiple,f'Multi-index implies multiple'
        df_subs = self.find_submission(assignment=assignment, problem=problem, multiple=multiple, only_grades=True)
        return self.get_grades(df_subs, multiple=multiple, only_grades=only_grades, tutorials=tutorials, multi_index=multi_index, compact_names=compact_names, empty=empty)
    def find_grades_long(self, assignment=None, problem=None, distribute_teams=False):
        '''Returns a long DataFrame with all requested grades.
        @param distribute_teams If True, each submission is paired with team data, and expanded appropriately to accommodate all team members.
        '''
        df_subs = self.find_submission(assignment=assignment, problem=problem, multiple=True, only_grades=True)
        return self.get_grades_long(df_subs, distribute_teams=distribute_teams)
    
    def get_grades(self, df_subs, multiple=None, only_grades=False, tutorials=None, multi_index=False, compact_names=False, empty=False):
        '''
        @param only_grades Only return MNs and grade columns
        '''
        if multiple is None:
            multiple = np.ndim(df_subs) > 1
        indices = [df_subs.TestingId] if not multiple else df_subs.TestingId.values
        assert compact_names + multi_index <= 1,f'compact names are incompatible with multi_index.'
        if not multi_index:
            name_with_testings = not compact_names
            named_columns = True
        else:
            name_with_testings = True
            named_columns = False
        
        df_g = self.get_testing(indices=indices, name_with_testings=name_with_testings, named_columns=named_columns, only_grades=only_grades, empty=empty)
        if multi_index:
            s_idx_map = df_subs[['TestingId','Assignment','Problem']].dropna().set_index('TestingId')
            idl_grd = df_g.columns.str.match('^testing_[0-9]+$')
            idx_tst = df_g.columns[idl_grd].str.replace('^testing_([0-9]+)$','\\1',regex=True).astype(int)
            cols_grd = [(f'Assignment {a}','Bonus' if p=='bonus' else f'Problem {p}') for _,(a,p) in s_idx_map.loc[idx_tst].iterrows()]
            cols = list(zip(['Student Information']*df_g.shape[1], df_g.columns))
            idx_inv = np.where(idl_grd)[0]
            for i,pair in enumerate(cols_grd):
                cols[idx_inv[i]] = pair
            cols = tuple(zip(*cols))
            df_g.columns = pd.MultiIndex.from_arrays(cols)
            mn_index = 'Student Information','mn'
        else:
            mn_index = 'mn'
        if only_grades:
            df_g = df_g.set_index(mn_index)
            return df_g if multiple else df_g.iloc[:,0]
        else:
            return df_g

    def _distribute_teams_st(self, df_g_long, df_st, only_graded=False, column='Points',column_expanded='PointsPartner'):
        how = 'left' if only_graded else 'outer'
        cols_key = ['MN','Assignment','Problem']
        
        def explode(df, origin):
            df = df.explode('MNs').assign(
                IsSub=lambda x:x.MN==x.MNs,
                Origin=origin if isinstance(origin,str) else (lambda x:np.r_[origin][x.IsSub.astype(int)]),
                MN=lambda x:x.MNs
            ).drop('MNs',axis=1)
            return self._fill_pair(df)
        
        df_st_ex = explode(df_st, origin='Direct')
        df_g_ex = df_g_long.merge(df_st_ex, on=cols_key, how=how) # graded, extended with team info and also nonexistent subs
        df_g_ex.loc[df_g_ex.SubId.isna(),'Origin'] = 'Direct'
    
        df_g_prt = explode(df_g_long.merge(df_st, how='left',on=cols_key), origin=['Partner','Direct'])
        df_g_prt = df_g_prt[~df_g_prt.MN.isna() & (df_g_prt.Origin == 'Partner')].reset_index()
    
        df_g_long_ex = pd.concat([df_g_ex,df_g_prt],axis=0,ignore_index=True).drop('index',axis=1).sort_values(cols_key).reset_index(drop=True)
        df_g_wide = df_g_long_ex[np.r_[[column,'Origin'],cols_key]].pivot_table(index=cols_key, columns='Origin').droplevel(0,axis=1)\
            .rename({'Direct':column,'Partner':column_expanded},axis=1)
        return df_g_long_ex.groupby(cols_key).agg('first').drop([column,'Origin'],axis=1).merge(df_g_wide, on=cols_key, how='left').reset_index()

    def _distribute_teams(self, df_g_long, df_subs, only_graded=False):
        df_st = self.get_submission_teams(df_subs,only_columns='compact_teams')
        return self._distribute_teams_st(df_g_long=df_g_long, df_st=df_st, only_graded=only_graded)

    @classmethod
    def _fill_pair(cls,df):
        df = df.copy()
        df.loc[df.PartnerMN==df.MN,'PairMN'] = df.loc[df.PartnerMN==df.MN].FounderMN
        df.loc[df.FounderMN==df.MN,'PairMN'] = df.loc[df.FounderMN==df.MN].PartnerMN
        return df

    def get_submission_teams(self,df_sub=None, only_columns='compact', extra=False, explode=False):
        if df_sub is None: df_sub = self.submissions_graded
        sub_items = np.unique(df_sub.Id)
        df_sub_items_bare = self.get_submission_item(sub_items)
        df_sub_items = df_sub_items_bare.merge(df_sub,left_on='SubId',right_on='Id',suffixes=['Sol','Sub'])
        assignments = np.unique(df_sub.Assignment)
        log.info(f'Found {df_sub_items.shape[0]} submissions (for assignment(s) {assignments})')
        df_teams = self.get_team(assignments)
        log.info(f'Found {df_teams.shape[0]} teams (for assignment(s) {assignments})')
        df = df_sub_items.drop('Id',axis=1).astype(dict(TeamId=str)).merge(
            df_teams.drop(['Assignment'],axis=1),left_on='TeamId',right_on='Id',how='left'
        ).assign(
            MNs=lambda x:[list(set(m)-{np.nan,None}) for m in x[['Mat.Nr','FounderMN','PartnerMN']].values.tolist()]
        ).drop('Id',axis=1).rename({'Mat.Nr':'MN'}, axis=1)
        df = self._fill_pair(df)
        if only_columns is not None:
            if isinstance(only_columns, str):
                try:
                    cols = self._column_sets[only_columns]
                except KeyError:
                    raise KeyError(f'You may only use key collections: {list(self._column_sets.keys())}.')
            else:
                cols = only_columns
            df = df[cols]
        if explode:
            df = df.explode('MNs').assign(IsSub=lambda x:x.MN==x.MNs, MN=lambda x: x.MNs).drop('MNs',axis=1)
        if extra:
            return SimpleNamespace(teams=df_teams, submission_items=df_sub_items, merged=df, submissions = df_sub)
        else:
            return df
        
    def get_grades_long(self, df_subs=None, distribute_teams=False):
        '''Return the provided grades in a long format'''
        if df_subs is None:
            df_subs = self.submissions_graded
        df = self.get_grades(df_subs=df_subs, multi_index=True)
        return self._elongate_grades(df, df_subs, distribute_teams=distribute_teams)
    
    def set_grades_long(self, df_g_long, dry=True):
        '''Return the provided grades in a long format'''
        cols_key = ['Assignment','Problem']
        groups = tuple(df_g_long.set_index(cols_key).index.unique())
        dct_upd = {(a,p):SimpleNamespace(
            info=self.find_submission(assignment=a,problem=p,multiple=False,only_grades=True),
            df=df_g_long.loc[(df_g_long.Assignment==a)&(df_g_long.Problem==p)]
        ) for (a,p) in groups}
        dct_raw = {idx:dict(mns = s.df.MN.values, points=s.df.Points.values, index=s.info.TestingId) for idx,s in dct_upd.items()}
        for (a,p),data in dct_raw.items():
            sverb = "Would submit" if dry else "Submitting" 
            log.info(f'{sverb} {len(data["mns"]):3} grades into testing {data["index"]:2} that corresponds to assignment {a}, problem {p}.')
            if not dry:
                self.cms.submit_grades(**data)
        return dct_raw

    def _elongate_grades(self, df, df_subs=None, distribute_teams=False, mn_col=('Student Information','mn')):
        if df_subs is None:
            df_subs = self.submissions_graded
        value_vars = list(df.columns[df.columns.get_level_values(0).str.lower().str.startswith('assignment')])
        df_long = pd.melt(df,id_vars=[mn_col],value_vars=value_vars,value_name='Points', var_name=['Assignment','Problem'])
        df_long = df_long.loc[df_long.Points!='']\
            .rename({mn_col:'MN'},axis=1)\
            .assign(Assignment=lambda x:x.Assignment.str.replace(r'.*\s','',regex=True).astype(int))\
            .assign(Problem=lambda x:x.Problem.str.lower().str.replace(r'.*\s','',regex=True))\
            .astype({'Assignment':'category','Problem':'category','Points':float})
        if distribute_teams:
            df_long = self._distribute_teams(df_long, df_subs)
        return df_long
    def report(self, sc, assignment=None, dry=False):
        df_gg = self.gather_grades(sc, assignment=assignment)
        rep = HTMLReport(self, df_gg, sc.view_exceptions.df)
        if not dry:
            self.cms.update_report(title='Grading Report', content=rep.html())
        return rep

    def gather_grades(self, sc, assignment=None, problem=None, intersect_columns=False, extra=False):
        df_cms = self.find_grades_long(assignment=assignment, problem=problem, distribute_teams=True)
        df_gs = sc.find_grades_long(self, assignment=assignment,problem=problem, distribute_teams=True)
        df = df_cms.merge(
            df_gs[['MN','Assignment','Problem','Points','PointsPartner']],on=['MN','Assignment','Problem'],suffixes=['CMS','GS']
        )
        if extra:
            return SimpleNamespace(merged=df, cms=df_cms, gs=df_gs)
        else:
            return df



class SheetController:
    from cmstools.constants import _sheets
    def __init__(self, gc, tag='devel'):
        self.gc = gc
        self.ss = gc.open_by_url(self._sheets[tag])
    def get_worksheet_by_title(self, title):
        return next(filter(lambda x:x.title.lower() == title.lower(), self.ss.worksheets()))
    @memoised_property
    def ws_grading(self): return self.get_worksheet_by_title('Assignment Grades')
    @memoised_property
    def ws_exceptions(self): return self.get_worksheet_by_title('Exceptions')
    @memoised_property
    def ws_workload(self): return self.get_worksheet_by_title('Tutor Workload')
    @memoised_property
    def view_grading(self): return GradesView(self.ws_grading)
    @memoised_property
    def view_exceptions(self): return TableView(self.ws_exceptions, index=['MN','Assignment','Problem'],row_sel=lambda i,x:len(i[0])==7)
    @memoised_property
    def view_workload(self): return TableView(self.ws_workload,index=[('Key','Assignment'),('Key','Problem')], headers=2)
    @classmethod
    def from_credentials(cls, filename='credentials.json', tag='shared'):
        gc = gspread.auth.service_account(filename=filename)
        return cls(gc=gc, tag=tag)

    def find_grades_long(self, cmc, assignment=None, problem=None, multiple=True, empty=False, zero=False, distribute_teams=True):
        df_gs = self.view_grading.get_grades(assignment=assignment, problem=problem, multiple=multiple, empty=empty, zero=zero)
        df_long = cmc._elongate_grades(df_gs.reset_index(),mn_col=('index',''),distribute_teams=distribute_teams)
        return df_long
    def annotate_from_cms(self, cmc, assignment=None, problem=None, notes=True, highlight=True):
        '''
        Control annotations for given assignment/problem combinations.
        @param notes:     Whether to annotate partners. True: yes, None: do nothing, False: clear
        @param highlight: Whether to highlight entries. True: yes, None: do nothing, False: clear
        '''
        df_subs = cmc.find_submission(assignment=assignment, problem=problem,multiple=True)
        df_subs = df_subs.loc[df_subs.Problem!='coc']
        for i,(ass,prb) in df_subs[['Assignment','Problem']].iterrows():
            steams = cmc.find_submission_teams(ass, prb, extra=True, only_columns='named')
            df_st = steams.merged
            s_sub = steams.submissions.iloc[0]
            if highlight is not None:
                self.view_grading.highlight_submissions(df_st, s_sub, styles= None if highlight is True else 'clear')
            if notes is not None:
                params = dict(fmt=lambda **kwargs:'') if notes is False else {}
                self.view_grading.annotate_partners(df_st, s_sub, **params)
        return df_subs
    def set_grades_long(self, df_g_long,raw=False):
        return self.view_grading.set_grades_long(df_g_long, raw=raw)
    def fill_workload_submissions(self, cmc):
        vw = self.view_workload
        s_ns = cmc.submissions.merge(cmc.submission_items,right_on='SubId',left_on='Id').astype({'Assignment':str}).groupby(['Assignment','Problem']).count().Id
        with vw.open() as df:
            df[('Difficulty','Submissions')] = s_ns[vw.df.index]
        return s_ns

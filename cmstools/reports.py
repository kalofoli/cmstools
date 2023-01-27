'''
Created on Jan 19, 2023

@author: janis
'''

from datetime import datetime

import pandas as pd

from cmstools.common import GatheredGradeSelector

from urllib.parse import urljoin


def collection_adder(fn):
    from functools import wraps
    @wraps(fn)
    def adder(self, *args, **kwargs):
        return self.add(fn(self, *args, **kwargs))
    return adder
class HTMLCollection():
    def __init__(self, parent=None, tag='body', depth=0):
        self._children = []
        self._parent = parent
        self._tag = tag
        self._depth = depth
    def add(self, what):
        self._children.append(what)
        return self
    def make_header(self, text, level=1):
        l = self._depth+level
        return f'<h{l}>{text}</h{l}>'
    add_header = collection_adder(make_header)
    def make_text(self, text, style=None):
        sstyle= f' style="{style}"' if style is not None else ''
        return f'<div>{text}</div>'
    add_text = collection_adder(make_text)
    def make_table(self, df):
        return df.to_html()
    add_table = collection_adder(make_table)
    def make_progress(self, x, total, text=None):
        spc = f'{x/total*100:.3f}% ' if total>0 else ''
        stext = text if text is not None else f'{spc}({x}/{total})'
        return f'<progress value="{x}" max="{total}">{stext}</progress>'
    add_progress = collection_adder(make_progress)
    def add_tag(self, tag):
        return self.__class__(self, tag=tag, depth=self._depth+1)
    def add_block(self):
        return self.add_tag('div')
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_value, tb):
        if exc_type is not None:
            return False # uncomment to pass exception through
        else:
            self._parent.add(self.data())
        return True
    def data(self):
        tag = self._tag
        prefix0 = '\t'*(self._depth)
        prefix1 = '\t'*(self._depth+1)
        sdata = ('\n'+prefix1).join(self._children)
        return f'<{tag}>\n{prefix1}{sdata}\n{prefix0}</{tag}>'
    def _ipython_display_(self):
        from IPython.display import display,HTML
        display(HTML(self.data()))



class HTMLReport:
    from cmstools.constants import _urls
    def __init__(self, cmc, df, df_exc=None, depth=0, max_entries=10):
        self.cmc = cmc
        self._df = df
        self._df_exc = df_exc
        self._depth = depth
        self.max_entries = max_entries
    
    def __call__(self):
        def add_progress_column(df, v, total, out='Completion', at=None):
            cols = list(df.columns)
            df[out] = tuple(zip(df[v],df[total]))
            if at is not None:
                cols_srt = cols[:at] + [out] + cols[at:] 
                df = df[cols_srt]
            return df.style.format(formatter={out:lambda x: b.make_progress(x[0],x[1])})
        assignments = sorted(self._df.Assignment.unique())
        c = HTMLCollection(depth=self._depth)
        c.add_text(f'Current grading status as of {str(datetime.now())}.')
        for a in assignments:
            s = GatheredGradeSelector(self._df[self._df.Assignment==a])
            if self._df_exc is not None:
                df_exc = self._df_exc[self._df_exc.index.get_level_values('Assignment')==str(a)].droplevel('Assignment',axis=0)
            else:
                df_exc = pd.DataFrame({'MN':[],'Problem':[],'Ignore':[],'Comment':[]})
            with c.add_block() as c_a:
                c_a.add_header(f'Assignment {a}')
                with c_a.add_block() as b:
                    b.add_header('Grades Completion (Google Sheets)', 2)
                    b.add_text('Below is the current completion of the submitted exercises in the google sheet.')
                    df_full = s.df.loc[s.df.IsSub==True]
                    df = df_full[['PointsGS','SubId','Problem']].groupby('Problem')\
                        .count().rename({'PointsGS':'Graded','SubId':'Submitted'},axis=1)
                    style = add_progress_column(df, 'Graded','Submitted')
                    b.add_table(style)
                    df = df_full.loc[df_full.PointsGS.isna()]
                    self.add_student_listing(b, df, desc=f'The first few students with missing grades are listed below.')
                with c_a.add_block() as b:
                    b.add_header('Grades Completion (Content Management System)', 2)
                    df_full = s.df
                    b.add_text('Below is the current completion of the submitted exercises in the CMS. This includes the grades expected from team-members, those grades in the GS and those having submitted a solution.')
                    df = df_full[['PointsCMS','Problem','MN']]\
                        .groupby('Problem').count().rename({'PointsCMS':'Graded','MN':'Submitted'},axis=1)
                    style = add_progress_column(df, 'Graded','Submitted')
                    b.add_table(style)
                    df = df_full[df_full.PointsCMS.isna()]
                    self.add_student_listing(b, df, desc='The first few students with missing grades are listed below.')
                    
                with c_a.add_block() as b:
                    b.add_header('Feedback Submission', 2)
                    b.add_text('Below is the current percentage of feedbacks submitted out of the submissions that have been graded so far in the sheets.')
                    df_full = s.df.loc[s.df.IsSub==True] 
                    df = df_full[['FeedbackSol','PointsGS','Problem']].rename({'FeedbackSol':'Feedbacks','PointsGS':'Graded'},axis=1).assign(Feedbacks=lambda x:x.Feedbacks=='yes')\
                        .groupby('Problem').agg(dict(Feedbacks='sum',Graded='count'))
                    df,ignore = self._process_exceptions(a, df, 'feedback', ignore=False)
                    assert ignore==0,f'Not supposed to ignore at the aggregated level'
                    style = add_progress_column(df, 'Feedbacks','Graded',at=2)
                    b.add_table(style)
                    df = df_full[df_full.FeedbackSol!='yes']
                    df,ignored = self._process_exceptions(a, df, 'feedback', ignore=True, ignore_bulk=True)
                    self.add_student_listing(b, df, desc=f'The first few submissions with missing feedbacks are listed below.',ignored=ignored)
                with c_a.add_block() as b:
                    b.add_header('Grades Without Submissions (Google Sheet)', 2)
                    df = s.df.loc[ pd.isna(s.df.SubId)]
                    df,ignored = self._process_exceptions(a, df, ['nosub'], ignore=True)
                    self.add_student_listing(b, df, 'Below is a list of students who have been graded in the Google Sheet in the Google sheet and/or in the CMS, despite having no record of a submission in our system.',ignored=ignored)
                with c_a.add_block() as b:
                    b.add_header('Submissions With Conflicting Source Values', 2)
                    df = s.df.loc[s.idl_err_collide_src & s.idl_nn_gs]
                    self.add_student_listing(b, df, 'Below is a list of students who have had mismatched grades between the sheet and the CMS.')
                with c_a.add_block() as b:
                    b.add_header('Submissions With Conflicting Team Grades', 2)
                    df = s.df.loc[s.idl_err_collide_teamcms]
                    self.add_student_listing(b, df, 'Below is a list of students who have had mismatched grades between the submitter and their partner.')
                    
        return c.data()
        
    def _add_student_data(self, df,col='MN', prefix=''):
        keys = ['Name','StudentUrl','Tutorial', 'Email']
        _keys = ['_'+k for k in keys]
        df = df.merge(self.cmc.students[keys].rename(dict(zip(keys,_keys)),axis=1), left_on=col,right_index=True, how='left')\
            .astype({k:str for k in _keys})
        zip_none = lambda a,b: [(x1,x2) if not pd.isna(x1) else None for x1,x2 in zip(a,b)]
        df[prefix+'MN'] = zip_none(df[col], df._StudentUrl)
        df[prefix+'Name'] = zip_none(df._Name.replace('nan',None), df._Email)
        return df.drop(_keys,axis=1)
    def _student_formatter(self, cols_mn=['MN'], cols_name=['Name']):
        return {
            **{mn:lambda x:(f'<a href="{self._mkurl(x[1])}">{x[0]}</a>' if x is not None else '-') for mn in cols_mn},
            **{name:lambda x:(f'<a href="mailto:{x[1]}">{x[0]}</a>' if x is not None else '-') for name in cols_name}
        }
    @classmethod
    def _mkurl(cls, x):
        return urljoin(cls._urls['base'],'./'+x)
    def student_listing(self, df, max_entries=None):
        if max_entries is None:
            max_entries = self.max_entries
        if max_entries != -1:
            df = df.iloc[:max_entries,:]
        df = self._add_student_data(df, col='MN')
        df = self._add_student_data(df, col='PairMN',prefix='Pair')
        df = df[['MN','Name','Problem','TeamId','SubId','PairMN','PairName','PointsCMS','PointsGS','PointsPartnerCMS','PointsPartnerGS']]\
            .rename({'PointsPartnerCMS':'PontsPairCMS','PointsPartnerGS':'PontsPairGS'},axis=1)
        style = df.style.format(precision=1,formatter={
            **self._student_formatter(cols_mn=['MN','PairMN'], cols_name=['Name','PairName'])
        }).hide(level=0)
        return df,style
    def add_student_listing(self, b, df, desc='The first few students are listed below.',ignored=0):
        if df.shape[0]>0:
            if desc is not None:
                b.add_text(desc)
            if ignored>0:
                b.add_text(f' From this listing {ignored} entries were omitted as known exceptions.')
            df,style = self.student_listing(df)
            b.add_table(style)
        else:
            b.add_text('No errors found.')
    
    _vals_ignore = ['yes','true','1','on']
    def _process_exceptions(self, assignment, df, tags, ignore=True, ignore_bulk=True):
        if self._df_exc is None:
            return df
        if isinstance(tags, str): tags = [tags]
        aggregated = 'MN' not in df
        ignored = 0
        def xs1(df, keys, level):
            return df[df.index.get_level_values(level).isin(keys)].droplevel(level,axis=0)
        def merge_ignore(df, df_exc, on):
            nonlocal ignored
            df = df.merge(df_exc,left_on=on,right_index=True,how='left')
            if ignore:
                idl_ni = ~df.Ignore.str.lower().isin(self._vals_ignore)
                ignored += (~idl_ni).sum()
                df = df.loc[idl_ni]
            return df.drop('Ignore',axis=1)\
                .assign(Comment=lambda x:x.Comment.astype(str).str.replace('nan',''))
        df_exc = xs1(self._df_exc, [str(assignment)], 'Assignment')
        df_exc = df_exc[df_exc.Kind.isin(tags)].drop('Kind',axis=1)
        if aggregated or ignore_bulk:
            df_exc_all = xs1(df_exc, ['all','any',''], 'MN')[['Comment','Ignore']]
            df = merge_ignore(df, df_exc_all, on='Problem')
            if not aggregated:
                df = df.drop('Comment',axis=1)
        if not aggregated:
            df = merge_ignore(df,df_exc,on=['MN','Problem'])
        return df, ignored
        
        
    html = __call__
    def _ipython_display_(self):
        from IPython.display import HTML, display
        html = self.html()
        display(HTML(html))

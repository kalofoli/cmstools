'''
Created on Jan 19, 2023

@author: janis
'''

from datetime import datetime

import pandas as pd

from cmstools.common import GatheredGradeSelector



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
    def __init__(self, cmc, df, df_exc=None, depth=0):
        self.cmc = cmc
        self._df = df
        self._df_exc = df_exc
        self._depth = depth
    
    def __call__(self):
        from urllib.parse import urljoin
        def add_progress_column(df, v, total, out='Completion'):
            df[out] = tuple(zip(df[v],df[total]))
            return df.style.format(formatter={out:lambda x: b.make_progress(x[0],x[1])})
        def mkurl(x):
            return urljoin(self._urls['base'],'./'+x)
        def add_student(df,cols=None):
            df = df.merge(self.cmc.students[['Name','StudentUrl','Tutorial', 'Email']], left_on='MN',right_index=True)\
                .assign(MN=lambda x:tuple(zip(x.MN, x.StudentUrl)), Name=lambda x:tuple(zip(x.Name, x.Email)))\
                .drop(['StudentUrl','Email'],axis=1)
            if cols is not None:
                df = df[cols]
            return df.style.format(precision=1,formatter=dict(
                MN=lambda x:f'<a href="{mkurl(x[1])}">{x[0]}</a>',
                Name=lambda x:f'<a href="mailto:{x[1]}">{x[0]}</a>'
            )).hide(level=0)
        def add_student_data(df,col='MN', prefix=''):
            keys = ['Name','StudentUrl','Tutorial', 'Email']
            _keys = ['_'+k for k in keys]
            df = df.merge(self.cmc.students[keys].rename(dict(zip(keys,_keys)),axis=1), left_on=col,right_index=True)
            df[prefix+'MN'] = tuple(zip(df[col], df._StudentUrl))
            df[prefix+'Name'] = tuple(zip(df._Name, df._Email))
            return df.drop(_keys,axis=1)
        def student_formatter(cols_mn=['MN'], cols_name=['Name']):
            return {
                **{mn:lambda x:f'<a href="{mkurl(x[1])}">{x[0]}</a>' for mn in cols_mn},
                **{name:lambda x:f'<a href="mailto:{x[1]}">{x[0]}</a>' for name in cols_name}
            }
        assignments = self._df.Assignment.unique()
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
                    b.add_header('Grades Completion', 2)
                    b.add_text('Below is the current completion of the submitted exercised in the google sheet.')
                    df = s.df.loc[ (s.df.IsSub==True),['PointsGS','SubId','Problem']].groupby('Problem').count().rename({'PointsGS':'Graded','SubId':'Submitted'},axis=1)
                    df = add_progress_column(df, 'Graded','Submitted')
                    b.add_table(df)
                with c_a.add_block() as b:
                    b.add_header('Grades Completion', 2)
                    b.add_text('Below is the current completion of the submitted exercised in the CMS.')
                    df = s.df.loc[(s.df.IsSub==False),['PointsCMS','SubId','Problem']].groupby('Problem').count().rename({'PointsCMS':'Graded','SubId':'Submitted'},axis=1)
                    df = add_progress_column(df, 'Graded','Submitted')
                    b.add_table(df)
                with c_a.add_block() as b:
                    b.add_header('Feedback Submission', 2)
                    b.add_text('Below is the current percentage of feedbacks submitted out of the submissions that have been graded so far in the sheets.')
                    df = s.df.loc[s.df.IsSub==True,['FeedbackSol','PointsGS','Problem']].rename({'FeedbackSol':'Feedbacks','PointsGS':'Graded'},axis=1).assign(Feedbacks=lambda x:x.Feedbacks=='yes')\
                        .groupby('Problem').agg(dict(Feedbacks='sum',Graded='count'))
                    df = add_progress_column(df, 'Feedbacks','Graded')
                    b.add_table(df)
                with c_a.add_block() as b:
                    b.add_header('Grades Without Submissions', 2)
                    b.add_text('Below is a list of students who have been graded but have no recorded submission in the system.')
                    df = s.df.loc[ pd.isna(s.df.SubId)]
                    df = df.merge(df_exc[df_exc.Kind.isin(['','nosub'])],left_on=['MN','Problem'],right_index=True,how='left')
                    df = df.loc[~df.Ignore.str.lower().isin(['yes','1','true','on'])].assign(Comment=lambda x:x.Comment.astype(str).str.replace('nan',''))
                    df = add_student(df, cols=['MN','Name','Tutorial','Problem','PointsCMS','PointsGS','Comment'])
                    b.add_table(df)
                with c_a.add_block() as b:
                    b.add_header('Submissions With Conflicting Source Values', 2)
                    b.add_text('Below is a list of students who have had mismatched grades between the sheet and the CMS.')
                    df = s.df.loc[s.idl_err_collide_src & s.idl_nn_gs]
                    df = add_student(df, cols=['MN','Name','Problem','PointsCMS','PointsGS'])
                    b.add_table(df)
                with c_a.add_block() as b:
                    b.add_header('Submissions With Conflicting Team Grades', 2)
                    b.add_text('Below is a list of students who have had mismatched grades between the submitter and their partner.')
                    df = s.df.loc[s.idl_err_collide_teamcms]
                    style = df.style
                    df = add_student_data(df, col='MN')
                    df = add_student_data(df, col='PairMN',prefix='Pair')
                    df = df[['MN','Name','Problem','TeamId','SubId','PairMN','PairName','PointsCMS','PointsGS','PointsPartnerCMS','PointsPartnerGS']]\
                        .rename({'PointsPartnerCMS':'PontsPairCMS','PointsPartnerGS':'PontsPairGS'},axis=1)
                    style = df.style.format(precision=1,formatter={
                        **student_formatter(cols_mn=['MN','PairMN'], cols_name=['Name','PairName'])
                    }).hide(level=0)
                    b.add_table(style)
        return c.data()
    
    html = __call__
    def _ipython_display_(self):
        from IPython.display import HTML, display
        html = self.html()
        display(HTML(html))

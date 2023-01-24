
import typing, re

import numpy as np
import pandas as pd

import gspread

from logging import getLogger
log = getLogger('cmstools')


def update_notes(self, notes):
        """Update the content of the notes located at `cell`.
        @param notes A sequence of (cell, content) pairs, where cell is a string with cell coordinates in A1 notation,
            e.g. 'D7', and content is a string with the text note to insert.
        """

        requests = []
        for cell, content in notes:
            grid_range = gspread.utils.a1_range_to_grid_range(cell, self.id)
            request = dict(updateCells={
                "range": grid_range,
                "rows": [{"values": [{"note": content}]}],
                "fields": "note",
            })
            requests.append(request)
        body = dict(requests=requests)
        self.spreadsheet.batch_update(body)
gspread.Worksheet.update_notes = update_notes

class ArrayView:
    def __init__(self, ws, _data=None, vro='formatted'):
        self.ws = ws
        self.vro = vro
        self.reload(data=_data)
    
    def reload(self, data=None):
        if data is None:
            _vro = getattr(gspread.utils.ValueRenderOption,self.vro)
            data = np.array(self.ws.get_values(value_render_option=_vro),dtype=object)
        else:
            data = np.array(data, dtype=object)
        self._data = data
    
    @classmethod
    def _set_min_shape(cls, a, ms, fill=''):
        if np.ndim(a) == 1:
            assert len(a) == 0, f'Cannot apply min shape to a vector'
            a = a[:,None]
        ss = a.shape
        pad = [[0,max(ms[0]-ss[0],0)],[0,max(ms[1]-ss[1],0)]]
        return np.pad(a, pad, mode='constant', constant_values=fill)
    
    class _ArrayViewOpenContext:
        def __init__(self, av, raw, min_shape, data_attr='data'):
            self.av = av
            self.data = None
            self.raw = raw
            self.min_shape = min_shape
            self.data_attr = data_attr
        
        def __enter__(self):
            data = getattr(self.av,self.data_attr)
            if self.min_shape is not None:
                data = self.av._set_min_shape(data, self.min_shape)
            self.data = data
            return data

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.av.update(self.data,raw=self.raw)
            self.data = None
    
    def open(self, raw=False, min_shape=None):
        return self._ArrayViewOpenContext(self, raw=raw, min_shape=min_shape)
    
    def update(self, data, raw=False):
        dst_data = self._set_min_shape(self._data, np.maximum(data.shape,self.shape))
        idl_ne = data != dst_data
        I,J = np.where(idl_ne)
        K = data[I,J]
        self._update_sparse(I,J,K)
        
    def _update_sparse(self, I,J,K, raw=False):
        cells = [gspread.Cell(i+1,j+1,k) for i,j,k in zip(I,J,K)]
        
        vio = gspread.utils.ValueInputOption.raw if raw else gspread.utils.ValueInputOption.user_entered
        self.ws.update_cells(cells, value_input_option=vio)
        self.reload()

    def _rowcol_to_a1(self, rows, cols):
        if np.isscalar(cols): cols = [cols]
        if np.isscalar(rows): rows = [rows]
        n = max(len(rows), len(cols))
        assert len(rows)==1 or len(cols)==1 or len(rows)==len(cols),f'Indics must be either same-dimensional vectors or scalars, but the rows are of size {len(rows)} and the columns of size {len(cols)}.'
        if len(cols) == 1: cols=list(cols)*n
        if len(rows) == 1: rows=list(rows)*n
        return [gspread.utils.rowcol_to_a1(r+1,c+1) for r,c in zip(rows,cols)]
    _styles = {}
    @classmethod
    def lookup_style(cls, s):
        try:
            return s if isinstance(s, dict) else cls._styles[s]
        except KeyError:
            ss = ",".join(cls._styles.keys())
            raise KeyError(f'No style {s} available. Try one of {ss}.')
    def highlight_rowcol(self, rows, cols, style='default'):
        ranges = self._rowcol_to_a1(rows, cols)
        if isinstance(style, typing.Sequence) and not isinstance(style, str):
            get_format = lambda i: self.lookup_style(style[i])
        else:
            fmt = self.lookup_style(style)
            get_format = lambda _: fmt
        formats = [{'range':r, 'format':get_format(i)} for i,r in enumerate(ranges)]
        self.ws.batch_format(formats)
        return formats
    def note_rowcol(self, rows, cols, texts):
        ranges = self._rowcol_to_a1(rows, cols)
        if isinstance(texts,str): texts = [texts]
        if len(texts) == 1: texts *= len(ranges)
        assert len(ranges)==len(texts),f'You must provide either a single annotation or one for each provided cell.'
        return self.ws.update_notes(list(zip(ranges, texts)))
    @property
    def data(self): return self._data.copy()
    @property
    def shape(self): return self._data.shape
    def __repr__(self): return f'<{type(self).__name__}({"x".join(map(str,self.shape))}) for sheet: "{self.ws.title}">'

class TableView(ArrayView):
    def __init__(self, ws, index=0, headers=None, _data=None, vro='formatted', row_sel=None, col_sel=None):
        '''
        @param headers The number of header rows to use. A value greater than 1 creates a multi-index.
        @param index The index column to use.
            If an integer is probvided, it marks the (0-based) index of the column in the data.
            If a string is used, it is matched against the first row.
            If a tuple of strings is used, a multi-index is assumed, and is matched against the columns of the first rows.
            Before the matching, centered cells are also expanded (assumed to be followed by 0 entries tot he right until the next ).
        '''
        self.headers = headers
        self.index = index
        self.row_sel = row_sel
        self.col_sel = col_sel
        super().__init__(ws, _data=_data, vro=vro)
        
    @property
    def df(self): return self._df.copy()
    def open(self, raw=False):
        return self._ArrayViewOpenContext(self, raw=raw, min_shape=None,data_attr='df')
    def update(self, df, raw=False):
        I,J = np.where(df != self._df)
        K = df.values[I,J]
        I_arr,J_arr = self._map_df2arr(I,J)
        self._update_sparse(I_arr,J_arr,K)
    def _row_iloc(self, key):
        '''Maps a vector of entries of the TableView key to absolute indices in the TableView.'''
        return self._row_key2df[key].values
    def _map_df2arr(self, i,j):
        '''Maps absolute indices of elements in the TableView to those in the underlying array.
        Note that although both are zero-based, the Google Sheets use 1-based indexing.
        '''
        return self._row_df2arr[i], self._col_df2arr[j]
    def _rowcol_to_a1(self, i,j):
        i_arr, j_arr = self._map_df2arr(i, j)
        return super()._rowcol_to_a1(i_arr,j_arr)
    @property
    def shape(self): return self._df.shape
    def __repr__(self): return f'<{type(self).__name__}({"x".join(map(str,self.shape))})>'
    
    @classmethod
    def _make_index(cls, data, name=None):
        if data.shape[0] == 1:
            return pd.Index(data[0,:],name=name)
        cols = []
        for row in data:
            row_len = np.r_[[len(s) for s in row],1]
            idx_nz = np.where(row_len>0)[0]

            col = np.concatenate([np.repeat(v,i) for i,v in zip(np.diff(idx_nz), row[idx_nz[:-1]])])
            cols.append(col)
        return pd.MultiIndex.from_arrays(cols, names=name)

    @classmethod
    def parse_index(cls, index, headers, data):
        if headers is None:
            if isinstance(index, tuple):
                headers = len(index)
            elif isinstance(index, typing.Sequence) and isinstance(index[0], tuple):
                headers = len(index[0])
        if headers is None: headers = 1
        columns = cls._make_index(data[:headers,:])
        idx_indices = []
        if index is not None:
            _index = [index] if isinstance(index, (tuple,str)) or not isinstance(index, typing.Sequence) else index
            for idx in _index:
                if isinstance(idx, int):
                    idx_index = int(idx)
                elif isinstance(idx, str):
                    assert headers == 1,f'When a string index is provided only a single header can be used.'
                    idl_m = idx == columns
                    assert np.sum(idl_m)==1,f'Index string "{idx}" must match exactly one column, but matched at indices: {np.where(idl_m)[0]}.'
                    idx_index = np.where(idl_m)[0][0]
                elif isinstance(idx, tuple):
                    assert headers == len(idx),f'When an index is specified as a tuple, it must have one element per header row.'
                    idl_m = idx == columns
                    assert np.sum(idl_m)==1,f'Index tuple "{idx}" must match exactly one column, but matched at indices: {np.where(idl_m)[0]}.'
                    idx_index = np.where(columns==idx)[0][0]
                else:
                    raise TypeError(f'Could not parse index from entry of type {type(idx).__name__}.')
                idx_indices.append(idx_index)
        columns_rest = columns[idx_indices]
        return headers, idx_indices, columns_rest, columns.drop(columns_rest)

    def reload(self, data=None):
        headers = self.headers
        index = self.index
        row_sel = self.row_sel
        col_sel = self.col_sel
        super().reload(data=data)
        data = self.data

        num_hdr, idx_indices, cols_idx, columns = self.parse_index(index, headers, data)
        
        row_offset = self.row_offset = num_hdr
        self.index_idx = idx_indices
        col_df2arr = np.delete(np.arange(data.shape[1]), idx_indices)
        data_index, data_body = data[num_hdr:,idx_indices], np.delete(data[num_hdr:,], idx_indices,axis=1)
        
        if data_index.shape[1]>1:
            d = list(zip(*data_index))
            row_index = pd.MultiIndex.from_arrays(d, names=cols_idx)
        elif data_index.shape[1] == 0:
            row_index = None
        else:
            row_index = pd.Index(data_index.flatten())
        
        df = pd.DataFrame(data_body, index=row_index, columns=columns)
        
        row_df2arr = np.arange(data.shape[0]-num_hdr)+num_hdr
        if row_sel is not None:
            idl = np.r_[[row_sel(*ir) for ir in df.iterrows()]]
            df = df.loc[idl,:]
            row_df2arr = row_df2arr[idl]
        if col_sel is not None:
            idl = np.r_[[col_sel(*ir) for ir in df.items()]]
            df = df.loc[:,idl]
            col_df2arr = col_df2arr[idl]
        self._df = df
        self._row_df2arr = row_df2arr
        self._row_key2df = pd.Series(np.arange(len(data_index)), index=data_index)
        self._col_df2arr = col_df2arr
        return self

class GradesView(TableView):
    rex_mn = re.compile('^[0-9]{7}$')
    rex_num = re.compile('^[0-9]+$')
    @classmethod
    def _row_sel(cls, i, row): return cls.rex_mn.match(i) is not None
    @classmethod
    def _col_sel(cls, i, col): return i[1].lower().strip() != 'total'

    def __init__(self, ws, _data=None, vro='formatted'):
        super().__init__(ws=ws, _data=_data, index=('Student Information','MN'), headers=2,vro=vro, row_sel=self._row_sel, col_sel=self._col_sel)
    _styles={
        'default':{'backgroundColorStyle':{'themeColor':None}, 'textFormat':{'foregroundColorStyle':{'themeColor':None}}},
        #'submitted':{'backgroundColorStyle':{'themeColor':'ACCENT1'}},
        #'partner':{'backgroundColorStyle':{'themeColor':'ACCENT2'}},
        'submitted':{'backgroundColorStyle':{'rgbColor':dict(red=.96,green=1,blue=.79,alpha=1)}, 'textFormat':{'foregroundColorStyle':{'rgbColor':dict(red=0,green=0,blue=0)},'bold':False}},
        #'partner':{'backgroundColorStyle':{'rgbColor':dict(red=1,green=.85,blue=.78,alpha=.1)}, 'textFormat':{'foregroundColorStyle':{'rgbColor':dict(red=.4,green=.4,blue=.4,alpha=1)}}},
        'partner':{'backgroundColorStyle':{'rgbColor':dict(red=.93,green=.95,blue=.92,alpha=.1)}, 'textFormat':{'foregroundColorStyle':{'rgbColor':dict(red=.5,green=.5,blue=.5,alpha=1)},'bold':False}},
        'unsubmitted':{'backgroundColorStyle':{'rgbColor':None}, 'textFormat':{'foregroundColorStyle':{'rgbColor':dict(red=5,green=0,blue=0,alpha=1)},'bold':True}},
    }
    def _column_index(self, assignment=None, problem=None, multiple=False):
        df = self.df
        idl = np.ones(self.shape[1],bool)
        cols_ass = df.columns.levels[0].str.lower()
        if assignment is None:
            ass_full = cols_ass[cols_ass.str.startswith('assignment')]
        else:
            if isinstance(assignment, typing.Sequence) and not isinstance(assignment, str):
                ass_full = [f'assignment {a}' for a in assignment]
            else:
                ass_full = [f'assignment {assignment}']
        idl = idl & df.columns.get_level_values(0).str.lower().isin(ass_full)
        if problem is not None:
            idl = idl & df.columns.get_level_values(1).str.lower().isin([f'problem {problem}', str(problem).lower()])
        if not multiple:
            assert np.sum(idl)==1,f'Filters matched {np.sum(idl)} columns {",".join(map(str,df.columns[idl]))} but a single one was required.'
        return idl
        
    def get_grades(self, assignment=None, problem=None, multiple=True, empty=False, zero=False):
        idl = self._column_index(assignment=assignment, problem=problem, multiple=multiple)
        df = self.df.loc[:,idl]
        idl = np.zeros(df.shape, bool)
        if not empty:
            idl |= df==''
        if not zero:
            idl |= df.replace('','NaN').astype(float)==0
        df = df.loc[~idl.all(axis=1)]
        if multiple:
            return df
        else:
            return df.iloc[:,0]
    
    def set_grades(self, data, assignment=None, problem=None):
        multiple = np.ndim(data)==2 and data.shape[1]>1
        idl = self._column_index(assignment=assignment, problem=problem, multiple=multiple)
        with self.open(raw=True) as df:
            if multiple:
                df.loc[data.index,idl] = data.values
            else:
                idx = np.where(idl)[0][0]
                df.loc[data.index,df.columns[idx]] = data
    
    def set_grades_long(self, df_g_long,raw=False):
        df_g_long = df_g_long.astype(dict(Assignment='category', Problem='category'))
        dct_ren_ass = {i:f'Assignment {i}' for i in df_g_long.Assignment.cat.categories}
        dct_ren_prb = {p:'Bonus' if p=='bonus' else f'Problem {p}' for p in df_g_long.Problem.cat.categories}
        midx_g = df_g_long.assign(
            Assignment=lambda x:x.Assignment.cat.rename_categories(dct_ren_ass),
            Problem=lambda x:x.Problem.cat.rename_categories(dct_ren_prb)
        ).set_index(['MN','Assignment','Problem']).index

        idx_cols = midx_g.droplevel(0).drop_duplicates()
        with self.open(raw=raw) as df:
            df_grd = df[idx_cols].stack(level=[0,1])
            df_grd[midx_g] = df_g_long.Points.values
            df[idx_cols] = df_grd.unstack(level=[1,2])[idx_cols]
        return self

    def highlight_submissions(self, df, s_info, styles=None):
        '''Highlights the grades based on the submissions.
        
        @param df A DataFrame with a set of submissions
        @param s_info a Series with the Assignment and the Problem for the provided submissions.
        @note These arguments can be returned directly from the CMController object from the method find_submission_teams
        by setting the argument extra=True.
        
        >>> cmc = CMSController()
        >>> steams = cmc.find_submission_teams(assignment=1,problem='bonus', extra=True)
        >>> vg.highlight_submissions(steams.merged, steams.info)
        '''
        
        idl_col = self._column_index(assignment=s_info.Assignment, problem=s_info.Problem, multiple=False)
        idx_col = np.where(idl_col)[0]
        mn_sub = set(df.MN)
        mn_par = set(df.MNs.explode()) - mn_sub
        mn_rest = set(self.df.index) - mn_sub - mn_par
        mns = list(mn_sub)+list(mn_par)+list(mn_rest)
        idx_rows = self._row_iloc(mns)
        if styles is None:
            styles = ['submitted']*len(mn_sub) + ['partner']*len(mn_par) + ['unsubmitted']*len(mn_rest)
        log.info(f'Highlighting assignment {s_info.Assignment} problem {s_info.Problem} of submission {s_info.Id} with {len(mn_sub)} submitters, {len(mn_par)} partners, and {len(mn_rest)} remaining non-submitters.')
        return self.highlight_rowcol(idx_rows, idx_col, styles)
    
    @staticmethod
    def _make_note(submitter, partner, team, submitter_name=None, partner_name=None):
        spar = str(submitter) if submitter_name is None else f'{submitter_name} ({submitter})'
        return f'Partner of submitter {spar} through team {team}'
    
    def annotate_partners(self, df, s_info, fmt=_make_note, with_names=None):
        '''Annotates the partner entries with the MN of their team member.
        
        @param df A DataFrame with a set of submissions
        @param s_info a Series with the Assignment and the Problem for the provided submissions.
        @note These arguments can be returned directly from the CMController object from the method find_submission_teams
        by setting the argument extra=True.
        
        >>> cmc = CMSController()
        >>> steams = cmc.find_submission_teams(assignment=1,problem='bonus', extra=True)
        >>> vg.annotate_partners(steams.merged, steams.info)
        '''
        if with_names is None:
            with_names = 'FounderMN' in df and 'PartnerMN' in df
        
        idl_col = self._column_index(assignment=s_info.Assignment, problem=s_info.Problem, multiple=False)
        idx_col = np.where(idl_col)[0]
        df_e = df[['MN','MNs','TeamId']].explode('MNs')
        df_e = df_e.loc[df_e.MN!=df_e.MNs]
        (_,s_sub),(_,s_par),(_,s_team) = df_e.items()
    
        if with_names:
            get_names = lambda df,n:df[[n+'MN',n+'Name']].dropna().rename({n+'MN':'MN',n+'Name':'Name'},axis=1)
            s_mn_name = pd.concat([
                get_names(df,'Founder'), get_names(df,'Partner'),
            ],axis=0).set_index('MN')
            get_text = lambda submitter, partner, team: fmt(
                submitter=submitter,partner=partner,team=team, submitter_name=s_mn_name.loc[submitter][0], partner_name=s_mn_name.loc[partner][0]
            )
        else:
            get_text = fmt
        
        texts = [get_text(submitter=s,partner=p,team=t) for s,p,t in zip(s_sub,s_par, s_team)]
        
        idx_rows = self._row_iloc(s_par)
        log.info(f'Annotating assignment {s_info.Assignment} problem {s_info.Problem} of submission {s_info.Id} with {len(s_par)} partners.')
        self.note_rowcol(idx_rows, idx_col, texts)
        



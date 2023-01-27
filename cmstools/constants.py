



_xpaths = dict(
    login_action = '/html/body/div[@class="container"]/div[contains(@class,"navbar")]/div[@class="navbar-inner"]/ul[@class="pull-right nav"]/li[last()]/form/@action'
)

_strings = dict(
    login_response = '/eml22/users/logout'
)

_urls = dict(
    base='https://cms.cispa.saarland/eml22/',
    logout='users/logout',
    teams='TeamGroupings/view/{index}',
    submission_items='SubmissionItems/index/limit:0/byTutorial:0/bySub:{index}',
    submissions='Submissions/index/limit:0',
    students='students/index/limit:0/bytutorial:0/byA:0/cols:Si~Sm~Uu~Uf~Ul~Ue~Sgn~St~Sj~Ss~Sf~Sh',
    testings='testingresults/enter/1',
    import_csv='testingresults/import/{index}',
    upload_report='contents/edit/12',
)

_sheets = dict(
    shared='https://docs.google.com/spreadsheets/d/1WfeZJiTT_BZ5O7aFjNWxNhs70lOMmSj8j5zbmk_ITMI',
    devel='https://docs.google.com/spreadsheets/d/1VJBvdoAGBSL3EtkSxBVsq4Zwtnto90WmtJDHvZxJF-A'    
)
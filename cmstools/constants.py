



_xpaths = dict(
    login_action = '/html/body/div[@class="container"]/div[contains(@class,"navbar")]/div[@class="navbar-inner"]/ul[@class="pull-right nav"]/li[last()]//form/@action'
)

_strings = dict(
    login_response = '/eml22/users/logout'
)

_urls = dict(
    base='https://cms.cispa.saarland/eml22/',
    origin='https://cms.cispa.saarland',
    logout='users/logout',
    teams='TeamGroupings/view/{index}',
    submission_items='SubmissionItems/index/limit:0/byTutorial:0/bySub:{index}',
    submissions='Submissions/index/limit:0',
    students='students/index/limit:0/bytutorial:0/byA:0/cols:Si~Sm~Uu~Uf~Ul~Ue~Sgn~St~Sj~Ss~Sf~Sh',
    student_cols='students/index/limit:0/bytutorial:0/byA:0/cols:Sm~{cols}/format:csv',
    student_view='students/view/{mn}',
    testings='testingresults/enter/1',
    import_csv='testingresults/import/{index}',
    upload_report='contents/edit/12',
    upload_submission='SubmissionItems/upload',
)

_sheets = dict(
    shared='https://docs.google.com/spreadsheets/d/1WfeZJiTT_BZ5O7aFjNWxNhs70lOMmSj8j5zbmk_ITMI',
    devel='https://docs.google.com/spreadsheets/d/1h_74R12As5cVkFkenz_cXC_hbJmZGuQ4fexELCHWyUU'    ,
)

_testing_ids = dict(
    theoretical = 37,
    practical = 40,
    admission1 = 55,
    exceptions1 = 56,
    seating = 58,
    exam_problem_1 = 60,
    exam_problem_2 = 61,
    exam_problem_3 = 62,
    exam_problem_4 = 63,
    exam_problem_5 = 64,
)

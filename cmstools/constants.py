



_xpaths = dict(
    login_action = '/html/body/div[@class="container"]/div[contains(@class,"navbar")]/div[@class="navbar-inner"]/ul[@class="pull-right nav"]/li[last()]/form/@action'
)

_strings = dict(
    login_response = '/eml22/users/logout'
)

_urls = dict(
    base='https://cms.cispa.saarland/eml22/',
    teams='TeamGroupings/view/{index}',
    submission_items='SubmissionItems/index/limit:0/byTutorial:0/bySub:{index}',
    submissions='Submissions/index/limit:0',
)
def operation(arm,q,used):
    if not q:return ('normal',False)
    if used and arm!='REPEAT':return ('normal',True)
    return ('early',0 if arm=='CONTROL' else 1)
def advance(used,r):
    return used or 'early' in r

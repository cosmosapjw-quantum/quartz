"""Fixed cell shrinkage, red scaffold."""
def shrink(rows,alpha=8):
    return {'global':[0,0,1,0,0],'cells':[[0,0,1,0,0] for i in range(16)],'gain_cost':[[0.,1.] for i in range(16)]}

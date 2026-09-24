"""Conservative design arithmetic, not certification or statistical power."""
import json,math,hashlib
from pathlib import Path


def radius(n,delta=.05,policies=4,width=2.):
    if n<1 or not 0<delta<1 or policies<1 or width<0:raise ValueError('invalid bound parameters')
    return width*math.sqrt(math.log(policies/delta)/(2*n))


def strict_margin_n(margin,delta=.05,policies=4,width=2.):
    if margin<=0:raise ValueError('positive empirical margin required')
    n=max(1,math.floor(width*width*math.log(policies/delta)/(2*margin*margin))+1)
    assert radius(n,delta,policies,width)<margin
    assert n==1 or radius(n-1,delta,policies,width)>=margin
    return n


def main():
    rows=[dict(goals=n,radius=radius(n),radius_pp=100*radius(n)) for n in [128,256,1024,4096,16384]]
    margins=[dict(empirical_improvement_pp=100*m,goals_for_radius_strictly_below_margin=strict_margin_n(m)) for m in [.01,.02,.05,.10]]
    # Analytic random mixing scales both expected paired loss and its range.
    mixing=[]
    for weight in [0.,.1,.5,1.]:
        delta_mean=-.02*weight;bound=delta_mean+radius(256,width=2*weight)
        mixing.append(dict(switch_probability=weight,hypothetical_empirical_excess_failure=delta_mean,upper_bound_arithmetic=bound))
    out=Path('outputs/maintrack/risk_bound_feasibility.json');assert not out.exists()
    result=dict(status='DESIGN_ARITHMETIC_ONLY_NOT_A_CERTIFICATE',delta=.05,finite_policy_count=4,loss='Goal-level mean across fixed actors of selector failure minus baseline failure, bounded [-1,1].',rows=rows,empirical_margin_design=margins,mixing=mixing,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                assumptions='Frozen finite policies; fresh independent goals from designated deployment distribution; valid paired bounded loss; Bonferroni with Hoeffding. Current consumed goals do not satisfy fresh calibration status.',
                scope='Conservative confidence-radius arithmetic only. Not a sample-complexity lower bound, power calculation, certificate for existing policies, or raw-physical-cost guarantee. K=4 is illustrative, not retrospective accounting for all research choices. More efficient tests may exist.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(rows=rows,margins=margins,mixing=mixing),indent=2))


if __name__=='__main__':main()

import unittest
import numpy as np
import pandas as pd
from src.online_guard import guard
from src.positive_revision import range_breakout

SPEC=dict(multiplier=10,tick=1,open_fee_rate=0,close_fee_rate=0,
          open_fee_fixed=2,close_fee_fixed=2)


def shadow_frame(values):
    starts=pd.date_range('2020-01-01 10:00',periods=len(values),freq='D')
    return pd.DataFrame(dict(signal_time=starts,entry_time=starts+pd.Timedelta(minutes=1),
                              exit_time=starts+pd.Timedelta(hours=1),instrument=['X']*len(values),
                              candidate_id=['rule']*len(values),net_return=values))


def synthetic_bars():
    frames=[]
    for j,date in enumerate(pd.bdate_range('2017-01-02',periods=35)):
        starts=pd.date_range(date+pd.Timedelta(hours=9,minutes=1),periods=72,freq='5min')
        base=100+j
        close=np.full(72,base+3.);close[:6]=base;close[-1]=base+5
        frames.append(pd.DataFrame(dict(start=starts,end=starts+pd.Timedelta(minutes=4),
            date=date,open=np.full(72,base),close=close,high=np.r_[np.full(6,base+1),np.full(66,base+6)],
            low=np.full(72,base-1),volume=np.full(72,1000),open_interest=np.full(72,10000),
            symbol=['X1701']*72,period=['train']*72)))
    return pd.concat(frames,ignore_index=True)


class RevisionTests(unittest.TestCase):
    def test_unfinished_trade_and_equal_exit_signal_are_not_available(self):
        t=shadow_frame([.01,.02,.03])
        t.loc[0,'exit_time']=t.signal_time.iloc[1]
        _,a=guard(t,window=2,min_observations=1)
        self.assertEqual(a.prior_shadow_count.iloc[1],0)
        self.assertEqual(a.prior_shadow_count.iloc[2],2)

    def test_future_profit_does_not_change_earlier_guard(self):
        t=shadow_frame([.01]*30);_,a=guard(t)
        t.loc[20:,'net_return']=-100.;_,b=guard(t)
        pd.testing.assert_series_equal(a.guard_allowed.iloc[:21],b.guard_allowed.iloc[:21])
        pd.testing.assert_series_equal(a.prior_shadow_mean.iloc[:21],b.prior_shadow_mean.iloc[:21])

    def test_negative_history_can_recover_using_blocked_shadow_trades(self):
        t=shadow_frame([-.01]*20+[.02]*30)
        _,a=guard(t)
        self.assertFalse(a.guard_allowed.iloc[20])
        self.assertTrue(a.guard_allowed.iloc[-1])

    def test_separate_rules_have_separate_histories(self):
        t=shadow_frame([.01]*20);t.loc[10:,'instrument']='Y'
        _,a=guard(t,min_observations=5)
        self.assertEqual(a.prior_shadow_count.iloc[10],0)

    def test_breakout_known_range_next_open_fixed_exit(self):
        bars=synthetic_bars();t=range_breakout(bars,SPEC,'X')
        self.assertGreater(len(t),0)
        self.assertTrue((t.entry_time-t.signal_time).eq(pd.Timedelta(minutes=1)).all())
        self.assertTrue(t.signal_time.dt.strftime('%H:%M').eq('09:35').all())
        self.assertTrue(t.exit_time.dt.strftime('%H:%M').eq('15:00').all())
        self.assertFalse(t.date.duplicated().any())
        altered=bars.copy();altered.loc[altered.end.dt.hour.ge(11),'high']=100000
        z=range_breakout(altered,SPEC,'X')
        pd.testing.assert_frame_equal(t,z)

    def test_missing_scheduled_exit_is_not_replaced_by_early_close(self):
        bars=synthetic_bars()
        bars=bars[~bars.end.dt.strftime('%H:%M').eq('15:00')]
        self.assertTrue(range_breakout(bars,SPEC,'X').empty)


if __name__=='__main__':unittest.main()

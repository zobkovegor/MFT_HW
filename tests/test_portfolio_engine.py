import unittest
import numpy as np
import pandas as pd
from src.universe_research import parse_fee, price_trades, make_candidate
from src.portfolio_engine import simulate, lot_count, performance, DEFAULT_RISK


SPEC=dict(multiplier=10,tick=1,open_fee_rate=.0001,close_fee_rate=.0002,
          open_fee_fixed=2,close_fee_fixed=3)


def example(direction=1, entry=100, exit=105, symbol='X'):
    return price_trades(pd.DataFrame([dict(instrument=symbol,symbol='X2001',
        candidate_id='test',direction=direction,entry_open=entry,exit_close=exit,
        signal_time=pd.Timestamp('2020-01-02 09:05'),
        entry_time=pd.Timestamp('2020-01-02 09:06'),
        exit_time=pd.Timestamp('2020-01-02 09:10'),
        volume=10000,open_interest=10000,prior_vol=.01)]),SPEC)


class PortfolioTests(unittest.TestCase):
    def test_fee_units(self):
        self.assertEqual(parse_fee('0.23‰'),(.00023,0))
        self.assertEqual(parse_fee('3 CNY'),(0,3))

    def test_long_short_symmetric_round_trip(self):
        for direction,close in [(1,105),(-1,95)]:
            t=example(direction=direction,exit=close).iloc[0]
            self.assertEqual(t.gross_pnl_cny,50)
            self.assertEqual(t.slippage_cny,20)
            self.assertAlmostEqual(t.net_pnl_cny,
                direction*(t.exit_fill-t.entry_fill)*10-t.fees_cny)

    def test_zero_size_no_forced_contract(self):
        t=example();t['volume']=0
        l,d,m=simulate(t,{'X':SPEC},'2020-01-01','2020-01-03')
        self.assertEqual(l.lots.iloc[0],0)
        self.assertEqual(d.equity.iloc[-1],3_000_000)
        self.assertFalse(performance(l,d,m)['all_required_pass'])

    def test_liquidity_size_and_margin(self):
        t=example().iloc[0];risk={**DEFAULT_RISK,'initial_capital':1000}
        n=lot_count(t,SPEC,1000,0,risk)
        self.assertLessEqual(n*t.entry_fill*10*.2,300)
        t['volume']=99
        self.assertEqual(lot_count(t,SPEC,1000,0,risk),0)

    def test_minute_marks_reconcile_and_include_initial_peak(self):
        t=example(exit=95)
        minute=pd.DataFrame(dict(date=pd.date_range('2020-01-02 09:06',periods=5,freq='min'),
                                 close=[100,90,92,94,95],symbol=['X2001']*5))
        l,d,m=simulate(t,{'X':SPEC},'2020-01-01','2020-01-03',minute_frames={'X':minute})
        self.assertAlmostEqual(d.equity.iloc[-1],3_000_000+l.account_pnl_cny.sum())
        self.assertAlmostEqual(m.equity.iloc[-1],d.equity.iloc[-1])
        self.assertGreater(performance(l,d,m)['max_drawdown_pct'],0)
        self.assertEqual(d['return'].iloc[0],0)

    def test_overlap_and_noncausal_entry(self):
        t=pd.concat([example(),example()],ignore_index=True)
        l,d,m=simulate(t,{'X':SPEC},'2020-01-01','2020-01-03')
        self.assertEqual(l.accepted.sum(),1)
        t.loc[0,'entry_time']=t.signal_time.iloc[0]
        with self.assertRaises(ValueError):simulate(t,{'X':SPEC},'2020-01-01','2020-01-03')

    def test_sizing_does_not_use_future_return(self):
        a=example().iloc[0];b=example(exit=1).iloc[0]
        self.assertEqual(lot_count(a,SPEC,3_000_000,0,DEFAULT_RISK),
                         lot_count(b,SPEC,3_000_000,0,DEFAULT_RISK))

    def test_margin_exit_uses_observed_mark_and_reconciles_actual_pnl(self):
        t=example(exit=80)
        minute=pd.DataFrame(dict(date=pd.date_range('2020-01-02 09:06',periods=5,freq='min'),
                                 close=[100,70,75,78,80],symbol=['X2001']*5))
        risk=dict(initial_capital=1000,per_asset_margin=.6,risk_fraction=1.)
        l,d,m=simulate(t,{'X':SPEC},'2020-01-01','2020-01-03',risk,{'X':minute})
        self.assertTrue(l.forced_margin_exit.iloc[0])
        self.assertEqual(l.exit_time.iloc[0],pd.Timestamp('2020-01-02 09:07'))
        self.assertEqual(l.exit_fill.iloc[0],69.)
        self.assertAlmostEqual(d.equity.iloc[-1],1000+l.account_pnl_cny.sum())
        self.assertAlmostEqual(l.net_pnl_cny.iloc[0],
                              l.gross_pnl_cny.iloc[0]-l.fees_cny.iloc[0]-l.slippage_cny.iloc[0])
        self.assertLessEqual(m.margin_fraction.max(),.60)

    def test_threshold_and_pretest_signals_ignore_test_prices(self):
        n=2400;stamps=pd.date_range('2018-12-20',periods=n,freq='5min')
        periods=np.where(np.arange(n)<1600,'train','test')
        b=pd.DataFrame(dict(start=stamps,end=stamps+pd.Timedelta(minutes=4),
                            open=np.full(n,100.),close=100+np.sin(np.arange(n)),
                            volume=np.full(n,1000),open_interest=np.full(n,1000),
                            symbol=['X2001']*n,block=np.zeros(n),period=periods))
        a,threshold=make_candidate(b,SPEC,holding=1,quantile=.8)
        b.loc[b.period.eq('test'),'close']*=10
        z,threshold2=make_candidate(b,SPEC,holding=1,quantile=.8)
        self.assertEqual(threshold,threshold2)
        # Boundary trades may exit into test; compare fully completed train.
        cutoff=stamps[1599]
        pd.testing.assert_frame_equal(a[a.exit_time<cutoff],z[z.exit_time<cutoff])


if __name__=='__main__':unittest.main()

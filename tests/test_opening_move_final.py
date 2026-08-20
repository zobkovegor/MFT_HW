"""Accounting, causality and allocation regression tests (no market files needed)."""
import unittest
import numpy as np
import pandas as pd
from src.opening_move_final import (
    build_setups, add_trade_costs, daily_equity, simulate_portfolio,
    select_opening, fee_parameters,
)


class BacktestTests(unittest.TestCase):
    def test_first_loss_drawdown_includes_initial_capital(self):
        days = pd.date_range('2020-01-01', periods=2)
        t = pd.DataFrame({'date': days[:1], 'portfolio_pnl_cny': [-100.]})
        d = daily_equity(t, days, 1000)
        self.assertAlmostEqual(d.drawdown.iloc[0], -.1)
        self.assertEqual(d.equity_cny.iloc[-1], 900)

    def test_bps_and_both_slippage_sides(self):
        t = pd.DataFrame({'date': [pd.Timestamp('2020-01-02')], 'instrument': ['IF'],
                          'entry_open': [4000.], 'exit_close': [4000.]})
        r = add_trade_costs(t, slippage_ticks=1)
        self.assertAlmostEqual(r.slippage_cny.iloc[0], 120.)
        self.assertAlmostEqual(r.entry_fee_cny.iloc[0], 3999.8*300*.23/10000)
        self.assertAlmostEqual(r.exit_fee_cny.iloc[0], 4000.2*300*3.45/10000)
        self.assertLess(r.net_return.iloc[0], 0)
        self.assertEqual(fee_parameters('2020-01-02','IF','workbook_per_mille')[:2], (2.3,34.5))
        self.assertEqual(fee_parameters('2016-01-02','IF')[:2], (.23,23.))

    def test_previous_close_retains_mixed_day_and_missing_open_rejected(self):
        # Three otherwise complete modern days; middle day changes contract.
        frames=[]; bars=[]
        for k, day in enumerate(pd.date_range('2020-01-06',periods=3)):
            stamps = list(pd.date_range(day+pd.Timedelta(hours=9,minutes=31),periods=120,freq='min'))
            stamps += list(pd.date_range(day+pd.Timedelta(hours=13,minutes=1),periods=120,freq='min'))
            symbols=['NEW']*240
            if k==1: symbols[:120]=['OLD']*120
            frames.append(pd.DataFrame({'date':stamps,'symbol':symbols,'close':100.+k}))
            for j in range(16):
                part=stamps[j*15:(j+1)*15]
                bars.append(dict(start=part[0],end=part[-1],open=100.+k,close=100.+k,
                                 symbol=symbols[j*15],session_block=k*2+j//8))
        minute=pd.concat(frames,ignore_index=True)
        result,_=build_setups(minute,pd.DataFrame(bars),'IF')
        self.assertFalse(result.valid_setup.iloc[1])
        self.assertTrue(result.valid_setup.iloc[2])
        self.assertEqual(result.prev_day_close.iloc[2],101.)
        damaged=minute[minute.date.ne(frames[2].date.iloc[0])]
        result,_=build_setups(damaged,pd.DataFrame(bars),'IF')
        self.assertFalse(result.valid_setup.iloc[2])

    def test_training_threshold_ignores_future(self):
        setup=pd.DataFrame({'instrument':['IF']*4,'valid_setup':[True]*4,
                            'period':['train','train','test','test'],
                            'opening_move':[-.01,.02,-.10,-.20],
                            'opening_move_abs':[.01,.02,.10,.20],
                            'first_bar_end':pd.date_range('2020-01-01',periods=4)})
        _, a=select_opening(setup)
        setup.loc[2:,'opening_move_abs']=1000
        _, b=select_opening(setup)
        pd.testing.assert_frame_equal(a,b)

    def test_simulation_reconciles_and_enforces_margin_without_minimum_one(self):
        date=pd.Timestamp('2020-01-02')
        t=pd.DataFrame(dict(date=[date],instrument=['IF'],period=['test'],strategy=['opening'],
                            entry_time=[date+pd.Timedelta(hours=10)],
                            exit_time=[date+pd.Timedelta(hours=10,minutes=1)],
                            entry_open=[4000.],exit_close=[3990.],prior_daily_vol=[.01]))
        t=add_trade_costs(t)
        minute=pd.DataFrame({'date':[t.entry_time.iloc[0],t.exit_time.iloc[0]],
                             'close':[4002.,3990.],'symbol':['IF2001']*2})
        a,d,m,s=simulate_portfolio(t,{'IF':minute},[date],3_000_000)
        self.assertEqual(a.contracts.iloc[0],1)
        self.assertAlmostEqual(d.equity_cny.iloc[-1]-3_000_000,a.portfolio_pnl_cny.sum())
        self.assertAlmostEqual(m.equity_cny.iloc[-1],d.equity_cny.iloc[-1])
        self.assertLess(s.minute_close_max_drawdown_pct,0)
        a,d,m,s=simulate_portfolio(t,{'IF':minute},[date],1000)
        self.assertEqual(a.contracts.iloc[0],0)
        self.assertEqual(d.equity_cny.iloc[-1],1000)

    def test_simultaneous_positions_share_margin_and_entry_cost_budget(self):
        date=pd.Timestamp('2020-01-02')
        t=pd.DataFrame(dict(date=[date]*2,instrument=['IF','IH'],period=['test']*2,
                            strategy=['opening']*2,
                            entry_time=[date+pd.Timedelta(hours=10)]*2,
                            exit_time=[date+pd.Timedelta(hours=10,minutes=1)]*2,
                            entry_open=[4000.]*2,exit_close=[4000.]*2,prior_daily_vol=[.01]*2))
        t=add_trade_costs(t)
        minute=pd.DataFrame({'date':[t.entry_time.iloc[0],t.exit_time.iloc[0]],
                             'close':[4000.,4000.],'symbol':['contract']*2})
        a,d,m,s=simulate_portfolio(t,{'IF':minute,'IH':minute},[date],3_000_000,
                                   max_portfolio_margin_fraction=.20,fixed_contracts=20)
        self.assertEqual(a.contracts.sum(),4)
        actual_cost=4*(t.entry_fee_cny.iloc[0]+t.exit_fee_cny.iloc[0]+t.slippage_cny.iloc[0])+2
        self.assertAlmostEqual(a.portfolio_pnl_cny.sum(),-actual_cost)
        self.assertLessEqual(s.maximum_margin_usage_pct,20.)
        self.assertEqual(s.margin_cap_breach_events,0)


if __name__=='__main__':
    unittest.main()

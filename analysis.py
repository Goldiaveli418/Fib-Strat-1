import numpy as np

trades_raw = [
    (1,"Short",1518.07),(2,"Short",2902.79),(3,"Short",7733.67),(4,"Short",-6533.46),
    (5,"Short",-4054.97),(6,"Short",3162.96),(7,"Short",12882.74),(8,"Long",-9748.13),
    (9,"Long",-9012.74),(10,"Long",-1125.22),(11,"Long",-7391.52),(12,"Long",9926.70),
    (13,"Short",-205.60),(14,"Short",-11899.73),(15,"Short",-778.12),(16,"Short",-628.09),
    (17,"Long",1280.31),(18,"Short",3560.50),(19,"Long",2005.26),(20,"Long",-3733.33),
    (21,"Long",-1739.32),(22,"Short",5250.44),(23,"Long",3565.73),(24,"Short",-736.34),
    (25,"Short",1507.64),(26,"Long",-2746.79),(27,"Long",-2147.03),(28,"Long",-256.60),
    (29,"Long",632.34),(30,"Long",132.24),(31,"Long",-214.65),(32,"Short",-4407.84),
    (33,"Short",2757.25),(34,"Short",4266.95),(35,"Long",-1811.13),(36,"Long",-215.72),
    (37,"Long",4601.68),(38,"Short",6112.36),(39,"Long",-2585.06),(40,"Short",3131.14),
    (41,"Short",-216.96),(42,"Long",-5220.43),(43,"Short",10037.50),(44,"Long",-7759.08),
    (45,"Short",11452.74),(46,"Long",-3968.55),(47,"Long",-207.56),(48,"Long",2644.64),
    (49,"Short",-3851.58),(50,"Short",-4411.63),(51,"Short",2759.38),(52,"Long",5444.14),
    (53,"Short",-5147.57),(54,"Short",4139.11),(55,"Long",5619.79),(56,"Short",-5967.35),
    (57,"Short",10954.84),(58,"Long",-2030.70),(59,"Long",9439.92),(60,"Long",598.15),
    (61,"Short",15840.84),(62,"Long",9125.71),(63,"Short",5127.02),(64,"Long",-4686.15),
    (65,"Long",-4125.81),(66,"Long",-1729.83),(67,"Long",6080.02),(68,"Short",3110.08),
    (69,"Long",-5253.26),(70,"Short",-16049.33),(71,"Short",-2880.56),
]

pnls      = [t[2] for t in trades_raw]
dirs      = [t[1] for t in trades_raw]
short_pnl = [p for d, p in zip(dirs, pnls) if d == "Short"]
long_pnl  = [p for d, p in zip(dirs, pnls) if d == "Long"]

def stats(pnl_list, label):
    arr   = np.array(pnl_list)
    wins  = arr[arr > 0]
    losses= arr[arr < 0]
    wr    = len(wins) / len(arr) * 100
    pf    = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else float("inf")
    exp   = arr.mean()
    cum   = arr.cumsum()
    peak  = np.maximum.accumulate(cum)
    dd    = (cum - peak).min()
    print(f"\n{'='*50}")
    print(f"  {label}  ({len(arr)} trades)")
    print(f"{'='*50}")
    print(f"  Win rate       {wr:.1f}%  ({len(wins)}W / {len(losses)}L)")
    print(f"  Net P&L        ${arr.sum():,.2f}")
    print(f"  Avg trade      ${exp:,.2f}")
    print(f"  Avg win        ${wins.mean():,.2f}")
    print(f"  Avg loss       ${losses.mean():,.2f}")
    print(f"  Profit factor  {pf:.2f}")
    print(f"  Max drawdown   ${dd:,.2f}")
    print(f"  Max win        ${wins.max():,.2f}")
    print(f"  Max loss       ${losses.min():,.2f}")
    return arr

all_arr   = stats(pnls,      "ALL TRADES")
short_arr = stats(short_pnl, "SHORTS ONLY")
long_arr  = stats(long_pnl,  "LONGS ONLY")

# ── Drawdown profile ─────────────────────────────────────────────────────────
cum   = all_arr.cumsum()
peak  = np.maximum.accumulate(cum)
dd    = cum - peak
worst = np.argmin(dd)
print(f"\n{'='*50}")
print(f"  DRAWDOWN PROFILE")
print(f"{'='*50}")
print(f"  Max drawdown     ${dd.min():,.2f}  (after trade #{worst+1})")
print(f"  DD / Net P&L     {abs(dd.min())/all_arr.sum()*100:.1f}%")
print(f"  Recovery from worst DD: trade #{worst+1} → trade #{np.argmax(cum[worst:] >= peak[worst]) + worst + 1 if any(cum[worst:] >= peak[worst]) else 'not yet'}")

# ── Monte Carlo ───────────────────────────────────────────────────────────────
np.random.seed(42)
N_SIM = 10000
all_arr_copy = all_arr.copy()
sim_finals = []
sim_maxdds  = []
for _ in range(N_SIM):
    seq   = np.random.choice(all_arr_copy, size=len(all_arr_copy), replace=True)
    cum_s = seq.cumsum()
    pk_s  = np.maximum.accumulate(cum_s)
    sim_finals.append(cum_s[-1])
    sim_maxdds.append((cum_s - pk_s).min())

finals = np.array(sim_finals)
dds    = np.array(sim_maxdds)
print(f"\n{'='*50}")
print(f"  MONTE CARLO  ({N_SIM:,} simulations, 71-trade sequences)")
print(f"{'='*50}")
print(f"  Final P&L")
print(f"    5th  pct   ${np.percentile(finals, 5):,.0f}")
print(f"    25th pct   ${np.percentile(finals, 25):,.0f}")
print(f"    Median     ${np.percentile(finals, 50):,.0f}")
print(f"    75th pct   ${np.percentile(finals, 75):,.0f}")
print(f"    95th pct   ${np.percentile(finals, 95):,.0f}")
print(f"  % ending positive  {(finals > 0).mean()*100:.1f}%")
print(f"  Max Drawdown")
print(f"    Median     ${np.percentile(dds, 50):,.0f}")
print(f"    95th pct   ${np.percentile(dds, 5):,.0f}  (worst 5% of runs)")
print(f"  P(ruin > -$30k)    {(dds < -30000).mean()*100:.1f}%")
print(f"  P(ruin > -$50k)    {(dds < -50000).mean()*100:.1f}%")

# ── Streak analysis ───────────────────────────────────────────────────────────
win_loss = [1 if p > 0 else -1 for p in pnls]
max_win_streak = max_loss_streak = cur = 0
for wl in win_loss:
    cur = cur + 1 if wl > 0 else 0
    max_win_streak = max(max_win_streak, cur)
cur = 0
for wl in win_loss:
    cur = cur + 1 if wl < 0 else 0
    max_loss_streak = max(max_loss_streak, cur)
print(f"\n{'='*50}")
print(f"  STREAKS")
print(f"{'='*50}")
print(f"  Max win streak    {max_win_streak}")
print(f"  Max loss streak   {max_loss_streak}")

# ── Fixed fractional sizing (1% risk) ────────────────────────────────────────
# Assume starting equity $50k, risk 1% per trade = $500 initial risk
# Scale each trade P&L proportionally to current equity
eq = 50000.0
risk_pct = 0.01
base_risk = eq * risk_pct
ff_equity = [eq]
for p in pnls:
    scale   = ff_equity[-1] * risk_pct / base_risk
    ff_equity.append(ff_equity[-1] + p * scale)
ff_arr = np.diff(ff_equity)
ff_cum = np.array(ff_equity[1:])
ff_peak= np.maximum.accumulate(ff_cum)
ff_dd  = (ff_cum - ff_peak).min()
print(f"\n{'='*50}")
print(f"  FIXED FRACTIONAL SIZING  (1% risk, $50k start)")
print(f"{'='*50}")
print(f"  Final equity     ${ff_equity[-1]:,.2f}")
print(f"  Net P&L          ${ff_equity[-1]-50000:,.2f}")
print(f"  Max drawdown     ${ff_dd:,.2f}")
print(f"  Return           {(ff_equity[-1]/50000-1)*100:.1f}%")

# ── Long trade clustering ─────────────────────────────────────────────────────
long_trades = [(i+1, p) for i, (d, p) in enumerate(zip(dirs, pnls)) if d == "Long"]
consec_losses = []
run = []
for _, p in long_trades:
    if p < 0:
        run.append(p)
    else:
        if len(run) >= 3:
            consec_losses.append(run[:])
        run = []
if len(run) >= 3:
    consec_losses.append(run)

print(f"\n{'='*50}")
print(f"  LONG TRADE AUDIT")
print(f"{'='*50}")
print(f"  Total long trades  {len(long_pnl)}")
print(f"  Winning longs      {sum(1 for p in long_pnl if p > 0)}")
print(f"  Losing longs       {sum(1 for p in long_pnl if p < 0)}")
print(f"  Consecutive loss runs (3+): {len(consec_losses)}")
for r in consec_losses:
    print(f"    Run of {len(r)}: ${sum(r):,.0f} total loss")
print(f"\n  Long P&L by trade:")
for tn, p in long_trades:
    flag = " ◀ big loss" if p < -5000 else (" ◀ big win" if p > 5000 else "")
    print(f"    Trade #{tn:>2}  ${p:>10,.2f}{flag}")

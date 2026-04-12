-- Shadow Report: Full cost-aware performance analysis
-- Run against the quantagent PostgreSQL database.

-- 1. Trade P&L with cost breakdown
SELECT
  COUNT(*) as total_trades,
  ROUND(SUM(pnl)::numeric, 2) as net_pnl_after_costs,
  ROUND(SUM(COALESCE(raw_pnl, pnl))::numeric, 2) as gross_pnl,
  ROUND(SUM(COALESCE(trading_fee, 0))::numeric, 2) as total_fees,
  ROUND(SUM(COALESCE(funding_cost, 0))::numeric, 2) as total_funding_cost,
  ROUND(AVG(COALESCE(trading_fee, 0))::numeric, 4) as avg_fee_per_trade
FROM trades WHERE is_shadow = true AND status = 'closed';

-- 2. LLM cost summary
SELECT
  COUNT(*) as total_cycles,
  ROUND(SUM(COALESCE(llm_cost_usd, 0))::numeric, 4) as total_llm_cost,
  ROUND(AVG(COALESCE(llm_cost_usd, 0))::numeric, 4) as avg_cost_per_cycle,
  SUM(COALESCE(llm_input_tokens, 0)) as total_input_tokens,
  SUM(COALESCE(llm_output_tokens, 0)) as total_output_tokens,
  ROUND(AVG(COALESCE(duration_ms, 0))::numeric, 0) as avg_cycle_ms
FROM cycles WHERE is_shadow = true;

-- 3. True profitability
SELECT
  ROUND(COALESCE(SUM(t.pnl), 0)::numeric, 2) as trading_pnl_after_fees,
  ROUND(COALESCE(SUM(c_agg.llm_cost), 0)::numeric, 2) as total_llm_cost,
  ROUND(COALESCE(SUM(t.pnl), 0)::numeric - COALESCE(SUM(c_agg.llm_cost), 0)::numeric, 2) as true_net_pnl
FROM trades t
LEFT JOIN (
  SELECT bot_id, SUM(COALESCE(llm_cost_usd, 0)) as llm_cost
  FROM cycles WHERE is_shadow = true
  GROUP BY bot_id
) c_agg ON t.bot_id = c_agg.bot_id
WHERE t.is_shadow = true AND t.status = 'closed';

-- 4. Conviction calibration
SELECT
  CASE
    WHEN conviction_score >= 0.70 THEN 'HIGH (0.70+)'
    WHEN conviction_score >= 0.55 THEN 'MED (0.55-0.69)'
    ELSE 'LOW (0.50-0.54)'
  END as conviction_tier,
  COUNT(*) as trades,
  ROUND(COUNT(*) FILTER (WHERE pnl > 0)::numeric /
    NULLIF(COUNT(*), 0) * 100, 1) as win_rate,
  ROUND(AVG(pnl)::numeric, 2) as avg_pnl,
  ROUND(AVG(forward_max_r)::numeric, 2) as avg_max_r
FROM trades WHERE is_shadow = true AND status = 'closed'
GROUP BY 1 ORDER BY 1;

-- 5. Per-symbol performance
SELECT
  symbol,
  COUNT(*) as trades,
  COUNT(*) FILTER (WHERE pnl > 0) as wins,
  ROUND(SUM(pnl)::numeric, 2) as net_pnl,
  ROUND(SUM(COALESCE(trading_fee, 0))::numeric, 2) as total_fees,
  ROUND(AVG(conviction_score)::numeric, 2) as avg_conviction,
  ROUND(AVG(forward_max_r)::numeric, 2) as avg_max_r
FROM trades WHERE is_shadow = true AND status = 'closed'
GROUP BY symbol ORDER BY net_pnl DESC;

-- 6. Daily P&L curve
SELECT
  DATE(exit_time) as day,
  COUNT(*) as trades,
  ROUND(SUM(pnl)::numeric, 2) as daily_pnl,
  SUM(SUM(pnl)) OVER (ORDER BY DATE(exit_time))::numeric(10,2) as cumulative_pnl
FROM trades WHERE is_shadow = true AND status = 'closed'
GROUP BY DATE(exit_time) ORDER BY day;

-- 7. SL vs TP breakdown
SELECT
  exit_reason,
  COUNT(*) as count,
  ROUND(AVG(pnl)::numeric, 2) as avg_pnl,
  ROUND(AVG(COALESCE(trading_fee, 0))::numeric, 4) as avg_fee,
  ROUND(AVG(forward_max_r)::numeric, 2) as avg_max_r
FROM trades WHERE is_shadow = true AND status = 'closed'
GROUP BY exit_reason;

-- 8. Skip rate and conviction distribution
SELECT
  action,
  COUNT(*) as count,
  ROUND(AVG(conviction_score)::numeric, 2) as avg_conviction,
  ROUND(AVG(COALESCE(llm_cost_usd, 0))::numeric, 4) as avg_llm_cost
FROM cycles WHERE is_shadow = true
GROUP BY action ORDER BY count DESC;

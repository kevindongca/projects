# Kevin Dong — Quant & CS Projects

Personal repository of quantitative finance, algorithms, and SQL projects built during my CS + Statistics (Quantitative Finance) degree at UTSC.

---

## Projects

### Multi-Factor Quantitative Backtest (`python/`)

A factor-based equity backtest that selects and quarterly rebalances a portfolio using three systematic signals, benchmarked against SPY.

**Factors used:**
- **Momentum** — 12-month return excluding the most recent month. Stocks that trended up over the past year tend to continue.
- **Reversal** — Negative of 1-month return. Stocks that dropped recently tend to mean-revert.
- **Low Volatility** — Negative of 3-month rolling volatility. Less volatile stocks tend to outperform on a risk-adjusted basis.

**How it works:**
1. Downloads price data via `yfinance`
2. Computes and percentile-ranks all three signals cross-sectionally
3. Averages rankings into a composite score
4. Quarterly rebalances into the top 3 stocks by composite score
5. Simulates equal-weight portfolio returns and compares to SPY

**Tech stack:** Python, pandas, numpy, yfinance, matplotlib

**Sample results (2020–2026):**
- CAGR: ~14%
- Sharpe Ratio: ~0.65
- Max Drawdown: ~-38%

**To run:**
```bash
pip install pandas numpy yfinance matplotlib
python python/lesson_9_backtest_from_scratch.py
```

---

### LeetCode (`leetcode/`)

Python solutions to LeetCode problems, including both brute-force and optimized approaches where applicable.

| # | Problem | Approach | Time | Space |
|---|---------|----------|------|-------|
| 26 | Remove Duplicates from Sorted Array | Two Pointer | O(n) | O(1) |
| 35 | Search Insert Position | Binary Search (iterative) | O(log n) | O(1) |
| 35 | Search Insert Position | Binary Search (recursive) | O(log n) | O(log n) |

---

### HackerRank SQL (`sql/`)

MySQL solutions to HackerRank SQL challenges.

| Problem | Category | Concepts |
|---------|----------|----------|
| Weather Observation Station 18 | Aggregation | Manhattan distance, ABS |
| Weather Observation Station 19 | Aggregation | Euclidean distance, SQRT, POW |
| Weather Observation Station 20 | Aggregation | Median, correlated subquery |
| Placements | Advanced Join | Multi-table JOIN, salary comparison |
| The Report | Basic Join | CASE WHEN, BETWEEN, conditional ORDER BY |
| Top Competitors | Basic Join | Multi-table JOIN, GROUP BY, HAVING |

---

## About

**Kevin Dong** — First-year CS Co-op student at UTSC pursuing a CS Major and Statistics Specialist (Quantitative Finance stream). Targeting quantitative finance roles at top trading firms.

- GitHub: [kevindongca](https://github.com/kevindongca)
- Certifications: Bloomberg Market Concepts, Oracle Cloud Infrastructure Foundations

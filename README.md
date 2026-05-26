# Kevin Dong — Quant, ML & CS Projects

Personal repository of quantitative finance, machine learning, and software development projects built during my CS + Statistics (Quantitative Finance) degree at UTSC.

---

## Projects

### Multi-Factor Quantitative Backtest (`python/`)

A systematic, factor-based equity strategy that selects and quarterly rebalances a concentrated portfolio using 4 price-based signals, with minimum variance weighting via eigendecomposition. Validated using walk-forward testing with transaction costs.

**Factors used:**
- **Momentum** — 12-month return excluding the most recent month. Stocks that trended up over the past year tend to continue.
- **Reversal** — Negative of 1-month return. Stocks that dropped recently tend to mean-revert.
- **Low Volatility** — Negative of 3-month rolling volatility. Less volatile stocks tend to outperform on a risk-adjusted basis.
- **52-Week High** — Price relative to 52-week high. Stocks near their peak tend to break out and continue higher.

**How it works:**
1. Downloads price data via `yfinance`
2. Computes and percentile-ranks all 4 signals cross-sectionally
3. Averages rankings into a composite score
4. Quarterly rebalances into the top 3 stocks by composite score
5. Weights stocks using minimum variance portfolio (covariance matrix eigendecomposition)
6. Validates using walk-forward testing (2yr train / 6mo test windows)
7. Compares to SPY benchmark with full risk metric suite

**Tech stack:** Python, pandas, numpy, scipy, scikit-learn, yfinance, matplotlib

**Walk-forward results (2022–2026, out-of-sample, with transaction costs):**

| Metric | Strategy | SPY |
|--------|----------|-----|
| CAGR | 8.18% | 14.82% |
| Sharpe | 0.94 | 0.77 |
| Sortino | 1.33 | 0.95 |
| Calmar | 0.56 | 0.44 |
| Max Drawdown | -13.64% | -33.72% |
| VaR (95%) | -2.02% | -1.81% |

Strategy beats SPY on every risk-adjusted metric. Lower CAGR reflects conservative factor signals without leverage.

**To run:**
```bash
pip install pandas numpy scipy scikit-learn yfinance matplotlib seaborn
python python/lesson_23_final_strategy.py
```

---

### LeetCode (`leetcode/`)

Python solutions to LeetCode problems, including both brute-force and optimized approaches where applicable.

| # | Problem | Approach | Time | Space |
|---|---------|----------|------|-------|
| 26 | Remove Duplicates from Sorted Array | Two Pointer | O(n) | O(1) |
| 27 | Remove Element | Two Pointer | O(n) | O(1) |
| 28 | Find Index of First Occurrence | Built-in + Sliding Window | O(n) | O(1) |
| 35 | Search Insert Position | Binary Search (iterative) | O(log n) | O(1) |
| 35 | Search Insert Position | Binary Search (recursive) | O(log n) | O(log n) |
| 58 | Length of Last Word | One liner | O(n) | O(n) |
| 66 | Plus One | Reverse iteration with carry | O(n) | O(1) |
| 67 | Add Binary | Manual conversion + built-in | O(n) | O(n) |
| 69 | Sqrt(x) | Binary Search | O(log n) | O(1) |
| 70 | Climbing Stairs | Iterative Fibonacci (DP) | O(n) | O(1) |
| 83 | Remove Duplicates from Sorted List | Two Pointer (linked list) | O(n) | O(1) |
| 100 | Same Tree | Recursive DFS | O(n) | O(h) |
| 101 | Symmetric Tree | Recursive mirror checker | O(n) | O(h) |
| 104 | Maximum Depth of Binary Tree | Recursive DFS | O(n) | O(h) |
| 108 | Convert Sorted Array to BST | Divide and Conquer | O(n) | O(log n) |
| 110 | Balanced Binary Tree | DFS with -1 sentinel | O(n) | O(h) |
| 111 | Minimum Depth of Binary Tree | BFS level order | O(n) | O(n) |
| 112 | Path Sum | DFS running sum | O(n) | O(h) |

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

**Kevin Dong** — First-year CS Co-op student at UTSC pursuing a CS Specialist (AI & Machine Learning stream) and Statistics Major (Quantitative Finance stream). Interested in quantitative finance, algorithmic trading, machine learning, and AI-driven systems.

- GitHub: [kevindongca](https://github.com/kevindongca)
- Certifications: Bloomberg Market Concepts, Oracle Cloud Infrastructure Foundations & Generative AI

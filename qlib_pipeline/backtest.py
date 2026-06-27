# -*- coding: utf-8 -*-
"""
Qlib Backtest Results Analysis - backtest.py

Generate charts and reports from backtest results.
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _extract_series(df, key):
    """Extract a numeric series from DataFrame, handling MultiIndex."""
    if df is None or df.empty:
        return None
    cols = df.columns
    target_col = None
    if isinstance(cols, pd.MultiIndex):
        matches = [c for c in cols if c[1] == key]
        if matches:
            target_col = matches[0]
    else:
        if key in cols:
            target_col = key
    if target_col is None:
        return None
    s = df[target_col]
    # Flatten if needed
    if hasattr(s, "ndim") and s.ndim > 1:
        s = s.iloc[:, 0] if s.shape[1] == 1 else s.mean(axis=1)
    s = pd.to_numeric(s, errors="coerce").dropna()
    return s


def generate_report_charts(pred_df, report_normal_df, analysis_df,
                          positions_df=None, output_dir: str = "output/qlib_charts"):
    """
    Generate all analysis charts from backtest results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    plt.style.use('dark_background')
    plt.rcParams.update({
        'figure.facecolor': '#0f172a',
        'axes.facecolor': '#1e293b',
        'text.color': '#f1f5f9',
        'axes.labelcolor': '#f1f5f9',
        'xtick.color': '#94a3b8',
        'ytick.color': '#94a3b8',
        'axes.edgecolor': '#475569',
        'grid.color': '#334155',
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
    })

    charts = []

    # Chart 1: Qlib report_graph (if available, e.g. in Jupyter)
    try:
        from qlib.contrib.report import analysis_position
        fig = analysis_position.report_graph(report_normal_df)
        if fig is not None:
            out = output_path / "qlib_01_report.png"
            if _save_figure(fig, out):
                charts.append("qlib_01_report.png")
                logger.info("图表生成: qlib_01_report.png")
    except Exception as e:
        logger.debug("report_graph 不可用: %s", e)

    # Chart 2: Qlib risk_analysis_graph (if available)
    try:
        from qlib.contrib.report import analysis_position
        fig = analysis_position.risk_analysis_graph(analysis_df, report_normal_df)
        if fig is not None:
            out = output_path / "qlib_02_risk_analysis.png"
            if _save_figure(fig, out):
                charts.append("qlib_02_risk_analysis.png")
                logger.info("图表生成: qlib_02_risk_analysis.png")
    except Exception as e:
        logger.debug("risk_analysis_graph 不可用: %s", e)

    # Chart 3: Cumulative return (manual matplotlib)
    try:
        if report_normal_df is not None and not report_normal_df.empty:
            fig, ax = plt.subplots(figsize=(14, 7))

            ret = _extract_series(report_normal_df, "return")
            bench = _extract_series(report_normal_df, "bench")

            if ret is not None and len(ret) > 0:
                cum_ret = (1 + ret).cumprod()
                ax.plot(cum_ret.index, cum_ret.values, label="Strategy", color="steelblue", linewidth=1.5)
            if bench is not None and len(bench) > 0:
                cum_bench = (1 + bench).cumprod()
                ax.plot(cum_bench.index, cum_bench.values, label="Benchmark", color="orange",
                        linewidth=1.5, linestyle="--")
            ax.set_title("Cumulative Return", fontsize=14)
            ax.set_xlabel("Date", fontsize=12)
            ax.set_ylabel("Cumulative Return", fontsize=12)
            ax.legend(fontsize=12)
            ax.grid(True, alpha=0.3)
            fig.autofmt_xdate()
            fig.tight_layout()
            out = output_path / "qlib_03_cumulative_return.png"
            fig.savefig(out, dpi=150, bbox_inches='tight')
            plt.close(fig)
            charts.append("qlib_03_cumulative_return.png")
            logger.info("图表生成: qlib_03_cumulative_return.png")
    except Exception as e:
        logger.error("累计收益图失败: %s", e)

    # Chart 4: Prediction distribution
    try:
        if pred_df is not None and not pred_df.empty:
            fig, ax = plt.subplots(figsize=(12, 6))
            pred_values = _extract_series(pred_df, "score")
            if pred_values is None:
                # Fallback: use first numeric column
                for c in pred_df.columns:
                    candidate = _extract_series(pred_df, c if isinstance(c, str) else c[-1] if isinstance(c, tuple) else str(c))
                    if candidate is not None and len(candidate) > 0:
                        pred_values = candidate
                        break
            if pred_values is not None and len(pred_values) > 0:
                ax.hist(pred_values, bins=min(100, max(20, len(pred_values)//10)), color="steelblue",
                        edgecolor="white", alpha=0.8)
                ax.set_title("Prediction Distribution", fontsize=14)
                ax.set_xlabel("Score", fontsize=12)
                ax.set_ylabel("Count", fontsize=12)
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                out = output_path / "qlib_04_pred_distribution.png"
                fig.savefig(out, dpi=150, bbox_inches='tight')
                plt.close(fig)
                charts.append("qlib_04_pred_distribution.png")
                logger.info("图表生成: qlib_04_pred_distribution.png")
    except Exception as e:
        logger.error("预测分布图失败: %s", e)

    # Chart 5: Monthly returns heatmap
    try:
        if ret is not None and len(ret) > 0:
            monthly = ret.resample('ME').apply(lambda x: (1+x).prod()-1)
            if len(monthly) > 0:
                fig, ax = plt.subplots(figsize=(14, 6))
                monthly_df = monthly.to_frame(name="return")
                monthly_df["year"] = monthly_df.index.year
                monthly_df["month"] = monthly_df.index.month
                pivot = monthly_df.pivot(index="year", columns="month", values="return")
                im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto")
                ax.set_xticks(range(len(pivot.columns)))
                ax.set_xticklabels(pivot.columns)
                ax.set_yticks(range(len(pivot.index)))
                ax.set_yticklabels(pivot.index)
                for i in range(len(pivot.index)):
                    for j in range(len(pivot.columns)):
                        val = pivot.iloc[i, j]
                        if pd.notna(val):
                            ax.text(j, i, f"{val:.2%}", ha="center", va="center", color="black", fontsize=8)
                plt.colorbar(im, ax=ax)
                ax.set_title("Monthly Returns Heatmap", fontsize=14)
                fig.tight_layout()
                out = output_path / "qlib_05_monthly_heatmap.png"
                fig.savefig(out, dpi=150, bbox_inches='tight')
                plt.close(fig)
                charts.append("qlib_05_monthly_heatmap.png")
                logger.info("图表生成: qlib_05_monthly_heatmap.png")
    except Exception as e:
        logger.error("月度热力图失败: %s", e)

    logger.info("图表生成完成: %d 张", len(charts))
    return charts


def _save_figure(fig, output_path: Path, dpi: int = 150):
    """Save figure, auto-detecting Plotly or Matplotlib."""
    if fig is None:
        return False
    if hasattr(fig, "write_image"):
        fig.write_image(str(output_path), width=1280, height=720, scale=2)
        return True
    if hasattr(fig, "savefig"):
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        return True
    if isinstance(fig, dict) and 'application/vnd.plotly.v1+json' in fig:
        try:
            import plotly.graph_objects as go
            fig_data = fig['application/vnd.plotly.v1+json']
            plotly_fig = go.Figure(data=fig_data.get('data', []), layout=fig_data.get('layout', {}))
            plotly_fig.write_image(str(output_path), width=1280, height=720, scale=2)
            return True
        except Exception as e:
            logger.warning("Plotly dict 保存失败: %s", e)
    return False


def print_summary(report_normal_df, analysis_df):
    """Print backtest summary metrics."""
    try:
        print("\n" + "=" * 60)
        print("回测结果摘要")
        print("=" * 60)

        if report_normal_df is not None and not report_normal_df.empty:
            print("\n[每日收益统计]")
            for key in ["return", "bench", "turnover"]:
                s = _extract_series(report_normal_df, key)
                if s is not None and len(s) > 0:
                    print(f"  {key}: 日均={s.mean():.6f}, 年化={s.mean()*252:.4f}, 夏普={s.mean()/s.std()*np.sqrt(252):.4f}")

        if analysis_df is not None and not analysis_df.empty:
            print("\n[绩效指标]")
            cols = analysis_df.columns
            if isinstance(cols, pd.MultiIndex):
                for col in cols:
                    val = analysis_df[col].iloc[-1] if len(analysis_df) > 0 else None
                    if val is not None and pd.notna(val):
                        col_name = "_".join(str(c) for c in col if c)
                        print(f"  {col_name}: {val:.4f}")
            else:
                for col in cols:
                    val = analysis_df[col].iloc[-1] if len(analysis_df) > 0 else None
                    if val is not None and pd.notna(val):
                        print(f"  {col}: {val:.4f}")
    except Exception as e:
        logger.error("打印摘要失败: %s", e)
# -*-# -*- coding: utf-8 -*-
"""
Qlib Backtest Results Analysis - back# -*- coding: utf-8 -*-
"""
Qlib Backtest Results Analysis - backtest.py

Generate charts and reports from back# -*- coding: utf-8 -*-
"""
Qlib Backtest Results Analysis - backtest.py

Generate charts and reports from backtest results using Plotly (avoids matplotlib/n# -*- coding: utf-8 -*-
"""
Qlib Backtest Results Analysis - backtest.py

Generate charts and reports from backtest results using Plotly (avoids matplotlib/numpy2 compat issues).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
# -*- coding: utf-8 -*-
"""
Qlib Backtest Results Analysis - backtest.py

Generate charts and reports from backtest results using Plotly (avoids matplotlib/numpy2 compat issues).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _extract_series(df, key# -*- coding: utf-8 -*-
"""
Qlib Backtest Results Analysis - backtest.py

Generate charts and reports from backtest results using Plotly (avoids matplotlib/numpy2 compat issues).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _extract_series(df, key):
    """Extract a numeric series from DataFrame, handling MultiIndex."""
    if df is None or df.empty:
        return None
    cols = df.columns# -*- coding: utf-8 -*-
"""
Qlib Backtest Results Analysis - backtest.py

Generate charts and reports from backtest results using Plotly (avoids matplotlib/numpy2 compat issues).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _extract_series(df, key):
    """Extract a numeric series from DataFrame, handling MultiIndex."""
    if df is None or df.empty:
        return None
    cols = df.columns
    target_col = None
    if isinstance(cols, pd.MultiIndex):
        matches# -*- coding: utf-8 -*-
"""
Qlib Backtest Results Analysis - backtest.py

Generate charts and reports from backtest results using Plotly (avoids matplotlib/numpy2 compat issues).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _extract_series(df, key):
    """Extract a numeric series from DataFrame, handling MultiIndex."""
    if df is None or df.empty:
        return None
    cols = df.columns
    target_col = None
    if isinstance(cols, pd.MultiIndex):
        matches = [c for c in cols if c[1] == key]
        if matches:
            target_col = matches[0]
# -*- coding: utf-8 -*-
"""
Qlib Backtest Results Analysis - backtest.py

Generate charts and reports from backtest results using Plotly (avoids matplotlib/numpy2 compat issues).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _extract_series(df, key):
    """Extract a numeric series from DataFrame, handling MultiIndex."""
    if df is None or df.empty:
        return None
    cols = df.columns
    target_col = None
    if isinstance(cols, pd.MultiIndex):
        matches = [c for c in cols if c[1] == key]
        if matches:
            target_col = matches[0]
    else:
        if key in cols:
            target_col = key
    if target# -*- coding: utf-8 -*-
"""
Qlib Backtest Results Analysis - backtest.py

Generate charts and reports from backtest results using Plotly (avoids matplotlib/numpy2 compat issues).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _extract_series(df, key):
    """Extract a numeric series from DataFrame, handling MultiIndex."""
    if df is None or df.empty:
        return None
    cols = df.columns
    target_col = None
    if isinstance(cols, pd.MultiIndex):
        matches = [c for c in cols if c[1] == key]
        if matches:
            target_col = matches[0]
    else:
        if key in cols:
            target_col = key
    if target_col is None:
        return None
    s = df[target_col]
    if hasattr# -*- coding: utf-8 -*-
"""
Qlib Backtest Results Analysis - backtest.py

Generate charts and reports from backtest results using Plotly (avoids matplotlib/numpy2 compat issues).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _extract_series(df, key):
    """Extract a numeric series from DataFrame, handling MultiIndex."""
    if df is None or df.empty:
        return None
    cols = df.columns
    target_col = None
    if isinstance(cols, pd.MultiIndex):
        matches = [c for c in cols if c[1] == key]
        if matches:
            target_col = matches[0]
    else:
        if key in cols:
            target_col = key
    if target_col is None:
        return None
    s = df[target_col]
    if hasattr(s, "ndim") and s.ndim > 1:
        s = s.iloc[:,# -*- coding: utf-8 -*-
"""
Qlib Backtest Results Analysis - backtest.py

Generate charts and reports from backtest results using Plotly (avoids matplotlib/numpy2 compat issues).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _extract_series(df, key):
    """Extract a numeric series from DataFrame, handling MultiIndex."""
    if df is None or df.empty:
        return None
    cols = df.columns
    target_col = None
    if isinstance(cols, pd.MultiIndex):
        matches = [c for c in cols if c[1] == key]
        if matches:
            target_col = matches[0]
    else:
        if key in cols:
            target_col = key
    if target_col is None:
        return None
    s = df[target_col]
    if hasattr(s, "ndim") and s.ndim > 1:
        s = s.iloc[:, 0] if s.shape[1] == 1 else s.mean(axis=1)
# -*- coding: utf-8 -*-
"""
Qlib Backtest Results Analysis - backtest.py

Generate charts and reports from backtest results using Plotly (avoids matplotlib/numpy2 compat issues).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _extract_series(df, key):
    """Extract a numeric series from DataFrame, handling MultiIndex."""
    if df is None or df.empty:
        return None
    cols = df.columns
    target_col = None
    if isinstance(cols, pd.MultiIndex):
        matches = [c for c in cols if c[1] == key]
        if matches:
            target_col = matches[0]
    else:
        if key in cols:
            target_col = key
    if target_col is None:
        return None
    s = df[target_col]
    if hasattr(s, "ndim") and s.ndim > 1:
        s = s.iloc[:, 0] if s.shape[1] == 1 else s.mean(axis=1)
    s = pd.to_numeric(s, errors="coerce").dropna()
    return# -*- coding: utf-8 -*-
"""
Qlib Backtest Results Analysis - backtest.py

Generate charts and reports from backtest results using Plotly (avoids matplotlib/numpy2 compat issues).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _extract_series(df, key):
    """Extract a numeric series from DataFrame, handling MultiIndex."""
    if df is None or df.empty:
        return None
    cols = df.columns
    target_col = None
    if isinstance(cols, pd.MultiIndex):
        matches = [c for c in cols if c[1] == key]
        if matches:
            target_col = matches[0]
    else:
        if key in cols:
            target_col = key
    if target_col is None:
        return None
    s = df[target_col]
    if hasattr(s, "ndim") and s.ndim > 1:
        s = s.iloc[:, 0] if s.shape[1] == 1 else s.mean(axis=1)
    s = pd.to_numeric(s, errors="coerce").dropna()
    return s


def _to_list(s):
    """Safely convert series to Python# -*- coding: utf-8 -*-
"""
Qlib Backtest Results Analysis - backtest.py

Generate charts and reports from backtest results using Plotly (avoids matplotlib/numpy2 compat issues).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _extract_series(df, key):
    """Extract a numeric series from DataFrame, handling MultiIndex."""
    if df is None or df.empty:
        return None
    cols = df.columns
    target_col = None
    if isinstance(cols, pd.MultiIndex):
        matches = [c for c in cols if c[1] == key]
        if matches:
            target_col = matches[0]
    else:
        if key in cols:
            target_col = key
    if target_col is None:
        return None
    s = df[target_col]
    if hasattr(s, "ndim") and s.ndim > 1:
        s = s.iloc[:, 0] if s.shape[1] == 1 else s.mean(axis=1)
    s = pd.to_numeric(s, errors="coerce").dropna()
    return s


def _to_list(s):
    """Safely convert series to Python list for Plotly."""
    if s is None:
        return []
# -*- coding: utf-8 -*-
"""
Qlib Backtest Results Analysis - backtest.py

Generate charts and reports from backtest results using Plotly (avoids matplotlib/numpy2 compat issues).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _extract_series(df, key):
    """Extract a numeric series from DataFrame, handling MultiIndex."""
    if df is None or df.empty:
        return None
    cols = df.columns
    target_col = None
    if isinstance(cols, pd.MultiIndex):
        matches = [c for c in cols if c[1] == key]
        if matches:
            target_col = matches[0]
    else:
        if key in cols:
            target_col = key
    if target_col is None:
        return None
    s = df[target_col]
    if hasattr(s, "ndim") and s.ndim > 1:
        s = s.iloc[:, 0] if s.shape[1] == 1 else s.mean(axis=1)
    s = pd.to_numeric(s, errors="coerce").dropna()
    return s


def _to_list(s):
    """Safely convert series to Python list for Plotly."""
    if s is None:
        return []
    return [float(v) for v in s.values]


def _save_plotly# -*- coding: utf-8 -*-
"""
Qlib Backtest Results Analysis - backtest.py

Generate charts and reports from backtest results using Plotly (avoids matplotlib/numpy2 compat issues).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _extract_series(df, key):
    """Extract a numeric series from DataFrame, handling MultiIndex."""
    if df is None or df.empty:
        return None
    cols = df.columns
    target_col = None
    if isinstance(cols, pd.MultiIndex):
        matches = [c for c in cols if c[1] == key]
        if matches:
            target_col = matches[0]
    else:
        if key in cols:
            target_col = key
    if target_col is None:
        return None
    s = df[target_col]
    if hasattr(s, "ndim") and s.ndim > 1:
        s = s.iloc[:, 0] if s.shape[1] == 1 else s.mean(axis=1)
    s = pd.to_numeric(s, errors="coerce").dropna()
    return s


def _to_list(s):
    """Safely convert series to Python list for Plotly."""
    if s is None:
        return []
    return [float(v) for v in s.values]


def _save_plotly_fig(fig, output_path: Path):
    """Save Plotly figure to PNG.# -*- coding: utf-8 -*-
"""
Qlib Backtest Results Analysis - backtest.py

Generate charts and reports from backtest results using Plotly (avoids matplotlib/numpy2 compat issues).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _extract_series(df, key):
    """Extract a numeric series from DataFrame, handling MultiIndex."""
    if df is None or df.empty:
        return None
    cols = df.columns
    target_col = None
    if isinstance(cols, pd.MultiIndex):
        matches = [c for c in cols if c[1] == key]
        if matches:
            target_col = matches[0]
    else:
        if key in cols:
            target_col = key
    if target_col is None:
        return None
    s = df[target_col]
    if hasattr(s, "ndim") and s.ndim > 1:
        s = s.iloc[:, 0] if s.shape[1] == 1 else s.mean(axis=1)
    s = pd.to_numeric(s, errors="coerce").dropna()
    return s


def _to_list(s):
    """Safely convert series to Python list for Plotly."""
    if s is None:
        return []
    return [float(v) for v in s.values]


def _save_plotly_fig(fig, output_path: Path):
    """Save Plotly figure to PNG."""
    try:
        fig.write_image(str(output_path), width=1280# -*- coding: utf-8 -*-
"""
Qlib Backtest Results Analysis - backtest.py

Generate charts and reports from backtest results using Plotly (avoids matplotlib/numpy2 compat issues).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _extract_series(df, key):
    """Extract a numeric series from DataFrame, handling MultiIndex."""
    if df is None or df.empty:
        return None
    cols = df.columns
    target_col = None
    if isinstance(cols, pd.MultiIndex):
        matches = [c for c in cols if c[1] == key]
        if matches:
            target_col = matches[0]
    else:
        if key in cols:
            target_col = key
    if target_col is None:
        return None
    s = df[target_col]
    if hasattr(s, "ndim") and s.ndim > 1:
        s = s.iloc[:, 0] if s.shape[1] == 1 else s.mean(axis=1)
    s = pd.to_numeric(s, errors="coerce").dropna()
    return s


def _to_list(s):
    """Safely convert series to Python list for Plotly."""
    if s is None:
        return []
    return [float(v) for v in s.values]


def _save_plotly_fig(fig, output_path: Path):
    """Save Plotly figure to PNG."""
    try:
        fig.write_image(str(output_path), width=1280, height=720, scale=2)
# -*- coding: utf-8 -*-
"""
Qlib Backtest Results Analysis - backtest.py

Generate charts and reports from backtest results using Plotly (avoids matplotlib/numpy2 compat issues).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _extract_series(df, key):
    """Extract a numeric series from DataFrame, handling MultiIndex."""
    if df is None or df.empty:
        return None
    cols = df.columns
    target_col = None
    if isinstance(cols, pd.MultiIndex):
        matches = [c for c in cols if c[1] == key]
        if matches:
            target_col = matches[0]
    else:
        if key in cols:
            target_col = key
    if target_col is None:
        return None
    s = df[target_col]
    if hasattr(s, "ndim") and s.ndim > 1:
        s = s.iloc[:, 0] if s.shape[1] == 1 else s.mean(axis=1)
    s = pd.to_numeric(s, errors="coerce").dropna()
    return s


def _to_list(s):
    """Safely convert series to Python list for Plotly."""
    if s is None:
        return []
    return [float(v) for v in s.values]


def _save_plotly_fig(fig, output_path: Path):
    """Save Plotly figure to PNG."""
    try:
        fig.write_image(str(output_path), width=1280, height=720, scale=2)
        return True
    except Exception as e:
        logger.warning("Plotly 保存失败# -*- coding: utf-8 -*-
"""
Qlib Backtest Results Analysis - backtest.py

Generate charts and reports from backtest results using Plotly (avoids matplotlib/numpy2 compat issues).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _extract_series(df, key):
    """Extract a numeric series from DataFrame, handling MultiIndex."""
    if df is None or df.empty:
        return None
    cols = df.columns
    target_col = None
    if isinstance(cols, pd.MultiIndex):
        matches = [c for c in cols if c[1] == key]
        if matches:
            target_col = matches[0]
    else:
        if key in cols:
            target_col = key
    if target_col is None:
        return None
    s = df[target_col]
    if hasattr(s, "ndim") and s.ndim > 1:
        s = s.iloc[:, 0] if s.shape[1] == 1 else s.mean(axis=1)
    s = pd.to_numeric(s, errors="coerce").dropna()
    return s


def _to_list(s):
    """Safely convert series to Python list for Plotly."""
    if s is None:
        return []
    return [float(v) for v in s.values]


def _save_plotly_fig(fig, output_path: Path):
    """Save Plotly figure to PNG."""
    try:
        fig.write_image(str(output_path), width=1280, height=720, scale=2)
        return True
    except Exception as e:
        logger.warning("Plotly 保存失败: %s", e)
        return False


def generate_report_charts(pred_df# -*- coding: utf-8 -*-
"""
Qlib Backtest Results Analysis - backtest.py

Generate charts and reports from backtest results using Plotly (avoids matplotlib/numpy2 compat issues).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _extract_series(df, key):
    """Extract a numeric series from DataFrame, handling MultiIndex."""
    if df is None or df.empty:
        return None
    cols = df.columns
    target_col = None
    if isinstance(cols, pd.MultiIndex):
        matches = [c for c in cols if c[1] == key]
        if matches:
            target_col = matches[0]
    else:
        if key in cols:
            target_col = key
    if target_col is None:
        return None
    s = df[target_col]
    if hasattr(s, "ndim") and s.ndim > 1:
        s = s.iloc[:, 0] if s.shape[1] == 1 else s.mean(axis=1)
    s = pd.to_numeric(s, errors="coerce").dropna()
    return s


def _to_list(s):
    """Safely convert series to Python list for Plotly."""
    if s is None:
        return []
    return [float(v) for v in s.values]


def _save_plotly_fig(fig, output_path: Path):
    """Save Plotly figure to PNG."""
    try:
        fig.write_image(str(output_path), width=1280, height=720, scale=2)
        return True
    except Exception as e:
        logger.warning("Plotly 保存失败: %s", e)
        return False


def generate_report_charts(pred_df, report_normal_df, analysis_df,
                          positions_df=None, output_dir: str = "output/qlib_charts"):
    """
    Generate all analysis charts from back# -*- coding: utf-8 -*-
"""
Qlib Backtest Results Analysis - backtest.py

Generate charts and reports from backtest results using Plotly (avoids matplotlib/numpy2 compat issues).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _extract_series(df, key):
    """Extract a numeric series from DataFrame, handling MultiIndex."""
    if df is None or df.empty:
        return None
    cols = df.columns
    target_col = None
    if isinstance(cols, pd.MultiIndex):
        matches = [c for c in cols if c[1] == key]
        if matches:
            target_col = matches[0]
    else:
        if key in cols:
            target_col = key
    if target_col is None:
        return None
    s = df[target_col]
    if hasattr(s, "ndim") and s.ndim > 1:
        s = s.iloc[:, 0] if s.shape[1] == 1 else s.mean(axis=1)
    s = pd.to_numeric(s, errors="coerce").dropna()
    return s


def _to_list(s):
    """Safely convert series to Python list for Plotly."""
    if s is None:
        return []
    return [float(v) for v in s.values]


def _save_plotly_fig(fig, output_path: Path):
    """Save Plotly figure to PNG."""
    try:
        fig.write_image(str(output_path), width=1280, height=720, scale=2)
        return True
    except Exception as e:
        logger.warning("Plotly 保存失败: %s", e)
        return False


def generate_report_charts(pred_df, report_normal_df, analysis_df,
                          positions_df=None, output_dir: str = "output/qlib_charts"):
    """
    Generate all analysis charts from backtest results using Plotly.
    """
    output_path = Path(output_dir)
# -*- coding: utf-8 -*-
"""
Qlib Backtest Results Analysis - backtest.py

Generate charts and reports from backtest results using Plotly (avoids matplotlib/numpy2 compat issues).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _extract_series(df, key):
    """Extract a numeric series from DataFrame, handling MultiIndex."""
    if df is None or df.empty:
        return None
    cols = df.columns
    target_col = None
    if isinstance(cols, pd.MultiIndex):
        matches = [c for c in cols if c[1] == key]
        if matches:
            target_col = matches[0]
    else:
        if key in cols:
            target_col = key
    if target_col is None:
        return None
    s = df[target_col]
    if hasattr(s, "ndim") and s.ndim > 1:
        s = s.iloc[:, 0] if s.shape[1] == 1 else s.mean(axis=1)
    s = pd.to_numeric(s, errors="coerce").dropna()
    return s


def _to_list(s):
    """Safely convert series to Python list for Plotly."""
    if s is None:
        return []
    return [float(v) for v in s.values]


def _save_plotly_fig(fig, output_path: Path):
    """Save Plotly figure to PNG."""
    try:
        fig.write_image(str(output_path), width=1280, height=720, scale=2)
        return True
    except Exception as e:
        logger.warning("Plotly 保存失败: %s", e)
        return False


def generate_report_charts(pred_df, report_normal_df, analysis_df,
                          positions_df=None, output_dir: str = "output/qlib_charts"):
    """
    Generate all analysis charts from backtest results using Plotly.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        import# -*- coding: utf-8 -*-
"""
Qlib Backtest Results Analysis - backtest.py

Generate charts and reports from backtest results using Plotly (avoids matplotlib/numpy2 compat issues).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _extract_series(df, key):
    """Extract a numeric series from DataFrame, handling MultiIndex."""
    if df is None or df.empty:
        return None
    cols = df.columns
    target_col = None
    if isinstance(cols, pd.MultiIndex):
        matches = [c for c in cols if c[1] == key]
        if matches:
            target_col = matches[0]
    else:
        if key in cols:
            target_col = key
    if target_col is None:
        return None
    s = df[target_col]
    if hasattr(s, "ndim") and s.ndim > 1:
        s = s.iloc[:, 0] if s.shape[1] == 1 else s.mean(axis=1)
    s = pd.to_numeric(s, errors="coerce").dropna()
    return s


def _to_list(s):
    """Safely convert series to Python list for Plotly."""
    if s is None:
        return []
    return [float(v) for v in s.values]


def _save_plotly_fig(fig, output_path: Path):
    """Save Plotly figure to PNG."""
    try:
        fig.write_image(str(output_path), width=1280, height=720, scale=2)
        return True
    except Exception as e:
        logger.warning("Plotly 保存失败: %s", e)
        return False


def generate_report_charts(pred_df, report_normal_df, analysis_df,
                          positions_df=None, output_dir: str = "output/qlib_charts"):
    """
    Generate all analysis charts from backtest results using Plotly.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Import# -*- coding: utf-8 -*-
"""
Qlib Backtest Results Analysis - backtest.py

Generate charts and reports from backtest results using Plotly (avoids matplotlib/numpy2 compat issues).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _extract_series(df, key):
    """Extract a numeric series from DataFrame, handling MultiIndex."""
    if df is None or df.empty:
        return None
    cols = df.columns
    target_col = None
    if isinstance(cols, pd.MultiIndex):
        matches = [c for c in cols if c[1] == key]
        if matches:
            target_col = matches[0]
    else:
        if key in cols:
            target_col = key
    if target_col is None:
        return None
    s = df[target_col]
    if hasattr(s, "ndim") and s.ndim > 1:
        s = s.iloc[:, 0] if s.shape[1] == 1 else s.mean(axis=1)
    s = pd.to_numeric(s, errors="coerce").dropna()
    return s


def _to_list(s):
    """Safely convert series to Python list for Plotly."""
    if s is None:
        return []
    return [float(v) for v in s.values]


def _save_plotly_fig(fig, output_path: Path):
    """Save Plotly figure to PNG."""
    try:
        fig.write_image(str(output_path), width=1280, height=720, scale=2)
        return True
    except Exception as e:
        logger.warning("Plotly 保存失败: %s", e)
        return False


def generate_report_charts(pred_df, report_normal_df, analysis_df,
                          positions_df=None, output_dir: str = "output/qlib_charts"):
    """
    Generate all analysis charts from backtest results using Plotly.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        logger.error("Plotly 未安装，无法生成图表。请运行: pip install plotly kaleido")
        return# -*- coding: utf-8 -*-
"""
Qlib Backtest Results Analysis - backtest.py

Generate charts and reports from backtest results using Plotly (avoids matplotlib/numpy2 compat issues).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _extract_series(df, key):
    """Extract a numeric series from DataFrame, handling MultiIndex."""
    if df is None or df.empty:
        return None
    cols = df.columns
    target_col = None
    if isinstance(cols, pd.MultiIndex):
        matches = [c for c in cols if c[1] == key]
        if matches:
            target_col = matches[0]
    else:
        if key in cols:
            target_col = key
    if target_col is None:
        return None
    s = df[target_col]
    if hasattr(s, "ndim") and s.ndim > 1:
        s = s.iloc[:, 0] if s.shape[1] == 1 else s.mean(axis=1)
    s = pd.to_numeric(s, errors="coerce").dropna()
    return s


def _to_list(s):
    """Safely convert series to Python list for Plotly."""
    if s is None:
        return []
    return [float(v) for v in s.values]


def _save_plotly_fig(fig, output_path: Path):
    """Save Plotly figure to PNG."""
    try:
        fig.write_image(str(output_path), width=1280, height=720, scale=2)
        return True
    except Exception as e:
        logger.warning("Plotly 保存失败: %s", e)
        return False


def generate_report_charts(pred_df, report_normal_df, analysis_df,
                          positions_df=None, output_dir: str = "output/qlib_charts"):
    """
    Generate all analysis charts from backtest results using Plotly.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        logger.error("Plotly 未安装，无法生成图表。请运行: pip install plotly kaleido")
        return []

    charts = []
    ret = _extract_series(report_normal_df, "return# -*- coding: utf-8 -*-
"""
Qlib Backtest Results Analysis - backtest.py

Generate charts and reports from backtest results using Plotly (avoids matplotlib/numpy2 compat issues).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _extract_series(df, key):
    """Extract a numeric series from DataFrame, handling MultiIndex."""
    if df is None or df.empty:
        return None
    cols = df.columns
    target_col = None
    if isinstance(cols, pd.MultiIndex):
        matches = [c for c in cols if c[1] == key]
        if matches:
            target_col = matches[0]
    else:
        if key in cols:
            target_col = key
    if target_col is None:
        return None
    s = df[target_col]
    if hasattr(s, "ndim") and s.ndim > 1:
        s = s.iloc[:, 0] if s.shape[1] == 1 else s.mean(axis=1)
    s = pd.to_numeric(s, errors="coerce").dropna()
    return s


def _to_list(s):
    """Safely convert series to Python list for Plotly."""
    if s is None:
        return []
    return [float(v) for v in s.values]


def _save_plotly_fig(fig, output_path: Path):
    """Save Plotly figure to PNG."""
    try:
        fig.write_image(str(output_path), width=1280, height=720, scale=2)
        return True
    except Exception as e:
        logger.warning("Plotly 保存失败: %s", e)
        return False


def generate_report_charts(pred_df, report_normal_df, analysis_df,
                          positions_df=None, output_dir: str = "output/qlib_charts"):
    """
    Generate all analysis charts from backtest results using Plotly.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        logger.error("Plotly 未安装，无法生成图表。请运行: pip install plotly kaleido")
        return []

    charts = []
    ret = _extract_series(report_normal_df, "return")
    bench = _extract_series(report_normal_df, "bench")

    # Chart# -*- coding: utf-8 -*-
"""
Qlib Backtest Results Analysis - backtest.py

Generate charts and reports from backtest results using Plotly (avoids matplotlib/numpy2 compat issues).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _extract_series(df, key):
    """Extract a numeric series from DataFrame, handling MultiIndex."""
    if df is None or df.empty:
        return None
    cols = df.columns
    target_col = None
    if isinstance(cols, pd.MultiIndex):
        matches = [c for c in cols if c[1] == key]
        if matches:
            target_col = matches[0]
    else:
        if key in cols:
            target_col = key
    if target_col is None:
        return None
    s = df[target_col]
    if hasattr(s, "ndim") and s.ndim > 1:
        s = s.iloc[:, 0] if s.shape[1] == 1 else s.mean(axis=1)
    s = pd.to_numeric(s, errors="coerce").dropna()
    return s


def _to_list(s):
    """Safely convert series to Python list for Plotly."""
    if s is None:
        return []
    return [float(v) for v in s.values]


def _save_plotly_fig(fig, output_path: Path):
    """Save Plotly figure to PNG."""
    try:
        fig.write_image(str(output_path), width=1280, height=720, scale=2)
        return True
    except Exception as e:
        logger.warning("Plotly 保存失败: %s", e)
        return False


def generate_report_charts(pred_df, report_normal_df, analysis_df,
                          positions_df=None, output_dir: str = "output/qlib_charts"):
    """
    Generate all analysis charts from backtest results using Plotly.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        logger.error("Plotly 未安装，无法生成图表。请运行: pip install plotly kaleido")
        return []

    charts = []
    ret = _extract_series(report_normal_df, "return")
    bench = _extract_series(report_normal_df, "bench")

    # Chart 1: Cumulative return
    try:
        if ret is not None and len(ret# -*- coding: utf-8 -*-
"""
Qlib Backtest Results Analysis - backtest.py

Generate charts and reports from backtest results using Plotly (avoids matplotlib/numpy2 compat issues).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _extract_series(df, key):
    """Extract a numeric series from DataFrame, handling MultiIndex."""
    if df is None or df.empty:
        return None
    cols = df.columns
    target_col = None
    if isinstance(cols, pd.MultiIndex):
        matches = [c for c in cols if c[1] == key]
        if matches:
            target_col = matches[0]
    else:
        if key in cols:
            target_col = key
    if target_col is None:
        return None
    s = df[target_col]
    if hasattr(s, "ndim") and s.ndim > 1:
        s = s.iloc[:, 0] if s.shape[1] == 1 else s.mean(axis=1)
    s = pd.to_numeric(s, errors="coerce").dropna()
    return s


def _to_list(s):
    """Safely convert series to Python list for Plotly."""
    if s is None:
        return []
    return [float(v) for v in s.values]


def _save_plotly_fig(fig, output_path: Path):
    """Save Plotly figure to PNG."""
    try:
        fig.write_image(str(output_path), width=1280, height=720, scale=2)
        return True
    except Exception as e:
        logger.warning("Plotly 保存失败: %s", e)
        return False


def generate_report_charts(pred_df, report_normal_df, analysis_df,
                          positions_df=None, output_dir: str = "output/qlib_charts"):
    """
    Generate all analysis charts from backtest results using Plotly.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        logger.error("Plotly 未安装，无法生成图表。请运行: pip install plotly kaleido")
        return []

    charts = []
    ret = _extract_series(report_normal_df, "return")
    bench = _extract_series(report_normal_df, "bench")

    # Chart 1: Cumulative return
    try:
        if ret is not None and len(ret) > 0:
            fig = go.Figure()
            cum_ret = (1# -*- coding: utf-8 -*-
"""
Qlib Backtest Results Analysis - backtest.py

Generate charts and reports from backtest results using Plotly (avoids matplotlib/numpy2 compat issues).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _extract_series(df, key):
    """Extract a numeric series from DataFrame, handling MultiIndex."""
    if df is None or df.empty:
        return None
    cols = df.columns
    target_col = None
    if isinstance(cols, pd.MultiIndex):
        matches = [c for c in cols if c[1] == key]
        if matches:
            target_col = matches[0]
    else:
        if key in cols:
            target_col = key
    if target_col is None:
        return None
    s = df[target_col]
    if hasattr(s, "ndim") and s.ndim > 1:
        s = s.iloc[:, 0] if s.shape[1] == 1 else s.mean(axis=1)
    s = pd.to_numeric(s, errors="coerce").dropna()
    return s


def _to_list(s):
    """Safely convert series to Python list for Plotly."""
    if s is None:
        return []
    return [float(v) for v in s.values]


def _save_plotly_fig(fig, output_path: Path):
    """Save Plotly figure to PNG."""
    try:
        fig.write_image(str(output_path), width=1280, height=720, scale=2)
        return True
    except Exception as e:
        logger.warning("Plotly 保存失败: %s", e)
        return False


def generate_report_charts(pred_df, report_normal_df, analysis_df,
                          positions_df=None, output_dir: str = "output/qlib_charts"):
    """
    Generate all analysis charts from backtest results using Plotly.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        logger.error("Plotly 未安装，无法生成图表。请运行: pip install plotly kaleido")
        return []

    charts = []
    ret = _extract_series(report_normal_df, "return")
    bench = _extract_series(report_normal_df, "bench")

    # Chart 1: Cumulative return
    try:
        if ret is not None and len(ret) > 0:
            fig = go.Figure()
            cum_ret = (1 + ret).cumprod()
            fig.add_trace(go.Scatter(
                x=list(cum_ret.index.astype(str)),
                y# -*- coding: utf-8 -*-
"""
Qlib Backtest Results Analysis - backtest.py

Generate charts and reports from backtest results using Plotly (avoids matplotlib/numpy2 compat issues).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _extract_series(df, key):
    """Extract a numeric series from DataFrame, handling MultiIndex."""
    if df is None or df.empty:
        return None
    cols = df.columns
    target_col = None
    if isinstance(cols, pd.MultiIndex):
        matches = [c for c in cols if c[1] == key]
        if matches:
            target_col = matches[0]
    else:
        if key in cols:
            target_col = key
    if target_col is None:
        return None
    s = df[target_col]
    if hasattr(s, "ndim") and s.ndim > 1:
        s = s.iloc[:, 0] if s.shape[1] == 1 else s.mean(axis=1)
    s = pd.to_numeric(s, errors="coerce").dropna()
    return s


def _to_list(s):
    """Safely convert series to Python list for Plotly."""
    if s is None:
        return []
    return [float(v) for v in s.values]


def _save_plotly_fig(fig, output_path: Path):
    """Save Plotly figure to PNG."""
    try:
        fig.write_image(str(output_path), width=1280, height=720, scale=2)
        return True
    except Exception as e:
        logger.warning("Plotly 保存失败: %s", e)
        return False


def generate_report_charts(pred_df, report_normal_df, analysis_df,
                          positions_df=None, output_dir: str = "output/qlib_charts"):
    """
    Generate all analysis charts from backtest results using Plotly.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        logger.error("Plotly 未安装，无法生成图表。请运行: pip install plotly kaleido")
        return []

    charts = []
    ret = _extract_series(report_normal_df, "return")
    bench = _extract_series(report_normal_df, "bench")

    # Chart 1: Cumulative return
    try:
        if ret is not None and len(ret) > 0:
            fig = go.Figure()
            cum_ret = (1 + ret).cumprod()
            fig.add_trace(go.Scatter(
                x=list(cum_ret.index.astype(str)),
                y=_to_list(cum_ret),
                mode='lines',
                name='Strategy',
# -*- coding: utf-8 -*-
"""
Qlib Backtest Results Analysis - backtest.py

Generate charts and reports from backtest results using Plotly (avoids matplotlib/numpy2 compat issues).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _extract_series(df, key):
    """Extract a numeric series from DataFrame, handling MultiIndex."""
    if df is None or df.empty:
        return None
    cols = df.columns
    target_col = None
    if isinstance(cols, pd.MultiIndex):
        matches = [c for c in cols if c[1] == key]
        if matches:
            target_col = matches[0]
    else:
        if key in cols:
            target_col = key
    if target_col is None:
        return None
    s = df[target_col]
    if hasattr(s, "ndim") and s.ndim > 1:
        s = s.iloc[:, 0] if s.shape[1] == 1 else s.mean(axis=1)
    s = pd.to_numeric(s, errors="coerce").dropna()
    return s


def _to_list(s):
    """Safely convert series to Python list for Plotly."""
    if s is None:
        return []
    return [float(v) for v in s.values]


def _save_plotly_fig(fig, output_path: Path):
    """Save Plotly figure to PNG."""
    try:
        fig.write_image(str(output_path), width=1280, height=720, scale=2)
        return True
    except Exception as e:
        logger.warning("Plotly 保存失败: %s", e)
        return False


def generate_report_charts(pred_df, report_normal_df, analysis_df,
                          positions_df=None, output_dir: str = "output/qlib_charts"):
    """
    Generate all analysis charts from backtest results using Plotly.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        logger.error("Plotly 未安装，无法生成图表。请运行: pip install plotly kaleido")
        return []

    charts = []
    ret = _extract_series(report_normal_df, "return")
    bench = _extract_series(report_normal_df, "bench")

    # Chart 1: Cumulative return
    try:
        if ret is not None and len(ret) > 0:
            fig = go.Figure()
            cum_ret = (1 + ret).cumprod()
            fig.add_trace(go.Scatter(
                x=list(cum_ret.index.astype(str)),
                y=_to_list(cum_ret),
                mode='lines',
                name='Strategy',
                line=dict(color='steelblue', width=# -*- coding: utf-8 -*-
"""
Qlib Backtest Results Analysis - backtest.py

Generate charts and reports from backtest results using Plotly (avoids matplotlib/numpy2 compat issues).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _extract_series(df, key):
    """Extract a numeric series from DataFrame, handling MultiIndex."""
    if df is None or df.empty:
        return None
    cols = df.columns
    target_col = None
    if isinstance(cols, pd.MultiIndex):
        matches = [c for c in cols if c[1] == key]
        if matches:
            target_col = matches[0]
    else:
        if key in cols:
            target_col = key
    if target_col is None:
        return None
    s = df[target_col]
    if hasattr(s, "ndim") and s.ndim > 1:
        s = s.iloc[:, 0] if s.shape[1] == 1 else s.mean(axis=1)
    s = pd.to_numeric(s, errors="coerce").dropna()
    return s


def _to_list(s):
    """Safely convert series to Python list for Plotly."""
    if s is None:
        return []
    return [float(v) for v in s.values]


def _save_plotly_fig(fig, output_path: Path):
    """Save Plotly figure to PNG."""
    try:
        fig.write_image(str(output_path), width=1280, height=720, scale=2)
        return True
    except Exception as e:
        logger.warning("Plotly 保存失败: %s", e)
        return False


def generate_report_charts(pred_df, report_normal_df, analysis_df,
                          positions_df=None, output_dir: str = "output/qlib_charts"):
    """
    Generate all analysis charts from backtest results using Plotly.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        logger.error("Plotly 未安装，无法生成图表。请运行: pip install plotly kaleido")
        return []

    charts = []
    ret = _extract_series(report_normal_df, "return")
    bench = _extract_series(report_normal_df, "bench")

    # Chart 1: Cumulative return
    try:
        if ret is not None and len(ret) > 0:
            fig = go.Figure()
            cum_ret = (1 + ret).cumprod()
            fig.add_trace(go.Scatter(
                x=list(cum_ret.index.astype(str)),
                y=_to_list(cum_ret),
                mode='lines',
                name='Strategy',
                line=dict(color='steelblue', width=2)
            ))
            if bench is not None and len(bench) ># -*- coding: utf-8 -*-
"""
Qlib Backtest Results Analysis - backtest.py

Generate charts and reports from backtest results using Plotly (avoids matplotlib/numpy2 compat issues).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _extract_series(df, key):
    """Extract a numeric series from DataFrame, handling MultiIndex."""
    if df is None or df.empty:
        return None
    cols = df.columns
    target_col = None
    if isinstance(cols, pd.MultiIndex):
        matches = [c for c in cols if c[1] == key]
        if matches:
            target_col = matches[0]
    else:
        if key in cols:
            target_col = key
    if target_col is None:
        return None
    s = df[target_col]
    if hasattr(s, "ndim") and s.ndim > 1:
        s = s.iloc[:, 0] if s.shape[1] == 1 else s.mean(axis=1)
    s = pd.to_numeric(s, errors="coerce").dropna()
    return s


def _to_list(s):
    """Safely convert series to Python list for Plotly."""
    if s is None:
        return []
    return [float(v) for v in s.values]


def _save_plotly_fig(fig, output_path: Path):
    """Save Plotly figure to PNG."""
    try:
        fig.write_image(str(output_path), width=1280, height=720, scale=2)
        return True
    except Exception as e:
        logger.warning("Plotly 保存失败: %s", e)
        return False


def generate_report_charts(pred_df, report_normal_df, analysis_df,
                          positions_df=None, output_dir: str = "output/qlib_charts"):
    """
    Generate all analysis charts from backtest results using Plotly.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        logger.error("Plotly 未安装，无法生成图表。请运行: pip install plotly kaleido")
        return []

    charts = []
    ret = _extract_series(report_normal_df, "return")
    bench = _extract_series(report_normal_df, "bench")

    # Chart 1: Cumulative return
    try:
        if ret is not None and len(ret) > 0:
            fig = go.Figure()
            cum_ret = (1 + ret).cumprod()
            fig.add_trace(go.Scatter(
                x=list(cum_ret.index.astype(str)),
                y=_to_list(cum_ret),
                mode='lines',
                name='Strategy',
                line=dict(color='steelblue', width=2)
            ))
            if bench is not None and len(bench) > 0:
                cum_bench = (1 + bench).cumprod()
                fig.add# -*- coding: utf-8 -*-
"""
Qlib Backtest Results Analysis - backtest.py

Generate charts and reports from backtest results using Plotly (avoids matplotlib/numpy2 compat issues).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _extract_series(df, key):
    """Extract a numeric series from DataFrame, handling MultiIndex."""
    if df is None or df.empty:
        return None
    cols = df.columns
    target_col = None
    if isinstance(cols, pd.MultiIndex):
        matches = [c for c in cols if c[1] == key]
        if matches:
            target_col = matches[0]
    else:
        if key in cols:
            target_col = key
    if target_col is None:
        return None
    s = df[target_col]
    if hasattr(s, "ndim") and s.ndim > 1:
        s = s.iloc[:, 0] if s.shape[1] == 1 else s.mean(axis=1)
    s = pd.to_numeric(s, errors="coerce").dropna()
    return s


def _to_list(s):
    """Safely convert series to Python list for Plotly."""
    if s is None:
        return []
    return [float(v) for v in s.values]


def _save_plotly_fig(fig, output_path: Path):
    """Save Plotly figure to PNG."""
    try:
        fig.write_image(str(output_path), width=1280, height=720, scale=2)
        return True
    except Exception as e:
        logger.warning("Plotly 保存失败: %s", e)
        return False


def generate_report_charts(pred_df, report_normal_df, analysis_df,
                          positions_df=None, output_dir: str = "output/qlib_charts"):
    """
    Generate all analysis charts from backtest results using Plotly.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        logger.error("Plotly 未安装，无法生成图表。请运行: pip install plotly kaleido")
        return []

    charts = []
    ret = _extract_series(report_normal_df, "return")
    bench = _extract_series(report_normal_df, "bench")

    # Chart 1: Cumulative return
    try:
        if ret is not None and len(ret) > 0:
            fig = go.Figure()
            cum_ret = (1 + ret).cumprod()
            fig.add_trace(go.Scatter(
                x=list(cum_ret.index.astype(str)),
                y=_to_list(cum_ret),
                mode='lines',
                name='Strategy',
                line=dict(color='steelblue', width=2)
            ))
            if bench is not None and len(bench) > 0:
                cum_bench = (1 + bench).cumprod()
                fig.add_trace(go.Scatter(
                    x=list(cum_bench.index.astype(str)),
# -*- coding: utf-8 -*-
"""
Qlib Backtest Results Analysis - backtest.py

Generate charts and reports from backtest results using Plotly (avoids matplotlib/numpy2 compat issues).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _extract_series(df, key):
    """Extract a numeric series from DataFrame, handling MultiIndex."""
    if df is None or df.empty:
        return None
    cols = df.columns
    target_col = None
    if isinstance(cols, pd.MultiIndex):
        matches = [c for c in cols if c[1] == key]
        if matches:
            target_col = matches[0]
    else:
        if key in cols:
            target_col = key
    if target_col is None:
        return None
    s = df[target_col]
    if hasattr(s, "ndim") and s.ndim > 1:
        s = s.iloc[:, 0] if s.shape[1] == 1 else s.mean(axis=1)
    s = pd.to_numeric(s, errors="coerce").dropna()
    return s


def _to_list(s):
    """Safely convert series to Python list for Plotly."""
    if s is None:
        return []
    return [float(v) for v in s.values]


def _save_plotly_fig(fig, output_path: Path):
    """Save Plotly figure to PNG."""
    try:
        fig.write_image(str(output_path), width=1280, height=720, scale=2)
        return True
    except Exception as e:
        logger.warning("Plotly 保存失败: %s", e)
        return False


def generate_report_charts(pred_df, report_normal_df, analysis_df,
                          positions_df=None, output_dir: str = "output/qlib_charts"):
    """
    Generate all analysis charts from backtest results using Plotly.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        logger.error("Plotly 未安装，无法生成图表。请运行: pip install plotly kaleido")
        return []

    charts = []
    ret = _extract_series(report_normal_df, "return")
    bench = _extract_series(report_normal_df, "bench")

    # Chart 1: Cumulative return
    try:
        if ret is not None and len(ret) > 0:
            fig = go.Figure()
            cum_ret = (1 + ret).cumprod()
            fig.add_trace(go.Scatter(
                x=list(cum_ret.index.astype(str)),
                y=_to_list(cum_ret),
                mode='lines',
                name='Strategy',
                line=dict(color='steelblue', width=2)
            ))
            if bench is not None and len(bench) > 0:
                cum_bench = (1 + bench).cumprod()
                fig.add_trace(go.Scatter(
                    x=list(cum_bench.index.astype(str)),
                    y=_to_list(cum_bench),
                    mode='lines',
                    name='Benchmark# -*- coding: utf-8 -*-
"""
Qlib Backtest Results Analysis - backtest.py

Generate charts and reports from backtest results using Plotly (avoids matplotlib/numpy2 compat issues).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _extract_series(df, key):
    """Extract a numeric series from DataFrame, handling MultiIndex."""
    if df is None or df.empty:
        return None
    cols = df.columns
    target_col = None
    if isinstance(cols, pd.MultiIndex):
        matches = [c for c in cols if c[1] == key]
        if matches:
            target_col = matches[0]
    else:
        if key in cols:
            target_col = key
    if target_col is None:
        return None
    s = df[target_col]
    if hasattr(s, "ndim") and s.ndim > 1:
        s = s.iloc[:, 0] if s.shape[1] == 1 else s.mean(axis=1)
    s = pd.to_numeric(s, errors="coerce").dropna()
    return s


def _to_list(s):
    """Safely convert series to Python list for Plotly."""
    if s is None:
        return []
    return [float(v) for v in s.values]


def _save_plotly_fig(fig, output_path: Path):
    """Save Plotly figure to PNG."""
    try:
        fig.write_image(str(output_path), width=1280, height=720, scale=2)
        return True
    except Exception as e:
        logger.warning("Plotly 保存失败: %s", e)
        return False


def generate_report_charts(pred_df, report_normal_df, analysis_df,
                          positions_df=None, output_dir: str = "output/qlib_charts"):
    """
    Generate all analysis charts from backtest results using Plotly.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        logger.error("Plotly 未安装，无法生成图表。请运行: pip install plotly kaleido")
        return []

    charts = []
    ret = _extract_series(report_normal_df, "return")
    bench = _extract_series(report_normal_df, "bench")

    # Chart 1: Cumulative return
    try:
        if ret is not None and len(ret) > 0:
            fig = go.Figure()
            cum_ret = (1 + ret).cumprod()
            fig.add_trace(go.Scatter(
                x=list(cum_ret.index.astype(str)),
                y=_to_list(cum_ret),
                mode='lines',
                name='Strategy',
                line=dict(color='steelblue', width=2)
            ))
            if bench is not None and len(bench) > 0:
                cum_bench = (1 + bench).cumprod()
                fig.add_trace(go.Scatter(
                    x=list(cum_bench.index.astype(str)),
                    y=_to_list(cum_bench),
                    mode='lines',
                    name='Benchmark',
                    line=dict(color='orange', width# -*- coding: utf-8 -*-
"""
Qlib Backtest Results Analysis - backtest.py

Generate charts and reports from backtest results using Plotly (avoids matplotlib/numpy2 compat issues).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _extract_series(df, key):
    """Extract a numeric series from DataFrame, handling MultiIndex."""
    if df is None or df.empty:
        return None
    cols = df.columns
    target_col = None
    if isinstance(cols, pd.MultiIndex):
        matches = [c for c in cols if c[1] == key]
        if matches:
            target_col = matches[0]
    else:
        if key in cols:
            target_col = key
    if target_col is None:
        return None
    s = df[target_col]
    if hasattr(s, "ndim") and s.ndim > 1:
        s = s.iloc[:, 0] if s.shape[1] == 1 else s.mean(axis=1)
    s = pd.to_numeric(s, errors="coerce").dropna()
    return s


def _to_list(s):
    """Safely convert series to Python list for Plotly."""
    if s is None:
        return []
    return [float(v) for v in s.values]


def _save_plotly_fig(fig, output_path: Path):
    """Save Plotly figure to PNG."""
    try:
        fig.write_image(str(output_path), width=1280, height=720, scale=2)
        return True
    except Exception as e:
        logger.warning("Plotly 保存失败: %s", e)
        return False


def generate_report_charts(pred_df, report_normal_df, analysis_df,
                          positions_df=None, output_dir: str = "output/qlib_charts"):
    """
    Generate all analysis charts from backtest results using Plotly.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        logger.error("Plotly 未安装，无法生成图表。请运行: pip install plotly kaleido")
        return []

    charts = []
    ret = _extract_series(report_normal_df, "return")
    bench = _extract_series(report_normal_df, "bench")

    # Chart 1: Cumulative return
    try:
        if ret is not None and len(ret) > 0:
            fig = go.Figure()
            cum_ret = (1 + ret).cumprod()
            fig.add_trace(go.Scatter(
                x=list(cum_ret.index.astype(str)),
                y=_to_list(cum_ret),
                mode='lines',
                name='Strategy',
                line=dict(color='steelblue', width=2)
            ))
            if bench is not None and len(bench) > 0:
                cum_bench = (1 + bench).cumprod()
                fig.add_trace(go.Scatter(
                    x=list(cum_bench.index.astype(str)),
                    y=_to_list(cum_bench),
                    mode='lines',
                    name='Benchmark',
                    line=dict(color='orange', width=2, dash='dash')
                ))
            fig.update_layout(
                title# -*- coding: utf-8 -*-
"""
Qlib Backtest Results Analysis - backtest.py

Generate charts and reports from backtest results using Plotly (avoids matplotlib/numpy2 compat issues).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _extract_series(df, key):
    """Extract a numeric series from DataFrame, handling MultiIndex."""
    if df is None or df.empty:
        return None
    cols = df.columns
    target_col = None
    if isinstance(cols, pd.MultiIndex):
        matches = [c for c in cols if c[1] == key]
        if matches:
            target_col = matches[0]
    else:
        if key in cols:
            target_col = key
    if target_col is None:
        return None
    s = df[target_col]
    if hasattr(s, "ndim") and s.ndim > 1:
        s = s.iloc[:, 0] if s.shape[1] == 1 else s.mean(axis=1)
    s = pd.to_numeric(s, errors="coerce").dropna()
    return s


def _to_list(s):
    """Safely convert series to Python list for Plotly."""
    if s is None:
        return []
    return [float(v) for v in s.values]


def _save_plotly_fig(fig, output_path: Path):
    """Save Plotly figure to PNG."""
    try:
        fig.write_image(str(output_path), width=1280, height=720, scale=2)
        return True
    except Exception as e:
        logger.warning("Plotly 保存失败: %s", e)
        return False


def generate_report_charts(pred_df, report_normal_df, analysis_df,
                          positions_df=None, output_dir: str = "output/qlib_charts"):
    """
    Generate all analysis charts from backtest results using Plotly.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        logger.error("Plotly 未安装，无法生成图表。请运行: pip install plotly kaleido")
        return []

    charts = []
    ret = _extract_series(report_normal_df, "return")
    bench = _extract_series(report_normal_df, "bench")

    # Chart 1: Cumulative return
    try:
        if ret is not None and len(ret) > 0:
            fig = go.Figure()
            cum_ret = (1 + ret).cumprod()
            fig.add_trace(go.Scatter(
                x=list(cum_ret.index.astype(str)),
                y=_to_list(cum_ret),
                mode='lines',
                name='Strategy',
                line=dict(color='steelblue', width=2)
            ))
            if bench is not None and len(bench) > 0:
                cum_bench = (1 + bench).cumprod()
                fig.add_trace(go.Scatter(
                    x=list(cum_bench.index.astype(str)),
                    y=_to_list(cum_bench),
                    mode='lines',
                    name='Benchmark',
                    line=dict(color='orange', width=2, dash='dash')
                ))
            fig.update_layout(
                title='Cumulative Return',
                xaxis_title
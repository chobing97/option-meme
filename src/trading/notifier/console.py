"""Console notifier: prints trade events to stdout."""

from src.trading.notifier.base import Notifier, TradeEvent


class ConsoleNotifier(Notifier):
    """Print trade events to console."""

    def notify(self, event: TradeEvent) -> None:
        d = event.details

        if event.event_type == "BUY":
            print(
                f"  -> BUY PUT {d['strike']:,.0f} "
                f"exp={d['expiry']} @ {d['fill_price']:,.0f} "
                f"(qty={d['quantity']})"
            )

        elif event.event_type == "SELL":
            print(
                f"  -> SELL PUT {d['strike']:,.0f} "
                f"@ {d['fill_price']:,.0f} "
                f"(PnL: {d['pnl_pct']:+.1%}, {d['reason']}, qty={d['quantity']})"
            )

        elif event.event_type == "SESSION_END":
            print(f"\n=== Session Summary: {event.market}/{event.symbol} ===")
            print(f"Trades: {d['buys']} buy, {d['sells']} sell")
            if d.get("total_cost", 0) > 0:
                pnl_pct = d["net_pnl"] / d["total_cost"]
                print(f"Net PnL: {d['net_pnl']:+,.0f} ({pnl_pct:+.1%})")
            else:
                print("Net PnL: 0 (no trades)")
            print(
                f"Signals: {d['peak_signals']} peaks, "
                f"{d['trough_signals']} troughs"
            )
            print(f"Cash balance: {d['cash_balance']:,.0f}")

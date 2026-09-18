from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExecutionCosts:
    spread_points: float = 0.0
    slippage_points: float = 0.0
    commission_per_order: float = 20.0
    transaction_cost_bps: float = 0.0


class ExecutionHandler:
    def __init__(self, portfolio, costs: ExecutionCosts):
        self.portfolio = portfolio
        self.costs = costs

    def _fill_price(self, market_price: float, direction: int) -> float:
        impact = self.costs.spread_points / 2.0 + self.costs.slippage_points
        return float(market_price) + direction * impact

    def _order_cost(self, market_price: float, units: float, point_value: float) -> float:
        notional = abs(float(market_price) * units * point_value)
        variable = notional * self.costs.transaction_cost_bps / 10_000.0
        return self.costs.commission_per_order + variable

    def execute(
        self,
        *,
        time,
        market_price: float,
        target_direction: int,
        units: float,
    ) -> None:
        if target_direction not in (-1, 0, 1):
            raise ValueError("target_direction must be -1, 0 or 1")

        current = self.portfolio.direction
        if current == target_direction:
            return

        if current != 0:
            close_price = self._fill_price(market_price, -current)
            close_cost = self._order_cost(
                close_price, self.portfolio.units, self.portfolio.point_value
            )
            self.portfolio.close(time, close_price, close_cost)

        if target_direction != 0:
            open_price = self._fill_price(market_price, target_direction)
            open_cost = self._order_cost(
                open_price, units, self.portfolio.point_value
            )
            self.portfolio.open(time, open_price, target_direction, units, open_cost)

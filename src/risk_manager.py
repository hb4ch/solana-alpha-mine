import polars as pl
import numpy as np
import logging
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class RiskConfig:
    """Risk management configuration parameters"""
    max_position_pct: float = 0.15  # Max single position as % of portfolio (15%)
    max_total_exposure_pct: float = 0.80  # Max aggregate exposure (80%)
    min_cash_reserve_pct: float = 0.20  # Minimum cash to maintain (20%)
    max_risk_per_trade_pct: float = 0.01  # Max risk per trade (1%)
    max_concurrent_positions: int = 5  # Maximum number of open positions
    volatility_scaling_enabled: bool = True  # Enable volatility-based position sizing
    volatility_lookback_factor: float = 2.0  # Factor for volatility adjustment
    correlation_limit: float = 0.7  # Maximum correlation between positions

@dataclass
class PositionRisk:
    """Individual position risk metrics"""
    market: str
    size: float
    entry_price: float
    current_price: float
    notional_value: float
    portfolio_pct: float
    unrealized_pnl: float
    unrealized_pnl_pct: float

@dataclass
class PortfolioRisk:
    """Portfolio-level risk metrics"""
    total_exposure: float
    total_exposure_pct: float
    available_cash: float
    available_cash_pct: float
    total_unrealized_pnl: float
    total_unrealized_pnl_pct: float
    num_positions: int
    largest_position_pct: float
    risk_score: float  # 0-100, where higher = riskier

class RiskManager:
    """Centralized risk management for trading strategies"""
    
    def __init__(self, config: RiskConfig = None):
        self.config = config or RiskConfig()
        logger.info(f"Risk Manager initialized with config: {self.config}")
    
    def calculate_safe_position_size(self, 
                                   portfolio_value: float,
                                   price: float,
                                   volatility: float,
                                   current_positions: Dict[str, Dict],
                                   market: str,
                                   leverage: float = 1.0) -> Tuple[float, str]:
        """
        Calculate position size with comprehensive risk management
        
        Returns:
            Tuple[float, str]: (position_size, reason_for_size)
        """
        if portfolio_value <= 0 or price <= 0:
            return 0.0, "Invalid portfolio value or price"
        
        # Calculate current portfolio risk state
        portfolio_risk = self._calculate_portfolio_risk(portfolio_value, current_positions, {market: price})
        
        # Check if we can open new positions
        if portfolio_risk.num_positions >= self.config.max_concurrent_positions:
            return 0.0, f"Max concurrent positions reached ({self.config.max_concurrent_positions})"
        
        if portfolio_risk.total_exposure_pct >= self.config.max_total_exposure_pct:
            return 0.0, f"Max total exposure reached ({self.config.max_total_exposure_pct:.1%})"
        
        if portfolio_risk.available_cash_pct < self.config.min_cash_reserve_pct:
            return 0.0, f"Insufficient cash reserve (min {self.config.min_cash_reserve_pct:.1%})"
        
        # Calculate position size based on multiple risk factors
        sizes_with_reasons = []
        
        # 1. Maximum position size as % of portfolio
        max_position_value = portfolio_value * self.config.max_position_pct
        max_position_size = max_position_value / price
        sizes_with_reasons.append((max_position_size, f"Max position % limit ({self.config.max_position_pct:.1%})"))
        
        # 2. Risk-per-trade limit (based on volatility as stop loss proxy)
        if volatility > 0:
            risk_amount = portfolio_value * self.config.max_risk_per_trade_pct
            # Use volatility as proxy for potential loss per unit
            risk_based_size = risk_amount / (price * volatility)
            sizes_with_reasons.append((risk_based_size, f"Risk per trade limit ({self.config.max_risk_per_trade_pct:.1%})"))
        
        # 3. Available cash limit (accounting for leverage)
        available_cash = portfolio_risk.available_cash
        cash_based_max_notional = available_cash * leverage
        cash_based_size = cash_based_max_notional / price
        sizes_with_reasons.append((cash_based_size, f"Available cash limit"))
        
        # 4. Remaining exposure capacity
        remaining_exposure_pct = self.config.max_total_exposure_pct - portfolio_risk.total_exposure_pct
        remaining_exposure_value = portfolio_value * remaining_exposure_pct
        exposure_based_size = remaining_exposure_value / price
        sizes_with_reasons.append((exposure_based_size, f"Remaining exposure capacity"))
        
        # 5. Volatility-adjusted sizing (smaller positions for higher volatility)
        if self.config.volatility_scaling_enabled and volatility > 0:
            # Scale down position for high volatility
            volatility_factor = 1.0 / (1.0 + volatility * self.config.volatility_lookback_factor)
            base_size = min([size for size, _ in sizes_with_reasons])
            volatility_adjusted_size = base_size * volatility_factor
            sizes_with_reasons.append((volatility_adjusted_size, f"Volatility adjustment (factor: {volatility_factor:.2f})"))
        
        # Take the minimum size from all constraints
        valid_sizes = [(size, reason) for size, reason in sizes_with_reasons if size > 0]
        if not valid_sizes:
            return 0.0, "All constraints resulted in zero position size"
        
        final_size, limiting_reason = min(valid_sizes, key=lambda x: x[0])
        
        # Final safety check
        final_notional = final_size * price
        final_portfolio_pct = final_notional / portfolio_value
        
        if final_portfolio_pct > self.config.max_position_pct:
            return 0.0, f"Final size check failed: {final_portfolio_pct:.1%} > {self.config.max_position_pct:.1%}"
        
        logger.debug(f"Position size for {market}: {final_size:.4f} units "
                    f"(${final_notional:.2f}, {final_portfolio_pct:.1%} of portfolio) - {limiting_reason}")
        
        return final_size, limiting_reason
    
    def _calculate_portfolio_risk(self, 
                                portfolio_value: float, 
                                positions: Dict[str, Dict],
                                current_prices: Dict[str, float]) -> PortfolioRisk:
        """Calculate comprehensive portfolio risk metrics"""
        
        position_risks = []
        total_exposure = 0.0
        total_unrealized_pnl = 0.0
        
        for market, pos_details in positions.items():
            if market in current_prices:
                current_price = current_prices[market]
                entry_price = pos_details['entry_price']
                size = pos_details['size']
                
                notional_value = size * current_price
                portfolio_pct = notional_value / portfolio_value
                unrealized_pnl = (current_price - entry_price) * size
                unrealized_pnl_pct = ((current_price / entry_price) - 1) * 100
                
                position_risk = PositionRisk(
                    market=market,
                    size=size,
                    entry_price=entry_price,
                    current_price=current_price,
                    notional_value=notional_value,
                    portfolio_pct=portfolio_pct,
                    unrealized_pnl=unrealized_pnl,
                    unrealized_pnl_pct=unrealized_pnl_pct
                )
                
                position_risks.append(position_risk)
                total_exposure += notional_value
                total_unrealized_pnl += unrealized_pnl
        
        total_exposure_pct = total_exposure / portfolio_value
        available_cash = portfolio_value - total_exposure
        available_cash_pct = available_cash / portfolio_value
        total_unrealized_pnl_pct = total_unrealized_pnl / portfolio_value * 100
        num_positions = len(positions)
        largest_position_pct = max([pr.portfolio_pct for pr in position_risks], default=0.0)
        
        # Calculate risk score (0-100, higher = riskier)
        risk_score = self._calculate_risk_score(
            total_exposure_pct, available_cash_pct, num_positions, largest_position_pct
        )
        
        return PortfolioRisk(
            total_exposure=total_exposure,
            total_exposure_pct=total_exposure_pct,
            available_cash=available_cash,
            available_cash_pct=available_cash_pct,
            total_unrealized_pnl=total_unrealized_pnl,
            total_unrealized_pnl_pct=total_unrealized_pnl_pct,
            num_positions=num_positions,
            largest_position_pct=largest_position_pct,
            risk_score=risk_score
        )
    
    def _calculate_risk_score(self, 
                            total_exposure_pct: float,
                            available_cash_pct: float, 
                            num_positions: int,
                            largest_position_pct: float) -> float:
        """Calculate portfolio risk score (0-100, higher = riskier)"""
        
        risk_factors = []
        
        # Exposure risk (0-40 points)
        exposure_risk = min(40, total_exposure_pct * 50)  # 80% exposure = 40 points
        risk_factors.append(exposure_risk)
        
        # Cash reserve risk (0-20 points)
        cash_risk = max(0, 20 - available_cash_pct * 100)  # <20% cash = risk
        risk_factors.append(cash_risk)
        
        # Concentration risk (0-25 points)
        concentration_risk = min(25, largest_position_pct * 100)  # 25% position = 25 points
        risk_factors.append(concentration_risk)
        
        # Diversification risk (0-15 points)
        if num_positions == 0:
            diversification_risk = 0
        elif num_positions == 1:
            diversification_risk = 15
        elif num_positions <= 3:
            diversification_risk = 10
        else:
            diversification_risk = max(0, 8 - num_positions)
        risk_factors.append(diversification_risk)
        
        total_risk = sum(risk_factors)
        return min(100, total_risk)
    
    def check_position_limits(self, 
                            portfolio_value: float,
                            positions: Dict[str, Dict],
                            current_prices: Dict[str, float]) -> Dict[str, any]:
        """Check if current positions violate risk limits"""
        
        portfolio_risk = self._calculate_portfolio_risk(portfolio_value, positions, current_prices)
        
        violations = {
            'has_violations': False,
            'violations': [],
            'warnings': [],
            'portfolio_risk': portfolio_risk
        }
        
        # Check hard limits (violations)
        if portfolio_risk.total_exposure_pct > self.config.max_total_exposure_pct:
            violations['has_violations'] = True
            violations['violations'].append(
                f"Total exposure {portfolio_risk.total_exposure_pct:.1%} exceeds limit {self.config.max_total_exposure_pct:.1%}"
            )
        
        if portfolio_risk.available_cash_pct < self.config.min_cash_reserve_pct:
            violations['has_violations'] = True
            violations['violations'].append(
                f"Cash reserve {portfolio_risk.available_cash_pct:.1%} below minimum {self.config.min_cash_reserve_pct:.1%}"
            )
        
        if portfolio_risk.largest_position_pct > self.config.max_position_pct:
            violations['has_violations'] = True
            violations['violations'].append(
                f"Largest position {portfolio_risk.largest_position_pct:.1%} exceeds limit {self.config.max_position_pct:.1%}"
            )
        
        # Check warnings (soft limits)
        if portfolio_risk.risk_score > 75:
            violations['warnings'].append(f"High portfolio risk score: {portfolio_risk.risk_score:.1f}/100")
        
        if portfolio_risk.num_positions >= self.config.max_concurrent_positions * 0.8:
            violations['warnings'].append(
                f"Approaching max positions: {portfolio_risk.num_positions}/{self.config.max_concurrent_positions}"
            )
        
        return violations
    
    def format_risk_report(self, 
                          portfolio_value: float,
                          positions: Dict[str, Dict],
                          current_prices: Dict[str, float]) -> str:
        """Generate a formatted risk report"""
        
        portfolio_risk = self._calculate_portfolio_risk(portfolio_value, positions, current_prices)
        violations = self.check_position_limits(portfolio_value, positions, current_prices)
        
        report = []
        report.append("=== PORTFOLIO RISK REPORT ===")
        report.append(f"Portfolio Value: ${portfolio_value:,.2f}")
        report.append(f"Total Exposure: ${portfolio_risk.total_exposure:,.2f} ({portfolio_risk.total_exposure_pct:.1%})")
        report.append(f"Available Cash: ${portfolio_risk.available_cash:,.2f} ({portfolio_risk.available_cash_pct:.1%})")
        report.append(f"Unrealized P&L: ${portfolio_risk.total_unrealized_pnl:,.2f} ({portfolio_risk.total_unrealized_pnl_pct:+.1f}%)")
        report.append(f"Open Positions: {portfolio_risk.num_positions}")
        report.append(f"Largest Position: {portfolio_risk.largest_position_pct:.1%}")
        report.append(f"Risk Score: {portfolio_risk.risk_score:.1f}/100")
        
        if violations['violations']:
            report.append("\n❌ VIOLATIONS:")
            for violation in violations['violations']:
                report.append(f"  • {violation}")
        
        if violations['warnings']:
            report.append("\n⚠️  WARNINGS:")
            for warning in violations['warnings']:
                report.append(f"  • {warning}")
        
        if not violations['has_violations'] and not violations['warnings']:
            report.append("\n✅ All risk limits within acceptable ranges")
        
        return "\n".join(report)

if __name__ == "__main__":
    # Example usage
    config = RiskConfig()
    risk_manager = RiskManager(config)
    print("Risk Manager initialized successfully")

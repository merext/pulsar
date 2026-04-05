# MarketMakerBidAskGlmStrategy — Полная документация

## Оглавление

1. [Обзор](#1-обзор)
2. [Принцип работы](#2-принцип-работы)
3. [Архитектура системы](#3-архитектура-системы)
4. [Ценообразование (Pricing)](#4-ценообразование)
5. [Управление инвентарём](#5-управление-инвентарём)
6. [Определение размера ордера (Sizing)](#6-определение-размера-ордера)
7. [Фильтры и защиты](#7-фильтры-и-защиты)
8. [Исполнение ордеров (Trader)](#8-исполнение-ордеров)
9. [Ребалансировка](#9-ребалансировка)
10. [Конфигурация](#10-конфигурация)
11. [Жизненный цикл тика](#11-жизненный-цикл-тика)
12. [Производительность](#12-производительность)
13. [Известные ограничения](#13-известные-ограничения)

---

## 1. Обзор

**MarketMakerBidAskGlmStrategy** — двусторонняя стратегия маркет-мейкинга для спотового рынка Binance. Постоянно котирует bid (покупка) и ask (продажа) одновременно, зарабатывая на спреде между ними.

### Ключевые характеристики

| Параметр | Значение |
|----------|----------|
| Торговая пара | DOGEFDUSD |
| Биржа | Binance Spot |
| Цена DOGE | ~$0.090 |
| Tick size | 0.00001 (~1.1 bps) |
| Step size (lot) | 1.0 DOGE |
| Maker fee | **0%** (промо-акция Binance для FDUSD пар) |
| Taker fee | 0.1% |
| Капитал | ~$110 |
| Размер ордера | 12 DOGE (~$1.08) на каждой стороне |

### Источник прибыли

```
Прибыль = Σ(spread_captured) - adverse_selection_losses - taker_fees_on_rebalance
```

При нулевой maker fee, каждая пара BUY+SELL fillov на 1-tick spread (~1.1 bps) даёт чистую прибыль ~$0.0001. За час ~30 таких пар = ~$0.003/час = ~$0.07/день.

---

## 2. Принцип работы

### Высокоуровневая схема

```
BookTicker (best_bid, best_ask)
         │
         ▼
┌─────────────────────────────┐
│  1. Fair price extraction   │  mid = (bid + ask) / 2
│  2. Slow EMA update        │  slow_ema = α·mid + (1-α)·slow_ema
│  3. Emergency exit check    │  stop_loss? panic_vol? max_hold?
│  4. A-S reservation price   │  r = mid - q·γ·σ²
│  5. Dynamic half-spread     │  hs = base_hs · vol_multiplier
│  6. Raw quotes              │  buy = r - hs, sell = r + hs
│  7. Join-best-price         │  clamp to BBO
│  8. Inventory offset        │  extra bid offset if ratio > 0.55
│  9. Spread-crossing guard   │  ensure buy < sell
│ 10. Buy-side filters        │  vol gate, trend filter, cooldown
│ 11. Flat sizing             │  buy_qty = sell_qty = min_qty (12)
│ 12. Balance checks          │  enough FDUSD? enough DOGE?
└─────────────────────────────┘
         │
         ▼
   QuoteBothSides {
     buy_price, buy_quantity,
     sell_price, sell_quantity
   }
         │
         ▼
┌─────────────────────────────┐
│  Trader (trader.rs)         │
│  - place_or_replace_side()  │
│  - cancel-replace via API   │
│  - fill detection           │
│  - rebalance trigger        │
└─────────────────────────────┘
```

### Двусторонняя модель

В отличие от направленных стратегий, MM **всегда** держит ордера на обеих сторонах:

- **BID** (buy limit order) — стоит на best_bid или ниже, ждёт seller'а
- **ASK** (sell limit order) — стоит на best_ask или выше, ждёт buyer'а

Когда оба ордера исполняются (BUY fill + SELL fill), разница цен = прибыль.

---

## 3. Архитектура системы

### Файловая структура

```
strategies/src/market_maker_bid_ask_glm.rs   # Стратегия (862 строки)
  ├── MarketMakerBidAskGlmConfig             # Конфигурация (deserialized из TOML)
  ├── GlmDiagnostics                         # Счётчики и метрики
  └── MarketMakerBidAskGlmStrategy           # Основная логика
      ├── fair_price_and_book()               # Извлечение цены из bookTicker
      ├── dynamic_half_spread_bps()           # Волатильностный спред
      ├── compute_inventory_ratio()           # Инвентарь [0, 1]
      ├── max_inventory_notional()            # Максимальный инвентарь
      ├── in_cooldown()                       # Проверка cooldown
      ├── check_buy_filters()                 # Фильтры adverse selection
      ├── check_emergency_exit()              # Аварийные выходы
      └── decide()                            # Главный метод — вызывается каждый тик

exchanges/binance/src/trader.rs              # Исполнение ордеров (3783 строки)
  ├── place_or_replace_side()                 # Размещение/замена одной стороны
  ├── poll_two_sided_orders()                 # Детекция филлов
  ├── rebalance()                             # Ребалансировка 50/50
  └── QuoteBothSides handler                  # Обработка QuoteBothSides intent

config/strategies/market_maker_bid_ask_glm.toml   # Параметры стратегии
config/trading_config_dogefdusd.toml              # Параметры торговли + биржи
```

### Поток данных

```
Binance WebSocket ──► bookTicker (best_bid, best_ask)
                  ──► trade (last trade price, qty, side)
                        │
                        ▼
                   MarketState
                   ├── top_of_book()        → TopOfBook { bid, ask }
                   ├── micro()              → MicrostructureState
                   │   ├── ema_mid_price
                   │   ├── realized_vol_bps
                   │   └── ema_spread_bps
                   └── trade_window_stats() → TradeWindowStats
                       ├── trade_count
                       ├── buyer_initiated_volume
                       └── seller_initiated_volume
                        │
                        ▼
                   Strategy.decide()
                        │
                        ▼
                   StrategyDecision {
                     intent: QuoteBothSides { ... }
                   }
                        │
                        ▼
                   Trader.execute()
                        │
                        ▼
                   Binance REST API
                   ├── POST /api/v3/order    (new order)
                   └── POST /api/v3/order/cancelReplace  (atomic cancel+replace)
```

---

## 4. Ценообразование

### 4.1. Avellaneda-Stoikov Reservation Price

Классическая модель A-S для оптимального маркет-мейкинга. Ключевая идея: **сдвинуть «справедливую цену» от mid_price в зависимости от инвентаря**.

**Формула:**

```
reservation_price = mid_price - q_normalized · γ · σ²
```

Где:
- `q_normalized` — нормализованный инвентарь [-1, +1]. При `inventory_ratio = 0.5` (нейтральный) → `q = 0` (симметричные котировки). При `ratio = 1.0` (полный) → `q = +1` (reservation price ниже mid → bid ещё ниже, ask ближе к mid).
- `γ` (gamma) — коэффициент risk aversion. `γ = 200` в текущем конфиге. Чем выше — тем агрессивнее стратегия стремится вернуть инвентарь к нейтральному.
- `σ` — realized volatility в ценовых единицах: `σ = realized_vol_bps / 10000 · mid_price`
- `τ = 1.0` — упрощение (непрерывный маркет-мейкинг без экспирации)

**Пример при типичных параметрах:**

```
mid = 0.09040, vol = 50 bps, gamma = 200, ratio = 0.70

q_normalized = 2 · 0.70 - 1 = 0.40
σ = 50/10000 · 0.09040 = 0.000452
reservation_shift = 0.40 · 200 · 0.000452² = 0.0000163
reservation_price = 0.09040 - 0.0000163 = 0.09038

→ Bid сдвигается вниз на ~1.8 bps
→ Ask подтягивается ближе к mid
```

**Код** (`market_maker_bid_ask_glm.rs:613-636`):

```rust
let q_normalized = 2.0 * inventory_ratio - 1.0;
let vol_price = micro.realized_vol_bps / 10_000.0 * mid_price;
let reservation_shift = q_normalized * gamma * vol_price * vol_price;
let reservation_price = mid_price - reservation_shift;

let mut buy_price = reservation_price * (1.0 - half_spread_bps / 10_000.0);
let mut sell_price = reservation_price * (1.0 + half_spread_bps / 10_000.0);
```

### 4.2. Dynamic Half-Spread

Полуспред масштабируется волатильностью:

```
vol_multiplier = clamp(realized_vol / reference_vol, min, max)
half_spread = base_half_spread · vol_multiplier
```

Текущие параметры:
- `base_half_spread_bps = 0.6` (базовый полуспред)
- `volatility_reference_bps = 50.0` (при vol=50 bps → множитель = 1.0)
- `volatility_multiplier_min = 0.5` (при низкой vol → спред сужается до 0.3 bps)
- `volatility_multiplier_max = 4.0` (при высокой vol → спред расширяется до 2.4 bps)
- Финальный clamp: `[min_half_spread_bps=0.3, max_half_spread_bps=10.0]`

### 4.3. Join-Best-Price

На парах с узким спредом (~1 bps), котировки далеко от лучшей цены никогда не исполнятся. Поэтому стратегия подтягивает цены к BBO (Best Bid/Offer).

**Правила (асимметричные):**

| Сторона | Правило | Логика |
|---------|---------|--------|
| BID | Join best_bid **только если** `inventory_ratio < 0.50` | При нейтральном/низком инвентаре — стоим на лучшем бид. При высоком — A-S сдвигает бид вниз, не подтягиваем. |
| ASK | **Всегда** join best_ask | Максимизируем вероятность sell-fill при любом инвентаре. |

**Код** (`market_maker_bid_ask_glm.rs:652-657`):

```rust
if buy_price < best_bid && inventory_ratio < 0.50 {
    buy_price = best_bid;
}
if sell_price > best_ask {
    sell_price = best_ask;
}
```

### 4.4. Inventory-Aware Bid Offset

Дополнительный сдвиг бида вниз при перевесе инвентаря. Стакуется поверх A-S shift.

```
Если ratio > 0.55:
  extra_offset = (ratio - 0.55) · 10 bps per 0.10 ratio
  buy_price *= (1 - extra_offset / 10000)
```

Примеры:
- `ratio = 0.55` → offset = 0 bps
- `ratio = 0.65` → offset = 1.0 bps
- `ratio = 0.75` → offset = 2.0 bps
- `ratio = 0.85` → offset = 3.0 bps

**Код** (`market_maker_bid_ask_glm.rs:663-667`):

```rust
if inventory_ratio > 0.55 {
    let excess = inventory_ratio - 0.55;
    let extra_offset_bps = excess * 10.0;
    buy_price *= 1.0 - extra_offset_bps / 10_000.0;
}
```

### 4.5. Spread-Crossing Guard

Последний шаг — гарантия, что buy < sell:

```rust
if buy_price >= best_ask { buy_price = best_ask - tick_size; }
if sell_price <= best_bid { sell_price = best_bid + tick_size; }
if sell_price <= buy_price { sell_price = buy_price + tick_size; }
```

---

## 5. Управление инвентарём

### 5.1. Inventory Ratio

```
inventory_ratio = (position_qty · mid_price) / max_inventory_notional
max_inventory_notional = initial_capital · max_inventory_fraction
```

С текущими параметрами:
- `initial_capital = 112.0 FDUSD`
- `max_inventory_fraction = 1.0`
- `max_inventory_notional = 112.0 FDUSD`

После ребалансировки ~50/50, типичная позиция ~600 DOGE · 0.090 = ~54 FDUSD → `ratio ≈ 0.48`.

### 5.2. Три уровня управления инвентарём

Стратегия управляет инвентарём исключительно через **цену**, не через количество:

| Уровень | Механизм | Когда включается | Эффект |
|---------|----------|-------------------|--------|
| 1. A-S reservation price | Сдвиг «fair value» от mid | Всегда | Bid/Ask симметрично сдвигаются от mid |
| 2. Inventory bid offset | Дополнительный сдвиг bid вниз | ratio > 0.55 | Bid уходит от best_bid, реже покупаем |
| 3. Join-best-bid блокировка | Bid не подтягивается к best_bid | ratio >= 0.50 | A-S цена не переопределяется join-best |

### 5.3. Rebalance Target

Ребалансировка целит в **50/50 split** по стоимости:

```
total_value = FDUSD_free + DOGE_free · market_price
target_per_side = total_value / 2
delta = target_per_side - (DOGE_free · market_price)

if delta > min_notional → BUY |delta|/price DOGE (market order)
if delta < -min_notional → SELL |delta|/price DOGE (market order)
```

---

## 6. Определение размера ордера

### Flat Min-Qty Sizing

Обе стороны **всегда** котируют минимально допустимое количество:

```
min_qty = ceil(min_notional / buy_price)    // в DOGE
        = ceil(1.0 / 0.0904)
        = ceil(11.06)
        = 12 DOGE

buy_quantity = 12 DOGE    (если buy_allowed)
sell_quantity = 12 DOGE   (если position >= 12)
```

**Почему flat sizing:**

Предыдущие модели (A-S quantity scaling, cycle-based sizing) создавали проблемы:
- A-S quantity scaling: `buy_qty = base_qty · (1 - ratio)` → при ratio > 0.667, buy_qty < min_qty → buy_qty = 0 → auto-rebalance каждые ~130 секунд
- Cycle-based sizing: opposite-side qty растёт линейно → эскалация до огромных ордеров (121 DOGE)

Flat sizing + A-S **price** skew = чистое решение: инвентарь управляется сдвигом цен, а не объёмом.

### Ограничения на sell_quantity

```rust
if position < min_qty {
    sell_quantity = 0.0;          // нечего продавать
} else {
    sell_quantity = min_qty.min(position);  // не больше, чем есть
}
```

### Проверка баланса для buy_quantity

```rust
let buy_notional = buy_quantity * buy_price;
if buy_notional > available_cash {
    let affordable_qty = floor(available_cash / buy_price);
    if affordable_qty < min_qty {
        buy_quantity = 0.0;       // не хватает FDUSD
    }
}
```

---

## 7. Фильтры и защиты

### 7.1. Trend Filter (Slow EMA)

Блокирует покупки в падающем рынке при высоком инвентаре.

**Механизм:**
- `slow_ema_mid` — медленная EMA средней цены. `α = 0.002`, halflife ≈ 346 тиков ≈ 35 секунд.
- `price_falling = mid_price < slow_ema_mid`

**Гистерезис (для предотвращения осцилляции):**
- **Активация**: `price_falling AND ratio > 0.65`
- **Деактивация**: `price_falling == false OR ratio < 0.55`
- Когда фильтр уже активен, он остаётся активным пока ratio > 0.55 И цена падает

**Код** (`market_maker_bid_ask_glm.rs:707-738`):

```rust
let trend_should_block = if self.trend_filter_was_active {
    price_falling && inventory_ratio > 0.55   // keep blocking
} else {
    price_falling && inventory_ratio > 0.65   // start blocking
};
```

### 7.2. Volatility Gate

```rust
if realized_vol_bps > max_vol_bps (800) → buy_quantity = 0
if realized_vol_bps > panic_vol_bps (2000) → no_action (обе стороны отключены)
```

### 7.3. Emergency Taker Exits

Проверяются **до** ценообразования. При срабатывании — немедленный IOC SELL всей позиции:

| Exit | Условие | Текущий параметр |
|------|---------|------------------|
| Stop Loss | `pnl_bps <= -stop_loss_bps` | -500 bps (-5%) |
| Panic Vol | `realized_vol_bps > panic_vol_bps` | > 2000 bps (20%) |
| Max Hold | `hold_time >= max_hold_millis` | **ОТКЛЮЧЁН** (0) |

После taker exit → cooldown 5000ms (без новых покупок).

### 7.4. Order Book Imbalance Filter

```
if order_book_imbalance < -max_imbalance → block buys
```

Текущее значение: `max_imbalance = 1.0` → **отключён** (imbalance всегда в [-1, 1]).

### 7.5. Trade Flow Filter

```
if seller_initiated_volume / total_volume < min_sell_flow_fraction → block buys
```

Текущее значение: `min_sell_flow_fraction = 0.0` → **отключён**.

### 7.6. Minimum Edge Check

```
quoted_spread_bps = (sell_price - buy_price) / mid_price · 10000
if quoted_spread_bps < min_edge_bps · 2 (0.6 bps) → no_action
```

### 7.7. Requote Threshold

```
if |buy_change| < 0.1 bps AND |sell_change| < 0.1 bps → prices unchanged
```

В live-режиме trader пропустит cancel-replace. В бэктесте всё равно генерирует QuoteBothSides для трекера.

---

## 8. Исполнение ордеров

### 8.1. QuoteBothSides Handler (`trader.rs:2536-2827`)

Получает от стратегии:
```rust
QuoteBothSides {
    buy_price, buy_quantity,
    sell_price, sell_quantity,
    expected_edge_bps
}
```

**Шаги обработки:**

1. **Instant-fill cooldown check** — после мгновенного fill, пауза `min_order_rest_millis` (1 сек) перед повторным размещением. **ИСКЛЮЧЕНИЕ**: если обе стороны пусты (`both_sides_empty`) — cooldown обходится, чтобы бот не замолк.

2. **place_or_replace_side(BUY)** — размещает или заменяет bid.

3. **place_or_replace_side(SELL)** — размещает или заменяет ask.

4. **Fill detection** — если cancel-replace вернул ошибку -2022, проверяет — может быть, старый ордер заполнился (recovered fill).

5. **Rebalance trigger check** — если одна сторона не может разместиться 3+ тиков подряд ИЛИ стратегия возвращает qty=0 на 100+ тиков подряд → rebalance.

### 8.2. place_or_replace_side() (`trader.rs:640-1039`)

Атомарная операция размещения или замены одного ордера.

**Логика:**

```
1. qty = 0? → cancel existing, return Failed
2. Existing order + same price + not stale? → return Resting (skip API)
3. Existing order + too young (< min_rest)? → return Resting (skip API)
4. Existing order + different price? → cancel-replace (atomic API call)
5. No existing order? → place new limit order
6. Cancel-replace failed (-2022)? → check if old order filled (recovered fill)
```

### 8.3. Same-Price Skip (`trader.rs:709-724`)

```rust
if (active.price - new_price).abs() < tick_size * 0.5
    && (active.quantity - new_qty).abs() < step_size * 0.5
    && resting_ms < max_order_rest_millis   // не стал stale
{
    return PlaceResult::Resting;   // ничего не делаем
}
```

Это критично: без этой логики бот спамил бы API cancel-replace каждый тик (~100ms).

### 8.4. Max Order Rest (Stale Order Refresh)

Когда ордер стоит дольше `max_order_rest_millis` (30 сек), same-price skip **не срабатывает** → ордер пере-размещается с той же ценой для обновления позиции в очереди.

```
Ордер стоит 29 сек → same-price skip → ничего
Ордер стоит 31 сек → skip не работает → cancel-replace → свежий ордер в конце очереди
```

**Компромисс**: refresh каждые 30 сек = ~2 API calls/мин на сторону. Но обеспечивает актуальность позиции в очереди.

### 8.5. Fill Detection

Два механизма:

1. **Синхронный**: order placement возвращает `status = FILLED` сразу → `immediate fill registered`
2. **Асинхронный**: `poll_two_sided_orders()` проверяет статус resting orders каждые 2 секунды → `async fill detected`

При cancel-replace failure (-2022, «Order cancel-replace failed»):
- Старый ордер уже заполнился до того, как cancel успел его отменить
- Бот проверяет статус старого ордера → обнаруживает fill → `recovered fill from old order`
- Это **не ошибка**, а нормальное race condition

### 8.6. Instant-Fill Cooldown (`trader.rs:2550-2566`)

Когда ордер исполняется мгновенно при размещении (цена пересекла спред), бот ждёт `min_order_rest_millis` (1 сек) перед повторным размещением на этой стороне.

**Без этого**: лавина покупок — 139 BID fills за 30 секунд (Test 004).

**Bypass при both_sides_empty**: когда оба ордера исполнились, cooldown обходится, иначе бот замолкает без котировок.

```rust
let both_sides_empty = self.active_bid_order.is_none()
    && self.active_ask_order.is_none();

let bid_in_cooldown = !both_sides_empty     // ← bypass
    && self.last_bid_instant_fill_millis > 0
    && now_millis - self.last_bid_instant_fill_millis < cooldown_ms;
```

---

## 9. Ребалансировка

### 9.1. Триггеры

Два независимых триггера (OR):

| Триггер | Условие | Порог |
|---------|---------|-------|
| **Failure** | Одна сторона не может разместить ордер (rejected, insufficient balance) | 3+ подряд + cooldown 30 сек |
| **Zero-qty** | Стратегия возвращает qty=0 на одной стороне (фильтры блокируют) | 100+ тиков (~100 сек) + cooldown 30 сек |

### 9.2. Механизм

```
1. Отменить все открытые ордера
2. Получить балансы: DOGE_free, FDUSD_free
3. Получить рыночную цену (REST API если нет WebSocket)
4. Вычислить:
   total_value = FDUSD + DOGE · price
   target = total_value / 2
   delta = target - DOGE · price
5. Если |delta| > min_notional:
   delta > 0 → BUY |delta|/price DOGE (market order, taker fee)
   delta < 0 → SELL |delta|/price DOGE (market order, taker fee)
6. Сбросить счётчики
```

**Стоимость**: каждый rebalance = taker fee (0.1%) на объём ордера + потеря spread.

### 9.3. CLI Rebalance

```bash
cargo run --release -- --strategy market-maker-bid-ask-glm \
  --config config/trading_config_dogefdusd.toml rebalance
```

Использует REST API для получения цены (нет WebSocket без `trade` режима).

---

## 10. Конфигурация

### 10.1. Файл стратегии (`config/strategies/market_maker_bid_ask_glm.toml`)

```toml
# --- Ценообразование ---
base_half_spread_bps = 0.6        # Базовый полуспред
min_half_spread_bps = 0.3         # Минимальный полуспред
max_half_spread_bps = 10.0        # Максимальный полуспред
volatility_reference_bps = 50.0   # Эталонная волатильность
volatility_multiplier_min = 0.5   # Мин. множитель волатильности
volatility_multiplier_max = 4.0   # Макс. множитель волатильности
min_edge_bps = 0.3                # Мин. полный спред для котировки

# --- Инвентарь ---
max_inventory_fraction = 1.0      # Макс. инвентарь = 100% капитала
inventory_skew_factor = 200.0     # A-S gamma (risk aversion)

# --- Размер ордера ---
cash_fraction = 0.30              # Доля кэша на вход (не используется при flat sizing)
dynamic_sizing = false            # Динамический sizing отключён

# --- Выход ---
min_exit_edge_bps = 0.0           # Entry-price floor ОТКЛЮЧЁН
stop_loss_bps = 500.0             # Стоп-лосс: -5%
panic_vol_bps = 2000.0            # Паника при vol > 20%
max_hold_millis = 0               # Таймер удержания ОТКЛЮЧЁН
max_vol_bps = 800.0               # Блокировка покупок при vol > 8%

# --- Cooldown ---
cooldown_millis = 5000            # 5 сек после taker exit

# --- Фильтры ---
max_imbalance = 1.0               # Imbalance фильтр ОТКЛЮЧЁН
min_sell_flow_fraction = 0.0      # Trade flow фильтр ОТКЛЮЧЁН

# --- Requoting ---
requote_threshold_bps = 0.1       # Мин. сдвиг цены для re-quote
trade_window_millis = 30000       # Окно для микроструктурных метрик
```

### 10.2. Файл торговли (`config/trading_config_dogefdusd.toml`)

Ключевые параметры:

```toml
[position_sizing]
max_position_notional = 3.0       # Макс. размер ордера (нотионал)

[exchange]
maker_fee = 0.0                   # ZERO maker fee
taker_fee = 0.001                 # 0.1% taker fee
tick_size = 0.00001               # Минимальный шаг цены
step_size = 1.0                   # Минимальный шаг количества
min_notional = 1.0                # Минимальный нотионал ордера

[order_execution]
min_order_rest_millis = 1000      # Мин. время жизни ордера (1 сек)
max_order_rest_millis = 30_000    # Макс. время жизни ордера (30 сек → refresh)

[backtest_settings]
initial_capital = 112.0           # Начальный капитал
```

### 10.3. Хардкодированные константы (`trader.rs`)

```rust
ZERO_QTY_REBALANCE_THRESHOLD = 100    // Тиков с нулевым qty до rebalance
MIN_REBALANCE_INTERVAL_MS = 30_000    // Мин. интервал между rebalance
FAILURE_REBALANCE_THRESHOLD = 3       // Подряд неудач до rebalance
TWO_SIDED_COOLDOWN_MS = 10_000        // Cooldown после двойного failure
```

### 10.4. Хардкодированные константы (стратегия)

```rust
SLOW_EMA_ALPHA = 0.002                // Halflife ~35 сек
join_best_bid_threshold = 0.50        // Join best_bid если ratio < 0.50
inventory_bid_offset_start = 0.55     // Extra offset если ratio > 0.55
inventory_bid_offset_scale = 10.0     // bps на 0.10 ratio excess
```

---

## 11. Жизненный цикл тика

Каждый bookTicker event (≈100ms) запускает цикл:

```
1.  [Strategy] Получить bookTicker → (best_bid, best_ask, mid)
2.  [Strategy] Обновить slow_ema_mid
3.  [Strategy] Проверить emergency exits → если да → IOC SELL
4.  [Strategy] Проверить panic vol → если да → no_action
5.  [Strategy] Вычислить A-S reservation_price
6.  [Strategy] Вычислить dynamic half_spread
7.  [Strategy] Вычислить raw buy_price, sell_price
8.  [Strategy] Применить join-best-price
9.  [Strategy] Применить inventory bid offset
10. [Strategy] Проверить spread-crossing guard
11. [Strategy] Проверить min_edge
12. [Strategy] Проверить buy filters (vol, trend, cooldown, imbalance, flow)
13. [Strategy] Вычислить quantities (flat min_qty)
14. [Strategy] Проверить balance constraints
15. [Strategy] Вернуть QuoteBothSides { buy_price, buy_qty, sell_price, sell_qty }

16. [Trader] Получить QuoteBothSides
17. [Trader] Проверить instant-fill cooldown (bypass если both_sides_empty)
18. [Trader] BID: place_or_replace_side()
    a. Qty = 0? → cancel existing
    b. Same price + not stale? → skip (Resting)
    c. Too young (< min_rest)? → skip (Resting)
    d. Different price / stale? → cancel-replace via REST API
    e. No existing? → new limit order via REST API
19. [Trader] ASK: place_or_replace_side() (аналогично)
20. [Trader] Обработать fill если cancel-replace failed (race condition recovery)
21. [Trader] Обновить active_bid_order, active_ask_order
22. [Trader] Проверить rebalance triggers
23. [Trader] poll_two_sided_orders() каждые 2 сек (async fill detection)
```

**Типичный тик** (без изменения цены):
- Стратегия вычисляет те же цены → same-price skip → **0 API calls**

**Тик с изменением цены на 1 tick:**
- Стратегия вычисляет новые цены → cancel-replace bid + cancel-replace ask → **2 API calls**

**Тик с fill:**
- cancel-replace вернул -2022 → recovered fill → новый ордер → **2-3 API calls**

---

## 12. Производительность

### Результаты тестов (последние)

| Тест | Длительность | BID:ASK | Fills | Rebalances | PnL | Win Rate |
|------|-------------|---------|-------|------------|-----|----------|
| 023 | 30 мин | 0.54:1 | 40 | 0 | +0.012 | 100% |
| 024 | 60 мин | **1.03:1** | **65** | **0** | **+0.022** | **94%** |

### Экономика

```
Fills per hour:     ~65
Matched pairs:      ~32/hour
Profit per pair:    ~0.0007 FDUSD (1 tick spread · 12 DOGE)
Hourly PnL:         ~0.022 FDUSD
Daily PnL:          ~0.52 FDUSD
Daily return:       ~0.47% on $110 capital
Annualized:         ~172% (без компаундирования)
```

### API Usage

```
API calls per minute (typical):
  - Stale refreshes: ~4 (2 per side, every 30 sec)
  - Price-change cancel-replaces: ~2-6
  - Fill recovery: ~1-2
  Total: ~7-12 calls/min ≈ 420-720 calls/hour
  Binance limit: 1200/min → используем <1%
```

---

## 13. Известные ограничения

### 13.1. Adverse Selection в трендовом рынке

При сильном тренде (>50 bps за 5 мин) одна сторона систематически заполняется чаще. Trend filter с гистерезисом частично защищает, но не полностью. При сильном тренде стратегия может накопить нежелательный инвентарь.

### 13.2. Flat sizing = минимальная прибыль на трейд

12 DOGE · 1 tick = ~$0.0001 прибыли за пару. Масштабирование размера увеличило бы прибыль, но предыдущие попытки (A-S scaling, cycle-based) создавали проблемы. Возможен следующий шаг: увеличение min_qty до 24-36 DOGE для удвоения/утроения PnL.

### 13.3. BNB расход

Taker fees при rebalance оплачиваются в BNB. Текущий баланс ~0.007 BNB. При 0 rebalances за последние тесты расход минимален, но при неблагоприятных условиях может потребоваться пополнение.

### 13.4. Queue Position

Обновление ордеров каждые 30 сек ставит свежий ордер в конец очереди. На ликвидных парах это может уменьшить fill probability. Но для DOGEFDUSD с малым объёмом на каждом price level это не критично.

### 13.5. Отключённые фильтры

Несколько фильтров adverse selection отключены:
- `max_imbalance = 1.0` → order book imbalance не проверяется
- `min_sell_flow_fraction = 0.0` → trade flow не проверяется
- `max_hold_millis = 0` → нет таймера удержания

Эти фильтры были отключены из-за ложных срабатываний, но могут быть полезны после дальнейшей калибровки.

---

## Запуск

### Live Trading (30 мин)
```bash
cargo run --release -- \
  --strategy market-maker-bid-ask-glm \
  --config config/trading_config_dogefdusd.toml \
  --duration-secs 1800 trade
```

### Rebalance (50/50)
```bash
cargo run --release -- \
  --strategy market-maker-bid-ask-glm \
  --config config/trading_config_dogefdusd.toml rebalance
```

### С логированием
```bash
truncate -s0 /tmp/live_trade.txt
cargo run --release -- \
  --strategy market-maker-bid-ask-glm \
  --config config/trading_config_dogefdusd.toml \
  --duration-secs 1800 trade 2>&1 | tee /tmp/live_trade.txt
```

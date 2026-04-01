# Тестирование моделей

Этот документ нужен как короткая и понятная карта: какие модели есть, как они тестировались, что у них сейчас происходит и почему.

## Как читать таблицы

| Поле | Что значит |
| --- | --- |
| `entries` | сколько раз стратегия открыла позицию |
| `closed_trades` | сколько сделок закрылось и попало в PnL |
| `realized_pnl` | итоговый реализованный PnL |
| `blocked_*` | сколько раз вход был отклонен конкретным фильтром |
| cost gate | новый честный фильтр: вход только если ожидаемое edge покрывает модельную round-trip taker cost |

## Какие модели сейчас есть

| Модель | Тип | Источник сигнала | Статус |
| --- | --- | --- | --- |
| `trade-flow-momentum` | taker | агрессивный buyer flow + drift | исследовательская, не прибыльна |
| `liquidity-sweep-reversal` | taker | локальный sweep вниз + reclaim вверх | исследовательская, не прибыльна |
| `trade-flow-reclaim` | taker | pullback от high + reclaim от low | исследовательская, не прибыльна |
| `microprice-imbalance-maker` | maker | microprice/book imbalance | только baseline/scaffolding, live profit не подтвержден |

## Текущий итог по моделям

Текущий основной батч для честного сравнения:

| Датасеты | Формат |
| --- | --- |
| `DOGEUSDT-trades-2025-06-28.zip` | trade-only |
| `DOGEUSDT-trades-2025-08-08.zip` | trade-only |
| `DOGEUSDT-trades-2026-03-30.zip` | trade-only |

### Что было до cost gate

| Модель | Закрытых сделок | Итоговый PnL | Что это означало |
| --- | --- | --- | --- |
| `trade-flow-momentum` | 19 | `-0.9666310465` | модель торгует чаще всех, но все закрытые сделки убыточны |
| `liquidity-sweep-reversal` | 4 | `-0.2152563187` | торгует реже, но edge все равно не покрывает издержки |
| `trade-flow-reclaim` | 1 | `-0.0435852762` | выглядела "лучше" только потому, что почти не торговала |

### Что стало после cost gate

| Модель | Закрытых сделок | Итоговый PnL | Что реально произошло |
| --- | --- | --- | --- |
| `trade-flow-momentum` | 0 | `0.0` | честный фильтр вырезал оставшиеся слабые входы |
| `liquidity-sweep-reversal` | 0 | `0.0` | rebound-сигналы обычно слишком слабы для taker round trip |
| `trade-flow-reclaim` | 0 | `0.0` | редкие сигналы тоже не проходят по edge-vs-cost |

## Почему модели сейчас не торгуют

Агрегированные блокировки по последнему `strategy-diagnostics` на том же 3-дневном батче:

| Модель | Главный блокер | Значение | Второй блокер | Значение | Cost gate | Что это значит |
| --- | --- | --- | --- | --- | --- | --- |
| `trade-flow-momentum` | `blocked_min_trades` | `1,742,822` | `blocked_flow` | `220,282` | `1,011` | подходящих окон очень мало, а оставшиеся часто все равно слабые |
| `liquidity-sweep-reversal` | `blocked_min_trades` | `1,901,773` | `blocked_sweep_drop` | `240,883` | `222` | сначала почти не находится полноценный sweep, потом edge часто мал |
| `trade-flow-reclaim` | `blocked_min_trades` | `1,796,392` | `blocked_pullback_band` | `337,215` | `12` | стратегия сверхселективна и почти не видит валидных сетапов |

Короткий вывод из таблицы:

| Наблюдение | Интерпретация |
| --- | --- |
| `blocked_min_trades` огромный у всех | хорошие окна редки сами по себе |
| `blocked_cost_gate` есть у всех taker-моделей | даже прошедшие сетапы часто не перекрывают taker drag |
| `trade-flow-reclaim` почти не блокируется cost gate | проблема не в скрытой прибыли, а в том, что модель почти не находит живых входов |

## Что происходит с maker-моделью

| Модель | Где проверялась | Результат | Ограничение |
| --- | --- | --- | --- |
| `microprice-imbalance-maker` | live validation `1m / 5m / 10m` | `0` входов, `0` сделок | текущая maker-симуляция еще baseline, queue/fill realism приближенный |
| `microprice-imbalance-maker` | rotated gzip replay smoke | `0` входов, `0` сделок | проверена инфраструктура replay, а не наличие alpha |

## Что показал новый captured replay

Новый dataset:

| Датасет | Формат | Что видно |
| --- | --- | --- |
| `live_train_20260401_210321.jsonl` | captured `trade + bookTicker + depth` | dense quote coverage, depth present, но рыночный режим оказался слишком слабым для текущих базовых моделей |

### Базовые стратегии на новом capture

| Модель | `entries` | `closed_trades` | Итог |
| --- | --- | --- | --- |
| `trade-flow-momentum` | `0` | `0` | сетапы иногда почти формируются, но edge после cost отрицательный |
| `liquidity-sweep-reversal` | `0` | `0` | sweep/rebound форма почти не возникает |
| `trade-flow-reclaim` | `0` | `0` | pullback/reclaim геометрия почти не появляется |
| `microprice-imbalance-maker` | `0` | `0` | maker baseline по-прежнему не дает входов |

### Главные блокеры по `strategy-diagnostics`

| Модель | Главный блокер | Что это означает |
| --- | --- | --- |
| `trade-flow-momentum` | `blocked_min_trades`, `blocked_drift_band`, `blocked_flow` | поток и drift слишком слабы, а остаточный edge не покрывает taker cost |
| `liquidity-sweep-reversal` | `blocked_sweep_drop`, `blocked_min_trades` | рынок редко дает настоящий sweep вниз нужного масштаба |
| `trade-flow-reclaim` | `blocked_pullback_band`, `blocked_min_trades` | не формируется нужный pullback/reclaim паттерн |

Короткий вывод:

| Наблюдение | Интерпретация |
| --- | --- |
| quote/depth есть | проблема не в отсутствии данных |
| все 4 модели дали `0` входов | текущие правила не находят edge в этом live-like режиме |
| momentum ближе всех к сигналу, но все равно не проходит | слабый microstructure move не выдерживает taker drag |

## Что тестирует каждая команда

| Команда | Что проверяет | Когда использовать |
| --- | --- | --- |
| `compare --uris ... --strategies ...` | итоговый PnL, сделки, ранжирование моделей на одинаковом батче | чтобы понять, кто "меньше плох" или лучше |
| `strategy-diagnostics --uris ... --strategies ...` | почему стратегия не входит: какие фильтры блокируют сигнал | когда непонятно, почему модель молчит |
| `trade-attribution --uris ... --strategies ...` | почему сделки убыточны: expected edge, fee, slippage, exit reason | когда модель торгует, но PnL плохой |
| `optimize --strategy ...` | подбор одного параметра на батче | когда уже понятно, какой фильтр хотим двигать |
| `walk-forward --strategy ...` | проверка, переносится ли настройка out-of-sample | чтобы не обманывать себя переоптимизацией |
| `features --uri ...` | выгрузка признаков для research/ML/regime analysis | когда нужно искать новые режимы или фильтры |

## Простое объяснение без терминов

| Вопрос | Ответ |
| --- | --- |
| Почему раньше были сделки, а теперь нет? | Потому что раньше модель могла входить даже в слабые сигналы, которые потом проигрывали на комиссиях и slippage. Теперь такие входы честно запрещены. |
| Это хорошо или плохо? | Это хорошо для честности исследования и плохо для текущего PnL: стало ясно, что у taker-моделей пока нет достаточного edge. |
| Какая модель лучшая? | Сейчас ни одна taker-модель не является прибыльной. `trade-flow-reclaim` была лишь "least bad" из-за почти полного отсутствия сделок. |
| Значит все сломано? | Нет. Инфраструктура работает, тесты проходят, но научный результат сейчас отрицательный: найденные сигналы не покрывают издержки. |

## Что делать дальше

| Направление | Почему это логично |
| --- | --- |
| `taker entry + maker exit` | уменьшает round-trip drag, который сейчас убивает edge |
| более жесткий regime filter | нужен, чтобы не торговать low-activity шум, как в `live_train_20260401_210321.jsonl` |
| quote/depth-first research | без книги и котировок трудно найти настоящий microstructure edge |

## Последний проверенный набор команд

| Команда | Итог |
| --- | --- |
| `cargo test -p strategies` | все strategy tests green |
| `cargo run -p binance-bot -- compare --uris data/binance/daily/trades/DOGEUSDT/DOGEUSDT-trades-2025-06-28.zip data/binance/daily/trades/DOGEUSDT/DOGEUSDT-trades-2025-08-08.zip data/binance/daily/trades/DOGEUSDT/DOGEUSDT-trades-2026-03-30.zip --strategies trade-flow-momentum,liquidity-sweep-reversal,trade-flow-reclaim` | после cost gate все три taker-модели дали `0` сделок |
| `cargo run -p binance-bot -- strategy-diagnostics --uris data/binance/daily/trades/DOGEUSDT/DOGEUSDT-trades-2025-06-28.zip data/binance/daily/trades/DOGEUSDT/DOGEUSDT-trades-2025-08-08.zip data/binance/daily/trades/DOGEUSDT/DOGEUSDT-trades-2026-03-30.zip --strategies trade-flow-momentum,liquidity-sweep-reversal,trade-flow-reclaim` | подтвердило, что модели блокируются и редкостью сетапов, и недостаточным edge после cost |
| `cargo run -p binance-bot -- compare --uris data/binance/capture/DOGEUSDT/live_train_20260401_210321.jsonl --strategies trade-flow-momentum,liquidity-sweep-reversal,trade-flow-reclaim,microprice-imbalance-maker` | на новом capture все 4 базовые стратегии дали `0` входов / `0` сделок |
| `cargo run -p binance-bot -- strategy-diagnostics --uris data/binance/capture/DOGEUSDT/live_train_20260401_210321.jsonl --strategies trade-flow-momentum,liquidity-sweep-reversal,trade-flow-reclaim,microprice-imbalance-maker` | показало, что проблема в слабом режиме и недостающем edge, а не в поломке replay |

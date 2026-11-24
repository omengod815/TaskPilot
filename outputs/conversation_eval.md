# Conversation multi-agent evaluation

- generated: `2026-11-24T11:47:53+00:00`
- mode: **real**
- pass: **8/8**

| task | status | agents | tools | model calls | sec | pass |
|---|---|---|---|---:|---:|---|
| csv_metrics | completed | data, writing | analyze_csv | 6 | 33.48 | ✅ |
| research_summary | completed | research, general, writing | search_web, search_web | 8 | 71.78 | ✅ |
| safe_calc | completed | general, writing | safe_calculate | 5 | 21.59 | ✅ |
| writing_only | completed | writing | - | 3 | 29.14 | ✅ |
| data_limitations | completed | data, writing | analyze_csv | 6 | 68.99 | ✅ |
| mixed_research_data | completed | research, data, writing | search_web, search_web, analyze_csv | 9 | 97.69 | ✅ |
| general_explain | completed | research, general, writing | search_web, search_web | 7 | 88.82 | ✅ |
| csv_follow_up | completed | data, writing, writing | analyze_csv | 8 | 67.99 | ✅ |

## Summary

- completed rate: 100.00%
- agent hit rate: 100.00%
- tool hit rate: 100.00%
- answer hit rate: 100.00%
- avg model calls: 6.5
- avg tokens: 11413.1
- avg elapsed: 59.93s

> This is a fixed regression suite, not a statistical benchmark. Mock mode validates routing/plumbing; real mode is an optional smoke-level signal.

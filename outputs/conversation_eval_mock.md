# Conversation multi-agent evaluation

- generated: `2026-11-24T11:39:43+00:00`
- mode: **mock**
- pass: **8/8**

| task | status | agents | tools | model calls | sec | pass |
|---|---|---|---|---:|---:|---|
| csv_metrics | completed | data, writing | analyze_csv | 0 | 0.09 | ✅ |
| research_summary | completed | research, writing | search_web | 0 | 0.09 | ✅ |
| safe_calc | completed | general, writing | safe_calculate | 0 | 0.06 | ✅ |
| writing_only | completed | writing | - | 0 | 0.05 | ✅ |
| data_limitations | completed | data, writing | analyze_csv | 0 | 0.06 | ✅ |
| mixed_research_data | completed | data, research, writing | analyze_csv, search_web | 0 | 0.1 | ✅ |
| general_explain | completed | general, writing | - | 0 | 0.05 | ✅ |
| csv_follow_up | completed | data, writing, writing | analyze_csv | 0 | 0.13 | ✅ |

## Summary

- completed rate: 100.00%
- agent hit rate: 100.00%
- tool hit rate: 100.00%
- answer hit rate: 100.00%
- avg model calls: 0.0
- avg tokens: 0.0
- avg elapsed: 0.08s

> This is a fixed regression suite, not a statistical benchmark. Mock mode validates routing/plumbing; real mode is an optional smoke-level signal.

"""Offline MCP agents.

These run on cron schedules outside the live trading pipeline. They
read from production storage but write only to designated output
files — never to code, config, or the database.

See ARCHITECTURE.md §13.1 (Quant Data Scientist) and §13.2 (Macro
Regime).
"""

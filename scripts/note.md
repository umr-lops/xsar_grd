# Fusionner les CSV

```bash
head -n 1 logs/l1c_*.csv | head -n 1 > l1c_report_all.csv
tail -n +2 -q logs/l1c_*.csv >> l1c_report_all.csv
```
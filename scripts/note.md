# Merge CSV

```bash
awk 'FNR==1 && NR!=1 {next} {print}' *.csv > l1c_report.csv
```
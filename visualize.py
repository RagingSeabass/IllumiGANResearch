
# Load visualization into csv
import re

file = 'output/_default/reports/train.log'
lines = [line.rstrip('\n') for line in open(file)]

for line in lines:

    # Get date: \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}
    date = re.match(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', line).group()
    
    







import subprocess

# pipdeptree çıktısını al ve filtrele
result = subprocess.run(['pipdeptree', '--freeze', '--warn', 'silence'], stdout=subprocess.PIPE)
output = result.stdout.decode('utf-8')

# Regex ile satırları filtrele
import re
filtered_lines = re.findall(r'^[\w0-9\-=.]+', output, re.MULTILINE)

# Sonucu requirements.txt'ye yaz
with open('requirements.txt', 'w') as f:
    f.write("\n".join(filtered_lines))

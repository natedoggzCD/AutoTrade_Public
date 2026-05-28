import os

file_path = 'autotrade/core/autonomous_agent.py'
with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

start_idx = 5022 # 5023 - 1
end_idx = 5345 # 5346 - 1

print(f"DEBUG: Start line {start_idx+1}: {lines[start_idx].strip()}")
print(f"DEBUG: End line {end_idx+1}: {lines[end_idx].strip()}")

# Verify we're at the right spot before deleting
if '_generate_chart_LEGACY_DELETE_ME' in lines[start_idx] and '_technical_analysis' in lines[end_idx]:
    new_lines = lines[:start_idx] + [
        '    def _generate_chart_LEGACY_DELETE_ME(self, symbol: str, backtest: Dict = None):\n',
        '        \"\"\"Deprecated legacy method, redirecting to new optimized version.\"\"\"\n',
        '        return self._generate_chart(symbol, backtest)\n',
        '\n'
    ] + lines[end_idx:]
    
    with open(file_path + '.tmp', 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    os.replace(file_path + '.tmp', file_path)
    print('SUCCESS: Legacy code removed and redirected.')
else:
    # Try a broader search if indices shifted
    found_start = -1
    found_end = -1
    for i, line in enumerate(lines):
        if 'def _generate_chart_LEGACY_DELETE_ME' in line:
            found_start = i
        if found_start != -1 and 'def _technical_analysis' in line:
            found_end = i
            break
            
    if found_start != -1 and found_end != -1:
        print(f"Adjusting: Found start at {found_start+1}, end at {found_end+1}")
        new_lines = lines[:found_start] + [
            '    def _generate_chart_LEGACY_DELETE_ME(self, symbol: str, backtest: Dict = None):\n',
            '        \"\"\"Deprecated legacy method, redirecting to new optimized version.\"\"\"\n',
            '        return self._generate_chart(symbol, backtest)\n',
            '\n'
        ] + lines[found_end:]
        with open(file_path + '.tmp', 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        os.replace(file_path + '.tmp', file_path)
        print('SUCCESS: Legacy code removed and redirected (with adjustment).')
    else:
        print(f'ERROR: Could not find method markers. Start found: {found_start+1}, End found: {found_end+1}')

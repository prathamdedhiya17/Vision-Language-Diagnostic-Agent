from pathlib import Path

root = Path("DeepPCB-master/PCBData")

results = []

for subdir in root.iterdir():
    if subdir.is_dir():
        for nested in subdir.iterdir():
            if nested.is_dir():
                file_count = sum(1 for f in nested.iterdir() if f.is_file())
                results.append((subdir.name, nested.name, file_count))

# print results
for parent, child, count in results:
    print(f"{parent}/{child}: {count}")
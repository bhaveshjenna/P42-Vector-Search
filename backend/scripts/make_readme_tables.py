import json
import argparse
import sys
import os
import re

def generate_table(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    avg = data['avg']
    std = data['std']
    
    def fmt(m_key):
        m = avg[m_key]
        s = std[m_key]
        return f"{m:.4f} ± {s:.4f}"

    lines = [
        "| Metric | Text -> Image (T2I) | Image -> Text (I2T) |",
        "|---|---|---|",
        f"| **Recall@1** | {fmt('t2i_r1')} | {fmt('i2t_r1')} |",
        f"| **Recall@5** | {fmt('t2i_r5')} | {fmt('i2t_r5')} |",
        f"| **Recall@10** | {fmt('t2i_r10')} | {fmt('i2t_r10')} |",
        f"| **MRR** | {fmt('t2i_mrr')} | {fmt('i2t_mrr')} |"
    ]
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true', help="Check against README.md")
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    json_1k = os.path.join(repo_root, "results_1000.json")
    json_full = os.path.join(repo_root, "results_full.json")

    table_1k = generate_table(json_1k)
    table_full = generate_table(json_full)
    
    generated = f"### 1K Gallery Table:\n{table_1k}\n\n### Full Gallery Table:\n{table_full}"

    print(generated)

    if args.check:
        readme_path = os.path.join(repo_root, "README.md")
        with open(readme_path, 'r', encoding='utf-8') as f:
            readme_content = f.read()
        
        if table_1k not in readme_content:
            print("ERROR: 1K table mismatch in README.md")
            sys.exit(1)
        if table_full not in readme_content:
            print("ERROR: Full table mismatch in README.md")
            sys.exit(1)
            
        print("README tables match exactly.")

if __name__ == '__main__':
    main()

import os
import zipfile
from pathlib import Path

def create_submission_zip(team_name: str = "top_k_resolution"):
    project_dir = Path(__file__).parent
    output_zip = project_dir / f"{team_name}_submission.zip"
    
    print(f"Creating submission zip: {output_zip}", flush=True)
    
    code_files = [
        "config.py",
        "io_utils.py",
        "preprocess.py",
        "blocking.py",
        "features.py",
        "model.py",
        "decision.py",
        "evaluate.py",
        "pipeline.py",
        "main.py"
    ]
    
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
        # 1. Add output/ files
        matching_file = project_dir / "output" / "matching_results.tsv"
        candidate_file = project_dir / "output" / "candidate_pairs.tsv"
        
        if matching_file.exists():
            zipf.write(matching_file, "output/matching_results.tsv")
        if candidate_file.exists():
            zipf.write(candidate_file, "output/candidate_pairs.tsv")
            
        # 2. Add Documentation_template.md
        doc_file = project_dir / "Documentation_template.md"
        if doc_file.exists():
            zipf.write(doc_file, "Documentation_template.md")
            
        # 3. Add code/business_entity_resolution/
        code_base = "code/business_entity_resolution"
        
        readme_file = project_dir / "README.md"
        req_file = project_dir / "requirements.txt"
        
        if readme_file.exists():
            zipf.write(readme_file, f"{code_base}/README.md")
        if req_file.exists():
            zipf.write(req_file, f"{code_base}/requirements.txt")
            
        # Package code files into src/ inside submission zip
        for cf in code_files:
            full_path = project_dir / cf
            if full_path.exists():
                zipf.write(full_path, f"{code_base}/src/{cf}")
                
    print(f"Successfully packaged: {output_zip} ({os.path.getsize(output_zip)} bytes)", flush=True)

if __name__ == "__main__":
    create_submission_zip()

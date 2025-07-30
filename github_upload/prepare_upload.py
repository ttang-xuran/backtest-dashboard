#!/usr/bin/env python3
"""
Upload Preparation Helper
This script prepares all files for easy GitHub upload
"""

import os
import shutil
import zipfile
from pathlib import Path

def prepare_upload():
    """Prepare files for GitHub upload"""
    print("🚀 Preparing your files for GitHub upload...")
    
    # Create upload directory
    upload_dir = Path("github_upload")
    if upload_dir.exists():
        shutil.rmtree(upload_dir)
    upload_dir.mkdir()
    
    # Create templates subdirectory
    templates_dir = upload_dir / "templates"
    templates_dir.mkdir()
    
    # Essential files for deployment
    essential_files = [
        "app.py",
        "comprehensive_backtest.py", 
        "requirements.txt",
        "Procfile",
        "runtime.txt",
        "config.json",
        "app.json",
        "vercel.json",
        ".gitignore"
    ]
    
    # All Python files
    python_files = [f for f in os.listdir(".") if f.endswith(".py") and not f.startswith("test_")]
    
    # All documentation files
    doc_files = [f for f in os.listdir(".") if f.endswith(".md")]
    
    # Combine all files to upload
    files_to_upload = list(set(essential_files + python_files + doc_files))
    
    # Copy files
    copied_files = []
    for file in files_to_upload:
        if os.path.exists(file):
            shutil.copy2(file, upload_dir / file)
            copied_files.append(file)
            print(f"✅ Copied: {file}")
    
    # Copy templates
    if os.path.exists("templates/dashboard.html"):
        shutil.copy2("templates/dashboard.html", templates_dir / "dashboard.html")
        copied_files.append("templates/dashboard.html")
        print("✅ Copied: templates/dashboard.html")
    
    # Create a zip file for easy upload
    print("\n📦 Creating zip file for easy upload...")
    with zipfile.ZipFile("backtest-dashboard.zip", "w") as zipf:
        for file in copied_files:
            if file.startswith("templates/"):
                zipf.write(upload_dir / "templates" / "dashboard.html", "templates/dashboard.html")
            else:
                zipf.write(upload_dir / file, file)
    
    print(f"\n🎉 SUCCESS! Prepared {len(copied_files)} files for upload")
    print("\n📁 Files ready in two ways:")
    print("   1. Individual files in 'github_upload/' folder")
    print("   2. Single zip file: 'backtest-dashboard.zip'")
    
    print("\n🚀 NEXT STEPS:")
    print("1. Go to github.com and create new repository 'backtest-dashboard'")
    print("2. Either:")
    print("   A) Upload the zip file and extract it, OR")
    print("   B) Drag files from 'github_upload/' folder")
    print("3. Deploy to Railway!")
    
    return upload_dir, copied_files

if __name__ == "__main__":
    try:
        upload_dir, files = prepare_upload()
        print(f"\n✅ Ready to upload {len(files)} files to GitHub!")
    except Exception as e:
        print(f"❌ Error: {e}")
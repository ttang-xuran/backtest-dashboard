# 📁 **SUPER SIMPLE FILE UPLOAD GUIDE**

## **Method 1: Drag & Drop (Easiest)**

### **Step 1: Go to GitHub**
1. Open your web browser
2. Go to **github.com**
3. Sign in to your account

### **Step 2: Create Repository**
1. Click the green **"New"** button (or **"+"** in top right → "New repository")
2. Repository name: `backtest-dashboard`
3. Make sure it's **Public** ✅
4. Click **"Create repository"**

### **Step 3: Upload Files**
1. You'll see a page that says "Quick setup"
2. Click **"uploading an existing file"** link
3. **This opens a file upload page**

### **Step 4: Select Files to Upload**
**IMPORTANT: Upload these files (ignore folders that start with dot or backtest_env):**

**✅ Upload these files:**
- `app.py` ← Main dashboard file
- `comprehensive_backtest.py` ← Backtest engine
- `requirements.txt` ← Dependencies
- `Procfile` ← Deployment config
- `runtime.txt` ← Python version
- `config.json` ← Settings (now safe!)
- All `.md` files (README.md, DEPLOYMENT_GUIDE.md, etc.)
- All `.py` files
- `templates/dashboard.html` ← Web interface

**❌ DON'T upload these:**
- `backtest_env/` folder (virtual environment)
- `venv/` folder  
- `__pycache__/` folders
- `.png` files (charts - they're generated automatically)
- `stat_arb_bot.log` (log file)

### **Step 5: How to Upload**

**Option A: Drag and Drop (Easiest)**
1. Open your file manager/finder
2. Navigate to `/home/ttang/tony-project/`
3. Select the files listed above
4. **Drag them into the GitHub upload box**
5. Wait for upload to complete

**Option B: Click to Select**
1. Click **"choose your files"** 
2. Navigate to `/home/ttang/tony-project/`
3. Select multiple files (Ctrl+Click or Cmd+Click)
4. Click **"Open"**

### **Step 6: Upload the templates folder**
1. After uploading the main files, click **"Upload files"** again
2. Create a folder called `templates`
3. Upload `dashboard.html` into that folder

### **Step 7: Commit**
1. Scroll down to "Commit changes"
2. Leave the default message
3. Click **"Commit new files"**

---

## **Method 2: Create Files One by One (If drag-drop doesn't work)**

1. In your GitHub repository, click **"Create new file"**
2. Type the filename (e.g., `app.py`)
3. Copy and paste the content from your local file
4. Click **"Commit new file"**
5. Repeat for each file

---

## **✅ You'll Know It Worked When:**
- You see all your files listed in the GitHub repository
- You can click on `app.py` and see the code
- You see the green **"Code"** button in your repo

---

## **🚀 After Upload is Complete:**
1. Go to **railway.app**
2. Login with GitHub
3. Click **"New Project"** → **"Deploy from GitHub repo"**
4. Select **"backtest-dashboard"**
5. **Your website goes live in 2-3 minutes!**

---

## **Need Help?**
If you get stuck:
1. Make sure you're signed into GitHub
2. Try refreshing the page
3. Use "Create new file" method instead of drag-drop
4. Upload 5-10 files at a time (not all at once)

**The most important files are: `app.py`, `comprehensive_backtest.py`, `requirements.txt`, `Procfile`, and `templates/dashboard.html`**
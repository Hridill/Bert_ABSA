# GitHub Upload Guide

## Prerequisites

1. **Configure Git** (if not already done):
   ```bash
   git config --global user.name "Your Name"
   git config --global user.email "your.email@example.com"
   ```

2. **Install GitHub CLI** (optional but recommended) or use GitHub website

## Steps to Upload to GitHub

### Step 1: Create a GitHub Repository

#### Option A: Using GitHub Website (Recommended for beginners)
1. Go to https://github.com and sign in
2. Click the "+" icon in the top right corner
3. Select "New repository"
4. Choose a repository name (e.g., "bert-sentiment-analysis" or "major")
5. **DO NOT** initialize with README, .gitignore, or license (since we already have these)
6. Click "Create repository"

#### Option B: Using GitHub CLI
```bash
gh repo create major --public --source=. --remote=origin --push
```

### Step 2: Add Remote and Push

After creating the repository on GitHub, you'll see instructions. Run these commands:

```bash
# Add the remote repository (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git

# Verify the remote was added
git remote -v

# Push to GitHub (if this is your first push)
git branch -M main
git push -u origin main
```

### Alternative: If repository already exists with files
If you accidentally initialized the GitHub repo with a README:
```bash
git pull origin main --allow-unrelated-histories
git push -u origin main
```

### Step 3: Verify Upload

1. Go to your repository on GitHub
2. You should see all your files uploaded
3. The README.md should be visible on the main page

## Current Repository Status

✅ Git initialized
✅ .gitignore created (excludes large files, logs, models, cache)
✅ Files staged
✅ Initial commit created

## What's Excluded from GitHub

The `.gitignore` file excludes:
- Large model files (.bin, .pth, .pt)
- Model versions directory
- Pre-trained BERT models in Input/
- Log files
- Python cache files (__pycache__)
- Virtual environment (venv/)
- IDE files (.vscode, .idea)

## Troubleshooting

### If you get authentication errors:
1. **Personal Access Token** (recommended):
   - Go to GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
   - Generate new token with `repo` scope
   - Use token as password when pushing

2. **GitHub CLI** (easier):
   ```bash
   gh auth login
   ```

### If push is rejected:
```bash
git pull origin main --rebase
git push -u origin main
```

### To check repository status:
```bash
git status
git log --oneline
git remote -v
```

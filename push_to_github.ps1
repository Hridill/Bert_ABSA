# PowerShell script to push to GitHub
# Replace YOUR_USERNAME and REPO_NAME with your actual GitHub username and repository name

param(
    [Parameter(Mandatory=$true)]
    [string]$GitHubURL
)

Write-Host "Adding remote repository..." -ForegroundColor Green
git remote add origin $GitHubURL

Write-Host "Verifying remote..." -ForegroundColor Green
git remote -v

Write-Host "Renaming branch to main (if needed)..." -ForegroundColor Green
git branch -M main

Write-Host "Pushing to GitHub..." -ForegroundColor Green
git push -u origin main

Write-Host "Done! Check your repository on GitHub." -ForegroundColor Green

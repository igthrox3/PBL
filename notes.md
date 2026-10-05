# this is for vidio 
https://youtu.be/OssY5pzOyo0?si=iK3_7j23-ls7Ykk1
#we can refer github 
https://github.com/iamtheazizul/RL-Traffic.git

2) bro if u are copyin any repo do this becz they might have their own git so it iwll be difficult to get the code content when u import it  to ur local machine bro 
 # Remove it from index first
git rm --cached RL_signals

# Delete the nested .git folder
rm -rf RL_signals/.git        # Linux/Mac
# Windows PowerShell:
# Remove-Item -Recurse -Force RL_signals/.git

# Now add everything normally
git add .
git commit -m "Add RL_signals as regular folder" 
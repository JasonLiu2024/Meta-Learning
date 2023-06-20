# **MetaLearning**

# **Setup and How-to-run**
<span style="color:#33FF9E">

**pov: you need to upload file F to folder T on your remote server**

</span>

>scp -rp path_of_F server_address:path_of_T

where: 
<br>&emsp;path_of_F is the path on your computer to F
<br>&emsp;server_address is the address to your remote server
<br>&emsp;(this is of the form username@servername <- servername usually ends in @edu)
<br>&emsp;example: jason@somelab.something.school.edu
<br>&emsp;path_of_T is the path on the remote server to T

<span style="color:#33FF9E">

**pov: you need to run jupyter notebook while you are away (you have tmux & nbconvert)**

</span>

<br>**tmux** is a tool to run remote server. When you open a 'session' and run stuff in that session, it keeps running–even if you are not connected!
<br>**nbconvert** does stuff to notebook from terminal
<br>to go to session:
>tmux attach-session -t session_name

<br>&emsp;Note: you cannot scroll while in a session

to detach from session:
>^b D

to stop running code:
>^c

too much stuff in terminal, can't see!
>*right click, 'clear'*

to run notebook N.ipynb (need to include extension!)
>jupyter nbconvert --execute --to notebook N.ipynb

in English, this does:
<br>&emsp;run the notebook ('execute') -> turn the result to notebook (so that you get the notebook printout <- works for graphs like matplotlib)

<span style="color:#33FF9E">

**pov: your jupyter notebook doesn't let you use stuff from a python file (no code errors)**

</span>

This is because your notebook need to refresh to access updated version of that python file. In Visual Studio Code, this is 'reload' the python kernel

<span style="color:#33FF9E">

**pov: you run into github authentication problem (ECONNREFUSED; authentication failed)**

</span>

need to follow: 
1. generate new token (github Settings->developer settings; use the classic one)
2. (after removing the current origin that doesn't work, re-add it as follows:)
>git remote add origin https://<token>@<git_url>.git

for example, if your token is abcd, your github link says https://github.com/user/this.git, then you enter:
>git remote add origin https://abcd@github.com/user/this.git
3. do a pull: 
>git pull https://<token>@<git_url>.git

following above example, this becomes:
>git pull https://abcd@github.com/user/this.git

NOTE: this lets you catch up where you left off (when you first got stuck at the authentication error)

reference: https://stackoverflow.com/questions/68193573/git-push-returns-missing-or-invalid-credentials-code-econnrefused-remote-r
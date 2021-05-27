---
title: "Just another Crystal Report bug - NULL bug in If Else statements"
date: "2017-06-07"
categories: 
  - "crystal-report"
coverImage: "/post_images/find-all-the-bugs.jpg"
---

This is the start of a series, "Just another Crystal Report bug".

Anyone who has been using Crystal Report for at least a couple of months probably stumble upon a bug at least once or twice and as someone who has been actively using Crystal Report on a daily basis for over a year, I have a huge loads of bugs to unload on this website. Why am I doing that? I'm hoping that someone who has hit this bug can stop wasting X more hours and restore his health and sanity by reading more about this bug here and find out how they could fix it RIGHT NOW. **So if you find it helpful, share it.**

**Bug: Crystal Report does not enter the code block in the NULL criteria in your If Else statements unless it's explicitly listed as the first criteria.**

What does this means? Consider the following example.

**Example Problem:**

\[php\] if {ORDERS.LOCATION\_C}=1 then "X" else if {ORDERS.LOCATION\_C}=2 then "Y" else( //if it comes in here,ORDERS.LOCATION\_C is probably a NULL //if it's NULL, I'll look at another formula value before deciding what to return if {@locName}="Singapore" Then "X" else "Y" ) \[/php\]

These code in the Formula Editor checks that value of ORDERS.LOCATION\_C and return the appropriate value back. If ORDERS.LOCATION\_C happens to be "NULL", it looks at another formula (locName) before deciding what to return. Seems logical and reasonable isn't it?

Nope, it doesn't work. What happens is that when ORDERS.LOCATION\_C is null, Crystal Report refuses to enter into the area (line 3 to 10).

**Workaround/Solution:**

\[php\] if IsNull({ORDERS.LOCATION\_C}) Then ( if {@locName}="Singapore" Then "X" else "Y" ) else if {ORDERS.LOCATION\_C}=1 then "X" else if {ORDERS.LOCATION\_C}=2 then "Y" else "check orders" \[/php\]

Now, this is what you have to do. Explicitly check for NULL using "IsNull" before any other if else statements you have, and move the codes into the NULL criteria block.

Again, the NULL criteria only gets checked and the block of code only get entered if it's the first criteria in the whole chunk of if else statements in your Formula Editor.

If this info has fixed your problem, go grab a cup of coffee and share the joy with your Facebook friends. If not, post it in comments and I'll see if I can help you.

Till the next Crystal Report bug ;)

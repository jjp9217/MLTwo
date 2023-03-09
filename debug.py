################################################################
# Machine Learning
# Programming Assignment - Bayesian Classifier
#
# Debug utilities (A2) - now with 8 message types:
# * check, ncheck, dcheck, dncheck
# * pcheck, npcheck, dpcheck, dnpcheck
#
# Prefix 'n': newline after message
# Prefix 'd': '[DEBUG] ' before message
#
# Author: R. Zanibbi
# Author: E. Lima
################################################################

# RZ Sept - Allow strings at start + end of messages
def check(message, expression, msg_start='', msg_end='',  pause=False):
    if msg_start != '':
        print(msg_start,message, ":" + msg_end, expression)
    else:
        print(message, ":" + msg_end, expression)

    # Add newline or pause message
    if pause:
        input("Press any key to continue...\n")

def pcheck(message, expression,  msg_start='', msg_end=''):
    check(message, expression, msg_start, msg_end, pause=True)

##################################
# Print newline after ':'
# ** Useful for matrices
##################################
def ncheck(message, expression):
    check(message, expression, msg_end='\n')

def npcheck(message, expression):
    pcheck(message, expression, msg_end='\n')

##################################
# Insert debug tag at start 
##################################

def dcheck(message, expression):
    check(message, expression,  msg_start='[DEBUG]', msg_end='')

def dncheck(message, expression):
    check(message, expression, msg_start='[DEBUG]', msg_end='\n')

def dpcheck(message, expression):
    pcheck(message, expression,msg_start='[DEBUG]', msg_end='')

def dnpcheck(message, expression):
    pcheck(message, expression, msg_start='[DEBUG]', msg_end='\n')


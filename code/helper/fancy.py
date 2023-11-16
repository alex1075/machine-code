import os
import time
from os import system, name

def reset_color():
    os.system("tput sgr0")

class bcolors:
    WARNING = '\033[93m'
    ERROR = '\033[91m'
    GOOD = '\033[92m'

def clear():
    # for windows
    if name == 'nt':
        _ = system('cls')
    # for mac and linux(here, os.name is 'posix')
    else:
        _ = system('clear')

def display_banner():
    clear()
    print(r'''
 __      __          ___                                        
/\ \  __/\ \        /\_ \                                       
\ \ \/\ \ \ \     __\//\ \     ___    ___     ___ ___      __   
 \ \ \ \ \ \ \  /'__`\\ \ \   /'___\ / __`\ /' __` __`\  /'__`\ 
  \ \ \_/ \_\ \/\  __/ \_\ \_/\ \__//\ \L\ \/\ \/\ \/\ \/\  __/ 
   \ `\___x___/\ \____\/\____\ \____\ \____/\ \_\ \_\ \_\ \____\
    '\/__//__/  \/____/\/____/\/____/\/___/  \/_/\/_/\/_/\/____/
                                                                         
''') # Larry 3D 
    time.sleep(2)
    clear()
          
def selection_program():
    for x in range (0,5):  
        b = "Loading" + "." * x
        print (b, end="\r")
        time.sleep(1)

def infer_banner():
    clear()
    print(r'''
 ______             ___                                                                    
/\__  _\          /'___\                     __                                            
\/_/\ \/     ___ /\ \__/   __   _ __   _ __ /\_\    ___      __                            
   \ \ \   /' _ `\ \ ,__\/'__`\/\`'__\/\`'__\/\ \ /' _ `\  /'_ `\                          
    \_\ \__/\ \/\ \ \ \_/\  __/\ \ \/ \ \ \/ \ \ \/\ \/\ \/\ \L\ \                         
    /\_____\ \_\ \_\ \_\\ \____\\ \_\  \ \_\  \ \_\ \_\ \_\ \____ \                        
    \/_____/\/_/\/_/\/_/ \/____/ \/_/   \/_/   \/_/\/_/\/_/\/___L\ \                       
                                                             /\____/                       
                                                             \_/__/                        
                  ____                 ___                                        ___      
                 /\  _`\    __        /\_ \                   __                 /\_ \     
  ___     ___    \ \ \L\ \ /\_\    ___\//\ \     ___      __ /\_\    ___     __  \//\ \    
 / __`\ /' _ `\   \ \  _ <'\/\ \  / __`\\ \ \   / __`\  /'_ `\/\ \  /'___\ /'__`\  \ \ \   
/\ \L\ \/\ \/\ \   \ \ \L\ \\ \ \/\ \L\ \\_\ \_/\ \L\ \/\ \L\ \ \ \/\ \__//\ \L\.\_ \_\ \_ 
\ \____/\ \_\ \_\   \ \____/ \ \_\ \____//\____\ \____/\ \____ \ \_\ \____\ \__/.\_\/\____\
 \/___/  \/_/\/_/    \/___/   \/_/\/___/ \/____/\/___/  \/___L\ \/_/\/____/\/__/\/_/\/____/
                                                          /\____/                          
                                                          \_/__/                           
 ____                                 ___                                                  
/\  _`\                              /\_ \                                                 
\ \,\L\_\     __      ___ ___   _____\//\ \      __                                        
 \/_\__ \   /'__`\  /' __` __`\/\ '__`\\ \ \   /'__`\                                      
   /\ \L\ \/\ \L\.\_/\ \/\ \/\ \ \ \L\ \\_\ \_/\  __/                                      
   \ `\____\ \__/.\_\ \_\ \_\ \_\ \ ,__//\____\ \____\                                     
    \/_____/\/__/\/_/\/_/\/_/\/_/\ \ \/ \/____/\/____/                                     
                                  \ \_\                                                    
                                   \/_/                                                    
''')   
    time.sleep(2)     
    clear()

def train_banner():
    clear()
    print(r'''
 ______                                                   
/\__  _\               __          __                     
\/_/\ \/ _ __    __   /\_\    ___ /\_\    ___      __     
   \ \ \/\`'__\/'__`\ \/\ \ /' _ `\/\ \ /' _ `\  /'_ `\   
    \ \ \ \ \//\ \L\.\_\ \ \/\ \/\ \ \ \/\ \/\ \/\ \L\ \  
     \ \_\ \_\\ \__/.\_\\ \_\ \_\ \_\ \_\ \_\ \_\ \____ \ 
      \/_/\/_/ \/__/\/_/ \/_/\/_/\/_/\/_/\/_/\/_/\/___L\ \
                                                   /\____/
                                                   \_/__/ 
''' ) # Larry 3D
    time.sleep(2)

def data_processing_banner():
    clear()
    print(r''' 
 ____              __                                                     
/\  _`\           /\ \__                                                  
\ \ \/\ \     __  \ \ ,_\    __                                           
 \ \ \ \ \  /'__`\ \ \ \/  /'__`\                                         
  \ \ \_\ \/\ \L\.\_\ \ \_/\ \L\.\_                                       
   \ \____/\ \__/.\_\\ \__\ \__/.\_\                                      
    \/___/  \/__/\/_/ \/__/\/__/\/_/                                      
                                                                          
                                                                          
 ____                                                                     
/\  _`\                                            __                     
\ \ \L\ \_ __   ___     ___     __    ____    ____/\_\    ___      __     
 \ \ ,__/\`'__\/ __`\  /'___\ /'__`\ /',__\  /',__\/\ \ /' _ `\  /'_ `\   
  \ \ \/\ \ \//\ \L\ \/\ \__//\  __//\__, `\/\__, `\ \ \/\ \/\ \/\ \L\ \  
   \ \_\ \ \_\\ \____/\ \____\ \____\/\____/\/\____/\ \_\ \_\ \_\ \____ \ 
    \/_/  \/_/ \/___/  \/____/\/____/\/___/  \/___/  \/_/\/_/\/_/\/___L\ \
                                                                   /\____/
                                                                   \_/__/ 
''') # Larry 3D
    time.sleep(2)    

def banner_goodbye():
    clear()
    print(bcolors.GOOD + r'''
 ____                        __  __                        
/\  _`\                     /\ \/\ \                       
\ \ \L\_\    ___     ___    \_\ \ \ \____  __  __     __   
 \ \ \L_L   / __`\  / __`\  /'_` \ \ '__`\/\ \/\ \  /'__`\ 
  \ \ \/, \/\ \L\ \/\ \L\ \/\ \L\ \ \ \L\ \ \ \_\ \/\  __/ 
   \ \____/\ \____/\ \____/\ \___,_\ \_,__/\/`____ \ \____\
    \/___/  \/___/  \/___/  \/__,_ /\/___/  `/___/> \/____/
                                               /\___/      
                                               \/__/       
''') # Larry 3D
    time.sleep(2)
    reset_color()

def test_banner():
    clear()
    print(r'''

 ______                __                                 
/\__  _\              /\ \__  __                          
\/_/\ \/    __    ____\ \ ,_\/\_\    ___      __          
   \ \ \  /'__`\ /',__\\ \ \/\/\ \ /' _ `\  /'_ `\        
    \ \ \/\  __//\__, `\\ \ \_\ \ \/\ \/\ \/\ \L\ \       
     \ \_\ \____\/\____/ \ \__\\ \_\ \_\ \_\ \____ \      
      \/_/\/____/\/___/   \/__/ \/_/\/_/\/_/\/___L\ \     
                                              /\____/     
                                              \_/__/      
                                __          ___           
            /'\_/`\            /\ \        /\_ \          
   __      /\      \    ___    \_\ \     __\//\ \         
 /'__`\    \ \ \__\ \  / __`\  /'_` \  /'__`\\ \ \        
/\ \L\.\_   \ \ \_/\ \/\ \L\ \/\ \L\ \/\  __/ \_\ \_      
\ \__/.\_\   \ \_\\ \_\ \____/\ \___,_\ \____\/\____\     
 \/__/\/_/    \/_/ \/_/\/___/  \/__,_ /\/____/\/____/     
                                                          
                                                          
''' ) # Larry 3D
    time.sleep(2)

def beta_banner():
    clear()
    print(r'''
 
 ____            __                   ___                         __                           
/\  _`\         /\ \__              /'___\                       /\ \__  __                    
\ \ \L\ \     __\ \ ,_\    __      /\ \__/  __  __    ___     ___\ \ ,_\/\_\    ___     ___    
 \ \  _ <'  /'__`\ \ \/  /'__`\    \ \ ,__\/\ \/\ \ /' _ `\  /'___\ \ \/\/\ \  / __`\ /' _ `\  
  \ \ \L\ \/\  __/\ \ \_/\ \L\.\_   \ \ \_/\ \ \_\ \/\ \/\ \/\ \__/\ \ \_\ \ \/\ \L\ \/\ \/\ \ 
   \ \____/\ \____\\ \__\ \__/.\_\   \ \_\  \ \____/\ \_\ \_\ \____\\ \__\\ \_\ \____/\ \_\ \_\
    \/___/  \/____/ \/__/\/__/\/_/    \/_/   \/___/  \/_/\/_/\/____/ \/__/ \/_/\/___/  \/_/\/_/
                                                                                               
                                                                                               
''' ) # Larry 3D
    time.sleep(2)

def warnings_banner():
    clear()
    print(bcolors.WARNING + r'''

 __      __  ______  ____    __  __  ______   __  __  ____    __     
/\ \  __/\ \/\  _  \/\  _`\ /\ \/\ \/\__  _\ /\ \/\ \/\  _`\ /\ \    
\ \ \/\ \ \ \ \ \L\ \ \ \L\ \ \ `\\ \/_/\ \/ \ \ `\\ \ \ \L\_\ \ \   
 \ \ \ \ \ \ \ \  __ \ \ ,  /\ \ , ` \ \ \ \  \ \ , ` \ \ \L_L\ \ \  
  \ \ \_/ \_\ \ \ \/\ \ \ \\ \\ \ \`\ \ \_\ \__\ \ \`\ \ \ \/, \ \_\ 
   \ `\___x___/\ \_\ \_\ \_\ \_\ \_\ \_\/\_____\\ \_\ \_\ \____/\/\_\
    '\/__//__/  \/_/\/_/\/_/\/ /\/_/\/_/\/_____/ \/_/\/_/\/___/  \/_/
                                                                     
                                                                     
''' ) 
    time.sleep(1)
    os.system("tput sgr0")

def error_banner():
    clear()
    print(bcolors.ERROR + r'''
 
 ____    ____    ____    _____   ____    __     
/\  _`\ /\  _`\ /\  _`\ /\  __`\/\  _`\ /\ \    
\ \ \L\_\ \ \L\ \ \ \L\ \ \ \/\ \ \ \L\ \ \ \   
 \ \  _\L\ \ ,  /\ \ ,  /\ \ \ \ \ \ ,  /\ \ \  
  \ \ \L\ \ \ \\ \\ \ \\ \\ \ \_\ \ \ \\ \\ \_\ 
   \ \____/\ \_\ \_\ \_\ \_\ \_____\ \_\ \_\/\_\
    \/___/  \/_/\/ /\/_/\/ /\/_____/\/_/\/ /\/_/
                                                
                                                
''')
    time.sleep(1)
    os.system("tput sgr0")

def train_complete_banner():
    print(bcolors.GOOD + r'''

 ______                                                           
/\__  _\               __          __                             
\/_/\ \/ _ __    __   /\_\    ___ /\_\    ___      __             
   \ \ \/\`'__\/'__`\ \/\ \ /' _ `\/\ \ /' _ `\  /'_ `\           
    \ \ \ \ \//\ \L\.\_\ \ \/\ \/\ \ \ \/\ \/\ \/\ \L\ \          
     \ \_\ \_\\ \__/.\_\\ \_\ \_\ \_\ \_\ \_\ \_\ \____ \         
      \/_/\/_/ \/__/\/_/ \/_/\/_/\/_/\/_/\/_/\/_/\/___L\ \        
                                                   /\____/        
                                                   \_/__/         
 ____                                ___           __             
/\  _`\                             /\_ \         /\ \__          
\ \ \/\_\    ___     ___ ___   _____\//\ \      __\ \ ,_\    __   
 \ \ \/_/_  / __`\ /' __` __`\/\ '__`\\ \ \   /'__`\ \ \/  /'__`\ 
  \ \ \L\ \/\ \L\ \/\ \/\ \/\ \ \ \L\ \\_\ \_/\  __/\ \ \_/\  __/ 
   \ \____/\ \____/\ \_\ \_\ \_\ \ ,__//\____\ \____\\ \__\ \____\
    \/___/  \/___/  \/_/\/_/\/_/\ \ \/ \/____/\/____/ \/__/\/____/
                                 \ \_\                            
                                  \/_/                            
''')
    time.sleep(2)

def end_program():
        print('Exiting')
        banner_goodbye()
        clear()
        exit()    